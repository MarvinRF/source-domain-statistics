"""Robust GTE-large encoder that bypasses the model's RoPE position buffers.

Background: with the pinned remote code of Alibaba-NLP/gte-large-en-v1.5
(``trust_remote_code=True``) and recent transformers/torch releases, the
non-persistent RoPE buffers of the embedding module (``position_ids``,
``cos_cached``, ``sin_cached``) can be left in an invalid state after loading:
either allocated with fewer positions than ``max_position_embeddings`` or
filled with uninitialized memory. The model's rotary module only recomputes
its cos/sin cache when the input is longer than the cached length, so a
buffer of the right size but garbage content is never repaired. The result
is silently corrupted embeddings (no exception) for virtually every real
input text. Re-populating the buffers in place, without resizing them, is
not sufficient.

Fix: load the model with ``transformers.AutoModel``, explicitly recompute
the RoPE cache and ``position_ids`` buffer after loading, and pass
``position_ids=torch.arange(seq_len)`` on every forward call, so the rotary
embeddings are indexed from the supplied positions rather than the internal
buffer. Pooling (CLS token) and L2 normalization replicate the model's
standard sentence-transformers configuration. The encoder was verified to
be deterministic across runs and to separate unrelated topics sensibly.

Usage:
    encoder = GTEEncoder(device="cuda")
    vectors = encoder.encode(texts, batch_size=32)  # (len(texts), 1024)
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
EMBEDDING_DIM = 1024
MAX_SEQ_LEN = 512


class GTEEncoder:
    """Minimal drop-in replacement for SentenceTransformer.encode() for
    GTE-large that avoids the RoPE position-buffer issue described above.

    Compatible with the call used in embedder.py: ``.encode(texts,
    batch_size=32, show_progress_bar=..., normalize_embeddings=True,
    convert_to_numpy=True)`` -> np.ndarray of shape (len(texts), 1024).
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.float16 if device == "cuda" else torch.float32
        # low_cpu_mem_usage=False makes the buffer issue less frequent but
        # does not eliminate it; _force_reinit_rope_buffers() is the actual fix.
        self.model = (
            AutoModel.from_pretrained(
                model_name, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=False
            )
            .to(device)
            .eval()
        )
        self._force_reinit_rope_buffers()
        self._assert_no_nan()

    def _force_reinit_rope_buffers(self) -> None:
        """Explicitly recompute the model's non-persistent RoPE buffers.

        Root cause: the remote ``RotaryEmbedding`` class computes
        ``cos_cached``/``sin_cached`` in ``__init__`` via
        ``_set_cos_sin_cache``, but ``forward`` only recomputes them when
        ``seq_len > self.max_seq_len_cached``. If the buffer has the expected
        size but its content is uninitialized memory, it is never repaired.
        This happens reproducibly when the process has allocated and freed a
        large amount of heap memory before the model is created (e.g. a
        ``json.loads`` of a file of 10 MB or more), because the uninitialized
        buffers then receive leftover bytes instead of zeros.

        The fix calls ``_set_cos_sin_cache`` (and recreates the
        ``position_ids`` buffer) on every submodule that exposes it, so the
        buffers are correct regardless of the process's memory state.
        """
        n_fixed = 0
        for module in self.model.modules():
            if hasattr(module, "_set_cos_sin_cache") and hasattr(module, "inv_freq"):
                module._set_cos_sin_cache(
                    seq_len=max(module.max_seq_len_cached, MAX_SEQ_LEN),
                    device=module.inv_freq.device,
                    dtype=module.cos_cached.dtype,
                )
                n_fixed += 1
            if hasattr(module, "position_ids") and module.__class__.__name__ == "NewEmbeddings":
                module.register_buffer(
                    "position_ids",
                    torch.arange(MAX_SEQ_LEN, device=self.device),
                    persistent=False,
                )
        if n_fixed == 0:
            log.warning(
                "GTEEncoder: no RotaryEmbedding module found to "
                "reinitialize; the remote code structure may have changed. "
                "Re-check the position_ids issue manually."
            )

    def _assert_no_nan(self) -> None:
        """Check that the buffer fix worked before using the encoder
        (defense in depth; not expected to trigger in practice)."""
        out = self._embed_batch_raw(["warmup probe sentence for rope buffer check"])
        if np.isnan(out).any():
            raise RuntimeError(
                "GTEEncoder: RoPE buffers still produce NaN after "
                "_force_reinit_rope_buffers(). This is a different issue; "
                "do not trust this encoder without further investigation."
            )

    def _embed_batch_raw(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt"
        ).to(self.device)
        seq_len = batch["input_ids"].shape[1]
        bsz = batch["input_ids"].shape[0]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(bsz, -1)
        with torch.no_grad():
            out = self.model(**batch, position_ids=position_ids)
        # gte-large-en-v1.5's own 1_Pooling/config.json specifies CLS-token
        # pooling (pooling_mode_cls_token=true, pooling_mode_mean_tokens=
        # false), not mean pooling. Mean pooling here would produce embeddings
        # inconsistent with the standard SentenceTransformer wrapper.
        pooled = out.last_hidden_state[:, 0, :]
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.float().cpu().numpy().astype(np.float32)

    def _embed_batch(self, texts: list[str], max_retries: int = 5) -> np.ndarray:
        for attempt in range(max_retries):
            out = self._embed_batch_raw(texts)
            if not np.isnan(out).any():
                return out
            log.warning(
                f"GTEEncoder: NaN detected in batch output (attempt {attempt + 1}/"
                f"{max_retries}) — non-deterministic RoPE buffer issue, retrying."
            )
        raise RuntimeError(
            f"GTEEncoder: batch still produces NaN after {max_retries} retries. "
            "Refusing to silently save corrupted embeddings."
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,  # kept for API compatibility; always normalizes
        convert_to_numpy: bool = True,  # likewise; always returns numpy
        **_ignored,
    ) -> np.ndarray:
        del normalize_embeddings, convert_to_numpy  # always True in this implementation
        n = len(texts)
        out = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
        iterator = range(0, n, batch_size)
        if show_progress_bar:
            try:
                from tqdm import tqdm

                iterator = tqdm(list(iterator), desc="GTEEncoder.encode")
            except ImportError:
                pass
        for i in iterator:
            chunk = texts[i : i + batch_size]
            out[i : i + len(chunk)] = self._embed_batch(chunk)
        return out
