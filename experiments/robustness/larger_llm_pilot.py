"""
experiments/robustness/larger_llm_pilot.py
==========================================
Pilot of the larger-LLM control for the query-side register gap. Is the
register gap (evocative LLM-generated H3 profiles losing to H2 mean pooling
even after catalog whitening) a property of LLM-generated profiles in
general, or an artefact of Llama 3.2 3B (a weak instruction-model class)?

Reuses the exact canonical H3 prompt (build_h3_prompt) and generation
protocol (same temperature, same book history, same GTE-large encoder for
embedding), changing ONLY the LLM backend to qwen2.5:14b (local Ollama,
~4.7x the parameters of the canonical llama3.2:3b, no API cost).

On a single consumer GPU (6GB, with the 14B model partially offloaded to
CPU, ~90 s/profile), generating the full canonical validation set
(n=2,731) is infeasible (~140 h). This script therefore runs a PILOT on a
fixed (seed=42) subsample of the validation set and compares
H3(qwen2.5:14b) with H3(llama3.2:3b) and H2 on EXACTLY the same users
(apples-to-apples), not on the full validation set.

Supports: the check that the register gap is not specific to a small LLM.

Prerequisite: an Ollama server with qwen2.5:14b.
Inputs: canonical H2/H3 profiles and album embeddings in data/processed/
        (none of them is modified).
Outputs (results/larger_llm_pilot/): h3_strong_qwen14b.json (generation
        cache; re-running resumes from it), user_profiles_h3_qwen14b.npy
        (+ index) and m2_pilot_results.json.

Usage:
    python -m experiments.robustness.larger_llm_pilot --n 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from acw.embeddings.profile_summarization.config import TEMPERATURE_H3  # noqa: E402
from acw.embeddings.profile_summarization.llm_client import OllamaClient  # noqa: E402
from acw.embeddings.profile_summarization.loader import (  # noqa: E402
    load_book_history,
    load_book_metadata,
)
from acw.embeddings.profile_summarization.prompts import build_h3_prompt  # noqa: E402
from acw.embeddings.profile_summarization.validation import validate_summary  # noqa: E402
from acw.evaluation.config import resolve_paths  # noqa: E402
from acw.evaluation.loader import build_genre_ground_truth, sample_test_users  # noqa: E402
from acw.evaluation.metrics import ndcg_at_k, precision_at_k  # noqa: E402
from acw.settings import ROOT  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "larger_llm_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
MAX_BOOKS_PROFILE = 10
STRONG_MODEL = "qwen2.5:14b"
K_VALUES = [5, 10, 20]


def generate_h3_strong(
    uids: list[str], book_history: dict, book_meta: dict, cache_path: Path
) -> dict:
    cache: dict[str, str | None] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    client = OllamaClient(model=STRONG_MODEL, timeout=300)
    pending = [u for u in uids if u not in cache]
    print(f"  Pending: {len(pending)}/{len(uids)} (model={STRONG_MODEL})")

    t0 = time.time()
    for i, uid in enumerate(pending):
        recent_asins = book_history.get(uid, [])[-MAX_BOOKS_PROFILE:]
        books_data = [book_meta[a] for a in recent_asins if a in book_meta]
        if not books_data:
            cache[uid] = None
            continue
        prompt = build_h3_prompt(books_data)
        try:
            output = client.generate(prompt, TEMPERATURE_H3)
            if not validate_summary(output, "h3"):
                output = None
        except Exception as exc:
            print(f"  FAILED {uid}: {exc}")
            output = None
        cache[uid] = output
        cache_path.write_text(json.dumps(cache, indent=2))
        elapsed = time.time() - t0
        eta = (len(pending) - i - 1) * elapsed / (i + 1) / 60
        print(
            f"  [{i + 1}/{len(pending)}] {uid[:20]}... elapsed={elapsed / 60:.1f}min ETA={eta:.1f}min"
        )

    return cache


def embed_profiles_local(summaries: dict[str, str | None], uids: list[str]):
    from sentence_transformers import SentenceTransformer

    print("  Loading GTE-large-en-v1.5 for embedding (cpu)...")
    sbert = SentenceTransformer(
        "Alibaba-NLP/gte-large-en-v1.5", device="cpu", trust_remote_code=True
    )
    _orig_tokenize = sbert._first_module().tokenize

    def _tokenize_no_tti(texts, **kw):
        out = _orig_tokenize(texts, **kw)
        if "token_type_ids" in out:
            out["token_type_ids"].zero_()
        return out

    sbert._first_module().tokenize = _tokenize_no_tti

    valid_uids = [u for u in uids if summaries.get(u)]
    texts = [summaries[u] for u in valid_uids]
    embs = sbert.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    idx_map = {u: i for i, u in enumerate(valid_uids)}
    return np.asarray(embs, dtype=np.float32), idx_map


def evaluate(profiles, idx_map, items_emb, idx_to_item, genre_gt, test_users, label):
    max_k = max(K_VALUES)
    metrics = {f"nDCG@{k}": [] for k in K_VALUES}
    metrics.update({f"P@{k}": [] for k in K_VALUES})
    n_eval = 0
    for uid in test_users:
        relevant = genre_gt.get(uid)
        if not relevant:
            continue
        row = idx_map.get(uid)
        if row is None:
            continue
        sims = items_emb @ profiles[row]
        top = np.argsort(sims)[::-1][:max_k]
        ranked = [idx_to_item[i] for i in top]
        for k in K_VALUES:
            metrics[f"nDCG@{k}"].append(ndcg_at_k(ranked, relevant, k))
            metrics[f"P@{k}"].append(precision_at_k(ranked, relevant, k))
        n_eval += 1
    avgs = {m: float(np.mean(v)) for m, v in metrics.items()}
    avgs["n_eval"] = n_eval
    print(f"  {label:<55}: nDCG@10={avgs['nDCG@10']:.4f}  P@10={avgs['P@10']:.4f}  (n={n_eval})")
    return avgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n", type=int, default=60, help="pilot size (subsample of the validation set)"
    )
    args = ap.parse_args()

    cfg = resolve_paths(domain="music")
    genre_gt = build_genre_ground_truth(cfg)
    val_users = sample_test_users(genre_gt, cfg)

    rng = np.random.default_rng(SEED)
    pilot_users = list(rng.choice(val_users, size=min(args.n, len(val_users)), replace=False))
    print(
        f"Larger-LLM pilot: {len(pilot_users)} users (seed=42 subsample of the val set, n={len(val_users)})"
    )

    book_history = load_book_history()
    book_meta = load_book_metadata()

    cache_path = OUT_DIR / "h3_strong_qwen14b.json"
    summaries = generate_h3_strong(pilot_users, book_history, book_meta, cache_path)

    n_valid = sum(1 for u in pilot_users if summaries.get(u))
    print(f"\nValid H3(qwen2.5:14b) profiles: {n_valid}/{len(pilot_users)}")

    profiles_strong, idx_strong = embed_profiles_local(summaries, pilot_users)
    np.save(OUT_DIR / "user_profiles_h3_qwen14b.npy", profiles_strong)
    json.dump(idx_strong, open(OUT_DIR / "user_profile_index_h3_qwen14b.json", "w"))

    # ── Evaluation: H3(qwen14b) vs H3(llama3.2:3b, canonical) vs H2, all on the SAME subsample ──
    album_id_to_idx = json.load(open(PROCESSED / "album_id_to_idx.json"))
    idx_to_album = {v: k for k, v in album_id_to_idx.items()}
    album_emb_whitened = np.load(PROCESSED / "alignment" / "album_embeddings_whitened.npy").astype(
        np.float32
    )

    h3_llama_profiles = np.load(PROCESSED / "user_profiles_h3.npy").astype(np.float32)
    h3_llama_idx = json.load(open(PROCESSED / "user_profile_index_h3.json"))

    h2_profiles = np.load(PROCESSED / "user_profiles_h2.npy").astype(np.float32)
    h2_idx = json.load(open(PROCESSED / "user_profile_index_h2.json"))
    album_emb_raw = np.load(PROCESSED / "album_embeddings.npy").astype(np.float32)

    results = {}
    print("\n[EVALUATION ON THE PILOT SUBSAMPLE]")
    results["H3_qwen14b_vs_whitened"] = evaluate(
        profiles_strong,
        idx_strong,
        album_emb_whitened,
        idx_to_album,
        genre_gt,
        pilot_users,
        "H3 (qwen2.5:14b) / WHITENED albums",
    )
    results["H3_llama3b_vs_whitened_SAME_SUBSAMPLE"] = evaluate(
        h3_llama_profiles,
        h3_llama_idx,
        album_emb_whitened,
        idx_to_album,
        genre_gt,
        pilot_users,
        "H3 (llama3.2:3b) / WHITENED albums [same subsample]",
    )
    results["H2_vs_whitened_SAME_SUBSAMPLE"] = evaluate(
        h2_profiles,
        h2_idx,
        album_emb_whitened,
        idx_to_album,
        genre_gt,
        pilot_users,
        "H2 raw / WHITENED albums [same subsample]",
    )
    results["H2_vs_raw_SAME_SUBSAMPLE"] = evaluate(
        h2_profiles,
        h2_idx,
        album_emb_raw,
        idx_to_album,
        genre_gt,
        pilot_users,
        "H2 raw / RAW albums [same subsample]",
    )

    results["pilot_n"] = len(pilot_users)
    results["pilot_users"] = pilot_users
    results["n_valid_summaries_qwen14b"] = n_valid

    out_json = OUT_DIR / "m2_pilot_results.json"
    json.dump(results, open(out_json, "w"), indent=2)
    print(f"\nResults saved to: {out_json}")


if __name__ == "__main__":
    main()
