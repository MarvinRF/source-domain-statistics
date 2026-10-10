"""Non-linear parametric whitening baseline (MLP bottleneck).

Replaces the linear ``W1`` of parametric whitening with a two-layer bottleneck
MLP, ``f(x) = normalize(GELU((x - b) @ W_down) @ W_up)`` with
``W_down`` of shape (D, 256) and ``W_up`` of shape (256, D). A linear map can
only rotate and rescale the profile space; the GELU lets the aligner learn a
non-linear mapping toward the target catalog, while the bottleneck acts as an
implicit regularizer with roughly half the parameters of the linear ``W1``.
``W_down``/``W_up`` are warm-started from the truncated SVD of a ZCA matrix so
that epoch 0 approximates the ZCA solution. Training uses the same InfoNCE
objective as :mod:`acw.baselines.parametric_whitening` (profile vs. centroid of
relevant items, in-batch negatives), with ``--input-zca-book`` applying the
books-catalog ZCA as a first stage.

Usage::

    python -m acw.baselines.nonlinear_whitening --heuristic h3
    python -m acw.baselines.nonlinear_whitening --heuristic h4 --input-zca-book
    python -m acw.baselines.nonlinear_whitening --heuristic h3 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from acw.baselines.common import (
    apply_zca_book,
    compute_centroids,
    domain_paths,
    evaluate,
    ground_truth,
)
from acw.settings import EMBEDDING_DIM as DIM

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
from acw.settings import ROOT  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
ALIGNMENT_DIR = PROCESSED / "alignment"
RESULTS_DIR = ROOT / "results" / "phase4"

ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
K_VALUES = [5, 10]
BOTTLENECK = 256
EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


class NonLinearWhitening(nn.Module):
    """
    MLP bottleneck aligner for cross-domain profile alignment.

    Architecture:
        f(x) = Norm(W_up · GELU(W_down · (x - b)))

    where:
        b      ∈ R^1024        — input bias (centering)
        W_down ∈ R^(1024, 256) — compression
        W_up   ∈ R^(256, 1024) — projection

    Total trainable params: ~525k (vs ~1.05M for linear PW)
    """

    def __init__(self, dim: int = DIM, bottleneck: int = BOTTLENECK):
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck

        self.b = nn.Parameter(torch.zeros(dim))
        self.W_down = nn.Parameter(torch.empty(dim, bottleneck))
        self.W_up = nn.Parameter(torch.empty(bottleneck, dim))

        # Default init: Xavier uniform
        nn.init.xavier_uniform_(self.W_down)
        nn.init.xavier_uniform_(self.W_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, dim) float32, L2-normalized input embeddings
        Returns:
            (B, dim) float32, L2-normalized aligned embeddings
        """
        h = (x - self.b) @ self.W_down  # (B, bottleneck)
        h = F.gelu(h)  # non-linearity
        out = h @ self.W_up  # (B, dim)
        return F.normalize(out, dim=-1)

    def warmstart_from_zca(self, zca_mean: np.ndarray, zca_W: np.ndarray):
        """
        Initialize from ZCA W1 matrix via SVD.
        W1 ≈ U · S · V^T
        W_down = V[:, :bottleneck]^T  (top-k right singular vectors)
        W_up   = U[:, :bottleneck] · diag(S[:bottleneck])
        """
        log.info(f"  Computing SVD of ZCA W1 ({zca_W.shape})...")
        U, S, Vt = np.linalg.svd(zca_W.astype(np.float64), full_matrices=False)
        k = self.bottleneck

        W_down_init = Vt[:k, :].T.astype(np.float32)  # (1024, k)
        W_up_init = (U[:, :k] * S[:k]).astype(np.float32)  # (1024, k) → transposed for W_up
        # W_up shape is (k, 1024) — transpose
        W_up_init = W_up_init.T  # (k, 1024)

        with torch.no_grad():
            self.b.copy_(torch.from_numpy(zca_mean.astype(np.float32)))
            self.W_down.copy_(torch.from_numpy(W_down_init))
            self.W_up.copy_(torch.from_numpy(W_up_init))

        explained = S[:k].sum() / S.sum()
        log.info(f"  SVD warm-start: top-{k} components explain {explained:.1%} of variance ✓")

    def apply_numpy(self, X: np.ndarray) -> np.ndarray:
        """Apply transformation to numpy array (no grad)."""
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).to(device)
            out = self.forward(x)
        return out.cpu().numpy()

    def save_params(self, prefix: str):
        """Save W_down, W_up, b to alignment dir."""
        np.save(
            ALIGNMENT_DIR / f"{prefix}_down_best.npy",
            self.W_down.detach().cpu().numpy(),
        )
        np.save(ALIGNMENT_DIR / f"{prefix}_up_best.npy", self.W_up.detach().cpu().numpy())
        np.save(ALIGNMENT_DIR / f"{prefix}_bias_best.npy", self.b.detach().cpu().numpy())

    def load_params(self, prefix: str):
        """Load saved params."""
        with torch.no_grad():
            self.W_down.copy_(torch.from_numpy(np.load(ALIGNMENT_DIR / f"{prefix}_down_best.npy")))
            self.W_up.copy_(torch.from_numpy(np.load(ALIGNMENT_DIR / f"{prefix}_up_best.npy")))
            self.b.copy_(torch.from_numpy(np.load(ALIGNMENT_DIR / f"{prefix}_bias_best.npy")))


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, profiles_w: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        B = profiles_w.shape[0]
        sim = profiles_w @ centroids.T / self.temperature
        labels = torch.arange(B, device=profiles_w.device)
        return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry diagnostic
# ─────────────────────────────────────────────────────────────────────────────


def geometry_diagnostic(profiles: np.ndarray, label: str, n: int = 500):
    """Report avg_sim to track collapse during training."""
    sample = profiles[:n]
    s = sample @ sample.T
    np.fill_diagonal(s, 0)
    log.info(f"  {label} geometry: avg_sim={s.mean():.4f}  std={s.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def train(
    heuristic: str = "h3",
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.07,
    weight_decay: float = 1e-4,
    warmstart: bool = True,
    eval_every: int = 20,
    dry_run: bool = False,
    device_str: str = "auto",
    bottleneck: int = BOTTLENECK,
    input_zca_book: bool = False,
    domain: str = "music",
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    assert heuristic in ("h3", "h4"), f"heuristic must be h3 or h4, got {heuristic}"

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )

    H = heuristic.upper()
    _epochs = 10 if dry_run else epochs

    log.info("=" * 70)
    log.info("Non-Linear Parametric Whitening")
    log.info("=" * 70)
    log.info(f"  Device        : {device}")
    log.info(f"  Domain        : {domain}")
    log.info(f"  Heuristic     : {H}")
    log.info(f"  Bottleneck    : {bottleneck}")
    log.info(f"  Epochs        : {_epochs}")
    log.info(f"  Batch size    : {batch_size}")
    log.info(f"  LR            : {lr}")
    log.info(f"  Temperature τ : {temperature}")
    log.info(f"  Warm start    : {warmstart}")
    log.info(f"  Input ZCA-book: {input_zca_book}")
    log.info(f"  Dry run       : {dry_run}")

    # ── Domain paths ──────────────────────────────────────────────────────
    defaults = domain_paths(domain)
    _items_emb_path = defaults["items_emb_path"]
    _items_id_map = defaults["items_id_map_path"]
    _gt_path = defaults["gt_path"]
    _items_jsonl = defaults["items_jsonl_path"]
    _domain_dir = defaults["domain_dir"]

    domain_tag = "" if domain == "music" else f"_{domain}"
    if input_zca_book:
        artifact_prefix = f"nlpw_zb_{heuristic}{domain_tag}"
        profile_suffix = f"_nlpw_zb{domain_tag}"
        csv_suffix = f"_zb{domain_tag}"
    else:
        artifact_prefix = f"nlpw_{heuristic}{domain_tag}"
        profile_suffix = f"_nlpw{domain_tag}"
        csv_suffix = domain_tag

    # ── Load data ─────────────────────────────────────────────────────────
    log.info("\nLoading data...")

    train_profiles_raw = np.load(PROCESSED / f"user_profiles_{heuristic}.npy")

    if input_zca_book:
        log.info("  Applying ZCA-book pre-transform...")
        train_profiles = apply_zca_book(train_profiles_raw)
        log.info(f"  Profiles shape after zca_book: {train_profiles.shape}")
    else:
        train_profiles = train_profiles_raw

    album_emb_w = np.load(_items_emb_path)

    with open(PROCESSED / f"user_profile_index_{heuristic}.json") as f:
        prof_idx = json.load(f)
    with open(_items_id_map) as f:
        album_id_to_idx = json.load(f)
    idx_to_album = {v: k for k, v in album_id_to_idx.items()}

    genre_gt = ground_truth(
        domain, items_id_map_path=_items_id_map, gt_path=_gt_path, items_jsonl_path=_items_jsonl
    )
    ordered_users = sorted(prof_idx.keys(), key=lambda u: prof_idx[u])
    log.info(f"  Genre GT users: {len(genre_gt):,}")

    # ── Canonical leakage-free split ──────────────────────────────────────
    canonical_train_path = PROCESSED / "canonical_train_users.json"
    canonical_val_path = PROCESSED / "canonical_val_users.json"

    if canonical_train_path.exists() and canonical_val_path.exists():
        with open(canonical_train_path) as _f:
            _train_set = set(json.load(_f))
        with open(canonical_val_path) as _f:
            _val_list = json.load(_f)
        ordered_users = [u for u in ordered_users if u in _train_set]
        _val_users_file = [u for u in _val_list if u in prof_idx]
        log.info(f"  Canonical split: {len(ordered_users):,} train / {len(_val_users_file):,} val")
        _use_file_split = True
    else:
        canonical_test_path = PROCESSED / "canonical_test_users.json"
        if canonical_test_path.exists():
            with open(canonical_test_path) as _f:
                _test_users = set(json.load(_f))
            ordered_users = [u for u in ordered_users if u not in _test_users]
            log.info(
                f"  Excluded {len(_test_users):,} canonical test users → {len(ordered_users):,} training pool"
            )
        else:
            log.warning("  canonical_test_users.json not found — potential leakage!")
        _use_file_split = False

    # Geometry diagnostic before training
    geometry_diagnostic(train_profiles, f"{H} profiles (input)")
    album_sample = album_emb_w[:500]
    s = album_sample @ album_sample.T
    np.fill_diagonal(s, 0)
    log.info(f"  Album space geometry: avg_sim={s.mean():.4f}  (target)")

    # ── Centroids (training pool only) ────────────────────────────────────
    log.info("\nComputing supervision centroids...")
    centroids = compute_centroids(album_emb_w, album_id_to_idx, genre_gt, ordered_users)
    valid_mask = centroids.any(axis=1)
    valid_idxs = np.where(valid_mask)[0]
    log.info(f"  Valid users for training: {len(valid_idxs):,} of {len(ordered_users):,}")

    # ── Train/val split ───────────────────────────────────────────────────
    if _use_file_split:
        # Canonical split: every valid_idx is a train user; val users come from the file
        train_idxs = np.arange(len(valid_idxs))
        h_train = train_profiles[valid_idxs[train_idxs]]
        cent_train = centroids[valid_idxs[train_idxs]]

        val_users = _val_users_file
        val_idx_map = {u: i for i, u in enumerate(val_users)}
        val_profiles = train_profiles[np.array([prof_idx[u] for u in val_users])]
        val_idxs = np.arange(len(val_users))  # only used for len() below
    else:
        # Fallback: seeded 80/20 split of the valid training pool
        rng = np.random.default_rng(SEED)
        all_idxs = np.arange(len(valid_idxs))
        rng.shuffle(all_idxs)
        n_val = max(int(len(all_idxs) * 0.2), 1)
        val_idxs = all_idxs[:n_val]
        train_idxs = all_idxs[n_val:]

        h_train = train_profiles[valid_idxs[train_idxs]]
        cent_train = centroids[valid_idxs[train_idxs]]

        val_users = [ordered_users[valid_idxs[i]] for i in val_idxs]
        val_idx_map = {u: i for i, u in enumerate(val_users)}
        val_profiles = train_profiles[valid_idxs[val_idxs]]

    log.info(f"  Train users: {len(train_idxs):,}  |  Val users: {len(val_idxs):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = NonLinearWhitening(dim=DIM, bottleneck=bottleneck).to(device)

    if warmstart:
        if input_zca_book:
            # Warm start from a ZCA fitted on the ZCA-book-transformed profiles,
            # so epoch 0 matches ZCA in the new input space
            log.info("  Warm-start: computing ZCA over zca_book-transformed profiles...")
            mean_zb = train_profiles.mean(axis=0)
            Xc = train_profiles.astype(np.float64) - mean_zb
            cov = np.cov(Xc.T)
            evals, evecs = np.linalg.eigh(cov)
            idx_sort = np.argsort(evals)[::-1]
            evals, evecs = evals[idx_sort], evecs[:, idx_sort]
            W_zb = evecs @ np.diag(1.0 / np.sqrt(evals + EPS)) @ evecs.T
            model.warmstart_from_zca(mean_zb, W_zb.astype(np.float32))
        else:
            zca_path = ALIGNMENT_DIR / f"whitening_profile_{heuristic}.npz"
            if zca_path.exists():
                data = np.load(zca_path)
                model.warmstart_from_zca(data["mean"], data["W"])
            else:
                book_zca_path = ALIGNMENT_DIR / "whitening_book.npz"
                if book_zca_path.exists():
                    data = np.load(book_zca_path)
                    model.warmstart_from_zca(data["mean"], data["W"])
                    log.info("  Used book ZCA as warm-start fallback")
                else:
                    log.warning("  No ZCA params found — using Xavier init")

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Trainable params: {n_params:,}  ({n_params / 1e6:.2f}M)")

    criterion = InfoNCELoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_epochs, eta_min=lr * 0.01
    )

    # ── Baselines ─────────────────────────────────────────────────────────
    log.info(f"\nBaseline ({H} + Linear PW):")
    pw_profiles_path = ALIGNMENT_DIR / f"user_profiles_{heuristic}_pw.npy"
    if pw_profiles_path.exists():
        pw_prof = np.load(pw_profiles_path)
        evaluate(f"{H} + Linear PW", pw_prof, album_emb_w, prof_idx, idx_to_album, genre_gt)
    else:
        log.info("  Linear PW profiles not found — skipping baseline")

    log.info(f"\nBaseline ({H} + NL-PW, before training):")
    with torch.no_grad():
        nlpw_init = model.apply_numpy(val_profiles)
    evaluate(
        f"{H} + NL-PW (epoch 0)",
        nlpw_init,
        album_emb_w,
        val_idx_map,
        idx_to_album,
        genre_gt,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    log.info(f"\nTraining for {_epochs} epochs...")
    t_start = time.time()
    best_ndcg = 0.0
    best_epoch = 0

    prof_tensor = torch.from_numpy(h_train).to(device)
    cent_tensor = torch.from_numpy(cent_train).to(device)
    N = len(train_idxs)

    for epoch in range(1, _epochs + 1):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx_batch = perm[start : start + batch_size]
            if len(idx_batch) < 2:
                continue

            x_batch = prof_tensor[idx_batch]
            c_batch = cent_tensor[idx_batch]

            profiles_w = model(x_batch)
            loss = criterion(profiles_w, c_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = (time.time() - t_start) / 60

        log.info(
            f"  Epoch {epoch:>4}/{_epochs}  "
            f"loss={avg_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"elapsed={elapsed:.1f}min"
        )

        if epoch % eval_every == 0 or epoch == _epochs:
            log.info(f"\n  → Evaluation at epoch {epoch} (val set, n={len(val_idxs)}):")
            with torch.no_grad():
                val_profiles_w = model.apply_numpy(val_profiles)

            # Geometry diagnostic on val set
            s = val_profiles_w @ val_profiles_w.T
            np.fill_diagonal(s, 0)
            log.info(
                f"  {H}+NL-PW (epoch {epoch}) geometry: avg_sim={s.mean():.4f}  std={s.std():.4f}"
            )

            res = evaluate(
                f"{H} + NL-PW (epoch {epoch})",
                val_profiles_w,
                album_emb_w,
                val_idx_map,
                idx_to_album,
                genre_gt,
            )
            ndcg10 = res.get("nDCG@10", 0)

            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                best_epoch = epoch
                if not dry_run:
                    model.save_params(artifact_prefix)
                    log.info(f"  ✓ Best saved (nDCG@10={best_ndcg:.4f})")
            log.info("")

    # ── Load best and apply to all heuristics ────────────────────────────
    log.info("=" * 70)
    log.info(f"Best epoch: {best_epoch}  ({H} nDCG@10={best_ndcg:.4f})")

    if not dry_run and best_epoch > 0:
        model.load_params(artifact_prefix)

    log.info("\nEvaluating on val set + saving full npy for pipeline integrity...")
    results = {}

    val_user_set = set(val_users)
    stage_label = "zca_book → NL-PW" if input_zca_book else "NL-PW"

    for h in ["h2", "h3", "h4"]:
        npy_path = PROCESSED / f"user_profiles_{h}.npy"
        index_path = PROCESSED / f"user_profile_index_{h}.json"
        if not npy_path.exists():
            continue

        profiles_full = np.load(npy_path)
        with open(index_path) as f:
            idx_map_full = json.load(f)

        # Apply the books-catalog ZCA to every heuristic before NL-PW
        if input_zca_book:
            profiles_in = apply_zca_book(profiles_full)
        else:
            profiles_in = profiles_full

        # ── 1. Strict evaluation on the val set ────────────────────────────
        common = [(u, idx_map_full[u]) for u in val_user_set if u in idx_map_full]
        if not common:
            log.info(f"  {h.upper()}: no val users found, skipping")
            continue

        common.sort(key=lambda x: x[1])
        uids_h = [u for u, _ in common]
        row_idxs = [idx for _, idx in common]

        val_profiles_h = profiles_in[row_idxs]
        val_idx_map_h = {u: i for i, u in enumerate(uids_h)}

        with torch.no_grad():
            profiles_nlpw_val = model.apply_numpy(val_profiles_h)

        res = evaluate(
            f"{h.upper()} + {stage_label} (val only, n={len(uids_h)})",
            profiles_nlpw_val,
            album_emb_w,
            val_idx_map_h,
            idx_to_album,
            genre_gt,
        )
        results[h] = res

        # ── 2. Save the full-profile .npy for downstream use ───────────────
        # Save only for the heuristic the model was trained on; saving for the
        # others would overwrite their own canonical profiles with this model.
        if not dry_run and h == heuristic:
            with torch.no_grad():
                profiles_nlpw_full = model.apply_numpy(profiles_in)
            out_path = _domain_dir / f"user_profiles_{h}{profile_suffix}.npy"
            np.save(out_path, profiles_nlpw_full)
            log.info(f"  Saved full {h}{profile_suffix}.npy ({len(profiles_full)} users)")

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    phase_label = "NL-PW + ZCA-BOOK" if input_zca_book else "NL-PW"
    log.info(f"{phase_label} SUMMARY — Non-Linear Parametric Whitening")
    log.info("=" * 70)

    log.info(f"\n  {'Method':<50} {'nDCG@10':>10}")
    log.info("  " + "─" * 62)
    refs = [
        ("H2 (GTE, baseline)", 0.2595),
        ("H3 + PW-ZB", 0.1933),
        ("H3 + Linear PW", 0.1443),
        ("H4 + PW-ZB", 0.1632),
        ("H4 + Linear PW", 0.1510),
    ]
    for label, val in refs:
        log.info(f"  {label:<50} {val:>10.4f}  (reference)")
    log.info("  " + "─" * 62)
    for h, res in results.items():
        ndcg = res.get("nDCG@10", 0)
        label = f"{h.upper()} + {stage_label} (best, epoch {best_epoch})"
        ref = {
            "h3": 0.1933 if input_zca_book else 0.1443,
            "h4": 0.1632 if input_zca_book else 0.1510,
            "h2": 0.2595,
        }
        flag = "↑ NEW BEST" if ndcg > ref.get(h, 0) else ""
        log.info(f"  {label:<50} {ndcg:>10.4f}  {flag}")

    if not dry_run:
        rows = [
            {
                "method": f"{h}{profile_suffix}",
                "best_epoch": best_epoch,
                "bottleneck": bottleneck,
                "input_zca_book": input_zca_book,
                **res,
            }
            for h, res in results.items()
        ]
        csv_name = f"nonlinear_whitening{csv_suffix}_evaluation.csv"
        csv_path = RESULTS_DIR / csv_name
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"\n  Results saved: {csv_path}")

    log.info("\n" + "=" * 70)
    log.info("DONE")
    log.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Apply only
# ─────────────────────────────────────────────────────────────────────────────


def apply_only(
    heuristic: str = "h3",
    device_str: str = "auto",
    bottleneck: int = BOTTLENECK,
    input_zca_book: bool = False,
):
    """
    Re-apply a saved model and write the full profile .npy files.

    Reports no metrics on purpose: valid evaluation only happens during
    training on the held-out val users; scoring these outputs would leak.
    """
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )

    prefix = f"nlpw_zb_{heuristic}" if input_zca_book else f"nlpw_{heuristic}"
    suffix = "_nlpw_zb" if input_zca_book else "_nlpw"
    model = NonLinearWhitening(dim=DIM, bottleneck=bottleneck).to(device)
    model.load_params(prefix)
    log.info(f"Loaded NL-PW params: {prefix}")

    for h in ["h2", "h3", "h4"]:
        npy_path = PROCESSED / f"user_profiles_{h}.npy"
        if not npy_path.exists():
            continue
        profiles_raw = np.load(npy_path)
        profiles_in = apply_zca_book(profiles_raw) if input_zca_book else profiles_raw
        with torch.no_grad():
            profiles_out = model.apply_numpy(profiles_in)
        out = ALIGNMENT_DIR / f"user_profiles_{h}{suffix}.npy"
        np.save(out, profiles_out)
        log.info(f"  Saved {out.name} ({len(profiles_out)} users)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Non-Linear Parametric Whitening (MLP bottleneck)")
    p.add_argument("--heuristic", type=str, default="h3", choices=["h3", "h4"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--bottleneck", type=int, default=BOTTLENECK)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--no-warmstart", action="store_true")
    p.add_argument("--apply-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--input-zca-book",
        action="store_true",
        help=(
            "Apply the books-catalog ZCA (fit on the book embeddings) as a first "
            "stage before NL-PW; it gives a less collapsed input geometry than "
            "the profile ZCA. Writes separate artifacts (nlpw_zb_*) and does "
            "not overwrite the plain NL-PW results."
        ),
    )
    p.add_argument(
        "--domain",
        default="music",
        choices=["music", "movies", "games"],
        help="Target domain (default: music)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.apply_only:
        apply_only(
            heuristic=args.heuristic,
            device_str=args.device,
            bottleneck=args.bottleneck,
            input_zca_book=args.input_zca_book,
        )
    else:
        train(
            heuristic=args.heuristic,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            weight_decay=1e-4,
            warmstart=not args.no_warmstart,
            eval_every=args.eval_every,
            dry_run=args.dry_run,
            device_str=args.device,
            bottleneck=args.bottleneck,
            input_zca_book=args.input_zca_book,
            domain=args.domain,
        )
