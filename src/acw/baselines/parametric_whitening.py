"""Parametric whitening baseline for cross-domain profile alignment.

Learns a whitening transform ``x_out = normalize((x - b) @ W1)`` over
book-derived user profiles, in the spirit of UniSRec (Hou et al., KDD 2022):
instead of deriving ``W1`` from the profile covariance (ZCA), ``b`` and ``W1``
are trained end-to-end with an InfoNCE loss that pulls each whitened profile
toward the L2-normalized centroid of the user's relevant target-domain items
(in-batch negatives, temperature 0.07). Parameters are warm-started from the
ZCA solution so training only fine-tunes an already reasonable geometry.
Optional variants apply the books-catalog ZCA before ``W1``
(``--input-zca-book``) and add the Wang & Isola (2020) uniformity term
(``--uniformity``). The best epoch is selected on the validation users.

Usage::

    python -m acw.baselines.parametric_whitening --heuristic h3
    python -m acw.baselines.parametric_whitening --heuristic h4 --input-zca-book
    python -m acw.baselines.parametric_whitening --domain movies --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

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
EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Parametric Whitening Module
# ─────────────────────────────────────────────────────────────────────────────


class ParametricWhitening(nn.Module):
    """
    Trainable whitening transformation inspired by UniSRec (Hou et al., KDD 2022).

    Transforms embeddings as:
        x_out = F.normalize((x - b) @ W1, dim=-1)

    where b and W1 are learnable parameters.

    Key differences from ZCA:
    - ZCA: W derived analytically from covariance; fixed after fitting
    - Parametric: W learned end-to-end via retrieval loss; task-specific

    Warm-start from ZCA allows convergence in few epochs without
    sacrificing the geometric properties ZCA already achieved.
    """

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(dim))
        self.W1 = nn.Parameter(torch.eye(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, dim) float32, L2-normalized input embeddings
        Returns:
            (batch, dim) float32, L2-normalized whitened embeddings
        """
        transformed = (x - self.b) @ self.W1
        return F.normalize(transformed, dim=-1)

    def warmstart_from_zca(self, zca_mean: np.ndarray, zca_W: np.ndarray):
        """Initialize ``b`` and ``W1`` from precomputed ZCA parameters."""
        with torch.no_grad():
            self.b.copy_(torch.from_numpy(zca_mean.astype(np.float32)))
            self.W1.copy_(torch.from_numpy(zca_W.astype(np.float32)))
        log.info("  Warm-started from ZCA parameters ✓")

    def get_numpy_params(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.b.detach().cpu().numpy(),
            self.W1.detach().cpu().numpy(),
        )

    def apply_numpy(self, X: np.ndarray) -> np.ndarray:
        """Apply whitening to numpy array (inference, no grad)."""
        b = self.b.detach().cpu().numpy().astype(np.float64)
        W1 = self.W1.detach().cpu().numpy().astype(np.float64)
        Xw = (X.astype(np.float64) - b) @ W1
        norms = np.linalg.norm(Xw, axis=1, keepdims=True)
        return (Xw / np.maximum(norms, EPS)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# InfoNCE Loss
# ─────────────────────────────────────────────────────────────────────────────


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for profile → album centroid alignment.

    For each user i in the batch:
        positive: centroid of user i's relevant albums (ZCA-whitened)
        negatives: centroids of all other users j≠i in the batch

    This formulation directly optimizes for retrieval:
        the whitened profile should be closest to its own relevant albums.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        profiles_w: torch.Tensor,  # (B, dim) whitened profiles
        centroids: torch.Tensor,  # (B, dim) L2-normalized album centroids
    ) -> torch.Tensor:
        B = profiles_w.shape[0]

        # Similarity matrix: profiles × centroids  (B, B)
        sim = profiles_w @ centroids.T / self.temperature

        # Labels: diagonal = positive pair
        labels = torch.arange(B, device=profiles_w.device)

        # Asymmetric: profile should find its centroid
        loss = F.cross_entropy(sim, labels)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Uniformity Loss (Wang & Isola, 2020)
# ─────────────────────────────────────────────────────────────────────────────


class UniformityLoss(nn.Module):
    """
    Uniformity loss from Wang & Isola (2020) "Understanding Contrastive
    Representation Learning through Alignment and Uniformity on the Hypersphere".

    L_unif = log E_{i,j}[exp(-t * ||f_i - f_j||²)]

    For L2-normalized vectors: ||f_i - f_j||² = 2 - 2*(f_i · f_j)
    So:  exp(-t * ||f_i - f_j||²) = exp(-2t * (1 - f_i·f_j))

    Intuition: pulls the distribution of transformed profiles toward a
    uniform distribution on the hypersphere, preventing representational
    collapse (where all profiles cluster in a narrow cone).

    Wang & Isola show that alignment + uniformity at λ=1 is geometrically
    natural — equal weighting is a principled prior, not a tuned hyperparameter.

    t=2 is the standard value from the original paper.
    """

    def __init__(self, t: float = 2.0):
        super().__init__()
        self.t = t

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, dim), assumed L2-normalized
        # sq_dists[i,j] = ||f_i - f_j||² = 2 - 2*(f_i·f_j)
        sq_dists = 2.0 - 2.0 * (features @ features.T)  # (B, B)
        return sq_dists.mul(-self.t).exp().mean().log()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def train(
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.07,
    weight_decay: float = 1e-5,
    warmstart: bool = True,
    eval_every: int = 20,
    dry_run: bool = False,
    device_str: str = "auto",
    heuristic: str = "h3",
    input_zca_book: bool = False,  # apply the books-catalog ZCA before W1
    lambda_unif: float = 0.0,  # Wang & Isola uniformity weight (0 = disabled, 1 = equal weighting)
    domain: str = "music",  # Target domain: music | movies | games
    items_emb_path: Path | None = None,  # Override: whitened item embeddings (.npy)
    items_id_map_path: Path | None = None,  # Override: item id → row index (.json)
    gt_path: Path | None = None,  # Override: ground truth (.json)
    items_jsonl_path: Path | None = None,  # Override: item metadata (.jsonl)
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    assert heuristic in ("h3", "h4"), f"heuristic must be h3 or h4, got {heuristic}"
    assert domain in ("music", "movies", "games"), (
        f"domain must be music|movies|games, got {domain}"
    )

    # ── Resolve domain-specific paths ─────────────────────────────────────
    defaults = domain_paths(domain)
    _items_emb_path = Path(items_emb_path) if items_emb_path else defaults["items_emb_path"]
    _items_id_map_path = (
        Path(items_id_map_path) if items_id_map_path else defaults["items_id_map_path"]
    )
    _gt_path = Path(gt_path) if gt_path else defaults["gt_path"]
    _items_jsonl_path = Path(items_jsonl_path) if items_jsonl_path else defaults["items_jsonl_path"]
    _domain_dir = defaults["domain_dir"]

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )

    log.info("=" * 70)
    log.info("Parametric Whitening (UniSRec-inspired)")
    log.info("=" * 70)
    log.info(f"  Device        : {device}")
    log.info(f"  Domain        : {domain}")
    log.info(f"  Heuristic     : {heuristic.upper()}")
    log.info(f"  Epochs        : {epochs}")
    log.info(f"  Batch size    : {batch_size}")
    log.info(f"  LR            : {lr}")
    log.info(f"  Temperature τ : {temperature}")
    log.info(f"  Warm start    : {warmstart}")
    log.info(f"  Input ZCA-book: {input_zca_book}")
    log.info(f"  λ_uniformity  : {lambda_unif}  (0=InfoNCE only; 1=Wang & Isola equal weighting)")
    log.info(f"  Dry run       : {dry_run}")
    log.info(f"  Items emb     : {_items_emb_path}")
    log.info(f"  GT path       : {_gt_path}")

    # Artifact naming keeps the ZCA-book, uniformity and per-domain variants apart.
    domain_tag = "" if domain == "music" else f"_{domain}"
    # A non-default items_emb_path gets its own tag so it can never overwrite the
    # canonical artifacts trained against the whitened target catalog.
    raw_target_tag = "_rawtarget" if items_emb_path else ""
    if input_zca_book and lambda_unif > 0:
        artifact_prefix = f"pw_zb_unif_{heuristic}{domain_tag}{raw_target_tag}"
        profile_suffix = f"_pw_zb_unif{domain_tag}{raw_target_tag}"
        csv_suffix = f"_zb_unif{domain_tag}{raw_target_tag}"
    elif input_zca_book:
        artifact_prefix = f"pw_zb_{heuristic}{domain_tag}{raw_target_tag}"
        profile_suffix = f"_pw_zb{domain_tag}{raw_target_tag}"
        csv_suffix = f"_zb{domain_tag}{raw_target_tag}"
    elif lambda_unif > 0:
        artifact_prefix = f"pw_unif_{heuristic}{domain_tag}{raw_target_tag}"
        profile_suffix = f"_pw_unif{domain_tag}{raw_target_tag}"
        csv_suffix = f"_unif{domain_tag}{raw_target_tag}"
    else:
        artifact_prefix = f"pw_{heuristic}{domain_tag}{raw_target_tag}"
        profile_suffix = f"_pw{domain_tag}{raw_target_tag}"
        csv_suffix = f"{domain_tag}{raw_target_tag}"

    _epochs = 10 if dry_run else epochs

    # ── Load data ─────────────────────────────────────────────────────────
    log.info("\nLoading data...")

    # Load profiles for the chosen heuristic (always book-based, domain-agnostic)
    train_profiles_raw = np.load(PROCESSED / f"user_profiles_{heuristic}.npy")
    album_emb_w = np.load(_items_emb_path)

    # Optional first stage: books-catalog ZCA before W1
    if input_zca_book:
        log.info("  Applying ZCA-book pre-transform...")
        train_profiles = apply_zca_book(train_profiles_raw)
        log.info(f"  Profiles shape after zca_book: {train_profiles.shape}")
    else:
        train_profiles = train_profiles_raw

    with open(PROCESSED / f"user_profile_index_{heuristic}.json") as f:
        prof_idx = json.load(f)
    with open(_items_id_map_path) as f:
        album_id_to_idx = json.load(f)
    idx_to_album = {v: k for k, v in album_id_to_idx.items()}

    genre_gt = ground_truth(
        domain,
        items_id_map_path=_items_id_map_path,
        gt_path=_gt_path,
        items_jsonl_path=_items_jsonl_path,
    )
    log.info(f"  Genre GT users: {len(genre_gt):,}")

    # Ordered user list (same order as profile matrix)
    ordered_users = sorted(prof_idx.keys(), key=lambda u: prof_idx[u])

    # ── Canonical leakage-free split ──────────────────────────────────────
    # Preferred: precomputed train/val user lists.
    # Fallback: drop canonical test users, then a seeded 80/20 train/val split.
    canonical_train_path = PROCESSED / "canonical_train_users.json"
    canonical_val_path = PROCESSED / "canonical_val_users.json"

    if canonical_train_path.exists() and canonical_val_path.exists():
        with open(canonical_train_path) as _f:
            _train_set = set(json.load(_f))
        with open(canonical_val_path) as _f:
            _val_list = json.load(_f)
        ordered_users = [u for u in ordered_users if u in _train_set]
        val_users_ordered = [u for u in _val_list if u in prof_idx]
        train_users_ordered = ordered_users
        log.info(
            f"  Canonical split: {len(train_users_ordered):,} train / {len(val_users_ordered):,} val"
        )
    else:
        # Fallback: inline split
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
        rng_split = np.random.default_rng(SEED)
        perm_split = rng_split.permutation(len(ordered_users))
        n_train = int(len(ordered_users) * 0.8)
        train_user_set = {ordered_users[i] for i in perm_split[:n_train]}
        val_user_set = {ordered_users[i] for i in perm_split[n_train:]}
        log.info(
            f"  Split: {len(train_user_set):,} train / {len(val_user_set):,} val  (seed={SEED})"
        )
        train_users_ordered = [u for u in ordered_users if u in train_user_set]
        val_users_ordered = [u for u in ordered_users if u in val_user_set]
        ordered_users = train_users_ordered

    # ── Compute supervision centroids (train only) ─────────────────────────
    log.info("\nComputing supervision centroids (train set only)...")
    # Centroids computed only for train users — val GT never touches training
    centroids_all = compute_centroids(album_emb_w, album_id_to_idx, genre_gt, ordered_users)
    # ordered_users is the filtered list (no test users); map user→position in it
    ordered_user_to_centroid_idx = {u: i for i, u in enumerate(ordered_users)}

    # Build train tensors: positions in the profile matrix for train users
    train_row_idxs = np.array([prof_idx[u] for u in train_users_ordered])
    train_centroid_idxs = np.array([ordered_user_to_centroid_idx[u] for u in train_users_ordered])
    train_centroids = centroids_all[train_centroid_idxs]
    valid_train_mask = train_centroids.any(axis=1)
    valid_train_idxs = np.where(valid_train_mask)[0]
    log.info(
        f"  Valid train users for optimization: {valid_train_mask.sum():,} of {len(train_users_ordered):,}"
    )

    h3_train = train_profiles[train_row_idxs[valid_train_idxs]]
    cent_train = train_centroids[valid_train_idxs]

    # Val index map: subset of prof_idx for val users
    val_idx_map = {u: prof_idx[u] for u in val_users_ordered if u in prof_idx}

    # ── Model ─────────────────────────────────────────────────────────────
    model = ParametricWhitening(dim=DIM).to(device)

    if warmstart:
        if input_zca_book:
            # Warm start: ZCA fitted on the ZCA-book-transformed train profiles only
            log.info("  Warm-start: computing ZCA over zca_book-transformed train profiles...")
            train_profiles_for_zca = train_profiles[train_row_idxs]
            mean_zb = train_profiles_for_zca.mean(axis=0)
            Xc = train_profiles_for_zca.astype(np.float64) - mean_zb
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
                # Fallback: compute ZCA on the fly
                log.info("  ZCA params not found, computing on the fly...")
                mean = train_profiles.mean(axis=0)
                Xc = train_profiles.astype(np.float64) - mean
                cov = np.cov(Xc.T)
                evals, evecs = np.linalg.eigh(cov)
                idx_sort = np.argsort(evals)[::-1]
                evals, evecs = evals[idx_sort], evecs[:, idx_sort]
                W = evecs @ np.diag(1.0 / np.sqrt(evals + EPS)) @ evecs.T
                model.warmstart_from_zca(mean, W.astype(np.float32))

    criterion = InfoNCELoss(temperature=temperature)
    unif_criterion = UniformityLoss(t=2.0) if lambda_unif > 0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_epochs, eta_min=lr * 0.01
    )

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Trainable params: {n_params:,}  ({n_params / 1e6:.1f}M)")

    # ── Baseline evaluation ───────────────────────────────────────────────
    H = heuristic.upper()
    log.info(f"\nBaseline ({H} + ZCA-profile):")
    zca_profiles_path = (
        _domain_dir / f"user_profiles_{heuristic}_profile_whitened.npy"
        if domain != "music"
        else ALIGNMENT_DIR / f"user_profiles_{heuristic}_profile_whitened.npy"
    )
    if zca_profiles_path.exists():
        zca_pw = np.load(zca_profiles_path)
        if zca_pw.shape[0] != len(prof_idx):
            log.info(
                f"  ZCA-profile is stale ({zca_pw.shape[0]:,} rows vs "
                f"{len(prof_idx):,} in the current index) — skipping baseline"
            )
        else:
            evaluate(
                f"{H} + ZCA-profile",
                zca_pw,
                album_emb_w,
                prof_idx,
                idx_to_album,
                genre_gt,
            )
    else:
        log.info("  ZCA-profile not found — skipping baseline")

    log.info(f"\nBaseline ({H} + PW, before training — val set n={len(val_idx_map):,}):")
    with torch.no_grad():
        pw_init = model.apply_numpy(train_profiles)
    evaluate(
        f"{H} + PW (epoch 0, warm-start) [VAL]",
        pw_init,
        album_emb_w,
        val_idx_map,
        idx_to_album,
        genre_gt,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    log.info(f"\nTraining for {_epochs} epochs (train set: {len(h3_train):,} users)...")
    log.info(f"  best_epoch selected by VAL set nDCG@10 ({len(val_idx_map):,} users)")
    t_start = time.time()
    best_ndcg = 0.0
    best_epoch = 0

    prof_tensor = torch.from_numpy(h3_train).to(device)
    cent_tensor = torch.from_numpy(cent_train).to(device)
    N = len(h3_train)

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
            if unif_criterion is not None:
                loss = loss + lambda_unif * unif_criterion(profiles_w)

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
            log.info(f"\n  → Val evaluation at epoch {epoch}:")
            with torch.no_grad():
                pw_profiles = model.apply_numpy(train_profiles)
            # best_epoch is selected on the val set only, to avoid test leakage
            res = evaluate(
                f"{H} + PW (epoch {epoch}) [VAL]",
                pw_profiles,
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
                    b_np, W1_np = model.get_numpy_params()
                    np.save(_domain_dir / f"{artifact_prefix}_b_best.npy", b_np)
                    np.save(_domain_dir / f"{artifact_prefix}_W1_best.npy", W1_np)
                    log.info(f"  ✓ Best saved (nDCG@10={best_ndcg:.4f})")
            log.info("")

    # ── Load best and final evaluation ───────────────────────────────────
    log.info("=" * 70)
    log.info(f"Best epoch: {best_epoch}  ({H} VAL nDCG@10={best_ndcg:.4f})")
    log.info(f"  *** Reported score = val set nDCG@10={best_ndcg:.4f} (n≈{len(val_idx_map)}) ***")

    if not dry_run and best_epoch > 0:
        b_best = np.load(_domain_dir / f"{artifact_prefix}_b_best.npy")
        W1_best = np.load(_domain_dir / f"{artifact_prefix}_W1_best.npy")
        with torch.no_grad():
            model.b.copy_(torch.from_numpy(b_best).to(device))
            model.W1.copy_(torch.from_numpy(W1_best).to(device))

    # Apply the best model to every heuristic and evaluate on the val set.
    stage_label = "zca_book → W1" if input_zca_book else "W1"
    log.info(f"\nApplying best [{stage_label}] to all heuristics...")
    results = {}

    for h in ["h2", "h3", "h4"]:
        npy_path = PROCESSED / f"user_profiles_{h}.npy"
        index_path = PROCESSED / f"user_profile_index_{h}.json"
        if not npy_path.exists():
            continue

        profiles_raw_h = np.load(npy_path)
        with open(index_path) as f:
            idx_map_h = json.load(f)

        if input_zca_book:
            profiles_in = apply_zca_book(profiles_raw_h)
        else:
            profiles_in = profiles_raw_h

        with torch.no_grad():
            profiles_pw = model.apply_numpy(profiles_in)

        # Save only for the heuristic the model was trained on; saving for the
        # others would overwrite their own canonical profiles with this model.
        if not dry_run and h == heuristic:
            out_path = ALIGNMENT_DIR / f"user_profiles_{h}{profile_suffix}.npy"
            np.save(out_path, profiles_pw)
            if domain != "music":
                # Also save in domain-specific dir for easy reference
                np.save(_domain_dir / f"user_profiles_{h}{profile_suffix}.npy", profiles_pw)

        # Val-set evaluation (the reported number)
        val_idx_h = {u: idx_map_h[u] for u in val_users_ordered if u in idx_map_h}
        res_val = evaluate(
            f"{h.upper()} + [{stage_label}] (best) [VAL]",
            profiles_pw,
            album_emb_w,
            val_idx_h,
            idx_to_album,
            genre_gt,
        )
        results[h] = res_val

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    phase_label = "ZCA-BOOK + W1" if input_zca_book else "W1"
    log.info(f"{phase_label} SUMMARY — Parametric Whitening")
    log.info("=" * 70)

    ref = {
        "H2 (baseline)": 0.2595,
        "H3 + PW Linear": 0.1443,
        "H4 + PW Linear": 0.1510,
        "H3 + MoE-PW": 0.1627,
    }

    log.info(f"\n  {'Method':<45} {'nDCG@10':>10}")
    log.info("  " + "─" * 57)
    for label, val in ref.items():
        log.info(f"  {label:<45} {val:>10.4f}  (reference)")
    log.info("  " + "─" * 57)
    for h, res in results.items():
        ndcg = res.get("nDCG@10", 0)
        label = f"{h.upper()} + [{stage_label}] (epoch {best_epoch})"
        flag = (
            "↑ NEW BEST"
            if (h == "h3" and ndcg > 0.1627)
            or (h == "h4" and ndcg > 0.1510)
            or (h == "h2" and ndcg > 0.2595)
            else ""
        )
        log.info(f"  {label:<45} {ndcg:>10.4f}  {flag}")

    if not dry_run:
        import csv

        rows = []
        for h, res in results.items():
            rows.append(
                {"method": f"{h}_pw{csv_suffix}", "best_epoch": best_epoch, "split": "val", **res}
            )
        csv_name = f"parametric_whitening{csv_suffix}_evaluation.csv"
        csv_path = (RESULTS_DIR / csv_name) if domain == "music" else (_domain_dir / csv_name)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"\n  Results saved: {csv_path}")

    log.info("\n" + "=" * 70)
    log.info("DONE")
    log.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Apply only (no training)
# ─────────────────────────────────────────────────────────────────────────────


def apply_only(device_str: str = "auto"):
    """Load saved PW params and apply to all heuristics."""
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )

    b_path = ALIGNMENT_DIR / "pw_b_best.npy"
    W1_path = ALIGNMENT_DIR / "pw_W1_best.npy"

    if not b_path.exists() or not W1_path.exists():
        log.error("Saved PW params not found. Run training first.")
        sys.exit(1)

    model = ParametricWhitening(dim=DIM).to(device)
    with torch.no_grad():
        model.b.copy_(torch.from_numpy(np.load(b_path)).to(device))
        model.W1.copy_(torch.from_numpy(np.load(W1_path)).to(device))

    album_emb_w = np.load(ALIGNMENT_DIR / "album_embeddings_whitened.npy")
    with open(PROCESSED / "album_id_to_idx.json") as f:
        album_id_to_idx = json.load(f)
    idx_to_album = {v: k for k, v in album_id_to_idx.items()}
    genre_gt = ground_truth()

    for h in ["h2", "h3", "h4"]:
        npy_path = PROCESSED / f"user_profiles_{h}.npy"
        index_path = PROCESSED / f"user_profile_index_{h}.json"
        if not npy_path.exists():
            continue
        profiles = np.load(npy_path)
        with open(index_path) as f:
            idx_map = json.load(f)
        with torch.no_grad():
            profiles_pw = model.apply_numpy(profiles)
        np.save(ALIGNMENT_DIR / f"user_profiles_{h}_pw.npy", profiles_pw)
        evaluate(
            f"{h.upper()} + PW",
            profiles_pw,
            album_emb_w,
            idx_map,
            idx_to_album,
            genre_gt,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Parametric Whitening (UniSRec-inspired)")
    p.add_argument("--heuristic", type=str, default="h3", choices=["h3", "h4"])
    p.add_argument(
        "--domain",
        type=str,
        default="music",
        choices=["music", "movies", "games"],
        help=(
            "Target recommendation domain. Profiles (H3/H4) are always book-based "
            "and shared across domains. Only item embeddings, GT, and item metadata "
            "change. Default paths per domain are inferred automatically; use "
            "--items-emb-path / --gt-path / --items-jsonl-path / --items-id-map-path "
            "to override."
        ),
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--no-warmstart", action="store_true")
    p.add_argument("--apply-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--uniformity",
        action="store_true",
        help=(
            "Add the Wang & Isola (2020) uniformity term with weight 1 (equal "
            "weighting of alignment and uniformity). Writes separate artifacts "
            "(*_unif*) and does not overwrite the PW-ZB ones."
        ),
    )
    p.add_argument(
        "--input-zca-book",
        action="store_true",
        help=(
            "Apply the books-catalog ZCA (fit on the book embeddings) as a first "
            "stage before training W1. Writes separate artifacts (pw_zb_*) and "
            "does not overwrite the plain PW results."
        ),
    )
    # Path overrides (optional — use when domain defaults don't match your layout)
    p.add_argument(
        "--items-emb-path",
        type=str,
        default=None,
        help="Path to whitened item embeddings (.npy). Overrides domain default.",
    )
    p.add_argument(
        "--gt-path",
        type=str,
        default=None,
        help="Path to ground truth JSON {uid: [item_ids]}. Overrides domain default.",
    )
    p.add_argument(
        "--items-jsonl-path",
        type=str,
        default=None,
        help="Path to items JSONL with parent_asin + categories. Overrides domain default.",
    )
    p.add_argument(
        "--items-id-map-path",
        type=str,
        default=None,
        help="Path to item_id→row_index JSON. Overrides domain default.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.apply_only:
        apply_only(device_str=args.device)
    else:
        train(
            heuristic=args.heuristic,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            warmstart=not args.no_warmstart,
            eval_every=args.eval_every,
            dry_run=args.dry_run,
            device_str=args.device,
            input_zca_book=args.input_zca_book,
            lambda_unif=1.0 if args.uniformity else 0.0,
            domain=args.domain,
            items_emb_path=args.items_emb_path,
            items_id_map_path=args.items_id_map_path,
            gt_path=args.gt_path,
            items_jsonl_path=args.items_jsonl_path,
        )
