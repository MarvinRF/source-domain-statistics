"""Mixture-of-experts parametric whitening baseline (G = 2 experts).

Generalizes parametric whitening to a gated mixture, following RUNSRec
(Zhang et al., ACM TOIS 2025): each expert ``k`` is its own whitening
``f_k(x) = normalize((x - b_k) @ W_k)``, and a noisy softmax router
``g = softmax(x @ W_router + noise)`` combines them as
``v = normalize(sum_k g_k * f_k(x))``. A single global ``W1`` fits the average
profile; experts can specialize on heterogeneous (multi-genre) profiles.
Expert 0 is warm-started from the trained linear PW (or ZCA) parameters,
expert 1 from the same parameters plus small Gaussian noise, and the router
starts uniform, so epoch 0 reproduces the linear model. The loss is InfoNCE
plus ``lambda_balance * var(mean_batch(g))`` to prevent one expert from taking
all inputs. G = 2 keeps capacity modest for a few thousand training users.

Usage::

    python -m acw.baselines.moe_whitening --heuristic h3
    python -m acw.baselines.moe_whitening --heuristic h4 --input-zca-book
    python -m acw.baselines.moe_whitening --heuristic h3 --dry-run
"""

from __future__ import annotations

import argparse
import csv
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
N_EXPERTS = 2
EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


class MoEWhitening(nn.Module):
    """
    Mixture-of-Experts Parametric Whitening.
    Inspired by RUNSRec (Zhang et al., ACM TOIS 2025, Eqs. 5-8).

    Each expert k learns an independent whitening transform:
        f_k(x) = L2-norm((x - b_k) @ W_k)

    The router combines the experts per input:
        g = Softmax(x @ W_router + noise)
        v = Σ_k g_k * f_k(x)

    The output v is L2-normalized for cosine retrieval.
    """

    def __init__(self, dim: int = DIM, n_experts: int = N_EXPERTS):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts

        # Expert parameters: one bias and one whitening matrix per expert
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(n_experts)])
        self.W = nn.ParameterList([nn.Parameter(torch.eye(dim)) for _ in range(n_experts)])

        # Router takes the raw (non-whitened) embedding as input
        # W_router ∈ R^(dim, n_experts)
        self.W_router = nn.Parameter(torch.zeros(dim, n_experts))
        # Noise router for load balancing (RUNSRec Eq. 8)
        self.W_noise = nn.Parameter(torch.zeros(dim, n_experts))

        nn.init.xavier_uniform_(self.W_router)
        nn.init.xavier_uniform_(self.W_noise)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, dim) float32, L2-normalized input embeddings

        Returns:
            v: (B, dim) float32, L2-normalized MoE-whitened embeddings
            g: (B, n_experts) float32, gating weights (for the load-balancing loss)
        """
        # ── Router ────────────────────────────────────────────────────────
        # Input-dependent noise scale (RUNSRec Eq. 8)
        noise = torch.randn_like(x @ self.W_noise) * F.softplus(x @ self.W_noise)
        logits = x @ self.W_router + noise  # (B, n_experts)
        g = F.softmax(logits, dim=-1)  # (B, n_experts)

        # ── Expert outputs ────────────────────────────────────────────────
        expert_outs = []
        for k in range(self.n_experts):
            x_k = F.normalize((x - self.b[k]) @ self.W[k], dim=-1)  # (B, dim)
            expert_outs.append(x_k)

        # ── Weighted combination ──────────────────────────────────────────
        # v = Σ_k g_k * f_k(x)
        # g: (B, n_experts) → (B, n_experts, 1) for broadcasting
        expert_stack = torch.stack(expert_outs, dim=1)  # (B, n_experts, dim)
        g_expanded = g.unsqueeze(-1)  # (B, n_experts, 1)
        v = (g_expanded * expert_stack).sum(dim=1)  # (B, dim)
        v = F.normalize(v, dim=-1)

        return v, g

    def warmstart_from_pw(self, b_np: np.ndarray, W1_np: np.ndarray, perturb_scale: float = 0.01):
        """
        Initialize from trained linear PW parameters.

        Expert 0: exact copy of the linear PW (stable starting point)
        Expert 1: linear PW + small random perturbation (lets the experts diverge)
        Router:   uniform weights (g_0 = g_1 = 0.5 initially)
        """
        rng = np.random.default_rng(SEED)
        with torch.no_grad():
            for k in range(self.n_experts):
                self.b[k].copy_(torch.from_numpy(b_np.astype(np.float32)))
                if k == 0:
                    self.W[k].copy_(torch.from_numpy(W1_np.astype(np.float32)))
                else:
                    noise = rng.normal(0, perturb_scale, W1_np.shape).astype(np.float32)
                    self.W[k].copy_(torch.from_numpy((W1_np + noise).astype(np.float32)))
            # Uniform router: W_router = 0 → softmax → [0.5, 0.5]
            nn.init.zeros_(self.W_router)
            nn.init.zeros_(self.W_noise)
        log.info("  Warm-started from Linear PW params ✓")
        log.info(f"  Expert 0: PW exact copy | Expert 1: PW + N(0, {perturb_scale})")

    def warmstart_from_zca(
        self, zca_mean: np.ndarray, zca_W: np.ndarray, perturb_scale: float = 0.01
    ):
        """Fallback: initialize from ZCA parameters."""
        rng = np.random.default_rng(SEED)
        with torch.no_grad():
            for k in range(self.n_experts):
                self.b[k].copy_(torch.from_numpy(zca_mean.astype(np.float32)))
                if k == 0:
                    self.W[k].copy_(torch.from_numpy(zca_W.astype(np.float32)))
                else:
                    noise = rng.normal(0, perturb_scale, zca_W.shape).astype(np.float32)
                    self.W[k].copy_(torch.from_numpy((zca_W + noise).astype(np.float32)))
            nn.init.zeros_(self.W_router)
            nn.init.zeros_(self.W_noise)
        log.info("  Warm-started from ZCA params ✓ (PW params not found)")

    def apply_numpy(self, X: np.ndarray) -> np.ndarray:
        """Apply the transform to a numpy array (inference, no grad)."""
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).to(device)
            v, _ = self.forward(x)
        return v.cpu().numpy()

    def save_params(self, prefix: str):
        """Save all MoE parameters."""
        for k in range(self.n_experts):
            np.save(
                ALIGNMENT_DIR / f"{prefix}_b{k}_best.npy",
                self.b[k].detach().cpu().numpy(),
            )
            np.save(
                ALIGNMENT_DIR / f"{prefix}_W{k}_best.npy",
                self.W[k].detach().cpu().numpy(),
            )
        np.save(
            ALIGNMENT_DIR / f"{prefix}_router_best.npy",
            self.W_router.detach().cpu().numpy(),
        )

    def load_params(self, prefix: str):
        """Load saved parameters."""
        with torch.no_grad():
            for k in range(self.n_experts):
                self.b[k].copy_(
                    torch.from_numpy(np.load(ALIGNMENT_DIR / f"{prefix}_b{k}_best.npy"))
                )
                self.W[k].copy_(
                    torch.from_numpy(np.load(ALIGNMENT_DIR / f"{prefix}_W{k}_best.npy"))
                )
            self.W_router.copy_(
                torch.from_numpy(np.load(ALIGNMENT_DIR / f"{prefix}_router_best.npy"))
            )


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────


class MoEInfoNCELoss(nn.Module):
    """
    InfoNCE + Load Balancing Loss.

    L_total = L_InfoNCE + λ_balance × L_balance

    L_InfoNCE: retrieval loss (same as the linear PW)
    L_balance: var(mean_batch(g)) — penalizes a dominant expert
        g = [1.0, 0.0] for every input → var = 0.5 (high penalty)
        g = [0.5, 0.5] for every input → var = 0.0 (no penalty)
    """

    def __init__(self, temperature: float = 0.07, lambda_balance: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.lambda_balance = lambda_balance

    def forward(
        self,
        profiles_w: torch.Tensor,  # (B, dim) whitened profiles
        centroids: torch.Tensor,  # (B, dim) album centroids
        gate_weights: torch.Tensor,  # (B, n_experts) gating weights
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = profiles_w.shape[0]

        # InfoNCE loss
        sim = profiles_w @ centroids.T / self.temperature  # (B, B)
        labels = torch.arange(B, device=profiles_w.device)
        l_infonce = F.cross_entropy(sim, labels)

        # Load-balancing loss: variance of the per-expert mean gate weight
        # gate_weights: (B, n_experts) → mean over batch: (n_experts,)
        mean_gate = gate_weights.mean(dim=0)  # (n_experts,)
        l_balance = mean_gate.var()

        l_total = l_infonce + self.lambda_balance * l_balance

        return l_total, l_infonce, l_balance


# ─────────────────────────────────────────────────────────────────────────────
# Gate diagnostic
# ─────────────────────────────────────────────────────────────────────────────


def gate_diagnostic(model: MoEWhitening, profiles: np.ndarray, label: str):
    """Log the gate-weight distribution to detect expert collapse."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(profiles.astype(np.float32)).to(device)
        _, g = model.forward(x)
    g_np = g.cpu().numpy()
    mean_g = g_np.mean(axis=0)
    std_g = g_np.std(axis=0)
    log.info(
        f"  {label} — gate weights: "
        f"expert0={mean_g[0]:.3f}±{std_g[0]:.3f}  "
        f"expert1={mean_g[1]:.3f}±{std_g[1]:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def train(
    heuristic: str = "h3",
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.07,
    lambda_balance: float = 0.1,
    weight_decay: float = 1e-4,
    perturb_scale: float = 0.01,
    eval_every: int = 20,
    dry_run: bool = False,
    device_str: str = "auto",
    input_zca_book: bool = False,
    domain: str = "music",
    items_emb_path: Path | None = None,  # Override: item embeddings (.npy), e.g.
    # the raw (unwhitened) catalog for a
    # raw-to-raw comparison against catalog
    # whitening.
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

    # ── Domain paths ──────────────────────────────────────────────────────
    defaults = domain_paths(domain)
    _items_emb_path = Path(items_emb_path) if items_emb_path else defaults["items_emb_path"]
    _items_id_map = defaults["items_id_map_path"]
    _gt_path = defaults["gt_path"]
    _items_jsonl = defaults["items_jsonl_path"]
    _domain_dir = defaults["domain_dir"]

    domain_tag = "" if domain == "music" else f"_{domain}"
    # A non-default items_emb_path gets its own tag so it can never overwrite the
    # canonical artifacts trained against the whitened target catalog.
    raw_target_tag = "_rawtarget" if items_emb_path else ""
    if input_zca_book:
        artifact_prefix = f"moe_zb_{heuristic}{domain_tag}{raw_target_tag}"
        profile_suffix = f"_moe_zb{domain_tag}{raw_target_tag}"
        csv_suffix = f"_zb{domain_tag}{raw_target_tag}"
    else:
        artifact_prefix = f"moe_{heuristic}{domain_tag}{raw_target_tag}"
        profile_suffix = f"_moe{domain_tag}{raw_target_tag}"
        csv_suffix = f"{domain_tag}{raw_target_tag}"

    log.info("=" * 70)
    log.info("MoE Parametric Whitening (G=2)")
    log.info("=" * 70)
    log.info(f"  Device         : {device}")
    log.info(f"  Domain         : {domain}")
    log.info(f"  Heuristic      : {H}")
    log.info(f"  N experts      : {N_EXPERTS}")
    log.info(f"  Epochs         : {_epochs}")
    log.info(f"  Batch size     : {batch_size}")
    log.info(f"  LR             : {lr}")
    log.info(f"  Temperature τ  : {temperature}")
    log.info(f"  λ_balance      : {lambda_balance}")
    log.info(f"  Perturb scale  : {perturb_scale}")
    log.info(f"  Input ZCA-book : {input_zca_book}")
    log.info(f"  Dry run        : {dry_run}")

    # ── Load data ──────────────────────────────────────────────────────────
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

    # ── Centroids (training pool only) ────────────────────────────────────
    log.info("\nComputing supervision centroids...")
    centroids = compute_centroids(album_emb_w, album_id_to_idx, genre_gt, ordered_users)
    valid_mask = centroids.any(axis=1)
    valid_idxs = np.where(valid_mask)[0]
    log.info(f"  Valid users for training: {len(valid_idxs):,} of {len(ordered_users):,}")

    # ── Train/val split (80/20) ────────────────────────────────────────────
    if _use_file_split:
        # Canonical split: every valid_idx is a train user; val users come from the file
        train_idxs_ = np.arange(len(valid_idxs))
        h_train = train_profiles[valid_idxs[train_idxs_]]
        cent_train = centroids[valid_idxs[train_idxs_]]

        val_users = _val_users_file
        val_idx_map = {u: i for i, u in enumerate(val_users)}
        val_profiles_arr = train_profiles[np.array([prof_idx[u] for u in val_users])]
        val_idxs_ = np.arange(len(val_users))  # only used for len() below
    else:
        # Fallback: seeded 80/20 split of the valid training pool
        rng_split = np.random.default_rng(SEED)
        all_idxs = np.arange(len(valid_idxs))
        rng_split.shuffle(all_idxs)
        n_val = max(int(len(all_idxs) * 0.2), 1)
        val_idxs_ = all_idxs[:n_val]
        train_idxs_ = all_idxs[n_val:]

        h_train = train_profiles[valid_idxs[train_idxs_]]
        cent_train = centroids[valid_idxs[train_idxs_]]

        val_users = [ordered_users[valid_idxs[i]] for i in val_idxs_]
        val_idx_map = {u: i for i, u in enumerate(val_users)}
        val_profiles_arr = train_profiles[valid_idxs[val_idxs_]]

    log.info(f"  Train users: {len(train_idxs_):,}  |  Val users: {len(val_idxs_):,}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = MoEWhitening(dim=DIM, n_experts=N_EXPERTS).to(device)

    # Warm start: prefer trained linear PW params, else ZCA
    if input_zca_book:
        # Warm start from PW-ZB params if available, otherwise from a ZCA
        # fitted on the ZCA-book-transformed profiles
        pw_zb_b = ALIGNMENT_DIR / f"pw_zb_{heuristic}_b_best.npy"
        pw_zb_W1 = ALIGNMENT_DIR / f"pw_zb_{heuristic}_W1_best.npy"
        if pw_zb_b.exists() and pw_zb_W1.exists():
            b_np = np.load(pw_zb_b)
            W1_np = np.load(pw_zb_W1)
            model.warmstart_from_pw(b_np, W1_np, perturb_scale=perturb_scale)
            log.info("  Warm-started from PW-ZB params ✓")
        else:
            log.info(
                "  PW-ZB params not found, computing ZCA over zca_book-transformed profiles..."
            )
            mean_zb = train_profiles.mean(axis=0)
            Xc = train_profiles.astype(np.float64) - mean_zb
            cov = np.cov(Xc.T)
            evals, evecs = np.linalg.eigh(cov)
            idx_sort = np.argsort(evals)[::-1]
            evals, evecs = evals[idx_sort], evecs[:, idx_sort]
            W_zb = evecs @ np.diag(1.0 / np.sqrt(evals + EPS)) @ evecs.T
            model.warmstart_from_zca(mean_zb, W_zb.astype(np.float32), perturb_scale=perturb_scale)
    else:
        pw_b_path = ALIGNMENT_DIR / f"pw_{heuristic}_b_best.npy"
        pw_W1_path = ALIGNMENT_DIR / f"pw_{heuristic}_W1_best.npy"
        zca_path = ALIGNMENT_DIR / f"whitening_profile_{heuristic}.npz"

        if pw_b_path.exists() and pw_W1_path.exists():
            b_np = np.load(pw_b_path)
            W1_np = np.load(pw_W1_path)
            model.warmstart_from_pw(b_np, W1_np, perturb_scale=perturb_scale)
        elif zca_path.exists():
            data = np.load(zca_path)
            model.warmstart_from_zca(data["mean"], data["W"], perturb_scale=perturb_scale)
        else:
            log.warning("  No warm-start params found — using Xavier init")

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Trainable params: {n_params:,}  ({n_params / 1e6:.2f}M)")

    criterion = MoEInfoNCELoss(temperature=temperature, lambda_balance=lambda_balance)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_epochs, eta_min=lr * 0.01
    )

    # ── Baselines ──────────────────────────────────────────────────────────
    log.info("\nBaseline (Linear PW):")
    pw_path = ALIGNMENT_DIR / f"user_profiles_{heuristic}_pw.npy"
    if pw_path.exists():
        pw_prof = np.load(pw_path)
        evaluate(f"{H} + Linear PW", pw_prof, album_emb_w, prof_idx, idx_to_album, genre_gt)
    else:
        log.info("  Linear PW profiles not found — skipping")

    log.info("\nBaseline (MoE-PW, before training):")
    with torch.no_grad():
        moe_init = model.apply_numpy(train_profiles)
    evaluate(
        f"{H} + MoE-PW (epoch 0)",
        moe_init,
        album_emb_w,
        prof_idx,
        idx_to_album,
        genre_gt,
    )
    gate_diagnostic(model, train_profiles[:200], f"{H} epoch 0")

    # ── Training loop ──────────────────────────────────────────────────────
    log.info(f"\nTraining for {_epochs} epochs...")
    t_start = time.time()
    best_ndcg = 0.0
    best_epoch = 0

    prof_tensor = torch.from_numpy(h_train).to(device)
    cent_tensor = torch.from_numpy(cent_train).to(device)
    N = len(train_idxs_)

    for epoch in range(1, _epochs + 1):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        epoch_l_infonce = 0.0
        epoch_l_balance = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx_batch = perm[start : start + batch_size]
            if len(idx_batch) < 2:
                continue

            x_batch = prof_tensor[idx_batch]
            c_batch = cent_tensor[idx_batch]

            profiles_w, g = model(x_batch)
            loss, l_infonce, l_balance = criterion(profiles_w, c_batch, g)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping matters with two full D×D matrices
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_l_infonce += l_infonce.item()
            epoch_l_balance += l_balance.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_infonce = epoch_l_infonce / max(n_batches, 1)
        avg_balance = epoch_l_balance / max(n_batches, 1)
        elapsed = (time.time() - t_start) / 60

        log.info(
            f"  Epoch {epoch:>4}/{_epochs}  "
            f"loss={avg_loss:.4f}  "
            f"(InfoNCE={avg_infonce:.4f} | balance={avg_balance:.4f})  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"elapsed={elapsed:.1f}min"
        )

        if epoch % eval_every == 0 or epoch == _epochs:
            log.info(f"\n  → Evaluation at epoch {epoch} (val set, n={len(val_idxs_)}):")
            with torch.no_grad():
                val_pw = model.apply_numpy(val_profiles_arr)

            gate_diagnostic(model, val_profiles_arr[:200], f"{H}+MoE (epoch {epoch})")

            res = evaluate(
                f"{H} + MoE-PW (epoch {epoch})",
                val_pw,
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

    # ── Load best and apply to all heuristics ──────────────────────────────
    log.info("=" * 70)
    log.info(f"Best epoch: {best_epoch}  ({H} nDCG@10={best_ndcg:.4f})")

    if not dry_run and best_epoch > 0:
        model.load_params(artifact_prefix)

    stage_label = "zca_book → MoE-PW" if input_zca_book else "MoE-PW"
    log.info("\nEvaluating on val set + saving full npy for pipeline integrity...")
    results = {}

    val_user_set = set(val_users)

    for h in ["h2", "h3", "h4"]:
        npy_path = PROCESSED / f"user_profiles_{h}.npy"
        index_path = PROCESSED / f"user_profile_index_{h}.json"
        if not npy_path.exists():
            continue

        profiles_full = np.load(npy_path)
        with open(index_path) as f:
            idx_map_full = json.load(f)

        # Apply the books-catalog ZCA to every heuristic before MoE
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
            profiles_moe_val = model.apply_numpy(val_profiles_h)

        res = evaluate(
            f"{h.upper()} + {stage_label} (val only, n={len(uids_h)})",
            profiles_moe_val,
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
                profiles_moe_full = model.apply_numpy(profiles_in)
            out_path = _domain_dir / f"user_profiles_{h}{profile_suffix}.npy"
            np.save(out_path, profiles_moe_full)
            log.info(f"  Saved full {h}{profile_suffix}.npy ({len(profiles_full)} users)")

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    phase_label = "MoE-PW + ZCA-BOOK" if input_zca_book else "MoE-PW"
    log.info(f"{phase_label} SUMMARY — MoE Parametric Whitening (G=2)")
    log.info("=" * 70)

    log.info(f"\n  {'Method':<50} {'nDCG@10':>10}")
    log.info("  " + "─" * 62)
    refs = [
        ("H2 (GTE, baseline)", 0.2595),
        ("H3 + PW-ZB", 0.1933),
        ("H3 + MoE-PW", 0.1627),
        ("H3 + Linear PW", 0.1443),
        ("H4 + PW-ZB", 0.1632),
        ("H4 + Linear PW", 0.1510),
    ]
    for label, val in refs:
        log.info(f"  {label:<50} {val:>10.4f}  (reference)")
    log.info("  " + "─" * 62)

    ref_map = {
        "h3": 0.1933 if input_zca_book else 0.1627,
        "h4": 0.1632 if input_zca_book else 0.1510,
        "h2": 0.2595,
    }
    for h, res in results.items():
        ndcg = res.get("nDCG@10", 0)
        ref = ref_map.get(h, 0)
        label = f"{h.upper()} + {stage_label} (best, epoch {best_epoch})"
        flag = "↑ NEW BEST" if ndcg > ref else ""
        log.info(f"  {label:<50} {ndcg:>10.4f}  {flag}")

    if not dry_run:
        rows = [
            {
                "method": f"{h}{profile_suffix}",
                "best_epoch": best_epoch,
                "n_experts": N_EXPERTS,
                "input_zca_book": input_zca_book,
                **res,
            }
            for h, res in results.items()
        ]
        csv_name = f"moe_whitening{csv_suffix}_evaluation.csv"
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


def apply_only(heuristic: str = "h3", device_str: str = "auto", input_zca_book: bool = False):
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

    prefix = f"moe_zb_{heuristic}" if input_zca_book else f"moe_{heuristic}"
    suffix = "_moe_zb" if input_zca_book else "_moe"
    model = MoEWhitening(dim=DIM, n_experts=N_EXPERTS).to(device)
    model.load_params(prefix)
    log.info(f"Loaded MoE-PW params: {prefix}")

    for h in ["h2", "h3", "h4"]:
        npy_path = PROCESSED / f"user_profiles_{h}.npy"
        if not npy_path.exists():
            continue
        profiles_raw = np.load(npy_path)
        profiles_in = apply_zca_book(profiles_raw) if input_zca_book else profiles_raw
        with torch.no_grad():
            profiles_moe = model.apply_numpy(profiles_in)
        out = ALIGNMENT_DIR / f"user_profiles_{h}{suffix}.npy"
        np.save(out, profiles_moe)
        log.info(f"  Saved {out.name} ({len(profiles_moe)} users)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="MoE Parametric Whitening (G=2, RUNSRec-inspired)")
    p.add_argument("--heuristic", type=str, default="h3", choices=["h3", "h4"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lambda-balance", type=float, default=0.1)
    p.add_argument("--perturb-scale", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--apply-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--input-zca-book",
        action="store_true",
        help=(
            "Apply the books-catalog ZCA (fit on the book embeddings) as a first "
            "stage before MoE-PW. Warm-starts from PW-ZB params if available. "
            "Writes separate artifacts (moe_zb_*) and does not overwrite the plain MoE-PW ones."
        ),
    )
    p.add_argument(
        "--domain",
        default="music",
        choices=["music", "movies", "games"],
        help="Target domain (default: music)",
    )
    p.add_argument(
        "--items-emb-path",
        type=str,
        default=None,
        help="Override: path to item embeddings (.npy). Defaults to the "
        "domain's whitened catalog; pass the raw (unwhitened) "
        "catalog path for a raw-to-raw comparison against catalog "
        "whitening.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.apply_only:
        apply_only(
            heuristic=args.heuristic,
            device_str=args.device,
            input_zca_book=args.input_zca_book,
        )
    else:
        train(
            heuristic=args.heuristic,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            lambda_balance=args.lambda_balance,
            weight_decay=1e-4,
            perturb_scale=args.perturb_scale,
            eval_every=args.eval_every,
            dry_run=args.dry_run,
            device_str=args.device,
            input_zca_book=args.input_zca_book,
            domain=args.domain,
            items_emb_path=args.items_emb_path,
        )
