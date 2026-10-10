"""Contrastive infomax alignment baseline (C2DSR-inspired).

Instead of pulling profiles toward item centroids with a temperature-scaled
contrastive loss, this baseline follows the mutual-information maximization
objective of C2DSR (Cao et al., CIKM 2022): a residual MLP ``ProfileEncoder``
maps (optionally ZCA-whitened) user profiles, and a bilinear discriminator
``D(p, c) = sigmoid(p @ W @ c^T)`` (Eq. 13) is trained with binary
cross-entropy (Eq. 12) to separate positive pairs (user i, centroid of user i's
relevant items) from in-batch negatives (centroids rolled by one position).
The encoder is trained on 80% of the users and the best epoch is selected on
the held-out 20% by nDCG@10.

Usage::

    python -m acw.baselines.infomax_alignment --heuristic h3
    python -m acw.baselines.infomax_alignment --heuristic h4 --no-zca
    python -m acw.baselines.infomax_alignment --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from acw.baselines.common import compute_centroids, ground_truth
from acw.baselines.common import evaluate as evaluate_profiles
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
# Models
# ─────────────────────────────────────────────────────────────────────────────


class ProfileEncoder(nn.Module):
    """
    Lightweight residual encoder for user profiles.
    Learns a domain-specific transformation of the (optionally whitened)
    profile embeddings, keeping the same dimensionality.

    Architecture: Linear → LayerNorm → GELU → Linear → residual + L2-norm
    """

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual + L2-normalize
        out = self.net(x) + x
        return F.normalize(out, dim=-1)


class InfomaxDiscriminator(nn.Module):
    """
    Bilinear discriminator for mutual information maximization.

    Adapted from C2DSR (Cao et al., CIKM 2022) Eq. 13:
        D(o_profile, o_centroid) = σ(o_profile @ W_disc @ o_centroid^T)

    where W_disc ∈ R^(d×d) is the learnable compatibility matrix.

    Initialized as identity so the discriminator starts measuring
    cosine similarity (a reasonable prior for retrieval).
    """

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))

    def forward(
        self,
        o_profile: torch.Tensor,  # (B, d) L2-normalized
        o_centroid: torch.Tensor,  # (B, d) L2-normalized
    ) -> torch.Tensor:
        """Returns (B,) scores in [0, 1]."""
        # Bilinear: profile @ W @ centroid^T → diagonal (B,)
        Wc = o_centroid @ self.W.T  # (B, d)
        scores = (o_profile * Wc).sum(dim=-1)  # (B,)
        return torch.sigmoid(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Infomax Loss (C2DSR Eq. 12)
# ─────────────────────────────────────────────────────────────────────────────


class InfomaxLoss(nn.Module):
    """
    Binary cross-entropy infomax loss (Jensen-Shannon MI estimator).

    For each user i in the batch:
        Positive: (profile_i, centroid_i) → label 1
        Negative: (profile_i, centroid_j) → label 0, where j = roll(i,1)

    Using roll(1) as corruption (C2DSR §3.3.2): shifts centroids by 1
    position so every sample gets a hard negative from the same batch.
    """

    def forward(
        self,
        discriminator: InfomaxDiscriminator,
        profiles: torch.Tensor,  # (B, d) encoded profiles
        centroids: torch.Tensor,  # (B, d) album centroids
    ) -> torch.Tensor:
        B = profiles.shape[0]

        # Positive scores
        pos_scores = discriminator(profiles, centroids)  # (B,)

        # Negative scores — roll centroids by 1 (in-batch corruption)
        neg_centroids = torch.roll(centroids, shifts=1, dims=0)
        neg_scores = discriminator(profiles, neg_centroids)  # (B,)

        # BCE loss
        pos_loss = F.binary_cross_entropy(pos_scores, torch.ones(B, device=profiles.device))
        neg_loss = F.binary_cross_entropy(neg_scores, torch.zeros(B, device=profiles.device))
        return pos_loss + neg_loss


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    label: str,
    profiles_np: np.ndarray,
    encoder: ProfileEncoder | None,
    album_emb_w: np.ndarray,
    idx_map: dict,
    idx_to_album: dict,
    genre_gt: dict,
    device: torch.device,
) -> dict[str, float]:
    """Encode profiles (optionally) and evaluate retrieval."""
    if encoder is not None:
        encoder.eval()
        prof_t = torch.from_numpy(profiles_np).to(device)
        encoded = []
        for i in range(0, len(prof_t), 4096):
            encoded.append(encoder(prof_t[i : i + 4096]).cpu().numpy())
        profiles_enc = np.concatenate(encoded, axis=0)
    else:
        profiles_enc = profiles_np

    return evaluate_profiles(label, profiles_enc, album_emb_w, idx_map, idx_to_album, genre_gt)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def train(
    heuristic: str = "h3",
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    use_zca: bool = True,
    eval_every: int = 20,
    dry_run: bool = False,
    device_str: str = "auto",
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    assert heuristic in ("h3", "h4")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    H = heuristic.upper()
    _epochs = 10 if dry_run else epochs
    # zca_tag keeps the artifacts of the ZCA and --no-zca variants apart, so
    # running one does not overwrite the other's checkpoint and profiles.
    zca_tag = "" if use_zca else "_noZCA"

    log.info("=" * 70)
    log.info("Contrastive Infomax (C2DSR-inspired)")
    log.info("=" * 70)
    log.info(f"  Device     : {device}")
    log.info(f"  Heuristic  : {H}")
    log.info(f"  Epochs     : {_epochs}")
    log.info(f"  Batch size : {batch_size}")
    log.info(f"  LR         : {lr}")
    log.info(f"  ZCA input  : {use_zca}")
    log.info(f"  Dry run    : {dry_run}")

    # ── Load data ─────────────────────────────────────────────────────────
    log.info("\nLoading data...")
    album_emb_w = np.load(ALIGNMENT_DIR / "album_embeddings_whitened.npy")

    with open(PROCESSED / "album_id_to_idx.json") as f:
        album_id_to_idx = json.load(f)
    idx_to_album = {v: k for k, v in album_id_to_idx.items()}

    genre_gt = ground_truth()
    log.info(f"  Genre GT users: {len(genre_gt):,}")

    # Load profiles — optionally ZCA-whitened
    if use_zca:
        zca_path = ALIGNMENT_DIR / f"user_profiles_{heuristic}_profile_whitened.npy"
        if zca_path.exists():
            profiles_raw = np.load(zca_path)
            log.info(f"  Profiles: ZCA-profile whitened {profiles_raw.shape}")
        else:
            log.warning("  ZCA-profile not found, using raw profiles")
            profiles_raw = np.load(PROCESSED / f"user_profiles_{heuristic}.npy")
    else:
        profiles_raw = np.load(PROCESSED / f"user_profiles_{heuristic}.npy")
        log.info(f"  Profiles: raw BGE {profiles_raw.shape}")

    with open(PROCESSED / f"user_profile_index_{heuristic}.json") as f:
        prof_idx = json.load(f)
    ordered_users = sorted(prof_idx.keys(), key=lambda u: prof_idx[u])

    # ── Centroids ─────────────────────────────────────────────────────────
    log.info("\nComputing supervision centroids...")
    centroids = compute_centroids(album_emb_w, album_id_to_idx, genre_gt, ordered_users)
    valid_mask = centroids.any(axis=1)
    valid_idxs = np.where(valid_mask)[0]
    log.info(f"  Total valid pairs: {len(valid_idxs):,}")

    # ── Train / Val split (80/20) — fixes data leakage ────────────────────
    # The encoder must generalize to unseen users; training and evaluation
    # on the same users causes the encoder to memorize user-specific mappings.
    rng_split = np.random.default_rng(SEED)
    shuffled = valid_idxs.copy()
    rng_split.shuffle(shuffled)
    split = int(0.8 * len(shuffled))
    train_idxs = shuffled[:split]  # 80% — encoder training
    val_idxs = shuffled[split:]  # 20% — held-out evaluation

    log.info(f"  Train users : {len(train_idxs):,}")
    log.info(f"  Val users   : {len(val_idxs):,}")

    prof_train = profiles_raw[train_idxs]
    cent_train = centroids[train_idxs]

    # Val idx_map: maps user_id → position in val profiles array
    val_users = [ordered_users[i] for i in val_idxs]
    val_idx_map = {u: i for i, u in enumerate(val_users)}
    prof_val = profiles_raw[val_idxs]

    # ── Models ────────────────────────────────────────────────────────────
    encoder = ProfileEncoder(DIM).to(device)
    discriminator = InfomaxDiscriminator(DIM).to(device)
    criterion = InfomaxLoss()

    params = list(encoder.parameters()) + list(discriminator.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_epochs, eta_min=lr * 0.01
    )

    enc_params = sum(p.numel() for p in encoder.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    log.info(f"  Encoder params      : {enc_params:,}")
    log.info(f"  Discriminator params: {disc_params:,}")

    # ── Baselines ─────────────────────────────────────────────────────────
    log.info(f"\nBaseline ({H} + ZCA-profile) — val set ({len(val_idxs)} users):")
    evaluate(
        f"{H} + ZCA-profile (ref, val)",
        prof_val,
        None,
        album_emb_w,
        val_idx_map,
        idx_to_album,
        genre_gt,
        device,
    )

    log.info(f"\nBaseline ({H} + Infomax encoder, epoch 0) — val set:")
    evaluate(
        f"{H} + Infomax (epoch 0, val)",
        prof_val,
        encoder,
        album_emb_w,
        val_idx_map,
        idx_to_album,
        genre_gt,
        device,
    )

    # ── Training ──────────────────────────────────────────────────────────
    log.info(f"\nTraining for {_epochs} epochs (on {len(train_idxs)} users)...")
    t_start = time.time()
    best_ndcg = 0.0
    best_epoch = 0
    N = len(train_idxs)

    prof_tensor = torch.from_numpy(prof_train).to(device)
    cent_tensor = torch.from_numpy(cent_train).to(device)

    for epoch in range(1, _epochs + 1):
        encoder.train()
        discriminator.train()

        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx_b = perm[start : start + batch_size]
            if len(idx_b) < 4:
                continue

            p_batch = prof_tensor[idx_b]
            c_batch = cent_tensor[idx_b]

            encoded = encoder(p_batch)
            loss = criterion(discriminator, encoded, c_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
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
            res = evaluate(
                f"{H} + Infomax (epoch {epoch}, val)",
                prof_val,
                encoder,
                album_emb_w,
                val_idx_map,
                idx_to_album,
                genre_gt,
                device,
            )
            ndcg10 = res.get("nDCG@10", 0)
            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                best_epoch = epoch
                if not dry_run:
                    torch.save(
                        encoder.state_dict(),
                        ALIGNMENT_DIR / f"infomax_{heuristic}{zca_tag}_encoder_best.pt",
                    )
                    torch.save(
                        discriminator.state_dict(),
                        ALIGNMENT_DIR / f"infomax_{heuristic}{zca_tag}_disc_best.pt",
                    )
                    log.info(f"  ✓ Best saved (val nDCG@10={best_ndcg:.4f})")
            log.info("")

    # ── Final ─────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info(f"Best epoch: {best_epoch}  ({H} val nDCG@10={best_ndcg:.4f})")

    if not dry_run and best_epoch > 0:
        encoder.load_state_dict(
            torch.load(
                ALIGNMENT_DIR / f"infomax_{heuristic}{zca_tag}_encoder_best.pt", map_location=device
            )
        )

    # Informative evaluation on all users (train + val); the val result is the unbiased one
    log.info("\nFinal evaluation — all users (best encoder):")
    log.info("  Note: train users seen during training, val users are held-out.")
    log.info(f"  Val set result (unbiased): nDCG@10={best_ndcg:.4f}")

    results = {}
    for h in ["h2", "h3", "h4"]:
        npy = PROCESSED / f"user_profiles_{h}.npy"
        ipath = PROCESSED / f"user_profile_index_{h}.json"
        if not npy.exists():
            continue
        zca_p = ALIGNMENT_DIR / f"user_profiles_{h}_profile_whitened.npy"
        p_np = np.load(zca_p) if (use_zca and zca_p.exists()) else np.load(npy)
        with open(ipath) as f:
            imap = json.load(f)
        res = evaluate(
            f"{h.upper()} + Infomax (full, best ep.{best_epoch})",
            p_np,
            encoder,
            album_emb_w,
            imap,
            idx_to_album,
            genre_gt,
            device,
        )
        results[h] = res

        # Save only for the heuristic the model was trained on; saving for the
        # others would overwrite their own canonical profiles with this model.
        if not dry_run and h == heuristic:
            with torch.no_grad():
                pt = torch.from_numpy(p_np).to(device)
                enc = []
                for i in range(0, len(pt), 4096):
                    enc.append(encoder(pt[i : i + 4096]).cpu().numpy())
                enc_np = np.concatenate(enc, axis=0)
            np.save(ALIGNMENT_DIR / f"user_profiles_{h}_infomax{zca_tag}.npy", enc_np)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("SUMMARY — Contrastive Infomax")
    log.info("=" * 70)
    ref = {
        "H2 original": 0.1969,
        "H3 original": 0.0695,
        "H4 original": 0.0478,
        "H3 + ZCA-profile": 0.1396,
        "H4 + ZCA-profile": 0.1195,
        "H3 + PW": 0.1664,
        "H4 + PW": 0.1525,
    }
    log.info(f"\n  {'Method':<42} {'nDCG@10':>10}")
    log.info("  " + "─" * 54)
    for label, val in ref.items():
        log.info(f"  {label:<42} {val:>10.4f}  (ref)")
    log.info("  " + "─" * 54)
    for h, res in results.items():
        ndcg = res.get("nDCG@10", 0)
        flag = ""
        if h == "h3" and ndcg > 0.1664:
            flag = "↑ NEW BEST"
        if h == "h4" and ndcg > 0.1525:
            flag = "↑ NEW BEST"
        log.info(f"  {h.upper()} + Infomax (epoch {best_epoch}){'':>18} {ndcg:>10.4f}  {flag}")

    if not dry_run:
        import csv

        rows = [
            {"method": f"{h}_infomax", "best_epoch": best_epoch, **res}
            for h, res in results.items()
        ]
        csv_path = RESULTS_DIR / "infomax_evaluation.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"\n  Saved: {csv_path}")

    log.info("\n" + "=" * 70)
    log.info("DONE")
    log.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Contrastive Infomax Alignment (C2DSR-inspired)")
    p.add_argument("--heuristic", type=str, default="h3", choices=["h3", "h4"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--no-zca", action="store_true", help="Use raw profiles instead of ZCA-whitened ones"
    )
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="auto")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        heuristic=args.heuristic,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_zca=not args.no_zca,
        eval_every=args.eval_every,
        dry_run=args.dry_run,
        device_str=args.device,
    )
