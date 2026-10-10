"""
Catalog whitening vs. raw retrieval under exact and genre ground truth.

Builds H2 profiles (mean of the user's last MAX_BOOKS_PROFILE book
embeddings, as in src/acw/embeddings/generate_h2_profiles) from any
book/album embedding set, fits ZCA on the album catalog
(acw.whitening.ZCAWhitening) and evaluates on the canonical validation users
with the standard pipeline metrics, under both exact-item and genre-level
ground truth. Reports H2 raw, H2 + catalog ZCA and H2 + profile ZCA, the
Random and trivial baselines, and a paired Wilcoxon test (catalog ZCA vs.
raw).

Supports: the check that the catalog-whitening gain is not an artefact of
the genre-level ground truth (it also holds under exact-item ground truth),
including with an alternative (description-based) item embedding.

Inputs: --domain {music,movies,games} (default music) selects the target
catalog via acw.evaluation.config.resolve_paths; the source side is always
data/processed/book_embeddings.npy + book_id_to_idx.json. For music the
canonical album embeddings must reproduce H2 + ZCA-catalog = 0.3476 on genre
GT; with --emb-dir (music only), {albums,books}_embeddings_qwen.npy and
*_id_to_idx_qwen.json from that directory.
Output: results/gate_exact_gt_<tag>.json (music),
results/gate_exact_gt_<domain>_<tag>.json (movies, games)

Usage:
    python -m experiments.main_results.gate_exact_gt
    python -m experiments.main_results.gate_exact_gt --domain movies
    python -m experiments.main_results.gate_exact_gt --emb-dir data/processed/desc --tag desc
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from acw.embeddings.generate_h2_profiles.io import load_user_history, user_history_path
from acw.evaluation.config import resolve_paths
from acw.evaluation.loader import build_ground_truth, sample_test_users
from acw.evaluation.pipeline import MAX_K, _score, compute_random_baseline, trivial_baselines
from acw.provenance import provenance
from acw.settings import (
    MAX_BOOKS_PROFILE,
    ROOT,  # noqa: E402
)
from acw.whitening import ZCAWhitening

P = ROOT / "data" / "processed"


def load(emb: Path, idx: Path) -> tuple[np.ndarray, dict[str, int]]:
    x = np.load(emb).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12), json.load(open(idx))


def h2_profiles(users, history, books, book_idx):
    prof, keep = [], []
    for u in users:
        rows = [book_idx[a] for a in history.get(u, [])[-MAX_BOOKS_PROFILE:] if a in book_idx]
        if rows:
            v = books[rows].mean(0)
            prof.append(v / np.linalg.norm(v))
            keep.append(u)
    return np.asarray(prof, dtype=np.float32), keep


def top_k(profiles, items, k=MAX_K, bs=512):
    out = []
    for i in range(0, len(profiles), bs):
        s = profiles[i : i + bs] @ items.T
        part = np.argpartition(-s, k, axis=1)[:, :k]
        order = np.take_along_axis(s, part, 1).argsort(1)[:, ::-1]
        out.append(np.take_along_axis(part, order, 1))
    return np.vstack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--emb-dir",
        type=Path,
        default=None,
        help="Dir with {albums,books}_embeddings_qwen.npy + _id_to_idx_qwen.json; default = canonical",
    )
    ap.add_argument("--tag", default="canonical")
    ap.add_argument("--domain", choices=["music", "movies", "games"], default="music")
    args = ap.parse_args()

    bok = (P / "book_embeddings.npy", P / "book_id_to_idx.json")
    over = {}
    if args.emb_dir:
        if args.domain != "music":
            ap.error("--emb-dir is only supported for --domain music")
        d = args.emb_dir if args.emb_dir.is_absolute() else ROOT / args.emb_dir
        over = {
            "items_emb_path": str(d / "albums_embeddings_qwen.npy"),
            "items_idx_path": str(d / "albums_id_to_idx_qwen.json"),
        }
        bok = (d / "books_embeddings_qwen.npy", d / "books_id_to_idx_qwen.json")

    cfg = resolve_paths(args.domain, **over)
    items, item_idx = load(cfg.items_emb_path, cfg.items_idx_path)
    books, book_idx = load(*bok)
    idx_to_item = {v: k for k, v in item_idx.items()}
    history = load_user_history(args.domain)

    # Genre GT of every user does not fit in RAM for movies; restrict it to the
    # canonical evaluation population (same priority as sample_test_users).
    pop_file = next(
        (p for p in (cfg.canonical_val_path, cfg.canonical_users_path) if p.exists()), None
    )
    pop = set(json.load(open(pop_file))) if pop_file else None
    gts = {
        "exact": build_ground_truth(cfg, "exact"),
        "genre": build_ground_truth(cfg, "genre", pop),
    }
    users = sample_test_users(gts["genre"], cfg)  # canonical val population (n=2,731)

    h2, users = h2_profiles(users, history, books, book_idx)
    items_w = ZCAWhitening().fit_transform(items)
    h2_w = ZCAWhitening().fit_transform(h2)
    ranked = {
        "H2 raw": top_k(h2, items),
        "H2+ZCA-catalog": top_k(h2, items_w),
        "H2+ZCA-profile": top_k(h2_w, items_w),
    }

    inputs = [cfg.items_emb_path, cfg.items_idx_path, *bok, cfg.items_meta_path, cfg.gt_path]
    inputs += [user_history_path(args.domain), *([pop_file] if pop_file else [])]
    report = {
        "tag": args.tag,
        "domain": args.domain,
        "n_users": len(users),
        "provenance": provenance(inputs),
    }
    for mode, gt in gts.items():
        ev = [u for u in users if gt.get(u)]
        rows = {"Random": compute_random_baseline(gt, ev, idx_to_item, len(idx_to_item))}
        rows.update({b.pop("label"): b for b in trivial_baselines(cfg, gt, ev, idx_to_item)})
        per_user = {}
        for name, top in ranked.items():
            m = defaultdict(list)
            for u, row in zip(users, top):
                if gt.get(u):
                    _score([idx_to_item[i] for i in row], gt[u], m)
            per_user[name] = np.array(m["nDCG@10"])
            rows[name] = {k: float(np.mean(v)) for k, v in m.items()}
        a, b = per_user["H2 raw"], per_user["H2+ZCA-catalog"]
        report[mode] = {
            "n_eval": len(ev),
            "rows": rows,
            "zca_vs_raw": {
                "mean_diff": float((b - a).mean()),
                "wins": int((b > a).sum()),
                "losses": int((b < a).sum()),
                "p_wilcoxon_pratt": float(wilcoxon(b, a, zero_method="pratt").pvalue),
            },
        }
        print(f"\n## {args.tag} | GT={mode} | n={len(ev)}")
        print(f"  {'method':<28}{'nDCG@10':>9}{'R@10':>9}{'R@50':>9}{'R@100':>9}")
        for name, r in rows.items():
            print(
                f"  {name:<28}"
                + "".join(f"{r.get(k, 0):>9.4f}" for k in ("nDCG@10", "R@10", "R@50", "R@100"))
            )
        print(f"  ZCA-catalog vs raw: {report[mode]['zca_vs_raw']}")

    name = args.tag if args.domain == "music" else f"{args.domain}_{args.tag}"
    out = ROOT / "results" / f"gate_exact_gt_{name}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
