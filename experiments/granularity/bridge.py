"""
Closed-form ridge bridge (EMCDR-style) from the book profile to the target catalog.

On the TRAIN users, a linear map W (with intercept) sends the H2 book profile x to the mean
embedding y of the user's train target items (exact interactions that are in the catalog; users
without one are left out): ``W = argmin ||Y - XW||^2 + lambda ||W||^2`` on centred X and Y, solved in
closed form from one eigendecomposition of X'X. A user is scored by the cosine between the mapped
profile ``(x - mean_X) W + mean_Y`` and the (L2-normalised) item embeddings. lambda = s * tr(X'X)/d
over the grid ``SCALES`` is chosen on the VAL users by mean exact nDCG@10 (the larger lambda on
ties). Val and test users never enter the fit.
The nonlinear bridge (``bridge_mlp``, EMCDR-style) has the same input and target: one hidden layer
(ReLU) trained with AdamW on the standardised data, early-stopped on the VAL loss (MSE to the mean
embedding of each VAL user's exact items); width and weight decay are chosen on the VAL users by
mean exact nDCG@10 at the early-stopped epoch (the first of the grid on ties). The raw and query-ZCA(U) conditions are re-scored
as in knn.py (checked against the grid, 1e-4); the bridge is tested against both.

The contrastive bridge (``bridgecl_*`` files, ``--contrastive``) has the same input but is trained
with InfoNCE: an MLP maps the profile of a TRAIN user to the target embedding space and is pulled,
by cosine / temperature, towards one of that user's train target items (a new one each epoch)
against the other positives of the batch plus ``CL_NEG`` catalog items sampled per batch. Width,
temperature and weight decay are chosen on the VAL users (mean exact nDCG@10 averaged over
``CL_SEEDS`` seeds; early stopping on the VAL nDCG@10 per seed); the TEST score of a user is the
mean over the seeds. ``bridge_cl_centred`` subtracts the mean bridge output of the train users from
the query (query centring after the bridge; its own configuration is chosen on VAL). Sanity check:
one train target item per train user (with two or more) is held out, a model with the selected
configuration is fitted without it, and the nDCG@10 of that item is reported for the bridge, raw,
query ZCA and the random expectation.

Outputs (results/granularity/; a text other than titlecat adds the suffix _<text>):
  bridge_{domain}_{encoder}.csv.gz / .json   per-user TEST metrics; lambda curve (VAL, TEST
                                             diagnostic), selection, means, provenance
  bridgecl_{domain}_{encoder}.csv.gz / .json  the contrastive bridge (same layout, plus sanity)
  stats_bridge.csv, summary_bridge.csv  (--stats)  paired tests on TEST (one Holm family per
                                             domain and metric)

Usage:
    python -m experiments.granularity.bridge --domain music --encoder gte
    python -m experiments.granularity.bridge --stats
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from acw.corrections import zca_power
from acw.evaluation.pipeline import MAX_K
from acw.evaluation.ranking import top_k
from acw.provenance import provenance
from acw.settings import SEED
from experiments.granularity import common as C
from experiments.granularity.knn import CONTENT, GRID_EQUIV, interaction_matrix
from experiments.granularity.run_corrections import REGISTRY

log = logging.getLogger("bridge")

SCALES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)  # lambda / (tr(X'X) / d)
BRIDGE = "bridge_ridge"
BRIDGE_MLP = "bridge_mlp"
MLP_WIDTHS = (256, 1024)
MLP_WDS = (0.0, 1e-2, 1e-1, 1.0)  # AdamW weight decay
MLP_MAX_EPOCHS, MLP_PATIENCE, MLP_BATCH, MLP_LR = 60, 5, 256, 1e-3
BRIDGE_CL, BRIDGE_CL_C = "bridge_cl", "bridge_cl_centred"
CL_WIDTHS, CL_TAUS, CL_WDS = (256, 1024), (0.05, 0.1), (0.0, 0.1)
CL_SEEDS = (SEED, SEED + 1, SEED + 2)
CL_MAX_EPOCHS, CL_PATIENCE, CL_BATCH, CL_NEG, CL_LR = 30, 4, 256, 512, 1e-3


def fit_ridge_path(X: np.ndarray, Y: np.ndarray, scales=SCALES):
    """Means and the ridge weights of every scale: ``{s: W}``, lambda = s * tr(X'X)/d on the
    centred data, W solved in closed form from the eigendecomposition of X'X."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = (X - mx).astype(np.float64), (Y - my).astype(np.float64)
    lam, V = np.linalg.eigh(Xc.T @ Xc)
    xy = V.T @ (Xc.T @ Yc)
    unit = lam.sum() / len(lam)
    return mx, my, {s: V @ (xy / (lam + s * unit)[:, None]) for s in scales}


def map_profiles(Q: np.ndarray, mx, my, W) -> np.ndarray:
    return ((Q - mx) @ W + my).astype(np.float32)


def fit_mlp(X, Y, Xv, Yv, width, wd, seed=SEED):
    """One-hidden-layer MLP ``x -> y`` on data standardised with the train mean and one scalar
    std per side; AdamW, early stopping on the MSE of (Xv, Yv). Returns ``(map, best_epoch)``
    where ``map(Q)`` maps raw profiles to the embedding space (float32, shape ``(n, d_y)``)."""
    torch.manual_seed(seed)
    mx, my = X.mean(0), Y.mean(0)
    sx, sy = float((X - mx).std()), float((Y - my).std())

    def tt(a, m, s):
        return torch.from_numpy(((a - m) / s).astype(np.float32))

    Xt, Yt, Xe, Ye = tt(X, mx, sx), tt(Y, my, sy), tt(Xv, mx, sx), tt(Yv, my, sy)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], width), torch.nn.ReLU(), torch.nn.Linear(width, Y.shape[1])
    )
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=wd)
    rng = np.random.default_rng(seed)
    best, best_ep, best_state = np.inf, 0, None
    for ep in range(1, MLP_MAX_EPOCHS + 1):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(Xt)))
        for j in range(0, len(perm), MLP_BATCH):
            b = perm[j : j + MLP_BATCH]
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(Xt[b]), Yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            loss = float(torch.nn.functional.mse_loss(net(Xe), Ye))
        if loss < best:
            best, best_ep = loss, ep
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        elif ep - best_ep >= MLP_PATIENCE:
            break
    net.load_state_dict(best_state)
    net.eval()

    def map_(Q: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return (net(tt(Q, mx, sx)).numpy() * sy + my).astype(np.float32)

    return map_, best_ep


def run(domain: str, encoder: str, text: str = "titlecat") -> None:
    t0 = time.time()
    src = REGISTRY[(domain, encoder, text)]
    pop = C.build_population(domain, src)
    emb = C.load_embeddings(src, pop, sources=("train_users",), shrinkage=False)
    mom = emb.moments["train_users"]
    inter = interaction_matrix(pop, pop.train_kept)
    n_i = np.asarray(inter.sum(axis=1)).ravel()
    keep = n_i > 0
    Y = (inter @ emb.items)[keep] / n_i[keep, None]
    X = emb.train_prof[keep]
    mx, my, Ws = fit_ridge_path(X, Y)
    items = emb.items / np.linalg.norm(emb.items, axis=1, keepdims=True)
    log.info(f"{domain}: ridge fitted on {len(X)} of {len(pop.train_kept)} train users")

    def tops(split: str) -> dict[str, np.ndarray]:
        q = emb.eval_prof[split]
        return {f"{BRIDGE}_s{s:g}": top_k(map_profiles(q, mx, my, W), items, MAX_K)
                for s, W in Ws.items()}  # fmt: skip

    val_curve = C.score(pop, "val", tops("val"))
    names = [f"{BRIDGE}_s{s:g}" for s in reversed(SCALES)]  # larger lambda first: wins ties
    best = C.select_best(val_curve, names)
    s_star = float(best.rsplit("_s", 1)[1])
    log.info(f"selected lambda scale on VAL: {s_star:g}")
    test_tops = tops("test")

    # nonlinear bridge: (width, weight decay) chosen on VAL, early stopping on the VAL loss
    inter_v = interaction_matrix(pop, pop.eval_users["val"])
    n_v = np.asarray(inter_v.sum(axis=1)).ravel()
    kv = n_v > 0
    Yv = (inter_v @ emb.items)[kv] / n_v[kv, None]
    Xv = emb.eval_prof["val"][kv]
    mlp_maps, mlp_epochs, mlp_val = {}, {}, {}
    for w in MLP_WIDTHS:
        for wd in MLP_WDS:
            name = f"{BRIDGE_MLP}_w{w}_wd{wd:g}"
            mlp_maps[name], mlp_epochs[name] = fit_mlp(X, Y, Xv, Yv, w, wd)
            mlp_val[name] = top_k(mlp_maps[name](emb.eval_prof["val"]), items, MAX_K)
            log.info(f"{name}: best epoch {mlp_epochs[name]}")
    mlp_curve = C.score(pop, "val", mlp_val)
    mlp_star = C.select_best(mlp_curve, list(mlp_maps))
    log.info(f"selected MLP on VAL: {mlp_star}")
    mlp_test = {n: top_k(m(emb.eval_prof["test"]), items, MAX_K) for n, m in mlp_maps.items()}
    content = {
        "raw": top_k(emb.eval_prof["test"], emb.items, MAX_K),
        "zca_train_users_a1.0": top_k(
            zca_power(emb.eval_prof["test"], *mom, 1.0), emb.items, MAX_K
        ),
    }
    test = C.score(
        pop, "test", {BRIDGE: test_tops[best], BRIDGE_MLP: mlp_test[mlp_star], **content},
        d2=True, strata=True,
    )  # fmt: skip
    test_curve = C.score(pop, "test", test_tops)
    mlp_test_curve = C.score(pop, "test", mlp_test)
    val_sel = C.score(pop, "val", {
        "raw": top_k(emb.eval_prof["val"], emb.items, MAX_K),
        "zca_train_users_a1.0": top_k(zca_power(emb.eval_prof["val"], *mom, 1.0), emb.items, MAX_K),
    })  # fmt: skip
    stem_grid = f"{domain}_{encoder}_{text}"
    grid_check = {**C.check_against_grid(val_sel, stem_grid, "val", GRID_EQUIV),
                  **C.check_against_grid(test, stem_grid, "test", GRID_EQUIV)}  # fmt: skip

    def curve(df: pd.DataFrame, cols: list[str] = names) -> dict:
        m = C.block_means(df)["exact"]
        return {n: m[n] for n in cols}

    summary = {
        "domain": domain, "encoder": encoder, "model": src.model, "text": text,
        "model_description": "ridge map of the H2 book profile to the mean embedding of the "
        "user's train target items (centred, with intercept); cosine to the items",
        "scales": list(SCALES), "selected_scale": s_star,
        "selection": "VAL users, mean exact nDCG@10, ties to the larger lambda",
        "n_fit_users": int(keep.sum()), "n_train_profiles": len(pop.train_kept),
        "n_eval_users": {s: len(u) for s, u in pop.eval_users.items()},
        "val_lambda_curve": curve(val_curve), "test_lambda_curve_diagnostic": curve(test_curve),
        "mlp_description": "one hidden layer (ReLU), AdamW lr 1e-3, batch 256, standardised "
        "inputs and targets, early stopping (patience 5, at most 60 epochs) on the VAL MSE",
        "mlp_grid": {"widths": list(MLP_WIDTHS), "weight_decays": list(MLP_WDS)},
        "mlp_selected": mlp_star, "mlp_best_epochs": mlp_epochs, "mlp_seed": SEED,
        "mlp_val_curve": curve(mlp_curve, list(mlp_maps)),
        "mlp_test_curve_diagnostic": curve(mlp_test_curve, list(mlp_maps)),
        "test": C.block_means(test), "grid_check_max_abs_diff": grid_check,
        "provenance": C.provenance_of(pop, *emb.files), **C.run_info(t0),
    }  # fmt: skip
    C.write_run(f"bridge_{domain}_{encoder}{C.text_tag(text)}", test, summary)


def val_ndcg10(top: np.ndarray, rel: sp.csr_matrix) -> float:
    """Mean exact nDCG@10 of ``top`` (n, >=10) against the binary relevance matrix ``rel`` (early
    stopping proxy of the pipeline's metric; the selection uses the pipeline itself)."""
    n = len(top)
    hit = np.asarray(rel[np.repeat(np.arange(n), 10), top[:, :10].ravel()]).reshape(n, 10)
    disc = 1.0 / np.log2(np.arange(2, 12))
    ideal = np.cumsum(disc)[np.minimum(np.asarray(rel.sum(1)).ravel(), 10).astype(int) - 1]
    return float(((hit * disc).sum(1) / ideal).mean())


def fit_contrastive(X, inter, items, Xv, rel_v, width, tau, wd, seed):
    """InfoNCE MLP ``x -> embedding``: row i of ``inter`` (binary users x catalog, train
    interactions) gives the positives of user i, one drawn per epoch; negatives are the other
    positives of the batch plus CL_NEG catalog items sampled per batch; logits are cosine / tau
    against the L2-normalised ``items``. Early stopping on the VAL nDCG@10. Returns
    ``(map, best_epoch, best_val)``; ``map(Q)`` is the raw network output (float32)."""
    torch.manual_seed(seed)
    mx = X.mean(0)
    sx = float((X - mx).std())
    tt = lambda a: torch.from_numpy(((a - mx) / sx).astype(np.float32))  # noqa: E731
    Xt, It = tt(X), torch.from_numpy(items.astype(np.float32))
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], width), torch.nn.ReLU(), torch.nn.Linear(width, items.shape[1])
    )
    opt = torch.optim.AdamW(net.parameters(), lr=CL_LR, weight_decay=wd)
    rng = np.random.default_rng(seed)
    ip, ix = inter.indptr, inter.indices
    deg = np.diff(ip)
    n_cat = len(items)

    def mapper(Q):
        with torch.no_grad():
            return net(tt(Q)).numpy()

    best, best_ep, best_state = -1.0, 0, None
    for ep in range(1, CL_MAX_EPOCHS + 1):
        net.train()
        pos = ix[ip[:-1] + (rng.random(len(deg)) * deg).astype(np.int64)]  # one positive per user
        perm = rng.permutation(len(Xt))
        for j in range(0, len(perm), CL_BATCH):
            b = perm[j : j + CL_BATCH]
            pb = torch.from_numpy(pos[b])
            cand = torch.cat([It[pb], It[torch.from_numpy(rng.integers(0, n_cat, CL_NEG))]])
            z = torch.nn.functional.normalize(net(Xt[b]), dim=1)
            logit = z @ cand.T / tau
            same = (pb[:, None] == pb[None, :]) & ~torch.eye(len(b), dtype=torch.bool)
            logit[:, : len(b)] = logit[:, : len(b)].masked_fill(same, -1e9)  # duplicate positives
            opt.zero_grad()
            torch.nn.functional.cross_entropy(logit, torch.arange(len(b))).backward()
            opt.step()
        net.eval()
        v = val_ndcg10(top_k(mapper(Xv), items, 10), rel_v)
        if v > best:
            best, best_ep = v, ep
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        elif ep - best_ep >= CL_PATIENCE:
            break
    net.load_state_dict(best_state)
    net.eval()
    return mapper, best_ep, best


def holdout(inter: sp.csr_matrix, seed: int):
    """(fit matrix, held rows, held cols): one random interaction removed from each user with
    two or more."""
    rng = np.random.default_rng(seed)
    deg = np.diff(inter.indptr)
    rows = np.flatnonzero(deg >= 2)
    cols = inter.indices[inter.indptr[rows] + (rng.random(len(rows)) * deg[rows]).astype(np.int64)]
    held = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=inter.shape)
    fit = (inter - held).tocsr()
    fit.eliminate_zeros()
    return fit, rows, cols


def rank_ndcg10(Q: np.ndarray, items: np.ndarray, target: np.ndarray) -> np.ndarray:
    """nDCG@10 of the single relevant item ``target[i]`` for query i, ranked over the catalog."""
    out = np.empty(len(Q))
    for i in range(0, len(Q), 512):
        s = Q[i : i + 512] @ items.T
        t = s[np.arange(len(s)), target[i : i + 512]]
        rank = (s > t[:, None]).sum(1)  # 0-based
        out[i : i + 512] = np.where(rank < 10, 1.0 / np.log2(rank + 2), 0.0)
    return out


def run_contrastive(domain: str, encoder: str, text: str = "titlecat") -> None:
    t0 = time.time()
    src = REGISTRY[(domain, encoder, text)]
    pop = C.build_population(domain, src)
    emb = C.load_embeddings(src, pop, sources=("train_users",), shrinkage=False)
    mom = emb.moments["train_users"]
    inter = interaction_matrix(pop, pop.train_kept)
    keep = np.asarray(inter.sum(axis=1)).ravel() > 0
    inter, X = inter[keep].tocsr(), emb.train_prof[keep]
    items = emb.items / np.linalg.norm(emb.items, axis=1, keepdims=True)
    rel_v = interaction_matrix(pop, pop.eval_users["val"]).tocsr()
    kv = np.asarray(rel_v.sum(axis=1)).ravel() > 0
    Xv_all, Xv, rel_vk = emb.eval_prof["val"], emb.eval_prof["val"][kv], rel_v[kv]
    log.info(f"{domain}: contrastive, {len(X)} fit users, {len(items)} items")

    def centre(m, Q):  # query centring after the bridge: minus the mean output of the train users
        return Q - m(X).mean(0)

    grid = [(w, t, wd) for w in CL_WIDTHS for t in CL_TAUS for wd in CL_WDS]
    maps, epochs, val_tops, val_tops_c = {}, {}, {}, {}
    for cfg in grid:
        for sd in CL_SEEDS:
            name = f"w{cfg[0]}_t{cfg[1]:g}_wd{cfg[2]:g}_s{sd}"
            m, epochs[name], _ = fit_contrastive(X, inter, items, Xv, rel_vk, *cfg, sd)
            maps[cfg, sd] = m
            val_tops[name] = top_k(m(Xv_all), items, MAX_K)
            val_tops_c[name] = top_k(centre(m, Xv_all), items, MAX_K)
            log.info(f"{name}: best epoch {epochs[name]}  ({time.time() - t0:.0f}s)")
    val_curve = C.score(pop, "val", val_tops)
    val_curve_c = C.score(pop, "val", val_tops_c)

    def pick(curve):
        cv = {cfg: float(np.mean([C.exact_ndcg(curve, f"w{cfg[0]}_t{cfg[1]:g}_wd{cfg[2]:g}_s{sd}")
                                  for sd in CL_SEEDS])) for cfg in grid}  # fmt: skip
        return max(cv, key=cv.get), {f"w{c[0]}_t{c[1]:g}_wd{c[2]:g}": v for c, v in cv.items()}

    star, cfg_val = pick(val_curve)  # the first of the grid on ties
    star_c, cfg_val_c = pick(val_curve_c)
    log.info(f"selected on VAL: plain {star}, centred {star_c}")
    q_test = emb.eval_prof["test"]
    tops = {f"{BRIDGE_CL}_s{sd}": top_k(maps[star, sd](q_test), items, MAX_K) for sd in CL_SEEDS}
    tops |= {f"{BRIDGE_CL_C}_s{sd}": top_k(centre(maps[star_c, sd], q_test), items, MAX_K)
             for sd in CL_SEEDS}  # fmt: skip
    tops |= {
        "raw": top_k(q_test, emb.items, MAX_K),
        "zca_train_users_a1.0": top_k(zca_power(q_test, *mom, 1.0), emb.items, MAX_K),
    }
    test = C.score(pop, "test", tops, d2=True, strata=True)
    is_seed = test["condition"].str.match(r"bridge_cl(_centred)?_s\d+$")
    avg = []
    for base in (BRIDGE_CL, BRIDGE_CL_C):  # the TEST score of a user: mean over the seeds
        part = test[test["condition"].str.match(rf"{base}_s\d+$")]
        a = part.groupby(["user_id", "split", "gt"], sort=False).mean(numeric_only=True)
        avg.append(a.reset_index().assign(condition=base))
    test = pd.concat([test[~is_seed], *avg], ignore_index=True)
    grid_check = C.check_against_grid(test, f"{domain}_{encoder}_{text}", "test", GRID_EQUIV)

    # sanity: one train target item per train user held out, refit without it, scored on it
    fit_m, hr, hc = holdout(inter, SEED)
    per_seed = []
    for sd in CL_SEEDS:
        m, _, _ = fit_contrastive(X, fit_m, items, Xv, rel_vk, *star, sd)
        per_seed.append(float(rank_ndcg10(m(X[hr]), items, hc).mean()))
    sanity = {
        "description": "nDCG@10 of one held-out train target item per train user (users with "
        "two or more), model refitted without it; random = expected value of a random ranking",
        "n_users": int(len(hr)), "random": float((1.0 / np.log2(np.arange(2, 12))).sum() / len(items)),
        "bridge_cl_per_seed": per_seed, "bridge_cl": float(np.mean(per_seed)),
        "raw": float(rank_ndcg10(X[hr], emb.items, hc).mean()),
        "zca_train_users_a1.0": float(rank_ndcg10(zca_power(X[hr], *mom, 1.0), emb.items, hc).mean()),
    }  # fmt: skip
    log.info(f"sanity: {sanity}")
    summary = {
        "domain": domain, "encoder": encoder, "model": src.model, "text": text,
        "model_description": "InfoNCE MLP from the H2 profile to the target embedding space "
        "(one hidden layer, ReLU), positives: one train target item of the user per epoch; "
        f"negatives: in-batch + {CL_NEG} sampled catalog items per batch; cosine / temperature",
        "grid": {"widths": list(CL_WIDTHS), "taus": list(CL_TAUS), "weight_decays": list(CL_WDS)},
        "seeds": list(CL_SEEDS), "batch": CL_BATCH, "n_neg": CL_NEG,
        "selection": "VAL users, exact nDCG@10 averaged over the seeds; early stopping on VAL",
        "selected": list(star), "selected_centred": list(star_c),
        "val_config_curve": cfg_val, "val_config_curve_centred": cfg_val_c,
        "best_epochs": epochs, "sanity": sanity, "n_fit_users": int(keep.sum()),
        "n_eval_users": {s: len(u) for s, u in pop.eval_users.items()},
        "test": C.block_means(test), "grid_check_max_abs_diff": grid_check,
        "provenance": C.provenance_of(pop, *emb.files), **C.run_info(t0),
    }  # fmt: skip
    C.write_run(f"bridgecl_{domain}_{encoder}{C.text_tag(text)}", test, summary)


def stats(encoder: str = "gte", pairset: str = "books") -> None:
    ptag, domains = C.PAIRSETS[pairset]
    rows, summ_rows, files = [], [], []
    for domain, text in C.gte_runs(domains):
        stem = f"bridge_{domain}_{encoder}{C.text_tag(text)}"
        path = C.OUT_DIR / f"{stem}.json"
        if not path.exists():
            log.warning(f"{stem}: no result, skipped")
            continue
        summ_rows += C.flat_rows(json.loads(path.read_text()))
        files.append(C.OUT_DIR / f"{stem}.csv.gz")
        cl = C.OUT_DIR / f"bridgecl_{domain}_{encoder}{C.text_tag(text)}.csv.gz"
        cl_wide = C.wide_test(cl.name.removesuffix(".csv.gz")) if cl.exists() else {}
        if cl_wide:
            files.append(cl)
            cj = json.loads(cl.with_name(cl.name.replace(".csv.gz", ".json")).read_text())
            summ_rows += [r for r in C.flat_rows(cj) if r["condition"].startswith(BRIDGE_CL)]
        for metric, wide in C.wide_test(stem).items():
            if cl_wide:
                w2 = cl_wide[metric][[BRIDGE_CL, BRIDGE_CL_C]]
                assert w2.index.equals(wide.index)
                wide = pd.concat([wide, w2], axis=1)
            meta = {"domain": domain, "encoder": encoder, "text": text, "metric": metric}
            pairs = [(f"{BRIDGE}_vs_{r}", BRIDGE, r) for r in (*CONTENT, "popularity")]
            rows += C.paired_rows(wide, pairs, meta, "bridge_vs_references")
            if BRIDGE_MLP in wide:  # a separate family: the ridge rows stay as they were
                pairs = [(f"{BRIDGE_MLP}_vs_{r}", BRIDGE_MLP, r)
                         for r in (*CONTENT, BRIDGE, "popularity")]  # fmt: skip
                rows += C.paired_rows(wide, pairs, meta, "bridge_mlp_vs_references")
            if BRIDGE_CL in wide:  # the contrastive bridges, again their own families
                pairs = [(f"{BRIDGE_CL}_vs_{r}", BRIDGE_CL, r)
                         for r in (*CONTENT, BRIDGE, BRIDGE_MLP, "popularity")]  # fmt: skip
                rows += C.paired_rows(wide, pairs, meta, "bridge_cl_vs_references")
                pairs = [(f"{BRIDGE_CL_C}_vs_{r}", BRIDGE_CL_C, r)
                         for r in (*CONTENT, BRIDGE, BRIDGE_CL)]  # fmt: skip
                rows += C.paired_rows(wide, pairs, meta, "bridge_cl_centred_vs_references")
        log.info(f"{stem}: tested")
    pd.DataFrame(rows).to_csv(C.OUT_DIR / f"stats_bridge{ptag}.csv", index=False, float_format="%.6g")
    pd.DataFrame(summ_rows).to_csv(C.OUT_DIR / f"summary_bridge{ptag}.csv", index=False,
                                   float_format="%.6g")  # fmt: skip
    (C.OUT_DIR / f"stats_bridge_provenance{ptag}.json").write_text(
        json.dumps({"provenance": provenance(files)}, indent=2))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=C.DOMAINS + C.DOMAINS2)
    ap.add_argument("--encoder", default="gte")
    ap.add_argument("--text", choices=C.ITEM_TEXTS, default="titlecat")
    ap.add_argument("--pairset", choices=list(C.PAIRSETS), default="books")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--contrastive", action="store_true", help="the InfoNCE bridge of --domain")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if args.stats:
        stats(args.encoder, args.pairset)
    elif args.domain and args.contrastive:
        run_contrastive(args.domain, args.encoder, args.text)
    elif args.domain:
        run(args.domain, args.encoder, args.text)
    else:
        ap.error("give --domain or --stats")


if __name__ == "__main__":
    main()
