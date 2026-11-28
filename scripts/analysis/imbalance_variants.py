"""Experiment 11 follow-up: three ways to get more out of the cut points, on saved predictions (no
model is trained). Every variant is fixed in advance and scored on the same entries (those every
source predicted, group split), against experiment 11's reference: `base` (per-labeler logistic
regression, raw inputs) with each labeler's own class shares.

  python -m scripts.analysis.imbalance_variants [--boot 1000] [--out outputs/experiments/10_clustering_v2/imbalance]

  1  the target shares of the cut points (scores of `base`):
       own        each labeler's class shares in the other groups (experiment 11)
       tempered   halfway between those and equal shares
       equal      equal shares over the scale
     With a weak signal, predicting the rare classes more often than they occur can raise macro-F1
     and balanced accuracy, at a cost to QWK; the three are reported side by side, none chosen.
  2  one pooled model: the pooled ridge of experiment 10 Appendix D (W0p: one model for all experts,
     normalised inputs, absolute rating), with each labeler's own shares; against the per-labeler
     logistic regression on the same inputs (`normalised`), so only pooling differs
  3  averaging models: the mean percentile rank of `base`, `agglomerative` and `sync+agglomerative`
     (ranks within each labeler and held-out group, so the models' scales do not matter), with each
     labeler's own shares

Metrics, intervals and the leakage guard as in imbalance_scoring (the cut points come from the held-
out group's own scores and the other groups' ratings). Written to <out>/variants.json and printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.imbalance_scoring import (BOOT_KEYS, boot_classes, class_metrics, class_shares, levels_of,
                                                 quantile_map, score_metrics, to_levels)

ENSEMBLE = ["base", "agglomerative", "sync+agglomerative"]


def shares_for(base: np.ndarray, mode: str) -> np.ndarray:
    """The target shares: own, tempered (halfway to equal) or equal."""
    eq = np.full(len(base), 1 / len(base))
    return {"own": base, "tempered": (base + eq) / 2, "equal": eq}[mode]


def cut(d: pd.DataFrame, levels: np.ndarray, mode: str = "own") -> np.ndarray:
    """Per held-out group and labeler: the labeler's scores in that group cut at the shares (`mode`)
    of the labeler's ratings in the other groups (pooled over labelers below 30 entries there)."""
    out = np.zeros(len(d))
    y = to_levels(d["rating"].to_numpy(float), levels)
    grp, lab, s = d["grp"].to_numpy(), d["labeler"].to_numpy(), d["pred"].to_numpy(float)
    for g in np.unique(grp):
        test = grp == g
        pooled = class_shares(y[~test], levels)
        for l in np.unique(lab[test]):
            m, tr = test & (lab == l), ~test & (lab == l)
            sh = class_shares(y[tr], levels) if tr.sum() >= 30 else pooled
            out[m] = quantile_map(s[m], shares_for(sh, mode), levels)
    return out


def rank_average(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """The mean percentile rank of several models' scores, ranked within each labeler and held-out
    group (on the entries all of them predicted)."""
    parts = []
    for f in frames:
        r = f.groupby(["labeler", "grp"])["pred"].rank(pct=True)
        parts.append(f.assign(pred=r).set_index("entry_id")["pred"])
    m = pd.concat(parts, axis=1, join="inner").mean(axis=1)
    base = frames[0].set_index("entry_id").loc[m.index].reset_index()
    return base.assign(pred=m.to_numpy())


def load_sources(abl: str, ws: str) -> dict:
    """track -> {source name: entries (entry_id, rating, pred, labeler, grp, cls)}, every source on
    the entries every source predicted (group split)."""
    a = pd.read_csv(abl)
    a = a[a["split"] == "group"].groupby(["arm", "track", "entry_id"], as_index=False).agg(
        rating=("rating", "first"), pred=("pred", "mean"))
    w = pd.read_csv(ws)
    w = w[(w["split"] == "group") & (w["arm"] == "W0p")][["track", "entry_id", "rating", "pred"]].assign(arm="pooled")
    allp = pd.concat([a, w], ignore_index=True)
    p = allp["entry_id"].str.split("/", expand=True)
    allp = allp.assign(grp=p[0], cls=p[0] + "/" + p[1], labeler=p[3])
    out = {}
    for track, g in allp.groupby("track"):
        srcs = {arm: h.drop(columns=["arm", "track"]).reset_index(drop=True) for arm, h in g.groupby("arm")}
        need = ["base", "normalised", "pooled"] + ENSEMBLE
        common = set.intersection(*(set(srcs[s]["entry_id"]) for s in need))
        srcs = {k: v[v["entry_id"].isin(common)].sort_values("entry_id").reset_index(drop=True)
                for k, v in srcs.items() if k in need}
        srcs["ensemble"] = rank_average([srcs[s] for s in ENSEMBLE]).sort_values("entry_id").reset_index(drop=True)
        out[track] = srcs
    return out


VARIANTS = [("reference: base, own shares", "base", "own"), ("1a base, tempered shares", "base", "tempered"),
            ("1b base, equal shares", "base", "equal"), ("2a normalised (per labeler), own shares", "normalised", "own"),
            ("2b pooled ridge, own shares", "pooled", "own"), ("3 average of 3 models, own shares", "ensemble", "own")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="outputs/experiments/10_clustering_v2/ablation/ablation_predictions.csv.gz")
    ap.add_argument("--within", default="outputs/experiments/10_clustering_v2/within_session/within_session_predictions.csv.gz")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/experiments/10_clustering_v2/imbalance")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    data = load_sources(args.predictions, args.within)
    res = {"settings": vars(args), "tracks": {}}
    for track, srcs in data.items():
        lv = levels_of(track)
        ref = srcs["base"]
        y = to_levels(ref["rating"].to_numpy(float), lv)
        n_cls = ref["cls"].nunique()
        W = rng.multinomial(n_cls, np.full(n_cls, 1 / n_cls), size=args.boot)
        cls = ref["cls"].to_numpy()
        res["tracks"][track] = {"entries": int(len(ref)), "variants": {}}
        print(f"\n=== {track}: {len(ref)} entries (every source predicted them)")
        boots = {}
        for name, src, mode in VARIANTS:
            d = srcs[src]
            assert (d["entry_id"].to_numpy() == ref["entry_id"].to_numpy()).all()
            c = cut(d, lv, mode)
            m = class_metrics(y, c, lv)
            m["score"] = score_metrics(y, d["pred"].to_numpy(float))
            m["per_labeler_qwk"] = {l: class_metrics(y[(d["labeler"] == l).to_numpy()], c[(d["labeler"] == l).to_numpy()], lv)["qwk"]
                                    for l in d["labeler"].unique() if (d["labeler"] == l).sum() >= 200}
            boots[name] = boot_classes(y, c, cls, lv, W)
            res["tracks"][track]["variants"][name] = m
        refb = boots[VARIANTS[0][0]]
        for name, _, _ in VARIANTS:
            m = res["tracks"][track]["variants"][name]
            if name != VARIANTS[0][0]:
                m["minus_reference"] = {k: {"diff": float(np.nanmean(boots[name][k] - refb[k])),
                                            "lo": float(np.nanpercentile(boots[name][k] - refb[k], 2.5)),
                                            "hi": float(np.nanpercentile(boots[name][k] - refb[k], 97.5))} for k in BOOT_KEYS}
            rec = " / ".join(f"{v:.0%}" for v in m["recall"].values())
            print(f"  {name:42s} QWK {m['qwk']:+.3f}  macro-F1 {m['macro_f1']:.3f}  balanced acc {m['balanced_acc']:.3f}  "
                  f"MAE {m['mae']:.3f}  recall {rec}  Spearman {m['score']['spearman']:+.3f}")
            if "minus_reference" in m:
                print("      vs reference: " + "  ".join(f"{k} {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"
                                                         for k, x in m["minus_reference"].items()))
            print("      per labeler QWK: " + ", ".join(f"{l} {v:+.3f}" for l, v in sorted(m["per_labeler_qwk"].items())))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "variants.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {out / 'variants.json'}")


if __name__ == "__main__":
    main()
