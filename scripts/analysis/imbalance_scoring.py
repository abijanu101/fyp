"""Experiment 10 follow-up: turn the models' scores into 1-5 ratings that use the scale as the
labelers do, and score them with metrics that do not reward always answering 3. Reads the saved
held-out predictions; no model is trained.

  python -m scripts.analysis.imbalance_scoring
         [--predictions outputs/experiments/10_clustering_v2/ablation/ablation_predictions.csv.gz]
         [--arms base normalised agglomerative sync+agglomerative normalised+sync] [--boot 1000]
         [--out outputs/experiments/10_clustering_v2/imbalance]

Group split (held-out students). A score becomes a rating three ways:
  round     the nearest whole rating, halves toward 3 (what every table so far reports)
  quantile  per labeler, cut points at the quantiles of the held-out group's OWN predictions (no
            labels) placed at the labeler's class shares in the OTHER groups' ratings: the top
            share-of-5s of the scores becomes a 5, and so on. Within a labeler the order is kept, so
            ranking cannot change; the predicted classes take the labeler's usual shares. (Fitting
            the cut points on the other groups' predictions instead would leak: their models were
            trained on this group's labels.)
  random    ratings drawn at random with the same shares (the mean of 20 draws): what matching the
            shares alone gives, the reference a mapping must beat
The experts' rating 1 (3 of 21,519 entries) is merged into 2, so their scale is 2-5; the self-report
keeps 1-5. Classes with fewer than 10 entries in the test data are left out of the per-class
averages.

Metrics: quadratic weighted kappa (QWK, the agreement measure for ordinal ratings), macro-F1,
balanced accuracy (mean recall per class), recall per class, MAE, macro-MAE (MAE per true class,
averaged), within one point, the predicted class shares; from the raw score: Spearman, AUC for
engaged (>= 3), and for detecting "highly engaged" (>= 4) the precision-recall AUC against its
base rate. Human reference on the same entries: another expert's rating in force at the entry's
time (as is), and the mean of the other experts' ratings in force mapped the same way (the rating
tracks are rebuilt from the saved entries, which lack moments when a student was not tracked: an
approximation, 0.156 against experiment 06's 0.143 for the mean). Intervals: bootstrap over class
sessions, paired for differences. Written to <out>/imbalance.json and imbalance.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

SELF = "self_labeling"
MIN_SUPPORT = 10


# ------------------------------------------------------------------ data
def load(path: str, arms: list[str]) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[(d["split"] == "group") & d["arm"].isin(arms)]
    d = d.groupby(["arm", "track", "entry_id"], as_index=False).agg(rating=("rating", "first"), pred=("pred", "mean"))
    p = d["entry_id"].str.split("/", expand=True)                      # group/experiment/subject/labeler/t
    return d.assign(grp=p[0], cls=p[0] + "/" + p[1], sess=p[0] + "/" + p[1] + "/" + p[2], labeler=p[3],
                    t=p[4].astype(float))


def levels_of(track: str) -> np.ndarray:
    return np.array([2, 3, 4, 5]) if track == "experts" else np.array([1, 2, 3, 4, 5])


def to_levels(y: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Ratings on the track's scale (the experts' 1 merged into 2)."""
    return np.clip(np.round(y), levels.min(), levels.max())


# ------------------------------------------------------------------ mappings
def round_map(score: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Nearest whole rating, halves toward 3, clipped to the scale."""
    f = np.floor(score)
    frac = score - f
    up = (frac > 0.5) | (np.isclose(frac, 0.5) & (f < 3))
    return np.clip(f + up, levels.min(), levels.max())


def class_shares(y: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.array([(y == k).mean() for k in levels]) if len(y) else np.full(len(levels), 1 / len(levels))


def quantile_map(score: np.ndarray, shares: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """The lowest share[0] of the scores become levels[0], the next share[1] levels[1], and so on
    (ties in the score share one class: average ranks)."""
    if len(score) == 0:
        return np.zeros(0)
    q = (pd.Series(score).rank(method="average").to_numpy() - 0.5) / len(score)
    cum = np.cumsum(shares / shares.sum())
    return levels[np.minimum(np.searchsorted(cum, q, side="left"), len(levels) - 1)]


def map_fold(d: pd.DataFrame, test_grp: str, levels: np.ndarray, method: str, rng=None) -> np.ndarray:
    """Classes for the entries of one held-out group: per labeler, the shares of that labeler's
    ratings in the other groups (pooled over labelers if they have none there)."""
    out = np.zeros(len(d))
    test = (d["grp"] == test_grp).to_numpy()
    y_all = to_levels(d["rating"].to_numpy(float), levels)
    pooled = class_shares(y_all[~test], levels)
    for lab in d.loc[test, "labeler"].unique():
        m = test & (d["labeler"] == lab).to_numpy()
        tr = ~test & (d["labeler"] == lab).to_numpy()
        sh = class_shares(y_all[tr], levels) if tr.sum() >= 30 else pooled
        s = d.loc[m, "pred"].to_numpy(float)
        if method == "quantile":
            out[m] = quantile_map(s, sh, levels)
        elif method == "random":
            out[m] = rng.choice(levels, size=m.sum(), p=sh / sh.sum())
        else:
            out[m] = round_map(s, levels)
    return out[test]


def mapped(d: pd.DataFrame, levels: np.ndarray, method: str, rng=None) -> np.ndarray:
    """Classes for every entry of d (one label set, one arm), group by group."""
    out = np.zeros(len(d))
    for g in d["grp"].unique():
        out[(d["grp"] == g).to_numpy()] = map_fold(d, g, levels, method, rng)
    return out


# ------------------------------------------------------------------ metrics
def rank_corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(pd.Series(x[ok]).rank(), pd.Series(y[ok]).rank())[0, 1])


def class_metrics(y: np.ndarray, c: np.ndarray, levels: np.ndarray) -> dict:
    """Metrics of the predicted classes c against the ratings y (both on the track's scale)."""
    present = [k for k in levels if (y == k).sum() >= MIN_SUPPORT]
    rec = {int(k): float(np.mean(c[y == k] == k)) for k in present}
    f1 = []
    for k in present:
        tp = np.sum((c == k) & (y == k))
        prec = tp / max(np.sum(c == k), 1)
        r = tp / max(np.sum(y == k), 1)
        f1.append(0.0 if prec + r == 0 else 2 * prec * r / (prec + r))
    return {"qwk": float(cohen_kappa_score(y, c, weights="quadratic", labels=levels)) if len(set(c)) > 1 or len(set(y)) > 1 else 0.0,
            "macro_f1": float(np.mean(f1)) if f1 else float("nan"),
            "balanced_acc": float(np.mean(list(rec.values()))) if rec else float("nan"),
            "recall": rec, "mae": float(np.mean(np.abs(c - y))),
            "macro_mae": float(np.mean([np.mean(np.abs(c[y == k] - k)) for k in present])) if present else float("nan"),
            "within_1": float(np.mean(np.abs(c - y) <= 1)),
            "pred_shares": {int(k): float(np.mean(c == k)) for k in levels},
            "class_spearman": rank_corr(c, y)}


def score_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    out = {"spearman": rank_corr(s, y)}
    ok = np.isfinite(s)
    for name, pos in (("auc_engaged", y >= 3), ("auc_high", y >= 4)):
        p = pos[ok]
        out[name] = float(roc_auc_score(p, s[ok])) if 0 < p.sum() < len(p) else float("nan")
    p = (y >= 4)[ok]
    out["pr_auc_high"] = float(average_precision_score(p, s[ok])) if 0 < p.sum() < len(p) else float("nan")
    out["high_base_rate"] = float(p.mean())
    return out


BOOT_KEYS = ("qwk", "macro_f1", "balanced_acc", "macro_mae")


def boot_classes(y, c, cls, levels, W) -> dict:
    """Bootstrap values (over class sessions, rows of W are class resample counts) of the class
    metrics in BOOT_KEYS, for paired differences."""
    codes = pd.factorize(cls)[0]
    out = {k: [] for k in BOOT_KEYS}
    idx = [np.flatnonzero(codes == j) for j in range(codes.max() + 1)]
    for w in W:
        pick = np.concatenate([np.repeat(idx[j], int(w[j])) for j in range(len(idx)) if w[j] > 0])
        m = class_metrics(y[pick], c[pick], levels)
        for k in BOOT_KEYS:
            out[k].append(m[k])
    return {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------ human reference
def in_force_reference(d: pd.DataFrame) -> pd.DataFrame:
    """Per entry: one other expert's rating in force at its time (the first other expert who had
    rated the student by then) and the mean of the other experts' ratings in force; for the self-
    report, the experts'. Step functions rebuilt from the saved entries."""
    from src.labels.attention import AttentionTrack

    base = d.drop_duplicates("entry_id")
    tracks = {(s, l): AttentionTrack(g["t"].to_numpy(), g["rating"].to_numpy())
              for (s, l), g in base.sort_values("t").groupby(["sess", "labeler"])}
    experts = {}
    for (s, l), tr in tracks.items():
        if l != SELF:
            experts.setdefault(s, []).append((l, tr))
    one, mean = [], []
    for s, l, t in zip(base["sess"], base["labeler"], base["t"]):
        v = [tr.at(t) for other, tr in sorted(experts.get(s, [])) if other != l]
        v = [x for x in v if np.isfinite(x)]
        one.append(v[0] if v else np.nan)
        mean.append(float(np.mean(v)) if v else np.nan)
    return pd.DataFrame({"entry_id": base["entry_id"].to_numpy(), "ref_one": one, "ref_mean": mean})


# ------------------------------------------------------------------ main
def evaluate(d: pd.DataFrame, track: str, rng, W: np.ndarray, n_random: int = 20) -> dict:
    """Every mapping of one arm's scores on one label set, with paired differences."""
    lv = levels_of(track)
    d = d.sort_values("entry_id").reset_index(drop=True)
    y = to_levels(d["rating"].to_numpy(float), lv)
    s = d["pred"].to_numpy(float)
    res = {"entries": int(len(d)), "true_shares": {int(k): float(np.mean(y == k)) for k in lv},
           "score": score_metrics(y, s)}
    cls = d["cls"].to_numpy()
    c_round, c_quant = mapped(d, lv, "round"), mapped(d, lv, "quantile")
    res["round"], res["quantile"] = class_metrics(y, c_round, lv), class_metrics(y, c_quant, lv)
    draws = [mapped(d, lv, "random", rng) for _ in range(n_random)]
    rnd = [class_metrics(y, c, lv) for c in draws]
    res["random"] = {k: float(np.nanmean([m[k] for m in rnd])) for k in ("qwk", "macro_f1", "balanced_acc", "mae",
                                                                          "macro_mae", "within_1")}
    res["constant_3"] = class_metrics(y, np.full(len(y), 3.0), lv)
    if len(W):
        br, bq = boot_classes(y, c_round, cls, lv, W), boot_classes(y, c_quant, cls, lv, W)
        bn = boot_classes(y, draws[0], cls, lv, W)
        res["differences"] = {}
        for name, a, b in (("quantile - round", bq, br), ("quantile - random", bq, bn)):
            res["differences"][name] = {
                k: {"diff": float(np.nanmean(a[k] - b[k])), "lo": float(np.nanpercentile(a[k] - b[k], 2.5)),
                    "hi": float(np.nanpercentile(a[k] - b[k], 97.5))} for k in BOOT_KEYS}
    res["per_labeler"] = {}
    for lab, g in d.groupby("labeler"):
        if len(g) >= 200:
            i = g.index.to_numpy()
            res["per_labeler"][lab] = {"entries": int(len(g)), "qwk_round": class_metrics(y[i], c_round[i], lv)["qwk"],
                                       "qwk_quantile": class_metrics(y[i], c_quant[i], lv)["qwk"],
                                       "pred_shares_quantile": class_metrics(y[i], c_quant[i], lv)["pred_shares"],
                                       "true_shares": {int(k): float(np.mean(y[i] == k)) for k in lv}}
    return res


def human(d: pd.DataFrame, ref: pd.DataFrame, track: str) -> dict:
    """The human reference on the same entries: another expert's rating as is, and the mean of the
    other experts mapped the same way as the models' scores."""
    lv = levels_of(track)
    h = d.drop_duplicates("entry_id").merge(ref, on="entry_id", how="inner").dropna(subset=["ref_mean"])
    h = h.sort_values("entry_id").reset_index(drop=True)
    y = to_levels(h["rating"].to_numpy(float), lv)
    out = {"entries": int(len(h)), "score_mean": score_metrics(y, h["ref_mean"].to_numpy(float))}
    one = h["ref_one"].to_numpy(float)
    ok = np.isfinite(one)
    if ok.sum() >= 30:
        out["one_expert_as_is"] = class_metrics(y[ok], to_levels(one[ok], lv), lv)
    out["mean_quantile"] = class_metrics(y, mapped(h.assign(pred=h["ref_mean"]), lv, "quantile"), lv)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="outputs/experiments/10_clustering_v2/ablation/ablation_predictions.csv.gz")
    ap.add_argument("--arms", nargs="*", default=["base", "normalised", "agglomerative", "sync+agglomerative", "normalised+sync"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/experiments/10_clustering_v2/imbalance")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    d = load(args.predictions, args.arms)
    ref = in_force_reference(d[d["arm"] == args.arms[0]])
    results, table = {"settings": vars(args), "tracks": {}}, []
    for track in ("experts", "self"):
        dt = d[d["track"] == track]
        n_cls = dt["cls"].nunique()
        W = rng.multinomial(n_cls, np.full(n_cls, 1 / n_cls), size=args.boot) if args.boot and n_cls >= 3 else np.zeros((0, n_cls))
        results["tracks"][track] = {"human": human(dt[dt["arm"] == args.arms[0]], ref, track), "arms": {}}
        hu = results["tracks"][track]["human"]
        print(f"\n=== {track} ({hu['entries']} entries also have a human reference)")
        for arm in args.arms:
            r = evaluate(dt[dt["arm"] == arm], track, rng, W)
            results["tracks"][track]["arms"][arm] = r
            for m in ("round", "quantile", "random", "constant_3"):
                v = r[m]
                table.append({"track": track, "arm": arm, "mapping": m, **{k: v.get(k) for k in
                              ("qwk", "macro_f1", "balanced_acc", "mae", "macro_mae", "within_1")}})
            print(f"  {arm:20s} true shares {r['true_shares']}  Spearman {r['score']['spearman']:+.3f}  "
                  f"PR-AUC(>=4) {r['score']['pr_auc_high']:.3f} (base rate {r['score']['high_base_rate']:.3f})")
            for m in ("constant_3", "round", "random", "quantile"):
                v = r[m]
                print(f"    {m:11s} QWK {v['qwk']:+.3f}  macro-F1 {v['macro_f1']:.3f}  balanced acc {v['balanced_acc']:.3f}  "
                      f"MAE {v['mae']:.3f}  macro-MAE {v['macro_mae']:.3f}  within-1 {v['within_1']:.3f}")
            for name, diffs in r.get("differences", {}).items():
                print(f"    {name:18s} " + "  ".join(f"{k} {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]" for k, x in diffs.items()))
        if "one_expert_as_is" in hu:
            v = hu["one_expert_as_is"]
            print(f"  {'human: one expert':20s} QWK {v['qwk']:+.3f}  macro-F1 {v['macro_f1']:.3f}  balanced acc {v['balanced_acc']:.3f}  "
                  f"MAE {v['mae']:.3f}  macro-MAE {v['macro_mae']:.3f}")
        v = hu["mean_quantile"]
        print(f"  {'human: mean, mapped':20s} QWK {v['qwk']:+.3f}  macro-F1 {v['macro_f1']:.3f}  balanced acc {v['balanced_acc']:.3f}  "
              f"MAE {v['mae']:.3f}  macro-MAE {v['macro_mae']:.3f}   Spearman {hu['score_mean']['spearman']:+.3f}  "
              f"PR-AUC(>=4) {hu['score_mean']['pr_auc_high']:.3f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "imbalance.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    pd.DataFrame(table).to_csv(out / "imbalance.csv", index=False)
    print(f"\nwrote {out / 'imbalance.json'}")


if __name__ == "__main__":
    main()
