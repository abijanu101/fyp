"""Experiment 10 follow-up: score the saved held-out predictions against better targets, the mean of
the experts and longer stretches of time, next to a human doing the same. No model is trained.

  python -m scripts.analysis.aggregated_scoring \
         [--predictions outputs/experiments/10_clustering_v2/ablation/ablation_predictions.csv.gz]
         [--split group] [--arms base normalised+sync agglomerative sync+agglomerative]
         [--bins 0 30 60 120 session] [--boot 1000] [--out <json>]

Bins: each student-session (group/activity/subject) is cut into stretches of B seconds from its first
entry (`0` = each entry on its own, `session` = the whole session). Per bin and labeler: the mean
rating and the mean prediction of that labeler's model. Then, pooled over bins:

  experts' labels
    model vs labeler     each labeler's model against that labeler's mean (the usual score)
    expert vs others     a labeler against the mean of the other experts in the same bin: the
                         human ceiling (each labeler weighted equally)
    model vs others      the model (mean over the labelers' models) against the same mean of the
                         other experts, on the same pairs: directly comparable with the row above
    model vs consensus   the model against the mean of every expert, bins with >= 2 experts
  self-report
    model vs self        the self-report model against the student's mean
    experts vs self      the mean expert rating in the bin against it (the human ceiling)
    expert model vs self the experts' models (mean) against it
    each also within student (both sides minus the student's mean over bins); the experts' rows also
    within student-session (each labeler's values minus that labeler's mean in the session: what
    changes from moment to moment)

`0` matches each entry with the other labelers' entries in the same bin of 4 s (one window), so it
is close to, not the same as, experiment 06's entry-level ceiling (which interpolates). Intervals:
percentile bootstrap over class sessions (group/activity), --boot resamples.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def load(path: str, split: str, arms: list[str]) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[(d["split"] == split) & d["arm"].isin(arms)]
    # one row per (track, arm, entry): the prediction averaged over the entry's rows, as ablation does
    d = d.groupby(["track", "arm", "entry_id"], as_index=False).agg(rating=("rating", "first"), pred=("pred", "mean"))
    p = d["entry_id"].str.split("/", expand=True)           # group/experiment/subject/labeler/t (s)
    return d.assign(cls=p[0] + "/" + p[1], student=p[0] + "/" + p[2], sess=p[0] + "/" + p[1] + "/" + p[2],
                    labeler=p[3], t=p[4].astype(float))


def add_bins(d: pd.DataFrame, width: str) -> pd.DataFrame:
    if width == "session":
        return d.assign(bin=0)
    w = float(width) or 4.0                                 # 0: one 4 s window per bin
    start = d.groupby("sess")["t"].transform("min")
    return d.assign(bin=np.floor((d["t"] - start) / w).astype(int))


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def demean(v: pd.Series, by: pd.Series) -> np.ndarray:
    return (v - v.groupby(by).transform("mean")).to_numpy(float)


def with_interval(pairs: pd.DataFrame, x: str, y: str, boot: int, rng, within: str | None = None) -> dict:
    """Spearman of pairs[x] against pairs[y]; percentile interval over class sessions. `within` (a
    column: student, sess): both sides are centred on that unit's mean over all pairs once, and the
    resamples keep that centring (a student spans several class sessions)."""
    if within:
        xs, ys = demean(pairs[x], pairs[within]), demean(pairs[y], pairs[within])
    else:
        xs, ys = pairs[x].to_numpy(float), pairs[y].to_numpy(float)
    out = {"rho": rank_corr(xs, ys), "n": int(len(pairs))}
    if boot and len(pairs) >= 20:
        cls = pairs["cls"].to_numpy()
        idx = [np.flatnonzero(cls == c) for c in np.unique(cls)]
        vals = []
        for _ in range(boot):
            pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
            vals.append(rank_corr(xs[pick], ys[pick]))
        out["lo"], out["hi"] = (float(v) for v in np.nanpercentile(vals, [2.5, 97.5]))
    return out


def expert_pairs(e: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (bin, labeler): rating, own model's prediction, the other experts' mean rating and the mean
    prediction of every labeler's model in the bin; and per bin: the consensus."""
    keys = ["cls", "student", "sess", "bin"]
    per = e.groupby(keys + ["labeler"], as_index=False).agg(r=("rating", "mean"), p=("pred", "mean"))
    g = per.groupby(keys)
    per["n_lab"] = g["r"].transform("size")
    tot_r = g["r"].transform("sum")
    per["others"] = np.where(per["n_lab"] > 1, (tot_r - per["r"]) / (per["n_lab"] - 1).clip(lower=1), np.nan)
    per["model"] = g["p"].transform("mean")
    # within a session, each labeler is centred on their own mean: centring a labeler's rating and the
    # mean of the *others* on one shared session mean would tie them negatively by construction
    per["sess_lab"] = per["sess"] + "|" + per["labeler"]
    cons = per.groupby(keys, as_index=False).agg(consensus=("r", "mean"), model=("p", "mean"), n_lab=("r", "size"))
    return per, cons


def score(d: pd.DataFrame, arm: str, width: str, boot: int, rng) -> dict:
    a = add_bins(d[d["arm"] == arm], width)
    e, s = a[a["track"] == "experts"], a[a["track"] == "self"]
    per, cons = expert_pairs(e)
    multi = per[per["n_lab"] > 1]
    keys = ["cls", "student", "sess", "bin"]
    sb = s.groupby(keys, as_index=False).agg(self_r=("rating", "mean"), model=("pred", "mean"))
    eb = per.groupby(keys, as_index=False).agg(experts=("r", "mean"), expert_model=("p", "mean"))
    sm = sb.merge(eb, on=keys, how="inner")
    res = {
        "experts": {
            "model_vs_labeler": with_interval(per, "p", "r", boot, rng),
            "expert_vs_others": with_interval(multi, "r", "others", boot, rng),
            "model_vs_others": with_interval(multi, "model", "others", boot, rng),
            "model_vs_consensus": with_interval(cons[cons["n_lab"] > 1], "model", "consensus", boot, rng),
            "model_vs_labeler_within": with_interval(per, "p", "r", boot, rng, within="sess_lab"),
            "expert_vs_others_within": with_interval(multi, "r", "others", boot, rng, within="sess_lab"),
            "model_vs_others_within": with_interval(multi, "model", "others", boot, rng, within="sess_lab"),
            "bins": int(len(cons)), "bins_multi": int((cons["n_lab"] > 1).sum()),
        },
        "self": {
            "model_vs_self": with_interval(sb, "model", "self_r", boot, rng),
            "model_vs_self_within": with_interval(sb, "model", "self_r", boot, rng, within="student"),
            "experts_vs_self": with_interval(sm, "experts", "self_r", boot, rng),
            "experts_vs_self_within": with_interval(sm, "experts", "self_r", boot, rng, within="student"),
            "expert_model_vs_self": with_interval(sm, "expert_model", "self_r", boot, rng),
            "expert_model_vs_self_within": with_interval(sm, "expert_model", "self_r", boot, rng, within="student"),
            "bins": int(len(sb)), "bins_with_experts": int(len(sm)),
        },
    }
    return res


def fmt(r: dict) -> str:
    if not np.isfinite(r["rho"]):
        return "–"
    ci = f" [{r['lo']:+.3f}, {r['hi']:+.3f}]" if "lo" in r else ""
    return f"{r['rho']:+.3f}{ci}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="outputs/experiments/10_clustering_v2/ablation/ablation_predictions.csv.gz")
    ap.add_argument("--split", default="group")
    ap.add_argument("--arms", nargs="*", default=["base", "normalised+sync", "agglomerative", "sync+agglomerative"])
    ap.add_argument("--bins", nargs="*", default=["0", "30", "60", "120", "session"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = load(args.predictions, args.split, args.arms)
    rng = np.random.default_rng(args.seed)
    results = {}
    for arm in args.arms:
        for w in args.bins:
            r = score(d, arm, w, args.boot, rng)
            results[f"{arm}|{w}"] = r
            ex, se = r["experts"], r["self"]
            print(f"\n{arm}, bins {w}{' s' if w not in ('session',) else ''}: {ex['bins']} expert bins "
                  f"({ex['bins_multi']} with 2+ experts), {se['bins']} self-report bins", flush=True)
            for k in ("model_vs_labeler", "expert_vs_others", "model_vs_others"):
                print(f"  experts  {k:24s} {fmt(ex[k])}   within session {fmt(ex[k + '_within'])}  (n {ex[k]['n']})")
            print(f"  experts  {'model_vs_consensus':24s} {fmt(ex['model_vs_consensus'])}  (n {ex['model_vs_consensus']['n']})")
            for k in ("model_vs_self", "experts_vs_self", "expert_model_vs_self"):
                print(f"  self     {k:24s} {fmt(se[k])}   within student {fmt(se[k + '_within'])}  (n {se[k]['n']})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"split": args.split, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
