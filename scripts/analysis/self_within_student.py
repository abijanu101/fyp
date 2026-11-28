"""Experiment 10, §3.7: how much of the self-report varies within a student, and who tracks it there.
Reads the held-out predictions that `scripts.clustering.ablation` saves; no model is trained.

  python -m scripts.analysis.self_within_student \
         [--predictions outputs/experiments/10_clustering_v2/ablation/ablation_predictions.csv.gz]
         [--split group] [--tolerance 5 15]

Prints, for the self-report entries of the split:
  - entries per student, distinct scores per student, and the share of the variance within a student
    (student = group/subject, the same person across activities) and within a student-session;
  - each arm's held-out Spearman, pooled and within student (the rating and the prediction each
    minus that student's mean; students with a single score left out);
  - the nearest expert rating, and the expert models' prediction at it (same student and session,
    within --tolerance seconds), against the self-report: pooled and within student;
  - the within-session Spearman of the rating with time, for both label sets.
"""

from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import spearmanr


def split_ids(d: pd.DataFrame) -> pd.DataFrame:
    p = d["entry_id"].str.split("/", expand=True)     # group/experiment/subject/labeler/t
    return d.assign(grp=p[0], student=p[0] + "/" + p[2], sess=p[0] + "/" + p[1] + "/" + p[2], t=p[4].astype(float))


def within_rho(student: pd.Series, rating: pd.Series, pred: pd.Series) -> float:
    g = pd.DataFrame({"s": student.to_numpy(), "r": rating.to_numpy(), "p": pred.to_numpy()})
    g = g[g.groupby("s")["r"].transform("nunique") > 1]
    return spearmanr(g["p"] - g.groupby("s")["p"].transform("mean"),
                     g["r"] - g.groupby("s")["r"].transform("mean")).statistic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="outputs/experiments/10_clustering_v2/ablation/ablation_predictions.csv.gz")
    ap.add_argument("--split", default="group")
    ap.add_argument("--tolerance", type=float, nargs="*", default=[5.0, 15.0])
    args = ap.parse_args()

    d = pd.read_csv(args.predictions)
    d = d[d["split"] == args.split]
    # one row per (track, arm, entry), the prediction averaged over the entry's rows, as ablation's
    # intervals do (an entry can be matched to more than one row)
    d = split_ids(d.groupby(["track", "arm", "entry_id"], as_index=False).agg(rating=("rating", "first"),
                                                                             pred=("pred", "mean")))
    s = d[(d["track"] == "self") & (d["arm"] == "base")]
    print(f"self-report entries {len(s)}, students {s.student.nunique()}, student-sessions {s.sess.nunique()}, "
          f"groups {s.grp.nunique()}")
    print("ratings:", s["rating"].value_counts().sort_index().to_dict())
    per = s.groupby("student")["rating"].agg(["size", "nunique", "std"])
    print(f"entries per student: median {per['size'].median():.0f}, min {per['size'].min()}, max {per['size'].max()}")
    print("distinct scores per student:", per["nunique"].value_counts().sort_index().to_dict())
    tot = s["rating"].var(ddof=0)
    for unit in ("student", "sess"):
        w = (s["rating"] - s.groupby(unit)["rating"].transform("mean")).var(ddof=0)
        print(f"variance within {unit}: {w:.3f} of {tot:.3f} ({w / tot:.0%})")
    print(f"within-student SD: median {per['std'].median():.2f}, quartiles {per['std'].quantile(.25):.2f}-"
          f"{per['std'].quantile(.75):.2f}")

    print("\narm: held-out Spearman pooled / within student")
    for arm in d["arm"].unique():
        g = d[(d["track"] == "self") & (d["arm"] == arm)]
        print(f"  {arm:30s} {spearmanr(g.pred, g.rating).statistic:+.3f} / {within_rho(g.student, g.rating, g.pred):+.3f}")

    e = d[(d["track"] == "experts") & (d["arm"] == "base")]
    e = e[["sess", "t", "rating", "pred"]].rename(columns={"t": "te", "rating": "exp_rating", "pred": "exp_pred"})
    for tol in args.tolerance:
        m = pd.merge_asof(s.sort_values("t"), e.sort_values("te"), left_on="t", right_on="te", by="sess",
                          direction="nearest", tolerance=tol).dropna(subset=["exp_rating"])
        print(f"\nnearest expert entry within {tol:g} s: {len(m)} self-report entries matched")
        for col, name in (("exp_rating", "expert rating"), ("exp_pred", "expert model's prediction")):
            print(f"  {name:28s} {spearmanr(m[col], m.rating).statistic:+.3f} / "
                  f"{within_rho(m.student, m.rating, m[col]):+.3f}")

    print()
    for track in ("self", "experts"):
        g = d[(d["track"] == track) & (d["arm"] == "base")]
        r = spearmanr(g.t - g.groupby("sess").t.transform("mean"),
                      g.rating - g.groupby("sess").rating.transform("mean")).statistic
        print(f"{track}: within-session Spearman of the rating with time {r:+.3f}")


if __name__ == "__main__":
    main()
