"""Analysis of the six-session run: where is the signal (if any)?
  python scripts/analysis/e2e6/analyse.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.run_pipeline import parse_sets
from scripts.train_models import engagement_rows, load_table
from src.core.config import load_config
from src.core.pipeline import VEC_PREFIX

cfg = load_config("configs/deliverable3.json",
                  parse_sets(['output_dir="tmp/e2e6"', "labels.tau_att=3", "labels.tau_share=0.6"]))
df = load_table(cfg)
vec_cols = sorted(c for c in df if c.startswith(VEC_PREFIX))
rows = engagement_rows(cfg, df, vec_cols)


def section(t):
    print(f"\n=== {t}")


# A. the window target: between- vs within-session variance of the labelled share
section("A. window target (labelled share engaged)")
s = df[["session", "exp", "share", "y"]].dropna(subset=["share"])
tot = s.share.var(ddof=0)
within = s.groupby("session").share.var(ddof=0).mean()
print(f"share variance: total {tot:.4f}, mean within-session {within:.4f} -> between-session share {1 - within / tot:.0%}")
print(s.groupby("session").agg(share_mean=("share", "mean"), share_sd=("share", "std"),
                               engaged_windows=("y", "mean")).round(2).to_string())
flips = s.groupby("session").y.apply(lambda v: int((np.diff(v.to_numpy()) != 0).sum()))
print("window-label changes per session:", flips.to_dict())

# B. student labels: who vs when
section("B. student labels (late-mode rows)")
r = pd.DataFrame({"window": rows.window, "y": rows.y})
r["session"] = df["session"].to_numpy()[rows.window]
import scripts.train_models as tm  # noqa: E402  (re-derive sids per row for the student grouping)
from src.core.store import load_track  # noqa: E402
from src.core.types import SessionKey  # noqa: E402
from src.datasets import DipserIndex  # noqa: E402
from src.tracking import associate_sids  # noqa: E402
index = DipserIndex.from_config(cfg)
sids = []
for session, sdf in df.groupby("session", sort=False):
    key = SessionKey(*session.split("/"))
    stem = tm.session_stem(key)
    t = pd.read_csv(tm.features_dir(cfg) / f"{stem}.tracks.csv")
    m = associate_sids(load_track(tm.tracks_dir(cfg) / f"{stem}.npz").tracklets, index.annotations(key),
                       cfg["association"]["min_iou"], cfg["association"]["min_match_share"])
    sids += [f"{session}/{m.get(int(tid))}" for tid in t["tid"]]
r["student"] = sids
lab = r.dropna(subset=["y"])
per_student = lab.groupby("student").y.agg(["mean", "count"])
constant = ((per_student["mean"] == 0) | (per_student["mean"] == 1)).mean()
print(f"labelled rows {len(lab)}, students {len(per_student)}, positive rate {lab.y.mean():.2f}")
print(f"students whose label never changes in the session: {constant:.0%}")
vy = lab.y.var(ddof=0)
wy = lab.groupby("student").y.var(ddof=0).mean()
print(f"student-label variance: between-student share {1 - wy / vy:.0%}, within-student {wy / vy:.0%}")

# C. per-student features vs per-student labels
section("C. single-feature AUC for a student's own label (trk.* features)")
X = pd.DataFrame(rows.X, columns=rows.cols)
trk = [c for c in rows.cols if c.startswith("trk.")]
out = []
for c in trk:
    v = X[c].to_numpy()
    ok = np.isfinite(v) & np.isfinite(rows.y)
    pooled = roc_auc_score(rows.y[ok], v[ok]) if len(np.unique(rows.y[ok])) == 2 else np.nan
    within = []
    for sess in r.session.unique():
        m = ok & (r.session.to_numpy() == sess)
        if len(np.unique(rows.y[m])) == 2 and m.sum() > 30:
            within.append(roc_auc_score(rows.y[m], v[m]))
    # between students: does a student's average feature rank their average label?
    st = pd.DataFrame({"s": r.student, "v": v, "y": rows.y})[ok].groupby("s").mean()
    between = st[["v", "y"]].corr(method="spearman").iloc[0, 1] if len(st) > 5 else np.nan
    out.append({"feature": c, "pooled_auc": pooled, "within_session_auc": np.mean(within) if within else np.nan,
                "between_student_rho": between})
o = pd.DataFrame(out)
o["strength"] = (o.pooled_auc - 0.5).abs()
print(o.sort_values("strength", ascending=False).drop(columns="strength").round(3).to_string(index=False))

# D. window-level features vs the share, within sessions
section("D. Vec features vs labelled share, within-session Spearman (mean over sessions)")
res = []
for c in [c for c in vec_cols if c.endswith(".mean") or c.endswith("n_tracks")]:
    rhos = [g[[c, "share"]].corr(method="spearman").iloc[0, 1] for _, g in df.groupby("session")
            if g[c].nunique() > 2 and g["share"].nunique() > 2]
    res.append((c, np.nanmean(rhos) if rhos else np.nan, np.nanstd(rhos) if rhos else np.nan))
for c, m, sd in sorted(res, key=lambda x: -abs(np.nan_to_num(x[1])))[:10]:
    print(f"{c:40s} rho {m:+.2f} (sd over sessions {sd:.2f})")
