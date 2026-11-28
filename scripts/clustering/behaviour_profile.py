"""Which behaviours do the labelers respond to, within a session and between students? Experiment 10
follow-up, next to seat_check: Appendix C found that the features carry a little information within
a session and none between students; this asks it of every feature on its own.

  python -m scripts.clustering.behaviour_profile [--config docs/experiments/10_clustering.json]
         [--set k=v ...] [--perms 2000] [--boot 1000] [--out outputs/experiments/10_clustering]

For every model input (the configured base columns) and three visibility measures, Spearman with the
ratings, for the experts and the self-report:
  within    an entry's feature value (its window; the mean of its rows when several cameras see the
            student) against its rating, both centred within the labeler-session: does the feature
            follow a student's own ups and downs? Interval: bootstrap over class sessions; p from the
            same resamples
  between   the student's mean of the feature over the whole session (every regular-grid window of
            their tracks, every camera: their behaviour profile, not tied to when labelers clicked)
            against their time-weighted rating level, both centred within (labeler, class session):
            are students who behave differently rated differently? p: the students permuted within
            each class session; interval: bootstrap over class sessions (seat_check's statistics)
Visibility (from the tracks, per student-session): coverage (share of the recording's frames with a
linked track), detection confidence, box diagonal in pixels (nearness, as in seat_check). BH within
each (analysis, label set). Written to <out>/behaviour_profile/ (json and csv), and printed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.clustering.common import SELF, base_columns, base_parser, bh_qvalues, label_table, load
from scripts.clustering.seat_check import joined, level_table, load_inputs, rank_corr, seat_effect
from scripts.clustering.within_session import grid_series

VISIBILITY = ["vis.coverage", "vis.conf", "vis.diag"]


# ------------------------------------------------------------------ data
def originals(rows: pd.DataFrame) -> pd.DataFrame:
    r = rows[~rows["aug"].astype(bool)].copy()
    parts = r["session"].str.split("/")
    r["cls"] = parts.str[0] + "/" + parts.str[1]
    r["sess"] = r["cls"] + "/" + r["sid"].astype(str)
    return r.reset_index(drop=True)


def student_profiles(cfg: dict, r: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per (cls, sid): the mean of every column over the student's regular-grid windows in every
    recording of the class session (tracks linked to the student in the label rows), and the three
    visibility measures (each recording's value, averaged over the recordings that see the student)."""
    from scripts.run_pipeline import tracks_dir

    link = r[["stem", "tid", "cls", "sid"]].drop_duplicates()
    parts = []
    for stem, lk in link.groupby("stem"):
        g = grid_series(cfg, stem, cols)
        m = g.merge(lk, on="tid", how="inner")
        feat = m.groupby(["cls", "sid"])[cols].mean()
        z = np.load(tracks_dir(cfg) / f"{stem}.npz", allow_pickle=True)
        tid, conf, xyxy = z["tid"], z["conf"], z["xyxy"]
        frames = max(len(z["frame_t"]), 1)
        diag = np.hypot(xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1])
        tr = pd.DataFrame({"tid": tid, "conf": conf, "diag": diag}).merge(lk, on="tid", how="inner")
        vis = tr.groupby(["cls", "sid"]).agg(**{"vis.coverage": ("tid", "size"), "vis.conf": ("conf", "mean"),
                                               "vis.diag": ("diag", "mean")})
        vis["vis.coverage"] = (vis["vis.coverage"] / frames).clip(upper=1.0)
        parts.append(feat.join(vis, how="outer").reset_index())
    if not parts:
        return pd.DataFrame(columns=["cls", "sid"] + cols + VISIBILITY)
    return pd.concat(parts, ignore_index=True).groupby(["cls", "sid"], as_index=False).mean()


def entry_table(r: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One row per entry: its rating, labeler, session keys and the mean of each column over its rows."""
    agg = {c: (c, "mean") for c in cols}
    agg.update(rating=("rating", "first"), labeler=("labeler", "first"), sess=("sess", "first"), cls=("cls", "first"))
    return r.groupby("entry_id", as_index=False).agg(**agg)


# ------------------------------------------------------------------ statistics
def within_profile(e: pd.DataFrame, cols: list[str], boot: int, rng) -> dict:
    """Per column: Spearman of the feature and the rating, both centred within the labeler-session,
    with a class-session bootstrap interval and p (twice the smaller share of resamples on either
    side of 0)."""
    key = e["labeler"] + "|" + e["sess"]
    y = (e["rating"] - e["rating"].groupby(key).transform("mean")).to_numpy(float)
    cl = e["cls"].to_numpy()
    idx = [np.flatnonzero(cl == c) for c in np.unique(cl)]
    picks = [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(boot)] if len(idx) >= 3 else []
    out = {}
    for c in cols:
        x = (e[c] - e[c].groupby(key).transform("mean")).to_numpy(float)
        s = {"rows": int(np.isfinite(x).sum()), "spearman": rank_corr(x, y)}
        if picks and np.isfinite(s["spearman"]):
            v = np.array([rank_corr(x[p], y[p]) for p in picks])
            v = v[np.isfinite(v)]
            if len(v) >= 20:
                s["lo"], s["hi"] = (float(q) for q in np.percentile(v, [2.5, 97.5]))
                s["p_boot"] = float(min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0)) + 1 / len(v)))
        out[c] = s
    return out


def between_profile(levels: pd.DataFrame, profiles: pd.DataFrame, cols: list[str], perms: int, boot: int, rng) -> dict:
    """Per column: seat_check's between-student test with the student's mean behaviour in place of
    the seat (centred within (labeler, class); students permuted within class; class bootstrap)."""
    d = joined(levels, profiles, "level", measures=cols)
    return {c: seat_effect(d, c, perms, boot, rng) for c in cols}


def add_q(res: dict, p_key: str) -> None:
    names = list(res)
    q = bh_qvalues(np.array([res[n].get(p_key, np.nan) for n in names], float))
    for n, v in zip(names, q):
        res[n]["q_bh"] = float(v)


def fmt(s: dict, p_key: str) -> str:
    if not np.isfinite(s.get("spearman", np.nan)):
        return "–"
    ci = f" [{s['lo']:+.3f}, {s['hi']:+.3f}]" if "lo" in s else ""
    q = f" q {s['q_bh']:.3f}" if np.isfinite(s.get("q_bh", np.nan)) else ""
    return f"{s['spearman']:+.3f}{ci}{q}"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg, out = load(args)
    rng = np.random.default_rng(args.seed)

    rows = label_table(cfg)
    cols = base_columns(cfg, rows)
    r = originals(rows)
    profiles = student_profiles(cfg, r, cols)
    _, tracks = load_inputs(cfg)
    levels = level_table(tracks)
    e = entry_table(r, cols)
    feats = cols + VISIBILITY
    print(f"{len(e)} entries, {len(profiles)} student profiles, {len(levels)} rating levels; {len(cols)} features "
          f"+ {len(VISIBILITY)} visibility measures", flush=True)
    res = {"settings": {k: v for k, v in vars(args).items() if k != "set"}, "within": {}, "between": {}}
    for track in ("experts", "self"):
        sel_e = (e["labeler"] != SELF) if track == "experts" else (e["labeler"] == SELF)
        sel_l = (levels["labeler"] != SELF) if track == "experts" else (levels["labeler"] == SELF)
        w = within_profile(e[sel_e].reset_index(drop=True), cols, args.boot, rng)
        add_q(w, "p_boot")
        b = between_profile(levels[sel_l], profiles, feats, args.perms, args.boot, rng)
        add_q(b, "p_perm")
        res["within"][track], res["between"][track] = w, b
        print(f"\n=== {track}: within {sum(v.get('q_bh', 1) < 0.05 for v in w.values())} of {len(w)} features "
              f"significant (q < 0.05); between {sum(v.get('q_bh', 1) < 0.05 for v in b.values())} of {len(b)}")
        for name, part, pk in (("within", w, "p_boot"), ("between", b, "p_perm")):
            top = sorted(part.items(), key=lambda kv: -abs(kv[1].get("spearman", 0) or 0))[:8]
            print(f"  {name}: " + "; ".join(f"{c} {fmt(s, pk)}" for c, s in top))
    table = []
    for c in feats:
        row = {"feature": c}
        for track in ("experts", "self"):
            for part in ("within", "between"):
                s = res[part][track].get(c, {})
                for k in ("spearman", "lo", "hi", "q_bh"):
                    row[f"{part}_{track}_{k}"] = s.get(k, np.nan)
        table.append(row)
    d = out / "behaviour_profile"
    d.mkdir(parents=True, exist_ok=True)
    (d / "behaviour_profile.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    pd.DataFrame(table).to_csv(d / "behaviour_profile.csv", index=False)
    print(f"\nwrote {d / 'behaviour_profile.json'}")


if __name__ == "__main__":
    main()
