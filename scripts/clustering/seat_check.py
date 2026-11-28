"""Do the ratings depend on where a student sits? Experiment 10 follow-up: Appendix C found that the
features carry nothing between students, while the experts agree between students (+0.25 at the
session level). Per-track normalisation removes seat and distance from the inputs on purpose; this
asks whether seat is part of what the labelers' between-student judgement holds.

  python -m scripts.clustering.seat_check [--config docs/experiments/10_clustering.json] [--set k=v ...]
         [--perms 5000] [--boot 2000] [--no-features] [--out outputs/experiments/10_clustering]

Seat, from the manual seat boxes (abi_labels: one static box per student and camera; no labels):
every in-scope camera is at the front of the room (cameras.yml: front_east / front_middle /
front_west) and the in-scope activities use the grid layout, so in every recording a bigger box is a
student nearer the front. Per recording, each student's percentile rank (0 back ... 1 front) of
  front    box area (nearer the front camera = bigger)
  low      the box's bottom edge in the image (nearer = lower)
  central  closeness of the box centre to the class's median centre, horizontally
averaged over the cameras that see the student (all three face the class from the front). A box is
also bigger for a bigger student; its bottom edge depends on where the student sits and hardly on
their size, so a real seat effect shows in both `front` and `low`.

Rating level, per labeler and student-session, from the rating tracks (src/labels, as the pipeline
reads them; ratings are change events, a value holds until the next): the time-weighted mean over
the class session (its first to last entry of any labeler, any student; from the labeler's own first
entry). Robustness: the plain mean of the labeler's entries, the session-start entry excluded.

Analyses (Spearman; between students of the same class, by the same labeler):
  A  seat against level: both centred within (labeler, class session), so neither a labeler's habit
     nor a class's overall level nor a camera's zoom can create it. Experts pooled, each labeler, and
     the self-report. p: permutation of the seat among the students of each class session (the same
     permutation for every labeler's rows, so a student keeps all their ratings); interval: bootstrap
     over class sessions. Pre-registered: `front`, experts and self-report, Bonferroni for the two;
     the other measures are exploratory, with BH over all of them.
  B  per camera: `front` from each camera alone (does the sign hold in all three views?)
  C  observer against self: the experts' and the self-report's seat effects side by side; seat in the
     experts' ratings and not in the students' own is an observer effect
  D  the experts' shared judgement: each expert's level against the mean of the other experts'
     levels for the same student (between students within class), before and after seat is
     regressed out of both: how much of the agreement seat explains
  E  the models: experiment 10's held-out predictions (ablation_predictions.csv.gz; `base`, raw
     inputs, and `normalised`), averaged per labeler and student-session, against seat
  F  the raw features: each student-session's mean of every model input against seat (do the
     movement features grow with nearness, as pixel-based flow would?); needs the label rows
  G  effect size: the mean centred level of the front third minus the back third, in rating points
  H  neighbours: is a student rated like the classmate sitting next to them (the nearest box centre in
     the same camera image, any camera), more than like a random classmate? Pairs by the same labeler,
     levels centred within (labeler, class); p: the neighbour graph kept as it is and the students
     shuffled among the seats of each class session (the same shuffle for every labeler)

Written to <out>/seat_check/seat_check.json (and seat_table.csv, levels.csv), and printed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.clustering.common import SELF, base_parser, bh_qvalues, load

MEASURES = ["front", "low", "central"]
PRIMARY = "front"
PRIMARY_ALPHA = 0.05 / 2


# ------------------------------------------------------------------ seat
def pct_rank(v: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] (0 smallest, 1 largest; ties averaged; one value -> 0.5)."""
    if len(v) == 1:
        return np.array([0.5])
    return (pd.Series(v).rank(method="average").to_numpy() - 1) / (len(v) - 1)


def seat_table(boxes: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """boxes: (cls, camera) -> {sid: xyxy}. Returns the per-camera ranks (cls, camera, sid, front,
    low, central) and, per student-session, their mean over the cameras (+ n_cameras)."""
    rows = []
    for (cls, cam), b in boxes.items():
        if len(b) < 3:                                   # too few students to rank
            continue
        sids = sorted(b)
        X = np.array([b[s] for s in sids], float)
        area = (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
        bottom = X[:, 3]
        cx = (X[:, 0] + X[:, 2]) / 2
        cent = -np.abs(cx - np.median(cx))
        for s, f, lo, ce in zip(sids, pct_rank(area), pct_rank(bottom), pct_rank(cent)):
            rows.append({"cls": cls, "camera": cam, "sid": s, "front": f, "low": lo, "central": ce})
    per_cam = pd.DataFrame(rows, columns=["cls", "camera", "sid"] + MEASURES)
    seat = per_cam.groupby(["cls", "sid"], as_index=False)[MEASURES].mean()
    seat["n_cameras"] = per_cam.groupby(["cls", "sid"]).size().to_numpy()
    return per_cam, seat


# ------------------------------------------------------------------ rating levels
def level_table(tracks: dict) -> pd.DataFrame:
    """tracks: (cls, sid, labeler) -> AttentionTrack. Per (cls, sid, labeler): the time-weighted mean
    rating over the class session (from max(session start, the labeler's first entry) to the session
    end: the first and last entry of any track in the class session), and the plain mean of the
    labeler's entries without the session-start entry (within 0.5 s of the session start)."""
    span = {}
    for (cls, _, _), tr in tracks.items():
        if len(tr.t):
            a, b = span.get(cls, (np.inf, -np.inf))
            span[cls] = (min(a, float(tr.t[0])), max(b, float(tr.t[-1])))
    rows = []
    for (cls, sid, lab), tr in tracks.items():
        if not len(tr.t) or cls not in span:
            continue
        t0, t1 = span[cls]
        a = max(t0, float(tr.t[0]))
        level = tr.mean_over(a, t1) if t1 > a else float(tr.value[-1])
        keep = np.abs(tr.t - t0) >= 0.5
        rows.append({"cls": cls, "sid": sid, "labeler": lab, "level": level,
                     "entries_mean": float(np.mean(tr.value[keep])) if keep.any() else np.nan,
                     "n_entries": int(keep.sum())})
    return pd.DataFrame(rows, columns=["cls", "sid", "labeler", "level", "entries_mean", "n_entries"])


# ------------------------------------------------------------------ statistics
def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def centre(v: pd.Series, keys: list[pd.Series]) -> np.ndarray:
    return (v - v.groupby(keys).transform("mean")).to_numpy(float)


def joined(levels: pd.DataFrame, seat: pd.DataFrame, value: str = "level",
           measures: list[str] | None = None) -> pd.DataFrame:
    """Rows (labeler, student-session) with a seat and a value; the value centred within (labeler,
    class) and each measure (default: the seat measures) centred the same way over these rows."""
    d = levels.merge(seat, on=["cls", "sid"], how="inner")
    d = d[np.isfinite(d[value])].reset_index(drop=True)
    d["y"] = centre(d[value], [d["labeler"], d["cls"]])
    for m in (MEASURES if measures is None else measures):
        d[f"{m}_c"] = centre(d[m], [d["labeler"], d["cls"]])
    return d


def seat_effect(d: pd.DataFrame, measure: str, perms: int, boot: int, rng) -> dict:
    """Spearman of the centred seat measure and the centred level, with a within-class permutation
    p (the seat permuted among the class's students, the same permutation for every labeler) and a
    class-session bootstrap interval; and the front-third minus back-third difference in rating
    points."""
    x, y = d[f"{measure}_c"].to_numpy(float), d["y"].to_numpy(float)
    out = {"rows": int(len(d)), "students": int(d[["cls", "sid"]].drop_duplicates().shape[0]),
           "classes": int(d["cls"].nunique()), "spearman": rank_corr(x, y)}
    if len(d) < 10 or not np.isfinite(out["spearman"]):
        return out
    # permutation within class sessions, at the level of students
    stu = d[["cls", "sid"]].drop_duplicates().reset_index(drop=True)
    stu_idx = d.merge(stu.reset_index(), on=["cls", "sid"])["index"].to_numpy()
    raw = d.groupby(["cls", "sid"])[measure].first().reindex(pd.MultiIndex.from_frame(stu)).to_numpy(float)
    by_cls = [np.flatnonzero(stu["cls"].to_numpy() == c) for c in stu["cls"].unique()]
    keys = [d["labeler"], d["cls"]]
    null = np.empty(perms)
    for p in range(perms):
        v = raw.copy()
        for i in by_cls:
            v[i] = v[i][rng.permutation(len(i))]
        xs = pd.Series(v[stu_idx])
        null[p] = rank_corr(centre(xs, keys), y)
    null = null[np.isfinite(null)]
    out["p_perm"] = float((1 + np.sum(np.abs(null) >= abs(out["spearman"]))) / (1 + len(null)))
    out["null_sd"] = float(np.std(null))
    # bootstrap over class sessions
    cl = d["cls"].to_numpy()
    idx = [np.flatnonzero(cl == c) for c in np.unique(cl)]
    if boot and len(idx) >= 3:
        vals = [rank_corr(x[s], y[s]) for s in
                (np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(boot))]
        out["lo"], out["hi"] = (float(v) for v in np.nanpercentile(vals, [2.5, 97.5]))
        out["lo_adj"], out["hi_adj"] = (float(v) for v in np.nanpercentile(vals, [100 * PRIMARY_ALPHA / 2,
                                                                                 100 * (1 - PRIMARY_ALPHA / 2)]))
    # front third minus back third of each class's students (by the raw measure), in rating points
    st = d.groupby(["cls", "sid"], as_index=False)[measure].first()
    st["third"] = st.groupby("cls")[measure].transform(
        lambda s: np.floor(3 * (s.rank(method="first") - 1) / len(s)) if len(s) >= 3 else np.nan)
    third = d.merge(st[["cls", "sid", "third"]], on=["cls", "sid"], how="left")["third"].to_numpy()
    if np.isfinite(third).any():
        out["front_minus_back_points"] = float(d.loc[third == 2, "y"].mean() - d.loc[third == 0, "y"].mean())
    return out


def per_labeler(d: pd.DataFrame, measure: str, min_rows: int = 20) -> dict:
    return {lab: {"rows": int(len(g)), "spearman": rank_corr(g[f"{measure}_c"].to_numpy(float), g["y"].to_numpy(float))}
            for lab, g in d.groupby("labeler") if len(g) >= min_rows}


def agreement_with_and_without_seat(levels: pd.DataFrame, seat: pd.DataFrame) -> dict:
    """Each expert's level for a student against the mean of the other experts' levels for the same
    student (both centred within (labeler, class)), before and after the three seat measures are
    regressed out of both (pooled OLS on the centred values)."""
    e = levels[levels["labeler"] != SELF]
    rows = []
    for (cls, sid), g in e.groupby(["cls", "sid"]):
        if g["labeler"].nunique() < 2:
            continue
        for _, row in g.iterrows():
            others = g.loc[g["labeler"] != row["labeler"], "level"]
            if others.notna().any():
                rows.append({"cls": cls, "sid": sid, "labeler": row["labeler"], "level": row["level"],
                             "others": float(others.mean())})
    a = pd.DataFrame(rows)
    if a.empty:
        return {}
    a = a.merge(seat, on=["cls", "sid"], how="inner").dropna(subset=["level", "others"]).reset_index(drop=True)
    keys = [a["labeler"], a["cls"]]
    y1, y2 = centre(a["level"], keys), centre(a["others"], keys)
    S = np.column_stack([centre(a[m], keys) for m in MEASURES])

    def resid(v):
        coef = np.linalg.lstsq(S, v, rcond=None)[0]
        return v - S @ coef
    before, after = rank_corr(y1, y2), rank_corr(resid(y1), resid(y2))
    return {"rows": int(len(a)), "agreement": before, "agreement_without_seat": after,
            "share_explained_by_seat": float(1 - after / before) if before and np.isfinite(before) and before > 0 else np.nan}


def model_against_seat(pred: pd.DataFrame, seat: pd.DataFrame) -> dict:
    """Experiment 10's held-out predictions (group split), averaged per labeler and student-session,
    against seat, for the `base` (raw inputs) and `normalised` arms."""
    out = {}
    p = pred[(pred["split"] == "group") & pred["arm"].isin(["base", "normalised"])].copy()
    if p.empty:
        return out
    parts = p["entry_id"].str.split("/", expand=True)                # group/experiment/subject/labeler/t
    p["cls"], p["sid"], p["labeler"] = parts[0] + "/" + parts[1], parts[2], parts[3]
    for arm, g in p.groupby("arm"):
        lv = g.groupby(["cls", "sid", "labeler"], as_index=False)["pred"].mean().rename(columns={"pred": "level"})
        for track, sel in (("experts", lv["labeler"] != SELF), ("self", lv["labeler"] == SELF)):
            d = joined(lv[sel], seat)
            if len(d) >= 10:
                out[f"{arm}|{track}"] = {m: rank_corr(d[f"{m}_c"].to_numpy(float), d["y"].to_numpy(float)) for m in MEASURES}
                out[f"{arm}|{track}"]["rows"] = int(len(d))
    return out


def features_against_seat(rows: pd.DataFrame, seat: pd.DataFrame, cols: list[str]) -> dict:
    """Each student-session's mean of every model input (original rows) against `front`, both centred
    within the class: which features grow with nearness to the camera."""
    r = rows[~rows["aug"].astype(bool)].copy()
    r["cls"] = r["group"] + "/" + r["session"].str.split("/").str[1]
    m = r.groupby(["cls", "sid"], as_index=False)[cols].mean().merge(seat, on=["cls", "sid"], how="inner")
    if len(m) < 10:
        return {}
    f = centre(m["front"], [m["cls"]])
    res = {c: rank_corr(centre(m[c], [m["cls"]]), f) for c in cols}
    return dict(sorted(res.items(), key=lambda kv: -abs(kv[1]) if np.isfinite(kv[1]) else 0))


def neighbour_pairs(boxes: dict) -> pd.DataFrame:
    """Per class session, each student and the classmate whose box centre is nearest in the same
    camera image (box widths as the scale), over every camera: unique (cls, a, b) pairs, a != b."""
    rows = set()
    for (cls, _), b in boxes.items():
        if len(b) < 3:
            continue
        sids = sorted(b)
        X = np.array([b[x] for x in sids], float)
        c = np.column_stack([(X[:, 0] + X[:, 2]) / 2, (X[:, 1] + X[:, 3]) / 2])
        D = np.linalg.norm(c[:, None] - c[None], axis=2)
        np.fill_diagonal(D, np.inf)
        for i, j in enumerate(D.argmin(1)):
            rows.add((cls, sids[i], sids[j]))
    return pd.DataFrame(sorted(rows), columns=["cls", "a", "b"])


def _pair_corr(pairs: pd.DataFrame, lv: pd.DataFrame) -> float:
    """Spearman of the centred levels of the two students of each pair, by the same labeler (both
    orders of a pair counted, since `pairs` holds each student's own nearest)."""
    m = pairs.merge(lv, left_on=["cls", "a"], right_on=["cls", "sid"]).merge(
        lv, left_on=["cls", "b", "labeler"], right_on=["cls", "sid", "labeler"], suffixes=("_a", "_b"))
    return rank_corr(m["y_a"].to_numpy(float), m["y_b"].to_numpy(float)), int(len(m))


def neighbour_agreement(boxes: dict, levels: pd.DataFrame, perms: int, rng) -> dict:
    """H: agreement of neighbours. The null keeps the neighbour graph exactly as it is (who sits next
    to whom, mutual pairs included) and shuffles which student sits in which seat within each class
    session, the same shuffle for every labeler; re-pairing students at random instead would change
    how many pairs are mutual and make the test too liberal."""
    pairs = neighbour_pairs(boxes)
    lv = levels.dropna(subset=["level"]).copy()
    lv["y"] = centre(lv["level"], [lv["labeler"], lv["cls"]])
    lv = lv[["cls", "sid", "labeler", "y"]]
    obs, n = _pair_corr(pairs, lv)
    if n < 10 or not np.isfinite(obs):
        return {"pairs": n}
    stu = {c: sorted(set(g["sid"])) for c, g in lv.groupby("cls")}
    null = []
    for _ in range(perms):
        mapping = {}
        for c, ss in stu.items():
            for x, y in zip(ss, rng.permutation(ss)):
                mapping[(c, x)] = y
        sh = lv.assign(sid=[mapping[(c, x)] for c, x in zip(lv["cls"], lv["sid"])])
        v, _ = _pair_corr(pairs, sh)
        null.append(v)
    null = np.array([v for v in null if np.isfinite(v)])
    return {"pairs": n, "neighbours": obs, "random_seating_mean": float(null.mean()),
            "excess": float(obs - null.mean()),
            "p_perm": float((1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (1 + len(null)))}


# ------------------------------------------------------------------ data
def load_inputs(cfg: dict) -> tuple[dict, dict]:
    """The seat boxes per (class session, camera) and every labeler's rating track per (class
    session, student), for the in-scope recordings."""
    from src.datasets import DipserIndex
    from src.labels.session import _load, collapse_key

    index, lcfg = DipserIndex.from_config(cfg), cfg["labels"]
    boxes, tracks = {}, {}
    d = cfg["dataset"]
    for g in d["groups"]:
        for e in d["experiments"]:
            cls = f"{g}/{e}"
            seen = set()
            for cam in d["cameras"]:
                from src.core.types import SessionKey
                b = index.annotations(SessionKey(g, e, cam))
                if b:
                    boxes[(cls, cam)] = b
                    seen |= set(b)
            zips = index.subject_zips(SessionKey(g, e, d["cameras"][0]))
            for sid in sorted(seen):
                if sid in zips:
                    for lab, tr in _load(zips[sid], float(lcfg["offset_s"]), collapse_key(lcfg)).items():
                        if len(tr.t):
                            tracks[(cls, sid, lab)] = tr
    return boxes, tracks


def fmt(s: dict) -> str:
    if not s or not np.isfinite(s.get("spearman", np.nan)):
        return "–"
    ci = f" [{s['lo']:+.3f}, {s['hi']:+.3f}]" if "lo" in s else ""
    p = f"  p {s['p_perm']:.4f}" if "p_perm" in s else ""
    pts = f"  front-back {s['front_minus_back_points']:+.3f} pts" if "front_minus_back_points" in s else ""
    return f"{s['spearman']:+.3f}{ci}{p}{pts}  (n {s['rows']} rows, {s['students']} students, {s['classes']} classes)"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--perms", type=int, default=5000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-features", action="store_true", help="skip analysis F (needs the label rows)")
    args = ap.parse_args()
    cfg, out = load(args)
    rng = np.random.default_rng(args.seed)

    boxes, tracks = load_inputs(cfg)
    per_cam, seat = seat_table(boxes)
    levels = level_table(tracks)
    print(f"seat boxes: {len(boxes)} recordings, {len(seat)} student-sessions "
          f"({seat['n_cameras'].value_counts().sort_index().to_dict()} by number of cameras); rating tracks: "
          f"{len(tracks)}; levels: {len(levels)} ({(levels['labeler'] != SELF).sum()} expert, "
          f"{(levels['labeler'] == SELF).sum()} self-report)", flush=True)
    res = {"settings": {k: v for k, v in vars(args).items() if k != "set"},
           "counts": {"recordings": len(boxes), "student_sessions": int(len(seat)), "levels": int(len(levels))}}

    # A, C, G: seat against level, experts and self-report, both level variants
    tests = []
    res["seat_effect"] = {}
    for value in ("level", "entries_mean"):
        for track in ("experts", "self"):
            lv = levels[(levels["labeler"] != SELF) if track == "experts" else (levels["labeler"] == SELF)]
            d = joined(lv, seat, value)
            for m in MEASURES:
                s = seat_effect(d, m, args.perms, args.boot, rng)
                if value == "level":
                    s["per_labeler"] = per_labeler(d, m) if track == "experts" else {}
                    tests.append((f"{track}|{m}", s.get("p_perm", np.nan)))
                res["seat_effect"][f"{value}|{track}|{m}"] = s
    q = bh_qvalues(np.array([p for _, p in tests], float))
    for (name, _), qv in zip(tests, q):
        res["seat_effect"][f"level|{name}"]["q_bh"] = float(qv)
    for track in ("experts", "self"):
        s = res["seat_effect"][f"level|{track}|{PRIMARY}"]
        s["primary"] = True
        s["primary_supported"] = bool("lo_adj" in s and (s["lo_adj"] > 0 or s["hi_adj"] < 0)
                                      and s.get("p_perm", 1) < PRIMARY_ALPHA)

    # B: per camera
    res["per_camera"] = {}
    for cam, pc in per_cam.groupby("camera"):
        sc = pc.drop(columns="camera")
        sc["n_cameras"] = 1
        for track in ("experts", "self"):
            lv = levels[(levels["labeler"] != SELF) if track == "experts" else (levels["labeler"] == SELF)]
            d = joined(lv, sc)
            res["per_camera"][f"{cam}|{track}"] = {"rows": int(len(d)),
                                                   "spearman": rank_corr(d["front_c"].to_numpy(float), d["y"].to_numpy(float))}

    # D: the experts' shared judgement with and without seat
    res["expert_agreement"] = agreement_with_and_without_seat(levels, seat)

    # E: the models
    pp = out / "ablation" / "ablation_predictions.csv.gz"
    res["models"] = model_against_seat(pd.read_csv(pp), seat) if pp.exists() else {}

    # F: the raw features
    res["features"] = {}
    if not args.no_features:
        try:
            from scripts.clustering.common import base_columns, label_table

            rows = label_table(cfg)
            res["features"] = features_against_seat(rows, seat, base_columns(cfg, rows))
        except (SystemExit, FileNotFoundError, KeyError) as exc:
            print(f"analysis F skipped: {exc}", flush=True)

    # H: neighbours
    res["neighbours"] = {}
    for track in ("experts", "self"):
        lv = levels[(levels["labeler"] != SELF) if track == "experts" else (levels["labeler"] == SELF)]
        res["neighbours"][track] = neighbour_agreement(boxes, lv, min(args.perms, 1000), rng)

    # report
    print("\nA. seat against rating level (between students of the same class, same labeler):")
    for key, s in res["seat_effect"].items():
        value, track, m = key.split("|")
        tag = "  PRIMARY" if s.get("primary") else ""
        q = f"  q {s['q_bh']:.4f}" if "q_bh" in s else ""
        print(f"  {value:12s} {track:7s} {m:8s} {fmt(s)}{q}{tag}")
        if s.get("primary"):
            if "lo_adj" in s:
                print(f"      98.75% interval [{s['lo_adj']:+.3f}, {s['hi_adj']:+.3f}], p < 0.025 needed: "
                      f"{'SUPPORTED' if s['primary_supported'] else 'not supported'}")
            else:
                print("      no interval (needs at least 3 class sessions): not tested")
        if s.get("per_labeler"):
            print("      per labeler: " + ", ".join(f"{k} {v['spearman']:+.3f} (n {v['rows']})" for k, v in s["per_labeler"].items()))
    print("\nB. per camera (front):")
    for key, s in res["per_camera"].items():
        print(f"  {key:24s} {s['spearman']:+.3f} (n {s['rows']})")
    a = res["expert_agreement"]
    if a:
        print(f"\nD. experts against the other experts, between students: {a['agreement']:+.3f}; with seat regressed "
              f"out {a['agreement_without_seat']:+.3f} (share explained by seat {a['share_explained_by_seat']:.1%}; n {a['rows']})")
    if res["models"]:
        print("\nE. experiment 10's held-out predictions against seat:")
        for key, s in res["models"].items():
            print(f"  {key:22s} " + ", ".join(f"{m} {s[m]:+.3f}" for m in MEASURES) + f"  (n {s['rows']})")
    if res["features"]:
        print("\nF. raw features against front (student-session means, centred within class): strongest")
        for c, v in list(res["features"].items())[:10]:
            print(f"  {c:28s} {v:+.3f}")
    print("\nH. neighbours (nearest classmate in the image), same labeler, against the students seated at random:")
    for track, h in res["neighbours"].items():
        if "neighbours" in h:
            print(f"  {track:7s} neighbours {h['neighbours']:+.3f}, random seating {h['random_seating_mean']:+.3f}, "
                  f"excess {h['excess']:+.3f}, p {h['p_perm']:.4f} (n {h['pairs']} pairs)")
    d = out / "seat_check"
    d.mkdir(parents=True, exist_ok=True)
    (d / "seat_check.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    seat.to_csv(d / "seat_table.csv", index=False)
    levels.to_csv(d / "levels.csv", index=False)
    print(f"\nwrote {d / 'seat_check.json'}")


if __name__ == "__main__":
    main()
