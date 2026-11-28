"""The within-session model: does a model asked only "is this student more or less engaged than
they usually are in this session?" do better at it than one asked for the rating itself?
Experiment 10 follow-up (docs/experiments/10_clustering.md §3.7 and Appendix C: the features carry
a small within-session signal and none between students).

  python -m scripts.clustering.within_session [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--split both] [--boot 2000] [--arms W0p W1 W2 W3 W1c T] [--candidates <dir>]
         [--out outputs/experiments/10_clustering]

Unit: a labeler's ratings of one student in one activity (a labeler-session). Only labeler-sessions
with at least 3 entries and 2 distinct ratings are used, in training and in scoring (others have no
within-session variation to learn or score; counted).

Arms (pooled ridge regression: one model for all experts, one for the self-report; inputs median-
imputed and standardised on the training rows; the ridge strength chosen by an inner cross-
validation grouped by class session, inside the training groups only, on the within-session score):
  W0p  absolute rating, `normalised` inputs (experiment 10's per-track standardised base columns)
  W1   deviation (the rating minus the labeler-session's mean rating), `normalised` inputs
  W2   deviation, `normalised` + the synchrony candidates (trk.cluster.sync.*)
  W3   deviation, `normalised` + the student's own normalised features averaged over the trailing
       30 s and 120 s
  W1c  deviation, causal inputs: each window standardised by the student's own features up to that
       moment only (expanding mean and SD, SD floor as TrackStandardiser), as a live system would
  T    deviation, the time since the student-session's first window only (and its square): what
       time alone gives
  W0   (context, no fit) experiment 10's `normalised` arm, read from <out>/ablation/
       ablation_predictions.csv.gz when it is there

Pre-registered (primary): W1 against W0p, within-session Spearman, group split, experts and
self-report, each at 98.75% (Bonferroni for the two). A gain counts only if, in the same direction,
(a) it holds after each labeler-session's linear time trend is removed from prediction and rating,
(b) W1 beats T, and (c) it holds for each of the three labelers with the most entries (experts).
Everything else is exploratory.

Scores, on original recordings, one value per entry (the mean prediction of its rows: an entry can
have a row in each camera, and in the activity split it is tested in several folds):
  within     Spearman of prediction and rating, each centred on its labeler-session mean
  detrended  the same after removing each labeler-session's linear trend in time from both
  per_session  the mean Spearman inside labeler-sessions with >= 5 entries
  per_labeler  within, for each labeler
  ceiling    the other experts' ratings in force at the entry's time (experts; for the self-report,
             the experts'), centred the same way: what a human reaches on the same entries
Intervals: cluster bootstrap over class sessions (whole classes resampled, so labeler-sessions stay
intact), paired on the same entries for differences. Written to <out>/within_session/.

Checks built in (tests/test_within_session.py): features shuffled within sessions score ~0; test
labels never reach the fit (changing them leaves the predictions unchanged); drift shared by
features and ratings disappears after detrending; a planted within-session effect is found.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.clustering.common import (SD_FLOOR, SELF, base_columns, base_parser, candidate_columns, folds,
                                       label_table, latest_candidates, load, load_candidates, per_track_normalised,
                                       split_parser_arg, splits, train_mask)
from scripts.clustering.feature_similarity import _weighted_pearson, boot_weights, in_force, load_tracks, rank_corr
from scripts.run_pipeline import features_dir

ALPHAS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
MIN_ENTRIES, MIN_LEVELS = 3, 2
PRIMARY_ALPHA = 0.05 / 2
ARMS = ["W0p", "W1", "W2", "W3", "W1c", "T"]
TARGET = {"W0p": "rating", "W1": "deviation", "W2": "deviation", "W3": "deviation", "W1c": "deviation", "T": "deviation"}
MAIN_LABELERS = 3


# ------------------------------------------------------------------ data
def prepare(rows: pd.DataFrame) -> pd.DataFrame:
    """Keys, the labeler-session mean and deviation, and eligibility (over the original entries)."""
    r = rows.reset_index(drop=True).copy()
    r["sess"] = r["group"] + "/" + r["exp"].astype(str) + "/" + r["sid"].astype(str)
    r["cls"] = r["group"] + "/" + r["exp"].astype(str)
    r["ls"] = r["labeler"] + "|" + r["sess"]
    r["track"] = np.where(r["labeler"] == SELF, "self", "experts")
    ent = r[~r["aug"].astype(bool)].drop_duplicates("entry_id")
    st = ent.groupby("ls")["rating"].agg(["mean", "size", "nunique"])
    r["ls_mean"] = r["ls"].map(st["mean"])
    r["deviation"] = r["rating"] - r["ls_mean"]
    ok = st[(st["size"] >= MIN_ENTRIES) & (st["nunique"] >= MIN_LEVELS)].index
    r["eligible"] = r["ls"].isin(ok)
    return r


def grid_series(cfg: dict, stem: str, cols: list[str]) -> pd.DataFrame:
    """The student's regular-grid windows of one recording with their centre time `t`."""
    fdir = features_dir(cfg)
    g = pd.read_csv(fdir / f"{stem}.tracks.csv")
    w = pd.read_csv(fdir / f"{stem}.csv", usecols=["t_start", "t_end"])
    mid = ((w["t_start"] + w["t_end"]) / 2).to_numpy(float)
    g = g[g["window"].between(0, len(mid) - 1)].copy()
    g["t"] = mid[g["window"].astype(int).to_numpy()]
    return g.reindex(columns=["tid", "t"] + cols).sort_values(["tid", "t"]).reset_index(drop=True)


def _cumsums(V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Running sums (with a leading zero row) of the values, their squares and the counts of finite
    values, per column: the mean or SD of rows a..b-1 is then a difference of two rows."""
    f = np.isfinite(V)
    v = np.where(f, V, 0.0)
    z = np.zeros((1, V.shape[1]))
    return (np.vstack([z, np.cumsum(v, 0)]), np.vstack([z, np.cumsum(v * v, 0)]),
            np.vstack([z, np.cumsum(f, 0)]))


def context_and_causal(cfg: dict, r: pd.DataFrame, cols: list[str], spans=(30.0, 120.0), min_rows: int = 5):
    """Per row, from the student's own regular-grid windows of the row's recording (label-free):
      ctx[span]  the per-track standardised features (as the `normalised` arm) averaged over the grid
                 windows with centre in (t - span, t]
      causal     the row's features standardised by the student's own windows with centre <= t
                 (expanding mean and SD; with fewer than min_rows, every student's windows of the
                 recording up to t), the SD at least SD_FLOOR x the SD of every student's windows up
                 to t: nothing after t is used
      start      the centre time of the student's first window in the recording (for T)
    Running sums per track, so each row is a lookup."""
    from scripts.clustering.common import TrackStandardiser

    n, D = len(r), len(cols)
    ctx = {sp: np.full((n, D), np.nan) for sp in spans}
    causal = np.full((n, D), np.nan)
    start = np.full(n, np.nan)
    Xrow = r.reindex(columns=cols).to_numpy(float)
    t_row, tid_row = r["t"].to_numpy(float), r["tid"].to_numpy()
    for stem, idx in r.groupby("stem").indices.items():
        g = grid_series(cfg, stem, cols)
        if g.empty:
            continue
        gz = TrackStandardiser(g, cols).transform(g)
        G = g[cols].to_numpy(float)
        tg_all = g["t"].to_numpy(float)
        o = np.argsort(tg_all, kind="stable")                      # every student, by time
        ta = tg_all[o]
        S1a, S2a, Na = _cumsums(G[o])
        tid_g = g["tid"].to_numpy()
        for tid in np.unique(tid_row[idx]):
            rows_i = idx[tid_row[idx] == tid]
            gi = np.flatnonzero(tid_g == tid)                       # grid is sorted by tid, t
            tq = t_row[rows_i]
            k_all = np.searchsorted(ta, tq, side="right")
            with np.errstate(invalid="ignore", divide="ignore"):
                na = np.maximum(Na[k_all], 1)
                mu_a = S1a[k_all] / na
                sd_a = np.sqrt(np.maximum(S2a[k_all] / na - mu_a ** 2, 0))
            if len(gi) == 0:
                mu, sd = mu_a, sd_a
            else:
                tg = tg_all[gi]
                start[rows_i] = tg[0]
                S1z, _, Nz = _cumsums(gz[gi])
                hi = np.searchsorted(tg, tq, side="right")
                for sp in spans:
                    lo = np.searchsorted(tg, tq - sp, side="right")
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ctx[sp][rows_i] = (S1z[hi] - S1z[lo]) / (Nz[hi] - Nz[lo])
                S1, S2, N = _cumsums(G[gi])
                with np.errstate(invalid="ignore", divide="ignore"):
                    nn = np.maximum(N[hi], 1)
                    mu_o = S1[hi] / nn
                    sd_o = np.sqrt(np.maximum(S2[hi] / nn - mu_o ** 2, 0))
                own = N[hi] >= min_rows
                mu, sd = np.where(own, mu_o, mu_a), np.where(own, sd_o, sd_a)
            sd = np.fmax(sd, SD_FLOOR * sd_a)
            with np.errstate(invalid="ignore", divide="ignore"):
                causal[rows_i] = np.where((sd > 1e-12) & (Na[k_all] >= 2), (Xrow[rows_i] - mu) / sd, np.nan)
    return ctx, causal, start


def session_position(r: pd.DataFrame, start: np.ndarray) -> np.ndarray:
    """Minutes since the student's first window in the recording, and its square (label-free:
    from the tracked video, not from when a labeler first rated)."""
    pos = (r["t"].to_numpy(float) - start) / 60.0
    return np.column_stack([pos, pos ** 2])


def build_inputs(cfg: dict, r: pd.DataFrame, arms: list[str], cands: list[str]) -> dict:
    """Arm -> input matrix (rows of r)."""
    base = base_columns(cfg, r)
    out = {}
    normed = per_track_normalised(cfg, r, base)[base].to_numpy(float)
    for arm in ("W0p", "W1"):
        if arm in arms:
            out[arm] = normed
    if "W2" in arms:
        sync = [c for c in cands if c.startswith("trk.cluster.sync.")]
        if not sync:
            raise SystemExit("W2 needs the synchrony candidates (run the candidates step, or drop W2 with --arms)")
        out["W2"] = np.column_stack([normed, r.reindex(columns=sync).to_numpy(float)])
    if {"W3", "W1c", "T"} & set(arms):
        ctx, causal, start = context_and_causal(cfg, r, base)
        if "W3" in arms:
            out["W3"] = np.column_stack([normed] + [ctx[sp] for sp in sorted(ctx)])
        if "W1c" in arms:
            out["W1c"] = causal
        if "T" in arms:
            out["T"] = session_position(r, start)
    return out


# ------------------------------------------------------------------ model
class Ridge:
    """Ridge regression on median-imputed, standardised inputs (statistics of the training rows)."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, X, y):
        from sklearn.linear_model import Ridge as R

        self.med = np.nanmedian(X, 0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Z = np.where(np.isfinite(X), X, self.med)
        self.mu, self.sd = Z.mean(0), Z.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        self.m = R(alpha=self.alpha).fit((Z - self.mu) / self.sd, y)
        return self

    def predict(self, X):
        Z = np.where(np.isfinite(X), X, self.med)
        return self.m.predict((Z - self.mu) / self.sd)


def per_entry(r: pd.DataFrame, mask: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """One row per entry of r[mask] (original rows): the mean prediction of its rows."""
    d = r.loc[mask, ["entry_id", "rating", "ls", "labeler", "sess", "cls", "t", "track"]].assign(pred=pred)
    return d.groupby("entry_id", as_index=False).agg(pred=("pred", "mean"), rating=("rating", "first"),
                                                     ls=("ls", "first"), labeler=("labeler", "first"),
                                                     sess=("sess", "first"), cls=("cls", "first"),
                                                     t=("t", "first"), track=("track", "first"))


def centred(v: pd.Series, key: pd.Series) -> np.ndarray:
    return (v - v.groupby(key).transform("mean")).to_numpy(float)


def detrended(v: pd.Series, t: pd.Series, key: pd.Series) -> np.ndarray:
    """v minus its least-squares line in t within each key (centred only where fewer than 3)."""
    out = np.full(len(v), np.nan)
    vv, tt = v.to_numpy(float), t.to_numpy(float)
    for _, i in pd.Series(np.arange(len(v))).groupby(key.to_numpy()).indices.items():
        if len(i) >= 3 and np.ptp(tt[i]) > 0:
            A = np.column_stack([np.ones(len(i)), tt[i] - tt[i].mean()])
            coef = np.linalg.lstsq(A, vv[i], rcond=None)[0]
            out[i] = vv[i] - A @ coef
        else:
            out[i] = vv[i] - vv[i].mean()
    return out


def within_score(e: pd.DataFrame) -> float:
    return rank_corr(centred(e["pred"], e["ls"]), centred(e["rating"], e["ls"]))


def choose_alpha(r: pd.DataFrame, X: np.ndarray, y: np.ndarray, train: np.ndarray, n_inner: int = 4) -> tuple[float, dict]:
    """The ridge strength with the best mean within-session score over inner folds grouped by class
    session, inside the training rows only (validated on their original entries)."""
    cls = r["cls"].to_numpy()
    tc = np.unique(cls[train])
    if len(tc) < 2:
        return ALPHAS[len(ALPHAS) // 2], {}
    rng = np.random.default_rng(0)
    order = rng.permutation(tc)
    parts = np.array_split(order, min(n_inner, len(tc)))
    aug = r["aug"].to_numpy(bool)
    scores = {a: [] for a in ALPHAS}
    for part in parts:
        val_c = np.isin(cls, part)
        fit_m, val_m = train & ~val_c, train & val_c & ~aug
        if fit_m.sum() < 20 or val_m.sum() < 10:
            continue
        for a in ALPHAS:
            p = Ridge(a).fit(X[fit_m], y[fit_m]).predict(X[val_m])
            s = within_score(per_entry(r, val_m, p))
            scores[a].append(s)
    mean = {a: float(np.nanmean(v)) if v and np.isfinite(v).any() else np.nan for a, v in scores.items()}
    best = max((a for a in ALPHAS if np.isfinite(mean[a])), key=lambda a: mean[a], default=ALPHAS[len(ALPHAS) // 2])
    return best, mean


def run_split(cfg: dict, r: pd.DataFrame, inputs: dict, split: str, track: str, log=print) -> tuple[pd.DataFrame, dict]:
    """Every arm's held-out predictions on one label set (one row per entry and arm, averaged over
    the folds an entry is tested in) and each fold's chosen ridge strength."""
    in_track = (r["track"] == track).to_numpy() & r["eligible"].to_numpy()
    preds, chosen = [], {}
    for name, tr, te in folds(r, split):
        tr = train_mask(cfg, r, tr) & in_track
        te = te & in_track
        if tr.sum() < 50 or te.sum() < 10:
            continue
        for arm, X in inputs.items():
            y = r[TARGET[arm]].to_numpy(float)
            fit = tr & np.isfinite(X).any(1)
            alpha, inner = choose_alpha(r, X, y, fit)
            model = Ridge(alpha).fit(X[fit], y[fit])
            e = per_entry(r, te, model.predict(X[te]))
            preds.append(e.assign(arm=arm, fold=name))
            chosen[f"{arm}|{name}"] = {"alpha": alpha, "inner": inner}
        log(f"  {split} {track} fold {name}: {int(tr.sum())} training rows, {int(te.sum())} test rows")
    if not preds:
        log(f"  {split} {track}: no fold with enough eligible rows (>= 50 training, >= 10 test); skipped")
        return pd.DataFrame(), chosen
    p = pd.concat(preds, ignore_index=True)
    p = p.groupby(["arm", "entry_id"], as_index=False).agg(
        pred=("pred", "mean"), rating=("rating", "first"), ls=("ls", "first"), labeler=("labeler", "first"),
        sess=("sess", "first"), cls=("cls", "first"), t=("t", "first"), track=("track", "first"))
    return p, chosen


# ------------------------------------------------------------------ scores
def scores(e: pd.DataFrame, Wb: np.ndarray, cls_code: dict, ceiling: pd.Series | None = None) -> dict:
    """Every score of one arm's per-entry predictions; `_boot*` keep resampled values for paired
    differences (whole class sessions resampled; ranks of the full sample)."""
    out = {"entries": int(len(e)), "labeler_sessions": int(e["ls"].nunique())}
    g = e["cls"].map(cls_code).to_numpy()
    for name, (x, y) in {"within": (centred(e["pred"], e["ls"]), centred(e["rating"], e["ls"])),
                         "detrended": (detrended(e["pred"], e["t"], e["ls"]), detrended(e["rating"], e["t"], e["ls"]))
                         }.items():
        out[name] = rank_corr(x, y)
        ok = np.isfinite(x) & np.isfinite(y)
        if len(Wb) and ok.sum() >= 20:
            vals = _weighted_pearson(pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy(),
                                     Wb[:, g[ok]])
            out[f"_boot_{name}"] = vals
            out[f"{name}_lo"], out[f"{name}_hi"] = (float(v) for v in np.nanpercentile(vals, [2.5, 97.5]))
    per = []
    for _, s in e.groupby("ls"):
        if len(s) >= 5 and s["rating"].nunique() >= 2 and s["pred"].nunique() >= 2:
            per.append(rank_corr(s["pred"].to_numpy(float), s["rating"].to_numpy(float)))
    out["per_session"] = float(np.nanmean(per)) if per else float("nan")
    out["per_session_n"] = len(per)
    out["per_labeler"] = {lab: {"within": within_score(s), "entries": int(len(s))}
                          for lab, s in e.groupby("labeler") if len(s) >= 30}
    return out


def diff(a: dict, b: dict, key: str, alpha: float = 0.05) -> dict:
    d = {"diff": float(a.get(key, np.nan) - b.get(key, np.nan))}
    if f"_boot_{key}" in a and f"_boot_{key}" in b and len(a[f"_boot_{key}"]) == len(b[f"_boot_{key}"]):
        v = a[f"_boot_{key}"] - b[f"_boot_{key}"]
        d["lo"], d["hi"] = (float(x) for x in np.nanpercentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        d["excludes_0"] = bool(d["lo"] > 0 or d["hi"] < 0)
    return d


def evaluate(pred: pd.DataFrame, rng, boot: int, ceiling: pd.DataFrame | None = None) -> dict:
    """Scores of every arm on the entries every arm predicted (paired), the ceiling on the same
    entries, and the pre-registered comparisons."""
    if pred.empty:
        return {}
    n_arms = pred["arm"].nunique()
    common = pred.groupby("entry_id")["arm"].nunique()
    keep = common[common == n_arms].index
    pred = pred[pred["entry_id"].isin(keep)]
    cls = sorted(pred["cls"].unique())
    code = {c: i for i, c in enumerate(cls)}
    Wb = boot_weights(len(cls), boot, rng)
    res = {"arms": {}}
    for arm, e in pred.groupby("arm"):
        res["arms"][arm] = scores(e.sort_values("entry_id").reset_index(drop=True), Wb, code)
    if ceiling is not None and not ceiling.empty:
        e = pred[pred["arm"] == pred["arm"].iloc[0]].drop(columns="pred").merge(ceiling, on="entry_id", how="inner")
        e = e[np.isfinite(e["pred"])].sort_values("entry_id").reset_index(drop=True)
        if len(e) >= 20:
            res["ceiling"] = scores(e, Wb, code)
    A = res["arms"]
    comp = {}
    for a, b in (("W1", "W0p"), ("W1", "T"), ("W2", "W1"), ("W3", "W1"), ("W1c", "W1"), ("W0", "W1")):
        if a in A and b in A:
            alpha = PRIMARY_ALPHA if (a, b) == ("W1", "W0p") else 0.05
            comp[f"{a} - {b}"] = {k: diff(A[a], A[b], k, alpha) for k in ("within", "detrended")}
    res["comparisons"] = comp
    for v in list(A.values()) + ([res["ceiling"]] if "ceiling" in res else []):
        for k in [k for k in v if k.startswith("_boot")]:
            v.pop(k)
    return res


def primary_verdict(res: dict, track: str, main_labelers: list[str]) -> dict:
    """The pre-registered verdict: W1 beats W0p (98.75%), and (a) after detrending, (b) W1 beats T,
    (c) for each main labeler (experts), all in the same direction."""
    c = res.get("comparisons", {})
    if "W1 - W0p" not in c:
        return {"verdict": "not run"}
    w, dt = c["W1 - W0p"]["within"], c["W1 - W0p"]["detrended"]
    beats_t = c.get("W1 - T", {}).get("within", {})
    A = res["arms"]
    labs = {}
    if track == "experts":
        for lab in main_labelers:
            a, b = A["W1"]["per_labeler"].get(lab, {}), A["W0p"]["per_labeler"].get(lab, {})
            labs[lab] = float(a.get("within", np.nan) - b.get("within", np.nan))
    gain = w.get("excludes_0") and w["diff"] > 0
    conds = {"detrended_same_sign": dt["diff"] > 0, "beats_T": beats_t.get("diff", np.nan) > 0,
             "main_labelers_same_sign": all(v > 0 for v in labs.values()) if labs else True}
    v = ("supported" if gain and all(conds.values()) else
         "not supported (fails: " + ", ".join(k for k, ok in conds.items() if not ok) + ")" if gain else
         "not supported")
    return {"verdict": v, "within": w, "detrended": dt, "W1_minus_T": beats_t, "per_labeler_diff": labs, "conditions": conds}


# ------------------------------------------------------------------ main
def fmt(s: dict, k: str) -> str:
    v = s.get(k, np.nan)
    ci = f" [{s[k + '_lo']:+.3f}, {s[k + '_hi']:+.3f}]" if f"{k}_lo" in s else ""
    return f"{v:+.3f}{ci}"


def main() -> None:
    ap = base_parser(__doc__)
    split_parser_arg(ap)
    ap.add_argument("--arms", nargs="*", default=ARMS, choices=ARMS)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", default=None)
    args = ap.parse_args()
    cfg, out = load(args)
    rng = np.random.default_rng(args.seed)

    rows = label_table(cfg)
    cands = []
    if "W2" in args.arms:
        rows = load_candidates(rows, latest_candidates(out, args.candidates))
        cands = candidate_columns(rows)
    r = prepare(rows)
    ent = r[~r["aug"].astype(bool)].drop_duplicates("entry_id")
    elig = ent.groupby("track")["eligible"].agg(["size", "sum"])
    print(f"{len(r)} rows ({int(r['aug'].astype(bool).sum())} augmented); entries per label set and eligible "
          f"(labeler-sessions with >= {MIN_ENTRIES} entries and >= {MIN_LEVELS} distinct ratings): "
          + "; ".join(f"{t} {int(v['sum'])} of {int(v['size'])}" for t, v in elig.iterrows()), flush=True)
    inputs = build_inputs(cfg, r, args.arms, cands)
    print("inputs: " + ", ".join(f"{a} {X.shape[1]} columns" for a, X in inputs.items()), flush=True)

    tracks = load_tracks(cfg, ent.rename(columns={}).reset_index(drop=True))
    w0 = None
    p0 = out / "ablation" / "ablation_predictions.csv.gz"
    if p0.exists():
        w0 = pd.read_csv(p0)
        w0 = w0[w0["arm"] == "normalised"]
    main_labs = (ent[ent["track"] == "experts"]["labeler"].value_counts().index[:MAIN_LABELERS].tolist())
    res_all, preds_all = {"settings": {k: v for k, v in vars(args).items() if k != "set"},
                          "eligible": {t: {"entries": int(v["size"]), "eligible": int(v["sum"])} for t, v in elig.iterrows()},
                          "main_labelers": main_labs, "splits": {}}, []
    for split in splits(args.split):
        res_all["splits"][split] = {}
        for track in ("experts", "self"):
            pred, chosen = run_split(cfg, r, inputs, split, track)
            if pred.empty:
                continue
            if w0 is not None:
                w = w0[(w0["split"] == split) & (w0["track"] == track)].groupby("entry_id", as_index=False)["pred"].mean()
                base = pred[pred["arm"] == pred["arm"].iloc[0]].drop(columns=["pred", "arm"])
                add = base.merge(w, on="entry_id", how="inner").assign(arm="W0")
                if len(add) >= 0.9 * len(base):
                    pred = pd.concat([pred, add[pred.columns]], ignore_index=True)
            e = pred[pred["arm"] == pred["arm"].iloc[0]].reset_index(drop=True)
            f = in_force(e.assign(sess=e["sess"]), tracks)
            ceiling = pd.DataFrame({"entry_id": e["entry_id"], "pred": f["mean"]})
            res = evaluate(pred, rng, args.boot, ceiling)
            res["alphas"] = chosen
            res["primary"] = primary_verdict(res, track, main_labs) if split == "group" else {}
            res_all["splits"][split][track] = res
            preds_all.append(pred.assign(split=split))
            print(f"\n=== {split} split, {track}: {res['arms'][next(iter(res['arms']))]['entries']} entries in "
                  f"{res['arms'][next(iter(res['arms']))]['labeler_sessions']} labeler-sessions (every arm)")
            if "ceiling" in res:
                print(f"  {'ceiling (humans)':18s} within {fmt(res['ceiling'], 'within')}   detrended {fmt(res['ceiling'], 'detrended')}")
            for arm, s in res["arms"].items():
                print(f"  {arm:18s} within {fmt(s, 'within')}   detrended {fmt(s, 'detrended')}   "
                      f"per-session {s['per_session']:+.3f} (n {s['per_session_n']})")
            for name, c in res["comparisons"].items():
                w, d = c["within"], c["detrended"]
                ci = f" [{w['lo']:+.3f}, {w['hi']:+.3f}]" if "lo" in w else ""
                print(f"  {name:12s} within {w['diff']:+.3f}{ci}   detrended {d['diff']:+.3f}")
            if res["primary"]:
                print(f"  PRIMARY (W1 vs W0p, 98.75%): {res['primary']['verdict']}; per labeler {res['primary']['per_labeler_diff']}")
            edge = [k for k, v in chosen.items() if v["alpha"] in (ALPHAS[0], ALPHAS[-1])]
            if edge:
                print(f"  note: ridge strength at the edge of the grid {ALPHAS[0]:g}-{ALPHAS[-1]:g} in "
                      f"{len(edge)} of {len(chosen)} fits ({', '.join(edge[:4])}...): widen ALPHAS")
    d = out / "within_session"
    d.mkdir(parents=True, exist_ok=True)
    (d / "within_session.json").write_text(json.dumps(res_all, indent=1, default=float), encoding="utf-8")
    if preds_all:
        pd.concat(preds_all, ignore_index=True).to_csv(d / "within_session_predictions.csv.gz", index=False)
    print(f"\nwrote {d / 'within_session.json'}")


if __name__ == "__main__":
    main()
