"""How much do DIPSER's labelers agree, and how well can one labeler be predicted from the others?
(docs/experiments/06_full_pipeline_and_labels.md, Part B.)

  python -m scripts.labels.label_agreement [stats|alpha|ceiling|coarse|all]
         [--model-entries]  [--set ...]  [--out outputs/labels/agreement.json]

Labels are read once per (group, experiment) from every subject ZIP (labels.students = "all"), so
each rating entry counts once; the session-start entry (every labeler's first rating, at one instant)
is excluded wherever entries are counted or predicted, as in label_entries. Ratings are compared on
the 1-5 scale; "binary" means rating >= 3 (tau_att 3).

  stats    label structure: entries, rating shares, entries per minute, run lengths, offline 1 fps
           entry timing, entries that repeat the previous value, and co-occurrence (how often
           another labeler entered something on the same student within +-k s).
  alpha    Krippendorff's ordinal alpha over a grid: unit (entry moments, per second, window
           midpoint, window mean) x interpolation (step, ramp, gauss) x labeler set (experts,
           experts + self) x scale (raw 1-5, binary), per activity and group and overall; plus the
           14 Nov setting (step, 8 s window midpoints) reproduced.
  ceiling  the human ceiling, scored exactly like the model (rating_metrics, tau 3): every entry of
           labeler A predicted from the other labelers' curves (each interpolation, global lags,
           a +-k s best-match upper bound), from the median / mean of the other experts, from a
           learned combination of them (ridge, leave-one-group-out), and from A's own other ratings
           of the same student (self-consistency); on entries and on run midpoints; pooled,
           macro over labelers, per pair, per activity; with --model-entries also on exactly the
           entries the pipeline was scored on (rebuilt from the cached labelwin.csv files).
  coarse   agreement on coarser units: student x activity means and student x k-minute means
           (pairwise Spearman / Pearson / MAE, alpha on rounded means, experts vs self), by
           activity; between- vs within-student share of each labeler's rating variance.

Headline metrics are the mean over the three groups of each group's score (the model's
leave-one-group-out summaries are fold means too); pooled values and confusion matrices are in the
JSON. Writes every table to --out and prints the headline tables (ASCII).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_pipeline import parse_sets
from src.aggregation import make_windows
from src.core.config import load_config
from src.datasets import DipserIndex
from src.evaluation import rating_metrics
from src.ingestion import ZipFrameSource, timing_audit
from src.labels.interp import interpolate, round_rating
from src.labels.session import _load

EXPERTS = ("labeler_01", "labeler_02", "labeler_03", "labeler_04", "labeler_05")
SELF = "self_labeling"
LABELERS = EXPERTS + (SELF,)
TAU = 3.0
# interpolation settings: name -> (mode, ramp_s, sigma_s). ramp width w matches gauss sigma w / sqrt(12).
MODES = {"step": ("step", 3.0, 2.0), "ramp3": ("ramp", 3.0, 2.0), "ramp7": ("ramp", 7.0, 2.0),
         "gauss1": ("gauss", 3.0, 1.0), "gauss2": ("gauss", 3.0, 2.0), "gauss4": ("gauss", 3.0, 4.0)}
LAGS = (-5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0)        # B's curve read at t + lag (B entered later if > 0)
LAG_MODES = ("step", "gauss2")
TOLERANCES = (2.0, 5.0, 10.0)
CO_WINDOWS = (0.5, 2.0, 5.0, 10.0)
START_TOL = 0.5                                      # an entry within 0.5 s of the student's start
SUMMARY = ("n", "mae", "rmse", "accuracy", "within_1", "macro_f1", "auc", "spearman", "balanced_acc")


# ----------------------------------------------------------------------------- loading

@dataclass
class Student:
    group: str
    exp: str                      # "01".."06"
    experiment: str               # "experiment_01"
    sid: str
    t_first: float                # session frame range (first camera)
    t_last: float
    tracks: dict = field(default_factory=dict)       # {labeler: (t, value)} sorted, values 1-5
    start: float = np.nan                            # the shared session-start instant

    def curve(self, labeler: str, t, mode: str = "step") -> np.ndarray:
        if labeler not in self.tracks:
            return np.full(np.shape(t), np.nan)
        tt, vv = self.tracks[labeler]
        m, ramp_s, sigma_s = MODES[mode]
        return interpolate(tt, vv, t, m, ramp_s, sigma_s)

    def entries(self, labeler: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t, value, index into the track) of the labeler's entries after the session start."""
        t, v = self.tracks[labeler]
        keep = np.abs(t - self.start) >= START_TOL
        return t[keep], v[keep], np.flatnonzero(keep)


def load_students(cfg: dict) -> tuple[list[Student], dict]:
    """Every labelled student of every (group, experiment), with the first camera's frame range."""
    index = DipserIndex.from_config(cfg)
    offset = float(cfg["labels"]["offset_s"])
    students, frames, seen = [], {}, set()
    for key in index.sessions():
        if (key.group, key.experiment) in seen:
            continue
        seen.add((key.group, key.experiment))
        a = timing_audit(ZipFrameSource(index.frames_zip(key)).timestamps)
        frames[(key.group, key.experiment)] = (float(a["t_first"]), float(a["t_last"]))
        for sid, path in sorted(index.subject_zips(key).items()):
            tracks = {n: (np.asarray(tr.t, float), np.clip(np.asarray(tr.value, float), 1, 5))
                      for n, tr in _load(path, offset).items() if len(tr.t)}
            if not tracks:
                continue
            s = Student(key.group, key.exp_id, key.experiment, sid, a["t_first"], a["t_last"], tracks)
            s.start = min(t[0] for t, _ in tracks.values())
            students.append(s)
    return students, frames


# ----------------------------------------------------------------------------- scoring helpers

def score(y, p, groups) -> dict:
    """rating_metrics as the model is summarised: mean over groups of each group's score
    (headline), plus pooled metrics and the pooled confusion matrix."""
    y, p, groups = np.asarray(y, float), np.asarray(p, float), np.asarray(groups)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, groups = y[ok], p[ok], groups[ok]
    if not len(y):
        return {"n": 0}
    pooled = rating_metrics(y, p, TAU)
    per = [rating_metrics(y[groups == g], p[groups == g], TAU) for g in sorted(set(groups.tolist()))]
    head = {k: float(np.nanmean([m[k] for m in per if m.get("n")])) for k in SUMMARY if k != "n"}
    return {"n": int(len(y)), **head,
            "pooled": {k: pooled[k] for k in SUMMARY},
            "confusion": pooled["confusion"], "confusion_binary": pooled["confusion_binary"]}


def fmt(v) -> str:
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else \
        (f"{v:.3f}" if isinstance(v, float) else str(v))


def table(rows: list[tuple[str, dict]], cols=("n", "mae", "accuracy", "macro_f1", "auc", "spearman")) -> str:
    out = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    out += [f"| {name} | " + " | ".join(fmt(m.get(c)) for c in cols) + " |" for name, m in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------- stats

def run_lengths(s: Student, labeler: str) -> np.ndarray:
    t, _ = s.tracks[labeler]
    return np.diff(np.r_[t, max(s.t_last, t[-1])])


def stats(students: list[Student]) -> dict:
    by_lab = defaultdict(lambda: defaultdict(list))
    by_exp = defaultdict(lambda: defaultdict(list))
    co = defaultdict(lambda: defaultdict(list))        # target track -> window -> [bool]
    for s in students:
        minutes = (s.t_last - s.start) / 60
        for lab in s.tracks:
            t, v = s.tracks[lab]
            te, ve, idx = s.entries(lab)
            prev = np.r_[np.nan, v][idx]                # the value before each entry
            d = by_lab[lab]
            d["students"].append(1)
            d["minutes"].append(minutes)
            d["values"] += ve.tolist()
            d["first_value"].append(v[0])
            d["repeat"] += (ve == prev).tolist()
            d["first_tenth"] += ((te % 1.0) < 0.1).tolist()
            d["runs"] += run_lengths(s, lab).tolist()
            d["never_changes"].append(bool(len(np.unique(v)) == 1))
            e = by_exp[s.exp]
            e["values"] += ve.tolist()
            e["minutes_" + ("self" if lab == SELF else "expert")].append(minutes)
            e["n_" + ("self" if lab == SELF else "expert")].append(len(te))
            # co-occurrence: another labeler's entry (not the session start) on this student nearby
            others = [s.entries(o)[0] for o in s.tracks if o != lab and o != SELF] + \
                     ([s.entries(SELF)[0]] if lab != SELF and SELF in s.tracks else [])
            other_t = np.sort(np.concatenate(others)) if others else np.zeros(0)
            other_exp = np.sort(np.concatenate([s.entries(o)[0] for o in s.tracks if o not in (lab, SELF)]
                                               or [np.zeros(0)]))
            track = "self" if lab == SELF else "experts"
            for w in CO_WINDOWS:
                for name, ot in (("any_other", other_t), ("other_expert", other_exp)):
                    if len(ot):
                        j = np.searchsorted(ot, te)
                        near = np.minimum(np.abs(ot[np.clip(j, 0, len(ot) - 1)] - te),
                                          np.abs(ot[np.clip(j - 1, 0, len(ot) - 1)] - te)) <= w
                    else:
                        near = np.zeros(len(te), bool)
                    co[f"{track}/{name}"][w] += near.tolist()
    out = {"labelers": {}, "activities": {}, "co_occurrence": {}}
    for lab in sorted(by_lab, key=lambda n: LABELERS.index(n) if n in LABELERS else 99):
        d = by_lab[lab]
        vals, runs = np.array(d["values"]), np.array(d["runs"])
        out["labelers"][lab] = {
            "student_sessions": len(d["students"]), "entries": int(len(vals)),
            "share_of_expert_entries": None,
            "rating_shares_1_5": [float(np.mean(vals == r)) if len(vals) else np.nan for r in range(1, 6)],
            "mean_rating": float(vals.mean()) if len(vals) else np.nan,
            "median_rating": float(np.median(vals)) if len(vals) else np.nan,
            "sd_rating": float(vals.std()) if len(vals) else np.nan,          # population SD (ddof 0)
            "entries_per_minute": float(len(vals) / max(np.sum(d["minutes"]), 1e-9)),
            "run_length_median_s": float(np.median(runs)), "runs_le_2s": float(np.mean(runs <= 2)),
            "entries_in_first_tenth_of_second": float(np.mean(d["first_tenth"])) if d["first_tenth"] else np.nan,
            "entries_repeating_previous_value": float(np.mean(d["repeat"])) if d["repeat"] else np.nan,
            "tracks_never_changing": float(np.mean(d["never_changes"])),
            "first_value_shares_1_5": [float(np.mean(np.array(d["first_value"]) == r)) for r in range(1, 6)],
        }
    n_exp = sum(v["entries"] for k, v in out["labelers"].items() if k != SELF)
    for k, v in out["labelers"].items():
        v["share_of_expert_entries"] = v["entries"] / n_exp if k != SELF and n_exp else None
    for exp in sorted(by_exp):
        e = by_exp[exp]
        vals = np.array(e["values"])
        out["activities"][exp] = {
            "entries": int(len(vals)), "mean_rating": float(vals.mean()),
            "median_rating": float(np.median(vals)), "sd_rating": float(vals.std()),
            "rating_shares_1_5": [float(np.mean(vals == r)) for r in range(1, 6)],
            "expert_entries_per_minute": float(np.sum(e["n_expert"]) / max(np.sum(e["minutes_expert"]), 1e-9)),
            "self_entries_per_minute": float(np.sum(e["n_self"]) / max(np.sum(e["minutes_self"]), 1e-9)),
        }
    for k, d in co.items():
        out["co_occurrence"][k] = {f"within_{w:g}s": float(np.mean(v)) for w, v in d.items()}
    return out


# ----------------------------------------------------------------------------- alpha

def fast_alpha(values: np.ndarray) -> tuple[float, int, int]:
    """Krippendorff's ordinal alpha of an (items x labelers) matrix of whole ratings (NaN missing):
    the coincidence-matrix computation of src.evaluation.agreement, vectorised. Returns (alpha,
    pairable items, pairable values)."""
    v = np.asarray(values, float)
    if v.ndim != 2 or not len(v):
        return np.nan, 0, 0
    levels = np.unique(v[np.isfinite(v)])
    if len(levels) < 2:
        return np.nan, 0, 0
    U = np.stack([(v == lv).sum(1) for lv in levels], 1).astype(float)     # items x levels
    m = U.sum(1)
    U, m = U[m >= 2], m[m >= 2]
    if not len(U):
        return np.nan, 0, 0
    W = U / (m - 1)[:, None]
    o = W.T @ U - np.diag(W.sum(0))                                        # coincidence matrix
    n_c = o.sum(1)
    n = n_c.sum()
    k = len(levels)
    cum = np.cumsum(n_c)
    lo, hi = np.minimum.outer(np.arange(k), np.arange(k)), np.maximum.outer(np.arange(k), np.arange(k))
    between = cum[hi] - cum[lo] + n_c[lo]
    delta = (between - (n_c[:, None] + n_c[None, :]) / 2) ** 2
    d_o = (o * delta).sum()
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n - 1)
    return (float(1 - d_o / d_e) if d_e > 0 else np.nan), int(len(U)), int(m.sum())


def window_means(grid_v: np.ndarray, grid_t: np.ndarray, windows) -> np.ndarray:
    """Mean of a curve sampled on grid_t over each window (NaN parts ignored; NaN if none defined)."""
    ok = np.isfinite(grid_v)
    cs_v = np.r_[0, np.cumsum(np.where(ok, grid_v, 0))]
    cs_n = np.r_[0, np.cumsum(ok)]
    a = np.searchsorted(grid_t, [w.t_start for w in windows])
    b = np.searchsorted(grid_t, [w.t_end for w in windows])
    n = cs_n[b] - cs_n[a]
    return np.where(n > 0, (cs_v[b] - cs_v[a]) / np.maximum(n, 1), np.nan)


def unit_values(s: Student, unit: str, mode: str, length_s: float, overlap: float) -> np.ndarray:
    """(items x 6 labelers) values of one student for one unit and interpolation."""
    if unit == "entry":                         # moments where any labeler entered something
        t = np.unique(np.concatenate([s.entries(l)[0] for l in s.tracks]))
    elif unit == "second":
        t = np.arange(np.ceil(s.start), s.t_last, 1.0)
    elif unit in ("midpoint", "window_mean"):
        ws = make_windows(s.t_first, s.t_last, length_s, overlap)
        if unit == "midpoint":
            t = np.array([(w.t_start + w.t_end) / 2 for w in ws])
        else:
            grid = np.arange(s.t_first, s.t_last, 0.2) + 0.1
            return np.column_stack([window_means(s.curve(l, grid, mode), grid, ws) for l in LABELERS])
    else:
        raise ValueError(unit)
    return np.column_stack([s.curve(l, t, mode) for l in LABELERS])


def alpha(students: list[Student], length_s: float, overlap: float) -> dict:
    units = ("entry", "second", "midpoint", "window_mean")
    cells = {}
    sets = {"experts": [LABELERS.index(l) for l in EXPERTS], "experts_self": list(range(len(LABELERS)))}
    for unit in units:
        for mode in ("step", "ramp3", "gauss1", "gauss2", "gauss4"):
            mats = [(s.group, s.exp, unit_values(s, unit, mode, length_s, overlap)) for s in students]
            for sname, cols in sets.items():
                for scale in ("raw", "binary"):
                    key = f"{unit}/{mode}/{sname}/{scale}"
                    cell = {}
                    for scope, pick in [("overall", lambda g, e: True)] + \
                            [(f"exp_{x}", (lambda x: lambda g, e: e == x)(x)) for x in sorted({s.exp for s in students})] + \
                            [(g_, (lambda g_: lambda g, e: g == g_)(g_)) for g_ in sorted({s.group for s in students})]:
                        rows = [m[:, cols] for g, e, m in mats if pick(g, e) and len(m)]
                        if not rows:
                            continue
                        v = np.vstack(rows)
                        v = round_rating(v) if scale == "raw" else np.where(np.isfinite(v), (round_rating(v) >= TAU) * 1.0, np.nan)
                        v = np.where(np.isfinite(np.vstack(rows)), v, np.nan)
                        a, n_items, n_vals = fast_alpha(v)
                        cell[scope] = {"alpha": a, "items": n_items, "values": n_vals}
                    cells[key] = cell
            print(f"  alpha {unit}/{mode} done", flush=True)
    # the 14 Nov setting (scripts/data_visualization/inspect_labels.py): step values at 8 s window
    # midpoints, 0.5 overlap, all entries, per activity, with and without the self-report
    legacy = {}
    for scale in ("raw",):
        for sname, cols in (("with_self", sets["experts_self"]), ("external_only", sets["experts"])):
            for x in sorted({s.exp for s in students}):
                rows = [unit_values(s, "midpoint", "step", 8.0, 0.5)[:, cols] for s in students if s.exp == x]
                legacy.setdefault(x, {})[sname] = fast_alpha(np.vstack(rows))[0]
    return {"window_length_s": length_s, "overlap": overlap, "cells": cells, "legacy_8s_midpoint_step": legacy}


# ----------------------------------------------------------------------------- ceiling

def ceiling_rows(students: list[Student]) -> pd.DataFrame:
    """One row per target (entry of labeler A after the session start, or midpoint of one of A's
    runs) with every predictor column: other labelers' curves under each interpolation and lag,
    best-match within +-k s (upper bound), and A's own other ratings (self-consistency)."""
    recs = []
    for s in students:
        for a in s.tracks:
            t_all, v_all = s.tracks[a]
            te, ve, idx = s.entries(a)
            ends = np.r_[t_all[1:], max(s.t_last, t_all[-1])]
            durations = np.maximum(ends - t_all, 0)
            tot_w, tot_v = durations.sum(), (durations * v_all).sum()
            for kind in ("entry", "midpoint"):
                if kind == "entry":
                    t, y, run = te, ve, idx
                else:
                    t, y, run = (t_all + ends) / 2, v_all, np.arange(len(t_all))
                    keep = durations > 0
                    t, y, run = t[keep], y[keep], run[keep]
                if not len(t):
                    continue
                cols = {"group": s.group, "exp": s.exp, "experiment": s.experiment, "sid": s.sid, "target": a,
                        "kind": kind, "t": t, "y": y,
                        "is_change": np.r_[True, v_all[1:] != v_all[:-1]][run],
                        # self-consistency: A's previous run, and A's time-weighted mean over its other runs
                        "own_prev": np.r_[np.nan, v_all][run],
                        "own_loro": np.where(tot_w - durations[run] > 0,
                                             (tot_v - durations[run] * v_all[run]) / np.maximum(tot_w - durations[run], 1e-9),
                                             np.nan)}
                for b in LABELERS:
                    if b == a or b not in s.tracks:
                        continue
                    for mode in MODES:
                        cols[f"{b}|{mode}|0"] = s.curve(b, t, mode)
                    for mode in LAG_MODES:
                        for lag in LAGS:
                            cols[f"{b}|{mode}|{lag:g}"] = s.curve(b, t + lag, mode)
                    for k in TOLERANCES:                # best match within +-k s: an upper bound (selects by y)
                        offs = np.arange(-k, k + 1e-9, 0.2)
                        grid = s.curve(b, t[:, None] + offs[None, :], "step")
                        diff = np.where(np.isfinite(grid), np.abs(grid - y[:, None]), np.inf)
                        best = grid[np.arange(len(t)), diff.argmin(1)]
                        cols[f"{b}|tol{k:g}"] = np.where(np.isfinite(diff.min(1)), best, np.nan)
                recs.append(pd.DataFrame(cols))
    df = pd.concat(recs, ignore_index=True)
    df["entry_id"] = [f"{g}/{e}/{s}/{l}/{t:.3f}" for g, e, s, l, t in
                      zip(df["group"], df["experiment"], df["sid"], df["target"], df["t"])]
    return df


def stack_pairs(df: pd.DataFrame, col_suffix: str, experts_only: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """'One other labeler' predictions: every (target row, other labeler B) pair as its own row.
    Returns (y, prediction, group, target labeler)."""
    ys, ps, gs, ts = [], [], [], []
    for b in (EXPERTS if experts_only else LABELERS):
        c = f"{b}|{col_suffix}"
        if c not in df:
            continue
        m = df[c].notna().to_numpy()
        ys.append(df["y"].to_numpy()[m]); ps.append(df[c].to_numpy()[m])
        gs.append(df["group"].to_numpy()[m]); ts.append(df["target"].to_numpy()[m])
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(gs), np.concatenate(ts)


def others(df: pd.DataFrame, col_suffix: str, how: str) -> np.ndarray:
    cols = [f"{b}|{col_suffix}" for b in EXPERTS if f"{b}|{col_suffix}" in df]
    v = df[cols].to_numpy(float)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(v, 1) if how == "median" else np.nanmean(v, 1)


def learned(df: pd.DataFrame, suffix: str = "step|0") -> tuple[np.ndarray, list[dict]]:
    """Ridge regression of A's rating on every other labeler's curve value (missing -> 3 plus a
    missing flag) and A's identity, leave-one-group-out. Returns held-out predictions and the
    training-fold scores."""
    from sklearn.linear_model import Ridge

    feats = []
    for b in LABELERS:
        c = f"{b}|{suffix}"
        v = df[c].to_numpy(float) if c in df else np.full(len(df), np.nan)
        feats += [np.where(np.isfinite(v), v - 3, 0.0), np.isfinite(v) * 1.0]
    feats += [(df["target"].to_numpy() == l) * 1.0 for l in LABELERS]
    X, y, g = np.column_stack(feats), df["y"].to_numpy(float), df["group"].to_numpy()
    pred, train_scores = np.full(len(df), np.nan), []
    for grp in sorted(set(g.tolist())):
        tr, te = g != grp, g == grp
        m = Ridge(alpha=1.0).fit(X[tr], y[tr])
        pred[te] = np.clip(m.predict(X[te]), 1, 5)
        train_scores.append(rating_metrics(y[tr], np.clip(m.predict(X[tr]), 1, 5), TAU))
    return pred, train_scores


def labeler_prior(df: pd.DataFrame) -> np.ndarray:
    """A's mean rating on the other groups' targets (leave-one-group-out)."""
    out = np.full(len(df), np.nan)
    for grp in df["group"].unique():
        tr = df["group"] != grp
        means = df[tr].groupby("target")["y"].mean()
        te = (df["group"] == grp).to_numpy()
        out[te] = df.loc[te, "target"].map(means).to_numpy(float)
    return out


def predictor_rows(df: pd.DataFrame) -> dict[str, tuple]:
    """{name: (y, prediction, group, target)} for every headline predictor on df's rows."""
    y, g, tg = df["y"].to_numpy(float), df["group"].to_numpy(), df["target"].to_numpy()
    out = {"constant 3": (y, np.full(len(y), 3.0), g, tg),
           "labeler prior (A's own mean, other groups)": (y, labeler_prior(df), g, tg),
           "self-consistency: A's previous value": (y, df["own_prev"].to_numpy(float), g, tg),
           "self-consistency: A's mean over its other runs": (y, df["own_loro"].to_numpy(float), g, tg)}
    for mode in MODES:
        out[f"one other expert, {mode}"] = stack_pairs(df, f"{mode}|0")
        out[f"median of other experts, {mode}"] = (y, others(df, f"{mode}|0", "median"), g, tg)
        out[f"mean of other experts, {mode}"] = (y, others(df, f"{mode}|0", "mean"), g, tg)
    for mode in LAG_MODES:
        for lag in LAGS:
            out[f"one other expert, {mode}, lag {lag:+g} s"] = stack_pairs(df, f"{mode}|{lag:g}")
            out[f"mean of other experts, {mode}, lag {lag:+g} s"] = (y, others(df, f"{mode}|{lag:g}", "mean"), g, tg)
    for k in TOLERANCES:
        out[f"UPPER BOUND one other expert, best match within +-{k:g} s"] = stack_pairs(df, f"tol{k:g}")
    out["learned combination of the others (ridge, step)"] = (y, learned(df, "step|0")[0], g, tg)
    out["learned combination of the others (ridge, gauss2)"] = (y, learned(df, "gauss2|0")[0], g, tg)
    return out


def score_predictors(df: pd.DataFrame) -> dict:
    res = {}
    for name, (y, p, g, tg) in predictor_rows(df).items():
        m = score(y, p, g)
        labs = sorted(set(tg.tolist()))
        macro = [score(y[tg == l], p[tg == l], g[tg == l]) for l in labs]
        m["macro_over_labelers"] = {k: float(np.nanmean([x.get(k, np.nan) for x in macro if x.get("n")]))
                                    for k in SUMMARY if k != "n"}
        # each target labeler's own entries (perspectivism: every labeler scored separately)
        m["by_labeler"] = {l: {k: x.get(k) for k in SUMMARY} for l, x in zip(labs, macro) if x.get("n")}
        res[name] = m
    return res


def model_entry_ids(cfg: dict) -> set[str]:
    """The entries the per-labeler pipeline is scored on (originals with a linked track), rebuilt
    from the cached labelwin.csv files exactly as train_models does."""
    from scripts.train_models import label_rows, load_table

    rows = label_rows(cfg, load_table(cfg))          # originals only: load_table has no augmented copies
    return set(rows["entry_id"].tolist())


def model_rows(results_dir: Path, hashes: dict[str, str]) -> dict:
    out = {}
    for name, h in hashes.items():
        found = sorted(results_dir.glob(f"{h}*.json"))
        if not found:
            continue
        r = json.loads(found[0].read_text(encoding="utf-8"))
        for track in ("experts", "self"):
            for key, label in ((f"{track}/configured/classified/own_labeler", "held-out"),
                               (f"{track}/configured/train/own_labeler", "training")):
                if key in r["evaluation"]:
                    s = r["evaluation"][key]["summary"]
                    out[f"{track}/{name} ({label})"] = {k: s[k]["mean"] for k in SUMMARY if k in s}
                    out[f"{track}/{name} ({label})"]["n"] = r["data"]["n_entries"] if label == "held-out" else None
    return out


def ceiling(students: list[Student], cfg: dict, model_entries: bool) -> dict:
    t0 = time.monotonic()
    df = ceiling_rows(students)
    print(f"  ceiling rows: {len(df)} ({time.monotonic() - t0:.0f} s)", flush=True)
    out = {"subsets": {}}
    subsets = {}
    for kind in ("entry", "midpoint"):
        for track, sel in (("experts", df["target"] != SELF), ("self", df["target"] == SELF)):
            subsets[f"{kind}/{track}"] = df[(df["kind"] == kind) & sel]
    subsets["entry/experts/changes_only"] = df[(df["kind"] == "entry") & (df["target"] != SELF) & df["is_change"]]
    if model_entries:
        ids = model_entry_ids(cfg)
        on = df["entry_id"].isin(ids)
        subsets["entry/experts/model_entries"] = df[(df["kind"] == "entry") & (df["target"] != SELF) & on]
        subsets["entry/self/model_entries"] = df[(df["kind"] == "entry") & (df["target"] == SELF) & on]
        out["model_entry_ids"] = len(ids)
    for name, sub in subsets.items():
        out["subsets"][name] = {"rows": int(len(sub)), "predictors": score_predictors(sub)}
        print(f"  ceiling {name}: {len(sub)} rows ({time.monotonic() - t0:.0f} s)", flush=True)
    # learned combination: training-fold scores (how much the others' curves explain in-sample)
    sub = subsets["entry/experts"]
    _, tr_scores = learned(sub, "step|0")
    out["learned_training_folds"] = {k: float(np.nanmean([m[k] for m in tr_scores])) for k in SUMMARY if k != "n"}
    # per pair (A <- B), step, entries, pooled
    pairs = {}
    ent = df[df["kind"] == "entry"]
    for a in LABELERS:
        for b in LABELERS:
            c = f"{b}|step|0"
            if a == b or c not in ent:
                continue
            m = (ent["target"] == a) & ent[c].notna()
            if m.sum() >= 50:
                r = rating_metrics(ent.loc[m, "y"].to_numpy(float), ent.loc[m, c].to_numpy(float), TAU)
                c3 = rating_metrics(ent.loc[m, "y"].to_numpy(float), np.full(int(m.sum()), 3.0), TAU)
                pairs[f"{a} <- {b}"] = {k: r[k] for k in SUMMARY} | {"mae_constant_3": c3["mae"]}
    out["pairs_step_entries_pooled"] = pairs
    # per activity, headline predictors, expert entries, pooled within the activity
    acts = {}
    for x in sorted(ent["exp"].unique()):
        s = ent[(ent["exp"] == x) & (ent["target"] != SELF)]
        rows = predictor_rows(s)
        acts[x] = {name: {k: v for k, v in rating_metrics(*[np.asarray(a)[np.isfinite(rows[name][1])]
                                                              for a in rows[name][:2]], TAU).items()
                          if k in SUMMARY}
                   for name in ("constant 3", "one other expert, step", "mean of other experts, step",
                                "mean of other experts, gauss2", "self-consistency: A's mean over its other runs")}
    out["activities_entries_experts"] = acts
    return out


# ----------------------------------------------------------------------------- coarse

def coarse(students: list[Student]) -> dict:
    def pairwise(M: np.ndarray) -> dict:
        """Mean pairwise agreement between the columns (labelers) of a units x labelers matrix."""
        from scipy.stats import pearsonr, spearmanr

        rho, r, mae, n = [], [], [], []
        for i in range(M.shape[1]):
            for j in range(i + 1, M.shape[1]):
                ok = np.isfinite(M[:, i]) & np.isfinite(M[:, j])
                if ok.sum() >= 10 and np.ptp(M[ok, i]) > 0 and np.ptp(M[ok, j]) > 0:
                    rho.append(spearmanr(M[ok, i], M[ok, j]).statistic)
                    r.append(pearsonr(M[ok, i], M[ok, j]).statistic)
                    mae.append(float(np.mean(np.abs(M[ok, i] - M[ok, j]))))
                    n.append(int(ok.sum()))
        return {"spearman": float(np.mean(rho)) if rho else np.nan, "pearson": float(np.mean(r)) if r else np.nan,
                "mae": float(np.mean(mae)) if mae else np.nan, "pairs": len(rho), "units_per_pair": float(np.mean(n)) if n else 0}

    def summarise(units: list[tuple[str, str, np.ndarray]]) -> dict:
        """units: (group, exp, row of 6 labeler means)."""
        res = {}
        for scope in ["overall"] + [f"exp_{x}" for x in sorted({u[1] for u in units})]:
            M = np.array([u[2] for u in units if scope == "overall" or scope == f"exp_{u[1]}"])
            if not len(M):
                continue
            E = M[:, [LABELERS.index(l) for l in EXPERTS]]
            med = np.nanmedian(np.where(np.isfinite(E), E, np.nan), 1) if np.isfinite(E).any() else np.full(len(E), np.nan)
            from scipy.stats import spearmanr
            okm = np.isfinite(med) & np.isfinite(M[:, -1])
            res[scope] = {
                "units": int(len(M)),
                "experts_pairwise": pairwise(E),
                "alpha_rounded_means_experts": fast_alpha(np.where(np.isfinite(E), round_rating(E), np.nan))[0],
                "alpha_rounded_means_experts_self": fast_alpha(np.where(np.isfinite(M), round_rating(M), np.nan))[0],
                "self_vs_expert_median_spearman": float(spearmanr(M[okm, -1], med[okm]).statistic) if okm.sum() > 10 else np.nan,
                "self_vs_expert_median_mae": float(np.mean(np.abs(M[okm, -1] - med[okm]))) if okm.sum() else np.nan,
                "sd_of_expert_means": float(np.nanstd(E)),
            }
        return res

    out = {}
    # student x activity: each labeler's time-weighted mean over the session (step), from the start
    units = []
    for s in students:
        grid = np.arange(np.ceil(s.start), s.t_last, 1.0)
        units.append((s.group, s.exp, np.array([np.nanmean(s.curve(l, grid)) if l in s.tracks else np.nan
                                                for l in LABELERS])))
    out["student_x_activity"] = summarise(units)
    for k in (1, 2, 5):
        units = []
        for s in students:
            edges = np.arange(np.ceil(s.start), s.t_last, 60.0 * k)
            for e0 in edges[:-1] if len(edges) > 1 else edges:
                grid = np.arange(e0, min(e0 + 60.0 * k, s.t_last), 1.0)
                if len(grid) < 30:
                    continue
                with np.errstate(all="ignore"):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        units.append((s.group, s.exp, np.array([np.nanmean(s.curve(l, grid)) if l in s.tracks
                                                                else np.nan for l in LABELERS])))
        out[f"student_x_{k}min"] = summarise(units)
    # variance decomposition per labeler: per-second step values, between vs within student
    var = {}
    for l in LABELERS:
        vals, sids = [], []
        for i, s in enumerate(students):
            if l not in s.tracks:
                continue
            v = s.curve(l, np.arange(np.ceil(s.start), s.t_last, 1.0))
            v = v[np.isfinite(v)]
            vals.append(v); sids.append(np.full(len(v), i))
        if not vals:
            continue
        v, sid = np.concatenate(vals), np.concatenate(sids)
        means = pd.Series(v).groupby(sid).transform("mean").to_numpy()
        total = v.var()
        var[l] = {"total_variance": float(total), "between_student_share": float(means.var() / total) if total > 0 else np.nan,
                  "seconds": int(len(v))}
    out["variance_between_vs_within"] = var
    return out


# ----------------------------------------------------------------------------- main

def print_headlines(res: dict) -> None:
    if "stats" in res:
        st = res["stats"]
        print("\n## Labelers\n")
        print("| labeler | student-sessions | entries | share of expert entries | mean | median | SD | shares 1-5 | per min | "
              "median run s | runs <=2 s | first 0.1 s | repeats | never changes |")
        print("|---|" + "---|" * 13)
        for l, d in st["labelers"].items():
            print(f"| {l} | {d['student_sessions']} | {d['entries']} | {fmt(d['share_of_expert_entries'])} | "
                  f"{d['mean_rating']:.2f} | {d.get('median_rating', np.nan):.0f} | {d.get('sd_rating', np.nan):.2f} | "
                  f"{' / '.join(f'{x:.2f}' for x in d['rating_shares_1_5'])} | "
                  f"{d['entries_per_minute']:.2f} | {d['run_length_median_s']:.1f} | {d['runs_le_2s']:.2f} | "
                  f"{d['entries_in_first_tenth_of_second']:.2f} | {d['entries_repeating_previous_value']:.2f} | "
                  f"{d['tracks_never_changing']:.2f} |")
        print("\n## Co-occurrence (share of entries with another entry on the same student within +-w)\n")
        for k, d in st["co_occurrence"].items():
            print(f"  {k}: " + "  ".join(f"{w} {v:.3f}" for w, v in d.items()))
    if "alpha" in res:
        a = res["alpha"]
        print("\n## Krippendorff's ordinal alpha (overall; items / values)\n")
        print("| unit / interpolation / labelers / scale | alpha | items |\n|---|---|---|")
        for key, cell in a["cells"].items():
            if key.split("/")[1] in ("step", "gauss2") and "overall" in cell:
                print(f"| {key} | {fmt(cell['overall']['alpha'])} | {cell['overall']['items']} |")
        print("\n14 Nov setting reproduced (8 s midpoints, step):", json.dumps(a["legacy_8s_midpoint_step"]))
    if "ceiling" in res:
        c = res["ceiling"]
        for sub in ("entry/experts", "entry/experts/model_entries", "midpoint/experts", "entry/self"):
            if sub not in c["subsets"]:
                continue
            preds = c["subsets"][sub]["predictors"]
            names = [n for n in preds if "lag" not in n and not n.startswith("UPPER")
                     and ("step" in n or "gauss2" in n or "," not in n)]
            print(f"\n## Ceiling: {sub} ({c['subsets'][sub]['rows']} rows; mean over groups)\n")
            print(table([(n, preds[n]) for n in names]))
        if "model" in c:
            print("\n## Model rows (results files, fold means)\n")
            print(table(list(c["model"].items())))
    if "coarse" in res:
        print("\n## Coarse units (overall)\n")
        for unit, d in res["coarse"].items():
            if unit.startswith("student_x"):
                o = d["overall"]
                print(f"  {unit}: units {o['units']}  experts pairwise rho {fmt(o['experts_pairwise']['spearman'])}  "
                      f"MAE {fmt(o['experts_pairwise']['mae'])}  alpha(experts) {fmt(o['alpha_rounded_means_experts'])}  "
                      f"self vs median rho {fmt(o['self_vs_expert_median_spearman'])}")
        for l, d in res["coarse"]["variance_between_vs_within"].items():
            print(f"  variance {l}: between-student share {d['between_student_share']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="all", choices=["stats", "alpha", "ceiling", "coarse", "all"])
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--model-entries", action="store_true",
                    help="also score the ceiling on exactly the entries the pipeline is scored on (cached labelwin.csv)")
    ap.add_argument("--models", nargs="*", default=["single=c5341cb6385e", "moe=38c985c0f859"],
                    help="name=results-hash rows to print next to the ceiling")
    ap.add_argument("--out", default="outputs/labels/agreement.json")
    args = ap.parse_args()
    cfg = load_config(args.config, {**parse_sets(args.set), "labels.students": "all"})
    t0 = time.monotonic()
    students, _ = load_students(cfg)
    print(f"{len(students)} student-sessions ({time.monotonic() - t0:.0f} s)", flush=True)
    out_path = Path(args.out)
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    if args.what in ("stats", "all"):
        res["stats"] = stats(students)
    if args.what in ("alpha", "all"):
        res["alpha"] = alpha(students, float(cfg["window"]["length_s"]), float(cfg["window"]["overlap"]))
    if args.what in ("ceiling", "all"):
        mcfg = load_config(args.config, {**parse_sets(args.set), "augmentation.copies": 1, "labels.tau_att": 3})
        res["ceiling"] = ceiling(students, mcfg, args.model_entries)
        res["ceiling"]["model"] = model_rows(Path(cfg["output_dir"]) / "results",
                                             dict(m.split("=", 1) for m in args.models))
    if args.what in ("coarse", "all"):
        res["coarse"] = coarse(students)
    res["meta"] = {"student_sessions": len(students), "tau": TAU, "modes": {k: list(v) for k, v in MODES.items()},
                   "window_length_s": cfg["window"]["length_s"], "written": time.strftime("%Y-%m-%d %H:%M")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print_headlines(res)
    print(f"\nwrote {out_path} ({time.monotonic() - t0:.0f} s)")


if __name__ == "__main__":
    main()
