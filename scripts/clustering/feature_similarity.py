"""How much do the features tell rating situations apart? Experiment 10, Appendix C (post hoc).

  python -m scripts.clustering.feature_similarity [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--spaces base base@learned normalised normalised@learned base+agglomerative ...] [--k 10] [--gap-s 20]
         [--moment-s 2] [--boot 200] [--candidates <dir>] [--max-rows 0] [--out outputs/experiments/10_clustering]

The question: when two rating entries have nearly the same feature vector, do they get the same
rating? Each entry's vector is its window's model inputs (original recordings only, no augmented
copies), standardised over the entries of its label set. An entry seen by several cameras has a row
in each; one stands for it (the camera that sees most of the student-session's entries), so every
entry counts once. Every neighbour has a reference that shares what it shares except the feature
match. Spearman over entries, mean |difference|, share equal. Intervals: a cluster bootstrap over
class sessions (ratings and features are autocorrelated within a class), twice: two-way (each pair
weighs its entry's class times its partner's: valid, conservative) and one-sided (the entry's class
only: narrow where pairs share dependence). On synthetic data with known answers the two bracket
the true interval (see agreement()). Every comparison of a nearest neighbour with its reference has
both intervals on the difference, from the same resamples, and a verdict: firm (the two-way
interval excludes 0), borderline (only the one-sided one does), or no difference:

  between students (new students: the test the group split makes; the headline)
    nn_other_group       nearest neighbour in another group (other students, other recordings)
    knn_other_group      the mean rating of the k nearest in other groups
    random_other_group   a random entry of another group
    nn_other_group_same / random_other_group_same   the same, both rated by the entry's own labeler
                         (experts: each labeler's own habits taken out)
  classmates
    nn_other             nearest neighbour in any other student-session: often a classmate (one class
                         shares the camera, the activity and the moment); reported for how often
    nn_classmate         the entries whose nearest is a classmate: that classmate's rating
    random_classmate_matched  for the same entries, a random classmate's entry at the same time gap
                         (within x1.25, at least --moment-s): what sharing the class and the time gives
                         without a feature match
  within a student-session (same recording)
    nn_within           nearest neighbour at least --gap-s away
    random_within_matched  a random entry of the same student-session as far in time from the entry
                        as nn_within's partner (within x1.25; nearby moments agree in rating and in
                        features through time alone, so the reference is matched on the time gap)
  the labels themselves
    same_moment         experts: another expert's rating in force at the entry's time, on the same
                        student (DIPSER ratings are change events: a value holds until the labeler's
                        next one), one pair per other expert who had rated the student by then; the
                        video is the same, so this is the disagreement of the labels
    same_moment_mean    the mean of the other experts' ratings in force (experiment 06's ceiling)
    experts_in_force    self-report: the mean of the experts' ratings in force at the entry's time

Reading it: a nearest neighbour no better than its reference means feature look-alikes are rated no
more alike than chance pairs; nn_other_group near same_moment would mean they agree as two experts on
the same moment do. Distances are Euclidean on standardised columns: every column counts the same, so
a column that carries nothing still decides who is "near". The @learned spaces weight each column by
how much it predicts the rating, learned without the entry's own group: for each group, a ridge
regression on the other two groups' entries, each column scaled by the size of its coefficient. "Near"
then means alike in what predicts ratings for other students, as a model trained on the group split
would see it; the held-out group's labels are never used. Unweighted and learned together separate
"the features carry nothing" from "the features carry it in a few columns".

Also, all on other groups: the distance to the nearest against random pairs, agreement by distance
(with the excess over a random partner, since Spearman inside a distance bin is range-restricted),
the share of near-duplicates and the rating spread among their neighbours, hub concentration
(how many distinct entries are someone's nearest), and how many dimensions the standardised features
use (PCA). Written to <out>/feature_similarity.json, and printed.

Missing values: a column missing in more than --max-col-nan of a label set's entries is dropped
from its space (trk.hp.matched, for instance); an entry missing any other value (no head pose found
in its window, for instance) is left out of every space, so a filled-in value cannot make entries
look alike and every space compares the same entries. The counts are reported.

Spaces: `base` (the configured model inputs, experiment 10's `base` arm); `normalised` (the same
columns per-track standardised, the `normalised` arm); `base+<family>` (plus that family's per-student
candidate columns: sync, agglomerative, hdbscan; needs the candidates step); `base:<p1>,<p2>` (only the
base columns starting with those prefixes). Any space with the suffix @learned: its columns weighted as
above.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.clustering.common import (PER_STUDENT, SELF, base_columns, base_parser, candidate_columns, label_table,
                                       latest_candidates, load, load_candidates, per_track_normalised)

FAMILIES = {"sync": f"{PER_STUDENT}sync.", "agglomerative": f"{PER_STUDENT}agg.", "hdbscan": f"{PER_STUDENT}hdb."}
REL_BINS = [0.0, 0.25, 0.5, 0.75, np.inf]
NEAR = 0.25            # a near-duplicate: nearest other-session neighbour closer than this share of the
                       # median random-pair distance


# ------------------------------------------------------------------ data
def entries(rows: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """One original-recording row per rating entry, with the keys the analysis needs. A student is
    often seen by 2-3 cameras, each giving the entry its own row (another viewpoint, another
    vector); each student-session keeps to one camera, the one that sees most of its entries (ties
    at random, --seed), so the entries compared within a session come from the same view; an entry
    that camera does not see takes its row from another one. Among a camera's linked tracks of the
    entry (a split track), one at random."""
    r = rows[~rows["aug"].astype(bool)]
    r = r.assign(sess=r["group"] + "/" + r["exp"].astype(str) + "/" + r["sid"].astype(str))
    rng = np.random.default_rng(seed)
    count = r.groupby(["sess", "stem"])["entry_id"].nunique()
    score = count + rng.uniform(0, 0.5, len(count))              # random tie-break, below one entry
    rank = score.groupby(level="sess").rank(ascending=False, method="first")
    pref = r.set_index(["sess", "stem"]).index.map(rank).to_numpy(float)
    r = (r.assign(_pref=pref, _tie=rng.uniform(size=len(r))).sort_values(["_pref", "_tie"])
         .drop_duplicates("entry_id").sort_index().drop(columns=["_pref", "_tie"]).reset_index(drop=True))
    return r.assign(cls=r["group"] + "/" + r["exp"].astype(str), grp=r["group"].astype(str),
                    track=np.where(r["labeler"] == SELF, "self", "experts"))


def space_columns(cfg: dict, r: pd.DataFrame, space: str) -> tuple[str, list[str]]:
    """(which inputs: rows or normed; the columns) of a space (an @learned suffix is handled by main)."""
    space = space.split("@")[0]
    base = base_columns(cfg, r)
    if ":" in space:
        head, prefixes = space.split(":", 1)
        if head != "base":
            raise SystemExit(f"unknown space {space!r}: a column filter is only for base (base:<prefixes>)")
        cols = [c for c in base if c.startswith(tuple(p for p in prefixes.split(",") if p))]
        if not cols:
            raise SystemExit(f"space {space!r}: no base column starts with {prefixes!r}")
        return "rows", cols
    parts = space.split("+")
    if parts[0] not in ("base", "normalised"):
        raise SystemExit(f"unknown space {space!r}: base, normalised, base+<family>, base:<prefixes>")
    cols = list(base)
    for fam in parts[1:]:
        if fam not in FAMILIES:
            raise SystemExit(f"unknown family {fam!r} in {space!r}; known: {sorted(FAMILIES)}")
        fam_cols = [c for c in candidate_columns(r) if c.startswith(FAMILIES[fam])]
        if not fam_cols:
            raise SystemExit(f"space {space!r}: no {fam} candidate columns (run the candidates step first)")
        cols += fam_cols
    return ("normed" if parts[0] == "normalised" else "rows"), cols


def standardise(X: np.ndarray, cols: list[str], max_nan: float = 0.05) -> tuple[np.ndarray, list[str], dict]:
    """Columns standardised over the given rows; columns missing in more than max_nan of the rows,
    or constant, dropped; any NaN left -> 0 (the column mean; main passes complete rows only)."""
    nan = np.isnan(X).mean(0)
    sd = np.nanstd(X, 0) if len(X) else np.zeros(X.shape[1])
    keep = (nan <= max_nan) & (np.nan_to_num(sd) > 1e-9)
    dropped = {"often_missing": [c for c, k, n in zip(cols, keep, nan) if not k and n > max_nan],
               "constant": [c for c, k, n in zip(cols, keep, nan) if not k and n <= max_nan]}
    X, cols = X[:, keep], [c for c, k in zip(cols, keep) if k]
    X = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)
    return np.nan_to_num(X, nan=0.0).astype(np.float32), cols, dropped


def complete_rows(raw: dict, max_nan: float) -> np.ndarray:
    """The rows with a value in every column of every space, columns missing in more than max_nan of
    the rows aside (they are dropped from their space). raw: space -> (rows, columns) array."""
    masks = [~np.isnan(X[:, np.isnan(X).mean(0) <= max_nan]).any(1) for X in raw.values()]
    return np.logical_and.reduce(masks) if masks else np.zeros(0, bool)


def dimensions(X: np.ndarray) -> dict:
    """How many principal components hold 80/90/95% of the variance of the standardised vectors."""
    if len(X) < 3 or X.shape[1] == 0:
        return {}
    s = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    cum = np.cumsum(s) / s.sum()
    return {"columns": int(X.shape[1]), **{f"pcs_{int(q * 100)}": int(np.searchsorted(cum, q) + 1) for q in (0.8, 0.9, 0.95)},
            "first_pc_share": float(s[0] / s.sum())}


# ------------------------------------------------------------------ neighbours
def neighbours(X: np.ndarray, keys: dict, k: int, gap_s: float, chunk: int = 512) -> dict:
    """Per row (exact, in chunks; -1 where there is none):
      knn_i / knn_d   the k nearest rows of other groups
      any_i / any_d   the nearest of any other student-session
      same_i          the nearest of another group rated by the same labeler
      within_i        the nearest of the same student-session and recording at least gap_s away
    keys: sess, stem, grp, labeler (arrays), t (seconds)."""
    n = len(X)
    sq = (X.astype(np.float64) ** 2).sum(1)
    code = {name: np.unique(keys[name], return_inverse=True)[1] for name in ("sess", "stem", "grp", "labeler")}
    t = keys["t"]
    k = max(1, min(k, n - 1))
    out = {"knn_i": np.full((n, k), -1), "knn_d": np.full((n, k), np.inf)}
    for name in ("any", "same", "within"):
        out[f"{name}_i"], out[f"{name}_d"] = np.full(n, -1), np.full(n, np.inf)
    Xd = X.astype(np.float64)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        d = np.sqrt(np.maximum(sq[a:b, None] + sq[None, :] - 2.0 * (Xd[a:b] @ Xd.T), 0.0))
        same_sess = code["sess"][a:b, None] == code["sess"][None, :]
        other_grp = np.where(code["grp"][a:b, None] == code["grp"][None, :], np.inf, d)
        part = np.argpartition(other_grp, k - 1, axis=1)[:, :k]
        pdist = np.take_along_axis(other_grp, part, 1)
        order = np.argsort(pdist, 1)
        out["knn_i"][a:b] = np.take_along_axis(part, order, 1)
        out["knn_d"][a:b] = np.take_along_axis(pdist, order, 1)
        for name, m in (("any", np.where(same_sess, np.inf, d)),
                        ("same", np.where(code["labeler"][a:b, None] == code["labeler"][None, :], other_grp, np.inf)),
                        ("within", np.where(same_sess & (code["stem"][a:b, None] == code["stem"][None, :])
                                            & (np.abs(t[a:b, None] - t[None, :]) >= gap_s), d, np.inf))):
            out[f"{name}_i"][a:b], out[f"{name}_d"][a:b] = m.argmin(1), m.min(1)
    for name in ("knn", "any", "same", "within"):
        out[f"{name}_i"][~np.isfinite(out[f"{name}_d"])] = -1
    return out


def _draw(pool: np.ndarray, ok, rng, tries: int = 50) -> int:
    """A random element of pool satisfying ok(j), by rejection (-1 if none found)."""
    if len(pool) == 0:
        return -1
    for _ in range(tries):
        j = int(pool[rng.integers(len(pool))])
        if ok(j):
            return j
    return -1


def random_partners(keys: dict, nb: dict, gap_s: float, moment_s: float, rng) -> dict:
    """Per row, the references (-1 where there is none):
      other_group       a random row of another group
      other_group_same  a random row of another group rated by the same labeler
      classmate         if the row's nearest (any other session) is a classmate: a random row of
                        another student of the class session at about the same time gap (x1.25,
                        at least moment_s either way)
      within_matched    a random row of the same student-session and recording at about the time gap
                        of the row's nearest there (x1.25, and >= gap_s)"""
    sess, stem, grp, lab, cls, t = (keys[k] for k in ("sess", "stem", "grp", "labeler", "cls", "t"))
    n = len(sess)
    out = {name: np.full(n, -1) for name in ("other_group", "other_group_same", "classmate", "within_matched")}
    idx = np.arange(n)
    by_lab = {v: idx[lab == v] for v in np.unique(lab)}
    by_rec = pd.Series(idx).groupby([sess, stem]).indices
    by_cls = {c: idx[cls == c] for c in np.unique(cls)}

    for i in range(n):
        t_i = t[i]
        out["other_group"][i] = _draw(idx, lambda j: grp[j] != grp[i], rng)
        out["other_group_same"][i] = _draw(by_lab[lab[i]], lambda j: grp[j] != grp[i], rng)
        j0 = nb["any_i"][i]
        if j0 >= 0 and cls[j0] == cls[i]:
            pool = by_cls[cls[i]]
            pool = pool[sess[pool] != sess[i]]
            gap = abs(t[j0] - t_i)
            dt = np.abs(t[pool] - t_i)
            keep = (dt <= max(moment_s, gap * 1.25)) & (dt >= (gap / 1.25 if gap / 1.25 > moment_s else 0.0))
            out["classmate"][i] = int(pool[keep][rng.integers(keep.sum())]) if keep.any() else -1
        j0 = nb["within_i"][i]
        if j0 >= 0:
            gap = abs(t[j0] - t_i)
            cand = by_rec[(sess[i], stem[i])]
            dt = np.abs(t[cand] - t_i)
            cand = cand[(dt >= max(gap_s, gap / 1.25)) & (dt <= gap * 1.25)]
            if len(cand):
                out["within_matched"][i] = int(cand[rng.integers(len(cand))])
    return out


def load_tracks(cfg: dict, r: pd.DataFrame) -> dict:
    """Every labeler's rating track (a step function) for the students of r, keyed by (sess,
    labeler), read from the DIPSER label archives as the pipeline reads them (src/labels: the
    configured offset_s and labels.collapse)."""
    from src.core.types import SessionKey
    from src.datasets import DipserIndex
    from src.labels.session import _load, collapse_key

    index, lcfg = DipserIndex.from_config(cfg), cfg["labels"]
    out = {}
    for (session, sid, sess), _ in r.groupby(["session", "sid", "sess"]).groups.items():
        zips = index.subject_zips(SessionKey(*session.split("#")[0].split("/")))
        if sid not in zips:
            continue
        for labeler, tr in _load(zips[sid], float(lcfg["offset_s"]), collapse_key(lcfg)).items():
            if len(tr.t):
                out[(sess, labeler)] = tr
    return out


def in_force(r: pd.DataFrame, tracks: dict) -> dict:
    """The experts' ratings in force at each entry's time, on the same student: `pairs` (row index,
    value), one per other expert who had rated the student by then (experts' rows: every expert but
    the entry's own; self-report rows: every expert), and `mean`, their mean per row (NaN if none)."""
    from src.labels.attention import SELF_LABELER

    by_sess: dict = {}
    for (sess, lab), tr in tracks.items():
        if lab != SELF_LABELER:
            by_sess.setdefault(sess, []).append((lab, tr))
    rows_i, vals, mean = [], [], np.full(len(r), np.nan)
    for i, (sess, lab, t) in enumerate(zip(r["sess"], r["labeler"], r["t"].astype(float))):
        v = [tr.at(t) for other, tr in by_sess.get(sess, []) if other != lab]
        v = [x for x in v if np.isfinite(x)]
        if v:
            rows_i += [i] * len(v)
            vals += v
            mean[i] = float(np.mean(v))
    return {"pairs": (np.asarray(rows_i, int), np.asarray(vals, float)), "mean": mean}


def learned_weights(X: np.ndarray, y: np.ndarray, train: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Column weights for the distance: the size of each column's ridge coefficient predicting the
    rating on the training rows only (the other groups), scaled to a mean square of 1. The held-out
    group's labels are never used, as in the group split."""
    from sklearn.linear_model import Ridge

    coef = np.abs(Ridge(alpha=alpha).fit(X[train], y[train]).coef_)
    ms = np.sqrt(np.mean(coef ** 2))
    return (coef / ms if ms > 0 else np.ones(X.shape[1])).astype(np.float32)


def neighbours_learned(X: np.ndarray, y: np.ndarray, keys: dict, k: int, gap_s: float, rng) -> tuple[dict, np.ndarray, dict]:
    """neighbours() with, for each group's entries, the columns weighted by learned_weights fitted on
    the other groups; also each row's distance scale (the median random other-group distance in its
    group's weighted space) and the weights per group."""
    n, grp = len(X), keys["grp"]
    nb, scale, weights = None, np.full(n, np.nan), {}
    for g in np.unique(grp):
        rows = grp == g
        w = learned_weights(X, y, ~rows)
        weights[str(g)] = w
        Xw = X * w
        part = neighbours(Xw, keys, k, gap_s)
        if nb is None:
            nb = {name: v.copy() for name, v in part.items()}
        else:
            for name, v in part.items():
                nb[name][rows] = v[rows]
        scale[rows] = median_random_distance(Xw, grp, rng)
    return nb, scale, weights


def median_random_distance(X: np.ndarray, grp: np.ndarray, rng, n_pairs: int = 20000) -> float:
    """The median distance between random rows of different groups (the scale for "near")."""
    i, j = rng.integers(len(X), size=n_pairs), rng.integers(len(X), size=n_pairs)
    ok = grp[i] != grp[j]
    return float(np.median(np.linalg.norm(X[i[ok]] - X[j[ok]], axis=1))) if ok.any() else float("nan")


# ------------------------------------------------------------------ statistics
def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_weights(n_classes: int, boot: int, rng) -> np.ndarray:
    """(boot, n_classes) counts of each class session in each resample (all ones as row 0 is not
    included: the point estimate is computed without weights)."""
    if not boot or n_classes < 3:
        return np.zeros((0, n_classes))
    return rng.multinomial(n_classes, np.full(n_classes, 1 / n_classes), size=boot).astype(float)


def _weighted_pearson(rx: np.ndarray, ry: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Pearson of rx and ry under each row of weights W (boot, n); NaN where degenerate."""
    sw = W.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = (W @ rx) / sw, (W @ ry) / sw
        cxy = (W @ (rx * ry)) / sw - mx * my
        vx = (W @ (rx * rx)) / sw - mx ** 2
        vy = (W @ (ry * ry)) / sw - my ** 2
        return cxy / np.sqrt(vx * vy)


def agreement(y: np.ndarray, z: np.ndarray, gq: np.ndarray, Wb: np.ndarray, gp: np.ndarray | None = None,
              discrete: bool = True) -> dict:
    """Spearman of y against z (rows with a partner), mean |difference|, share equal (when z is a
    rating, not a mean of ratings), and percentile intervals for Spearman from the class-session
    resamples Wb (boot, n_classes), ranks those of the full sample:
      lo / hi      two-way: each pair weighs the resample count of its entry's class (gq, codes) times
                   that of its partner's class (gp; 2-D for several partners: their mean). Valid but
                   conservative: on synthetic data with 12 class sessions it covered 97-100% and was
                   1.4-1.8x wider than the true sampling spread
      lo_1 / hi_1  one-sided: the entry's class only. Narrow: it covered only 87-93% where pairs
                   share dependence (a shared moment, drift in time)
    The true interval lies between; a difference is firm if the two-way interval excludes 0,
    borderline if only the one-sided one does. `_boot`, `_boot1` keep the resampled values for
    paired differences (removed before writing)."""
    ok = np.isfinite(z)
    out = {"n": int(ok.sum()), "spearman": rank_corr(y[ok], z[ok])}
    if out["n"] == 0:
        return out
    out["mean_abs_diff"] = float(np.mean(np.abs(y[ok] - z[ok])))
    if discrete:
        out["share_equal"] = float(np.mean(np.isclose(y[ok], z[ok])))
    if len(Wb) and out["n"] >= 20:
        rx = pd.Series(y[ok]).rank().to_numpy()
        ry = pd.Series(z[ok]).rank().to_numpy()
        w1 = Wb[:, gq[ok]]
        w2 = w1
        if gp is not None:
            g = gp[ok]
            w2 = w1 * (Wb[:, g].mean(2) if g.ndim == 2 else Wb[:, g])
        for tag, w in (("", w2), ("1", w1)):
            vals = _weighted_pearson(rx, ry, w)
            out[f"_boot{tag}"] = vals
            if np.isfinite(vals).sum() >= 10:
                out[f"lo{'_' + tag if tag else ''}"], out[f"hi{'_' + tag if tag else ''}"] = (
                    float(v) for v in np.nanpercentile(vals, [2.5, 97.5]))
    return out


def difference(a: dict, b: dict) -> dict:
    """a's Spearman minus b's, with a percentile interval from the shared resamples."""
    d = {"diff": float(a.get("spearman", np.nan) - b.get("spearman", np.nan))}
    for tag, suffix in (("", ""), ("1", "_1")):
        if f"_boot{tag}" in a and f"_boot{tag}" in b:
            v = a[f"_boot{tag}"] - b[f"_boot{tag}"]
            if np.isfinite(v).sum() >= 10:
                d[f"lo{suffix}"], d[f"hi{suffix}"] = (float(x) for x in np.nanpercentile(v, [2.5, 97.5]))
    lo, hi, lo1, hi1 = (d.get(x, np.nan) for x in ("lo", "hi", "lo_1", "hi_1"))
    d["verdict"] = ("firm" if lo > 0 or hi < 0 else "borderline" if lo1 > 0 or hi1 < 0
                    else "no difference" if np.isfinite(lo1) else "no interval")
    return d


def partner_rating(y: np.ndarray, j: np.ndarray) -> np.ndarray:
    z = np.full(len(y), np.nan)
    z[j >= 0] = y[j[j >= 0]]
    return z


COMPARISONS = [("nn_other_group", "random_other_group"), ("knn_other_group", "random_other_group"),
               ("nn_other_group_same", "random_other_group_same"), ("nn_classmate", "random_classmate_matched"),
               ("nn_within_paired", "random_within_matched"), ("nn_other_group", "same_moment"),
               ("nn_other_group", "experts_in_force")]


def analyse_track(r: pd.DataFrame, X: np.ndarray, k: int, gap_s: float, boot: int, rng, moment_s: float = 2.0,
                  forced: dict | None = None, learned: bool = False) -> dict:
    """Every agreement of one label set. forced: in_force(r, tracks) for the label references
    (None: left out). learned: distances in columns weighted per group by learned_weights (fitted on
    the other groups)."""
    y = r["rating"].to_numpy(float)
    keys = {c: r[c].to_numpy() for c in ("sess", "stem", "grp", "labeler", "cls", "tid")}
    keys["t"] = r["t"].to_numpy(float)
    cls, t = keys["cls"], keys["t"]
    cc = np.unique(cls, return_inverse=True)[1]                     # class-session codes, for the resamples
    Wb = boot_weights(int(cc.max()) + 1, boot, rng)
    weights = None
    if learned:
        nb, scale, weights = neighbours_learned(X, y, keys, k, gap_s, rng)
        rel = nb["knn_d"][:, 0] / scale
        med = float(np.nanmedian(scale))
    else:
        nb = neighbours(X, keys, k, gap_s)
        med = median_random_distance(X, keys["grp"], rng)
        rel = nb["knn_d"][:, 0] / med if med > 0 else np.full(len(y), np.nan)
    rp = random_partners(keys, nb, gap_s, moment_s, rng)
    ky = np.where(nb["knn_i"] >= 0, y[np.maximum(nb["knn_i"], 0)], np.nan)
    first, anyn = nb["knn_i"][:, 0], nb["any_i"]

    def ag(j, **kw):
        return agreement(y, partner_rating(y, j), cc, Wb, cc[np.maximum(j, 0)], **kw)

    has_any = anyn >= 0
    mate = has_any & (cls[np.maximum(anyn, 0)] == cls)
    res = {
        "rows": int(len(r)), "student_sessions": int(r["sess"].nunique()), "groups": int(r["grp"].nunique()),
        "label_sd": float(y.std()), "median_random_distance": med,
        "nn_distance_relative": ({q: float(np.nanpercentile(rel[np.isfinite(rel)], p)) for q, p in
                                  (("p10", 10), ("median", 50), ("p90", 90))} if np.isfinite(rel).any() else {}),
        "within_distance_relative_median": (float(np.median(nb["within_d"][np.isfinite(nb["within_d"])] / med))
                                            if med > 0 and np.isfinite(nb["within_d"]).any() else float("nan")),
        "nn_any_is_classmate": float(mate.sum() / max(1, has_any.sum())),
        "nn_classmate_median_time_gap_s": float(np.median(np.abs(t[anyn[mate]] - t[mate]))) if mate.any() else float("nan"),
        "nn_distinct_share": float(len(np.unique(first[first >= 0])) / max(1, (first >= 0).sum())),
        "agreement": {
            "nn_other_group": ag(first),
            "knn_other_group": agreement(y, np.nanmean(ky, 1), cc, Wb, cc[np.maximum(nb["knn_i"], 0)], discrete=False),
            "random_other_group": ag(rp["other_group"]),
            "nn_other": ag(anyn),
            "nn_classmate": ag(np.where(mate, anyn, -1)),
            "random_classmate_matched": ag(np.where(mate, rp["classmate"], -1)),
            "nn_within": ag(nb["within_i"]),
            "random_within_matched": ag(np.where(rp["within_matched"] >= 0, nb["within_i"], -1)),
            "random_within_matched_ref": ag(rp["within_matched"]),
        },
    }
    # nn_within on exactly the entries that got a matched reference, so the two compare like for like
    res["agreement"]["nn_within_paired"] = res["agreement"].pop("random_within_matched")
    res["agreement"]["random_within_matched"] = res["agreement"].pop("random_within_matched_ref")
    experts = (r["track"] == "experts").all()
    if experts:
        res["agreement"]["nn_other_group_same"] = ag(nb["same_i"])
        res["agreement"]["random_other_group_same"] = ag(rp["other_group_same"])
    if forced is not None:
        i, v = forced["pairs"]
        # the other experts' ratings are on the same student, so the partner's class is the entry's own
        if experts:
            res["agreement"]["same_moment"] = agreement(y[i], v, cc[i], Wb, cc[i])
            res["agreement"]["same_moment_mean"] = agreement(y, forced["mean"], cc, Wb, cc, discrete=False)
        else:
            res["agreement"]["experts_in_force"] = agreement(y, forced["mean"], cc, Wb, cc, discrete=False)
    res["comparisons"] = {f"{a} - {b}": difference(res["agreement"][a], res["agreement"][b])
                          for a, b in COMPARISONS if a in res["agreement"] and b in res["agreement"]}
    for a in res["agreement"].values():
        a.pop("_boot", None)
        a.pop("_boot1", None)
    if weights is not None:
        res["learned_weights"] = {g: w.tolist() for g, w in weights.items()}
    # agreement by how close the nearest (other group) is; Spearman inside a bin is range-restricted
    # (the closest pairs sit where the data are dense), so each bin also has `excess`: the mean
    # |difference| to a random other-group partner of the same entries minus that to the nearest
    z1, zr = partner_rating(y, first), partner_rating(y, rp["other_group"])
    bins = []
    for lo, hi in zip(REL_BINS[:-1], REL_BINS[1:]):
        m = (rel >= lo) & (rel < hi)
        a = agreement(y[m], z1[m], cc[m], Wb[:0]) if m.sum() else {"n": 0, "spearman": float("nan")}
        ok = m & np.isfinite(z1) & np.isfinite(zr)
        if a["n"] and ok.any():
            a["mean_abs_diff_random"] = float(np.mean(np.abs(y[ok] - zr[ok])))
            a["excess"] = float(np.mean(np.abs(y[ok] - zr[ok])) - np.mean(np.abs(y[ok] - z1[ok])))
        bins.append({"from": lo, "to": None if np.isinf(hi) else hi, "share": float(m.mean()), **a})
    res["by_distance"] = bins
    near = rel < NEAR
    res["near_duplicates"] = {
        "share": float(near.mean()),
        "neighbour_rating_sd_median": float(np.nanmedian(np.nanstd(ky[near], 1))) if near.any() else float("nan"),
        "neighbour_rating_sd_median_all": float(np.nanmedian(np.nanstd(ky, 1))),
        "by_activity": r.loc[near, "exp"].astype(str).value_counts(normalize=True).round(3).to_dict() if near.any() else {},
    }
    return res


# ------------------------------------------------------------------ output
ORDER = [("same_moment", "labels: another expert's rating in force"),
         ("same_moment_mean", "labels: mean of the other experts in force"),
         ("experts_in_force", "labels: mean of the experts in force"),
         ("nn_other_group", "nearest, another group"),
         ("knn_other_group", "mean of the k nearest, other groups"),
         ("random_other_group", "  random, another group"),
         ("nn_other_group_same", "nearest, another group, same labeler"),
         ("random_other_group_same", "  random, another group, same labeler"),
         ("nn_other", "nearest, any other session"),
         ("nn_classmate", "  ... where that is a classmate"),
         ("random_classmate_matched", "  random classmate, same time gap"),
         ("nn_within", "nearest, same session, >= gap"),
         ("nn_within_paired", "  ... entries with a matched reference"),
         ("random_within_matched", "  random, same session, same time gap")]


def fmt(a: dict) -> str:
    if not a or a.get("n", 0) == 0 or not np.isfinite(a.get("spearman", np.nan)):
        return "–" + (f"  (n {a['n']})" if a and a.get("n") else "")
    ci = f" [{a['lo']:+.3f}, {a['hi']:+.3f}]" if "lo" in a else ""      # the two-way (conservative) interval
    eq = f"  equal {a['share_equal']:.0%}" if "share_equal" in a else ""
    return f"{a['spearman']:+.3f}{ci}  |diff| {a['mean_abs_diff']:.2f}{eq}  (n {a['n']})"


def report(space: str, track: str, res: dict, dropped: dict) -> None:
    print(f"\n=== {space}, {track}: {res['rows']} entries, {res['student_sessions']} student-sessions, "
          f"{res['groups']} groups, label SD {res['label_sd']:.2f}")
    print(f"  dimensions: {res['dimensions']}; dropped: {sum(len(v) for v in dropped.values())} columns "
          f"({', '.join(dropped['often_missing'] + dropped['constant'])[:300]})")
    nd = res["nn_distance_relative"]
    if nd:
        print(f"  nearest in another group, distance / median random other-group distance: p10 {nd['p10']:.2f}, "
              f"median {nd['median']:.2f}, p90 {nd['p90']:.2f}; nearest within session (>= gap): "
              f"{res['within_distance_relative_median']:.2f}")
    print(f"  the nearest in any other session is a classmate for {res['nn_any_is_classmate']:.0%} of entries "
          f"(median time gap {res['nn_classmate_median_time_gap_s']:.1f} s); distinct nearest (other groups): "
          f"{res['nn_distinct_share']:.0%} of entries")
    for key, label in ORDER:
        if key in res["agreement"]:
            print(f"  {label:42s} {fmt(res['agreement'][key])}")
    print("  differences (Spearman; 95% intervals: two-way (conservative) / one-sided (narrow)):")
    for name, d in res.get("comparisons", {}).items():
        ci = (f" [{d['lo']:+.3f}, {d['hi']:+.3f}] / [{d['lo_1']:+.3f}, {d['hi_1']:+.3f}]"
              if "lo" in d and "lo_1" in d else "")
        print(f"    {name:48s} {d['diff']:+.3f}{ci}  {d['verdict']}")
    if "learned_top" in res:
        print("  heaviest learned weights (mean over the groups' fits): " +
              ", ".join(f"{c} {w:.1f}" for c, w in list(res["learned_top"].items())[:6]))
    print("  by distance of the nearest in another group (relative to random other-group pairs):")
    for b in res["by_distance"]:
        rng_ = f"{b['from']:g}-{'' if b['to'] is None else format(b['to'], 'g')}"
        if b["n"] and "excess" in b:
            print(f"    {rng_:8s} {b['share']:5.0%} of entries: rho {b['spearman']:+.3f}, |diff| {b['mean_abs_diff']:.2f} "
                  f"vs random {b['mean_abs_diff_random']:.2f} (excess {b['excess']:+.2f})")
        else:
            print(f"    {rng_:8s} none")
    nd2 = res["near_duplicates"]
    print(f"  near-duplicates (nearest < {NEAR:g} of the random distance): {nd2['share']:.1%} of entries; rating SD "
          f"among their k nearest: median {nd2['neighbour_rating_sd_median']:.2f} (all entries "
          f"{nd2['neighbour_rating_sd_median_all']:.2f}); by activity {nd2['by_activity']}")


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--spaces", nargs="*", default=["base", "base@learned", "normalised", "normalised@learned",
                                                     "base+agglomerative", "base+agglomerative@learned"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--gap-s", type=float, default=20.0, help="within a session, neighbours at least this far apart")
    ap.add_argument("--moment-s", type=float, default=2.0,
                    help="a classmate at the same moment: within this many seconds")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", default=None, help="candidate directory (default: the latest under <out>)")
    ap.add_argument("--max-col-nan", type=float, default=0.05,
                    help="columns missing in more than this share of entries are dropped; the rest of the "
                         "entries with a missing value are excluded")
    ap.add_argument("--max-rows", type=int, default=0, help="subsample each label set to this many rows (0: all)")
    args = ap.parse_args()
    cfg, out = load(args)
    rng = np.random.default_rng(args.seed)

    rows = label_table(cfg)
    if any("+" in s for s in args.spaces):
        rows = load_candidates(rows, latest_candidates(out, args.candidates))
    r = entries(rows, args.seed)
    tracks = load_tracks(cfg, r)
    main_cam = r.groupby("sess")["stem"].transform(lambda s: s.value_counts().index[0])
    results_main_cam = float((r["stem"] == main_cam).mean())
    print(f"entries from their student-session's main camera: {results_main_cam:.0%} (the rest from another camera: "
          f"compared between sessions, but within a session only with entries of the same camera); "
          f"label tracks loaded: {len(tracks)}", flush=True)
    print(f"{len(rows[~rows['aug'].astype(bool)])} original-recording rows -> {len(r)} entries, one row each "
          f"({(r['track'] == 'experts').sum()} expert, "
          f"{(r['track'] == 'self').sum()} self-report), {r['sess'].nunique()} student-sessions, "
          f"{r['grp'].nunique()} groups", flush=True)
    normed = None
    spaces = {}
    for space in args.spaces:
        which, cols = space_columns(cfg, r, space)
        if which == "normed" and normed is None:
            normed = per_track_normalised(cfg, r, base_columns(cfg, r))
        spaces[space] = (normed if which == "normed" else r, cols)
    results = {"settings": {k: v for k, v in vars(args).items() if k not in ("set",)}, "spaces": {s: {"tracks": {}} for s in spaces},
               "entries": {}, "main_camera_share": results_main_cam, "label_tracks": len(tracks)}
    for track in ("experts", "self"):
        m = (r["track"] == track).to_numpy()
        if args.max_rows and m.sum() > args.max_rows:
            keep = rng.choice(np.flatnonzero(m), args.max_rows, replace=False)
            m = np.zeros(len(r), bool)
            m[keep] = True
        # a filled-in value would make entries missing the same columns (e.g. no head pose found) look
        # alike, so only entries complete in every space are compared, the same entries in each space;
        # columns missing in more than --max-col-nan of the entries are dropped first (standardise)
        raw = {s: data.loc[m].reindex(columns=cols).to_numpy(float) for s, (data, cols) in spaces.items()}
        complete = complete_rows(raw, args.max_col_nan)
        results["entries"][track] = {"entries": int(m.sum()), "complete": int(complete.sum()),
                                     "excluded_missing_values": int((~complete).sum())}
        print(f"\n{track}: {int(m.sum())} entries, {int((~complete).sum())} excluded for missing values "
              f"({(~complete).mean():.1%}), {int(complete.sum())} compared in every space", flush=True)
        if complete.sum() < 10:
            print(f"  too few; skipped")
            continue
        rt = r[m].reset_index(drop=True)[complete].reset_index(drop=True)
        forced = in_force(rt, tracks)
        for space, (data, cols) in spaces.items():
            X, kept, dropped = standardise(raw[space][complete], cols, args.max_col_nan)
            res = analyse_track(rt, X, args.k, args.gap_s, args.boot, rng, args.moment_s, forced,
                                learned=space.endswith("@learned"))
            res["dimensions"], res["columns"], res["dropped"] = dimensions(X), kept, dropped
            if "learned_weights" in res:
                mean_w = np.mean([np.asarray(w) for w in res["learned_weights"].values()], 0)
                res["learned_top"] = {kept[i]: float(mean_w[i]) for i in np.argsort(-mean_w)[:10]}
            results["spaces"][space]["tracks"][track] = res
            report(space, track, res, dropped)
    path = out / "feature_similarity.json"
    path.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
