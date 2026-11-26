"""Class-relative features: one student against the students around them in the same window
(guide Chapter 15). Pure functions on small arrays, so the screening scripts (scripts/clustering/)
and a future CFMU extractor (guide 15 §15.7) compute exactly the same numbers.

Two spaces (guide 15 §15.4):
  descriptor space  X (n, D): one row per visible student, their window descriptors, standardised
                    per track over the session (requirement R6) before anything here sees them
  synchrony space   S (n, T): one row per student, a per-frame channel over the window's frames
                    (NaN where the student is not tracked); "similar" = rising and falling together

Every function is permutation invariant (R1): reordering the students reorders the per-student
outputs the same way and leaves the window outputs unchanged. NaN rules (§15.6):
  n < 3                              per-student deviation and synchrony are NaN
  n < 4                              clustering features are NaN
  < MIN_COMMON shared frames         that pair's (or that student's) correlations are NaN
"""

from __future__ import annotations

import numpy as np

from .spread import spread_decomposition

MIN_COMMON = 8        # frames two series must share for a correlation (about 1.6 s at 5 fps)
MIN_SYNC_N = 3
MIN_CLUSTER_N = 4


# ------------------------------------------------------------------ synchrony space (option 0)
def _corr(a: np.ndarray, b: np.ndarray, min_common: int) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_common:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else np.nan


def pairwise_corr(S: np.ndarray, min_common: int = MIN_COMMON) -> np.ndarray:
    """(n, n) Pearson correlation of every pair of series on their common frames; NaN on the
    diagonal, for pairs with fewer than min_common common frames, and for flat series."""
    S = np.asarray(S, float)
    n = len(S)
    if n == 0:
        return np.zeros((0, 0))
    # vectorised over all pairs: sums over each pair's common frames via mask products
    M = np.isfinite(S).astype(float)
    X = np.where(M > 0, S, 0.0)
    N = M @ M.T                               # common frames per pair
    Sx = X @ M.T                              # sum of x_i over the frames i and j share
    Sxx = (X * X) @ M.T
    Sxy = X @ X.T
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Sxy - Sx * Sx.T / N
        vx = Sxx - Sx * Sx / N
        vy = vx.T
        den = np.sqrt(vx * vy)
        # flat series: the same 1e-12 cut as _corr, relative to the scale of the values
        C = np.where((den > 1e-12) & (N >= min_common), cov / den, np.nan)
    np.fill_diagonal(C, np.nan)
    return np.clip(C, -1.0, 1.0)


def loo_median(S: np.ndarray) -> np.ndarray:
    """(n, T): for each student, the frame-by-frame median of the other students' series."""
    S = np.asarray(S, float)
    n = len(S)
    out = np.full_like(S, np.nan)
    for i in range(n):
        others = np.delete(S, i, axis=0)
        ok = np.isfinite(others).any(axis=0)
        if ok.any():
            out[i, ok] = np.nanmedian(others[:, ok], axis=0)
    return out


def sync_with_class(S: np.ndarray, min_common: int = MIN_COMMON) -> np.ndarray:
    """(n,) corr(s_i, median of the others); NaN for n < 3."""
    S = np.asarray(S, float)
    if len(S) < MIN_SYNC_N:
        return np.full(len(S), np.nan)
    M = loo_median(S)
    return np.array([_corr(S[i], M[i], min_common) for i in range(len(S))])


def lag_behind_class(S: np.ndarray, max_lag: int, min_common: int = MIN_COMMON) -> tuple[np.ndarray, np.ndarray]:
    """(lag, peak corr) per student: the shift k in [-max_lag, max_lag] frames maximising
    corr(s_i(t), m_-i(t - k)). Positive k = the student moves k frames after the class (the
    sleepers' lag, Raca et al.). NaN for n < 3 or where no shift has min_common common frames."""
    S = np.asarray(S, float)
    n, T = S.shape if S.ndim == 2 else (0, 0)
    lag, peak = np.full(n, np.nan), np.full(n, np.nan)
    if n < MIN_SYNC_N:
        return lag, peak
    M = loo_median(S)
    for i in range(n):
        best, best_k = -np.inf, np.nan
        for k in range(-min(max_lag, T - 1), min(max_lag, T - 1) + 1):   # |k| < T: short windows
            if k >= 0:
                a, b = S[i, k:], M[i, :T - k]
            else:
                a, b = S[i, :T + k], M[i, -k:]
            r = _corr(a, b, min_common)
            # ties go to the smallest |k|: a flat cross-correlation is not evidence of a lag
            if np.isfinite(r) and (r > best + 1e-12 or (abs(r - best) <= 1e-12 and abs(k) < abs(best_k))):
                best, best_k = r, k
        if np.isfinite(best):
            lag[i], peak[i] = best_k, best
    return lag, peak


def graph_degree(C: np.ndarray) -> np.ndarray:
    """(n,) each student's mean correlation with every other student (NaN pairs skipped)."""
    C = np.asarray(C, float)
    if len(C) < 2:
        return np.full(len(C), np.nan)
    with np.errstate(invalid="ignore"):
        return np.array([np.nanmean(np.delete(C[i], i)) if np.isfinite(np.delete(C[i], i)).any() else np.nan
                         for i in range(len(C))])


def class_synchrony(C: np.ndarray, sync: np.ndarray, threshold: float = 0.3) -> dict[str, float]:
    """Window level: the class's mean pairwise correlation, and the share of students whose
    synchrony with the class exceeds `threshold` (over students where it is defined)."""
    iu = np.triu_indices(len(C), 1)
    vals = np.asarray(C, float)[iu]
    vals = vals[np.isfinite(vals)]
    s = np.asarray(sync, float)
    s = s[np.isfinite(s)]
    return {"mean_corr": float(vals.mean()) if len(vals) else np.nan,
            "share_synced": float(np.mean(s > threshold)) if len(s) else np.nan}


def sync_features(S: np.ndarray, max_lag: int, min_common: int = MIN_COMMON,
                  threshold: float = 0.3) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Option 0 in synchrony space: per student {corr, lag, lag_corr, degree}, per window
    {mean_corr, share_synced}."""
    S = np.asarray(S, float).reshape(len(S), -1) if len(S) else np.zeros((0, 0))
    n = len(S)
    C = pairwise_corr(S, min_common)
    sync = sync_with_class(S, min_common)
    lag, peak = lag_behind_class(S, max_lag, min_common)
    deg = graph_degree(C) if n >= MIN_SYNC_N else np.full(n, np.nan)
    win = class_synchrony(C, sync, threshold) if n >= MIN_SYNC_N else {"mean_corr": np.nan, "share_synced": np.nan}
    return {"corr": sync, "lag": lag, "lag_corr": peak, "degree": deg}, win


# ------------------------------------------------------------------ descriptor space (option 0)
def deviation_from_class(X: np.ndarray) -> np.ndarray:
    """(n,) median over dimensions of |x_id - median_{j != i} x_jd| / MAD_d, where MAD_d is the
    class's median absolute deviation in dimension d. Dimensions with MAD 0 or NaN values are
    skipped per student; NaN for n < 3."""
    X = np.asarray(X, float)
    n = len(X)
    if n < MIN_SYNC_N:
        return np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mad = np.nanmedian(np.abs(X - np.nanmedian(X, axis=0)), axis=0)
        out = np.full(n, np.nan)
        for i in range(n):
            med = np.nanmedian(np.delete(X, i, axis=0), axis=0)
            d = np.abs(X[i] - med) / mad
            d = d[np.isfinite(d) & (mad > 1e-12)]
            if len(d):
                out[i] = float(np.median(d))
    return out


# ------------------------------------------------------------------ clustering (options 1, 2)
def correlation_distance(C: np.ndarray) -> np.ndarray:
    """1 - r, with undefined pairs at distance 1 (no evidence of moving together); 0 on the diagonal."""
    D = 1.0 - np.asarray(C, float)
    D[~np.isfinite(D)] = 1.0
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 2.0)


def _rms_scale(X: np.ndarray) -> np.ndarray:
    """Descriptors with NaN filled by the class median (0 if all NaN), divided by sqrt(D), so a
    distance threshold is in RMS standardised units whatever the number of descriptors."""
    X = np.asarray(X, float).copy()
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0) if np.isfinite(X).any() else np.zeros(X.shape[1])
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(X)
    X[bad] = np.take(med, np.nonzero(bad)[1])
    return X / np.sqrt(max(X.shape[1], 1))


def agglomerative_labels(tau: float, X: np.ndarray | None = None, D: np.ndarray | None = None,
                         linkage: str | None = None) -> np.ndarray:
    """Cluster ids from hierarchical clustering cut at distance tau (option 1). Give X (descriptor
    space; Ward by default, on RMS-scaled descriptors) or D (a distance matrix; average linkage by
    default, Ward being invalid for non-Euclidean distances). n < 2 gives one cluster per student."""
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import linkage as hlinkage
    from scipy.spatial.distance import squareform

    n = len(X) if X is not None else len(D)
    if n < 2:
        return np.zeros(n, int)
    if X is not None:
        Z = hlinkage(_rms_scale(X), method=linkage or "ward")
    else:
        Z = hlinkage(squareform(np.asarray(D, float), checks=False), method=linkage or "average")
    return fcluster(Z, t=tau, criterion="distance").astype(int) - 1


def hdbscan_labels(X: np.ndarray | None = None, D: np.ndarray | None = None,
                   min_cluster_size: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """(labels, membership strength) from HDBSCAN (option 2); label -1 = noise, "fits no group of
    classmates". allow_single_cluster, so a class all doing the same thing is one cluster."""
    from sklearn.cluster import HDBSCAN

    n = len(X) if X is not None else len(D)
    if n < max(MIN_CLUSTER_N, min_cluster_size):
        return np.full(n, -1), np.zeros(n)
    if X is not None:
        model = HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True, copy=True).fit(_rms_scale(X))
    else:
        model = HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True, copy=True,
                        metric="precomputed").fit(np.asarray(D, float))
    return model.labels_.astype(int), np.asarray(model.probabilities_, float)


def cluster_features(labels: np.ndarray, X: np.ndarray | None = None, D: np.ndarray | None = None,
                     strength: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Features from one window's clustering that depend only on group relations, never on cluster
    ids (R4). Noise (label -1) counts as its own singleton group.

    per student: share (n_c / n), singleton, in_largest, centroid_dist (X: RMS distance to the
                 cluster mean; D: mean distance to the other members; 0 for singletons),
                 strength (HDBSCAN membership, when given)
    per window:  n_clusters (/ n), largest_share, noise_share, and with X the spread decomposition:
                 chaos (mean within-cluster SD), fragmentation (between-cluster share of variance)
    NaN everywhere for n < 4."""
    labels = np.asarray(labels)
    n = len(labels)
    nan_s = {k: np.full(n, np.nan) for k in ("share", "singleton", "in_largest", "centroid_dist")}
    nan_w = {k: np.nan for k in ("n_clusters", "largest_share", "noise_share", "chaos", "fragmentation")}
    if strength is not None:
        nan_s["strength"] = np.full(n, np.nan)
    if n < MIN_CLUSTER_N:
        return nan_s, nan_w
    # noise points become their own groups, so they are singletons
    lab = labels.copy()
    noise = lab < 0
    lab[noise] = lab.max(initial=-1) + 1 + np.arange(noise.sum())
    ids, counts = np.unique(lab, return_counts=True)
    size = dict(zip(ids.tolist(), counts.tolist()))
    largest = counts.max()
    share = np.array([size[c] / n for c in lab])
    per = {"share": share, "singleton": (share * n == 1).astype(float),
           "in_largest": np.array([float(size[c] == largest and largest > 1) for c in lab])}
    dist = np.zeros(n)
    if X is not None:
        Xs = _rms_scale(X)
        for c in ids:
            m = lab == c
            if m.sum() > 1:
                dist[m] = np.sqrt(((Xs[m] - Xs[m].mean(axis=0)) ** 2).sum(axis=1))
    elif D is not None:
        D = np.asarray(D, float)
        for c in ids:
            idx = np.flatnonzero(lab == c)
            if len(idx) > 1:
                sub = D[np.ix_(idx, idx)]
                dist[idx] = sub.sum(axis=1) / (len(idx) - 1)
    per["centroid_dist"] = dist
    if strength is not None:
        per["strength"] = np.asarray(strength, float)
    win = {"n_clusters": len(ids) / n, "largest_share": largest / n, "noise_share": float(noise.mean()),
           "chaos": np.nan, "fragmentation": np.nan}
    if X is not None:
        sp = spread_decomposition(_rms_scale(X), lab, blocks={"all": list(range(np.asarray(X).shape[1]))})
        win["chaos"] = float(sp["within_sd"].mean())
        win["fragmentation"] = float(sp["between_share"]["all"])
    return per, win


# ------------------------------------------------------------------ prototype codebook (option 4)
class PrototypeCodebook:
    """K behaviour prototypes fitted once (k-means) on many standardised descriptors pooled over
    sessions, without labels (R7: fit it on the training groups of each fold). In each window every
    student gets a soft assignment to the fixed prototypes, so a prototype means the same thing in
    every window and session (R4)."""

    def __init__(self, k: int = 8, seed: int = 0, temperature: float | None = None):
        self.k, self.seed, self.temperature = k, seed, temperature
        self.centres: np.ndarray | None = None
        self.fill: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PrototypeCodebook":
        from sklearn.cluster import KMeans

        X = np.asarray(X, float)
        self.fill = np.where(np.isfinite(np.nanmedian(X, axis=0)), np.nanmedian(X, axis=0), 0.0)
        Xf = self._filled(X)
        km = KMeans(n_clusters=self.k, n_init=4, random_state=self.seed).fit(Xf)
        self.centres = km.cluster_centers_
        if self.temperature is None:          # the median squared distance to the nearest prototype
            self.temperature = float(np.median(self._sqdist(Xf).min(axis=1))) or 1.0
        return self

    def _filled(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float).copy()
        bad = ~np.isfinite(X)
        X[bad] = np.take(self.fill, np.nonzero(bad)[1])
        return X

    def _sqdist(self, Xf: np.ndarray) -> np.ndarray:
        return ((Xf[:, None, :] - self.centres[None, :, :]) ** 2).sum(axis=2)

    def assign(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(soft assignment (n, K), squared distance to the nearest prototype (n,))."""
        d2 = self._sqdist(self._filled(X))
        z = -(d2 - d2.min(axis=1, keepdims=True)) / self.temperature
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True), d2.min(axis=1)

    def window_features(self, X: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """per student: conformity (soft overlap with the mean assignment of the others),
        hard_share (share of the others in the student's nearest prototype), entropy (of the
        student's own assignment), nearest_dist; per window: entropy of the class's prototype
        distribution and the K shares (proto_<k>). NaN conformity for n < 3."""
        X = np.asarray(X, float)
        n = len(X)
        win = {"entropy": np.nan, **{f"proto_{j}": np.nan for j in range(self.k)}}
        if n == 0:
            return {k: np.zeros(0) for k in ("conformity", "hard_share", "entropy", "nearest_dist")}, win
        p, d2 = self.assign(X)
        hard = p.argmax(axis=1)
        ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)
        conf, hs = np.full(n, np.nan), np.full(n, np.nan)
        if n >= MIN_SYNC_N:
            total = p.sum(axis=0)
            for i in range(n):
                others = (total - p[i]) / (n - 1)
                conf[i] = float(p[i] @ others)
                hs[i] = float((np.delete(hard, i) == hard[i]).mean())
        dist = p.mean(axis=0)
        win["entropy"] = float(-(dist * np.log(np.clip(dist, 1e-12, None))).sum())
        win.update({f"proto_{j}": float(dist[j]) for j in range(self.k)})
        return {"conformity": conf, "hard_share": hs, "entropy": ent, "nearest_dist": np.sqrt(d2)}, win


# ------------------------------------------------------------------ screening helpers
def circular_shift(S: np.ndarray, rng: np.random.Generator, min_shift: int = 2) -> np.ndarray:
    """Each student's series rolled by its own random offset in [min_shift, T - min_shift]: keeps
    every student's own signal, breaks the timing between students (the synchrony surrogate)."""
    S = np.asarray(S, float)
    T = S.shape[1] if S.ndim == 2 else 0
    if T <= 2 * min_shift:
        return S.copy()
    return np.stack([np.roll(s, int(rng.integers(min_shift, T - min_shift + 1))) for s in S])


def noise_ratio(series: np.ndarray) -> float:
    """Within-window SD over the white-noise SD estimated from first differences
    (MAD(diff) / (0.6745 * sqrt 2)). About 1 for pure noise; larger when the series moves more
    slowly than frame-to-frame jitter. NaN with fewer than 4 finite frames."""
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    d = np.diff(x)
    noise = np.median(np.abs(d - np.median(d))) / (0.6745 * np.sqrt(2))
    sd = x.std()
    if noise <= 1e-12:
        return np.nan if sd <= 1e-12 else np.inf
    return float(sd / noise)


def lag1_autocorr(series: np.ndarray) -> float:
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    if len(x) < 4 or x.std() <= 1e-12:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):    # x[:-1] or x[1:] can be flat when x is not: NaN
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def laplacian_score(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Laplacian score per column (He, Cai and Niyogi, NIPS 2005): how well a feature preserves the
    local neighbourhood structure of the rows, without labels. Lower is better. X must be finite."""
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, float)
    n = len(X)
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    t = np.mean(dist[:, 1:] ** 2) or 1.0
    W = np.zeros((n, n))
    for i in range(n):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            w = np.exp(-d * d / t)
            W[i, j] = max(W[i, j], w)
            W[j, i] = W[i, j]
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    out = np.full(X.shape[1], np.nan)
    for r in range(X.shape[1]):
        f = X[:, r] - (X[:, r] @ deg) / deg.sum()
        den = f @ (deg * f)
        if den > 1e-12:
            out[r] = float(f @ L @ f / den)
    return out
