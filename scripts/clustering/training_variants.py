"""Three ways of training that might give better 1-5 ratings (experiment 11 follow-up), each against a
reference that differs from it in one thing only, all with experiment 11's output (per-labeler cut
points) and scoring:

  python -m scripts.clustering.training_variants [--config docs/experiments/10_clustering.json]
         [--set k=v ...] [--boot 1000] [--out outputs/experiments/10_clustering_v2]

  ordinal    O1  per labeler, proportional-odds (cumulative-logit) model: one set of feature weights and
                 a threshold per boundary between ratings; its score is the expected rating
             O0  per labeler, ridge regression on the rating          (only the model type differs)
  consensus  C1  one model for all experts, ridge on the mean of every expert's rating in force at
                 the entry's time (the experts' consensus)
             C0  the same model on each entry's own rating             (only the target differs)
  posture    P1  per labeler, ridge on the base inputs + body posture (scripts/clustering/posture.py)
             P0  the same without posture                             (only the inputs differ)
                 Posture exists for the original recordings only, so P0 and P1 train on originals;
                 the others train on the augmented copies too, as every model so far.

Inputs: the configured base columns (experiment 10's `base` arm), median-imputed and standardised on
the training rows. Group split (held-out students). Every model's strength (ridge alpha, ordinal L2)
is chosen by an inner cross-validation grouped by class session, inside the training groups only,
on the mean Spearman within labelers. An entry's score is the mean over its rows (cameras); scores
become ratings by experiment 11's cut points (the held-out group's own scores at each labeler's class
shares in the other groups; scripts/analysis/imbalance_variants.cut). The experts' rating 1 is
merged into 2. Metrics: QWK, macro-F1, balanced accuracy, recall per class, MAE, Spearman of the score.

Fixed before the run: three primary comparisons on the experts' labels, QWK, O1 - O0, C1 - C0 and
P1 - P0, each at 98.33% (Bonferroni for three); intervals by bootstrap over class sessions, paired on
the same entries. The self-report is reported for O and P (the consensus is the experts'). For
context, experiment 10's saved `base` predictions (per-labeler logistic regression) on the same entries.
Written to <out>/training_variants/.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from scripts.analysis.imbalance_scoring import BOOT_KEYS, boot_classes, class_metrics, levels_of, to_levels
from scripts.analysis.imbalance_variants import cut
from scripts.clustering.common import SELF, base_columns, base_parser, folds, label_table, load
from scripts.clustering.within_session import Ridge

PRIMARY_ALPHA = 0.05 / 3
RIDGE_GRID = [1.0, 10.0, 100.0, 1e3, 1e4]
ORD_GRID = [1e-4, 1e-3, 1e-2, 1e-1]


# ------------------------------------------------------------------ the ordinal model
class Ordinal:
    """Proportional-odds model: P(y <= level_j) = sigmoid(theta_j - x.w), thresholds increasing, L2
    penalty lam on w (mean log-likelihood). Inputs median-imputed and standardised on the training
    rows. score() is the expected rating."""

    def __init__(self, lam: float = 1e-3):
        self.lam = lam

    def _prep(self, X):
        Z = np.where(np.isfinite(X), X, self.med)
        return (Z - self.mu) / self.sd

    @staticmethod
    def objective(p, Z, j, k, lam):
        """Mean negative log-likelihood + lam/2 |w|^2 and its gradient, for parameters p = (w, theta_1,
        log increments of the other thresholds); Z standardised inputs, j class indices 0..k-1."""
        n, D = Z.shape
        w, a, b = p[:D], p[D], p[D + 1:]
        th = np.concatenate([[a], a + np.cumsum(np.exp(b))])
        eta = Z @ w
        hi_ok, lo_ok = j < k - 1, j > 0
        Fh = np.ones(n)
        Fl = np.zeros(n)
        Fh[hi_ok] = expit(th[j[hi_ok]] - eta[hi_ok])
        Fl[lo_ok] = expit(th[j[lo_ok] - 1] - eta[lo_ok])
        pr = np.maximum(Fh - Fl, 1e-12)
        fh, fl = np.where(hi_ok, Fh * (1 - Fh), 0.0), np.where(lo_ok, Fl * (1 - Fl), 0.0)
        loss = -np.mean(np.log(pr)) + 0.5 * lam * w @ w
        gw = -(Z.T @ ((-fh + fl) / pr)) / n + lam * w
        gth = np.zeros(k - 1)
        np.add.at(gth, j[hi_ok], -(fh / pr)[hi_ok] / n)
        np.add.at(gth, j[lo_ok] - 1, (fl / pr)[lo_ok] / n)
        gb = np.array([gth[m:].sum() * np.exp(b[m - 1]) for m in range(1, k - 1)])
        return loss, np.concatenate([gw, [gth.sum()], gb])

    def fit(self, X, y):
        self.med = np.nanmedian(X, 0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Z0 = np.where(np.isfinite(X), X, self.med)
        self.mu, self.sd = Z0.mean(0), Z0.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        Z = (Z0 - self.mu) / self.sd
        self.levels = np.unique(y)
        k = len(self.levels)
        j = np.searchsorted(self.levels, y)
        D = Z.shape[1]
        if k < 2:
            self.w, self.theta = np.zeros(D), np.zeros(0)
            return self
        cum = np.clip(np.cumsum(np.bincount(j, minlength=k))[:-1] / len(j), 1e-3, 1 - 1e-3)
        th0 = np.log(cum / (1 - cum))
        p0 = np.concatenate([np.zeros(D), [th0[0]], np.log(np.maximum(np.diff(th0), 1e-3))])
        r = minimize(self.objective, p0, args=(Z, j, k, self.lam), jac=True, method="L-BFGS-B",
                     options={"maxiter": 500})
        self.w = r.x[:D]
        self.theta = np.concatenate([[r.x[D]], r.x[D] + np.cumsum(np.exp(r.x[D + 1:]))])
        return self

    def proba(self, X):
        eta = self._prep(X) @ self.w
        if len(self.theta) == 0:
            return np.ones((len(eta), 1))
        F = expit(self.theta[None, :] - eta[:, None])
        F = np.column_stack([np.zeros(len(eta)), F, np.ones(len(eta))])
        return np.diff(F, axis=1)

    def predict(self, X):
        return self.proba(X) @ self.levels


# ------------------------------------------------------------------ helpers
def rank_corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(pd.Series(x[ok]).rank(), pd.Series(y[ok]).rank())[0, 1])


def entry_scores(r: pd.DataFrame, mask: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """One row per entry of r[mask]: the mean score of its rows."""
    d = r.loc[mask, ["entry_id", "rating", "labeler", "group", "cls"]].assign(pred=pred)
    return d.groupby("entry_id", as_index=False).agg(pred=("pred", "mean"), rating=("rating", "first"),
                                                     labeler=("labeler", "first"), grp=("group", "first"),
                                                     cls=("cls", "first"))


def within_labeler_spearman(e: pd.DataFrame) -> float:
    v = [rank_corr(g["pred"], g["rating"]) for _, g in e.groupby("labeler") if len(g) >= 20]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def make(kind: str, strength: float):
    return Ordinal(strength) if kind == "ordinal" else Ridge(strength)


def fit_predict(r, X, y, train, test, kind, per_labeler, grid, n_inner: int = 4, rng=None):
    """Fit (per labeler or pooled) with the strength chosen by grouped inner CV inside `train`;
    returns the test rows' scores and the chosen strengths."""
    lab, cls, aug = r["labeler"].to_numpy(), r["cls"].to_numpy(), r["aug"].to_numpy(bool)
    units = np.unique(lab[train]) if per_labeler else [None]
    pred = np.full(len(r), np.nan)
    chosen = {}
    tc = np.unique(cls[train])
    parts = np.array_split(np.random.default_rng(0).permutation(tc), min(n_inner, len(tc)))
    for u in units:
        um = (lab == u) if u is not None else np.ones(len(r), bool)
        tr = train & um & np.isfinite(y)
        if tr.sum() < 30:
            continue
        scores = {}
        for s in grid:
            vals = []
            for part in parts:
                vc = np.isin(cls, part)
                fm, vm = tr & ~vc, tr & vc & ~aug
                if fm.sum() < 30 or vm.sum() < 20 or len(np.unique(y[fm])) < 2:
                    continue
                p = make(kind, s).fit(X[fm], y[fm]).predict(X[vm])
                vals.append(within_labeler_spearman(entry_scores(r, vm, p)))
            scores[s] = float(np.nanmean(vals)) if vals and np.isfinite(vals).any() else np.nan
        best = max((s for s in grid if np.isfinite(scores[s])), key=lambda s: scores[s], default=grid[len(grid) // 2])
        te = test & um
        if te.any():
            pred[te] = make(kind, best).fit(X[tr], y[tr]).predict(X[te])
        chosen[str(u)] = {"strength": best, "inner": scores}
    return pred, chosen


def consensus_from_tracks(ent: pd.DataFrame, tracks: dict) -> dict:
    """{entry_id: the mean of every expert's rating in force at the entry's time on the same student}
    for the experts' entries (ent: entry_id, sess, labeler, t; tracks: (sess, labeler) -> track)."""
    by_sess = {}
    for (sess, lab), tr in tracks.items():
        if lab != SELF:
            by_sess.setdefault(sess, []).append(tr)
    val = {}
    for eid, sess, lab, t in zip(ent["entry_id"], ent["sess"], ent["labeler"], ent["t"].astype(float)):
        if lab == SELF:
            continue
        v = [x for x in (tr.at(t) for tr in by_sess.get(sess, [])) if np.isfinite(x)]
        if v:
            val[eid] = float(np.mean(v))
    return val


def consensus_target(cfg: dict, r: pd.DataFrame) -> np.ndarray:
    """Per row: the experts' consensus (consensus_from_tracks; rating tracks as the pipeline reads
    them); NaN for self-report rows or without any expert."""
    from scripts.clustering.feature_similarity import load_tracks

    ent = r[~r["aug"].astype(bool)].drop_duplicates("entry_id")
    return r["entry_id"].map(consensus_from_tracks(ent, load_tracks(cfg, ent))).to_numpy(float)


def evaluate(e_by_arm: dict, track: str, W: np.ndarray, pairs: list, alpha_of) -> dict:
    """Class metrics of every arm (per-labeler cut points) on the entries every arm scored, and the
    paired differences of `pairs`."""
    lv = levels_of(track)
    common = set.intersection(*(set(e["entry_id"]) for e in e_by_arm.values()))
    out, boots, y = {"arms": {}, "differences": {}}, {}, None
    for arm, e in e_by_arm.items():
        e = e[e["entry_id"].isin(common)].sort_values("entry_id").reset_index(drop=True)
        y = to_levels(e["rating"].to_numpy(float), lv)
        c = cut(e, lv, "own")
        m = class_metrics(y, c, lv)
        m["spearman"] = rank_corr(e["pred"], e["rating"])
        m["spearman_within_labeler"] = within_labeler_spearman(e)
        m["entries"] = int(len(e))
        m["per_labeler_qwk"] = {l: class_metrics(y[(e["labeler"] == l).to_numpy()], c[(e["labeler"] == l).to_numpy()], lv)["qwk"]
                                for l in e["labeler"].unique() if (e["labeler"] == l).sum() >= 100}
        out["arms"][arm] = m
        if len(W):
            boots[arm] = boot_classes(y, c, e["cls"].to_numpy(), lv, W)
    for a, b in pairs:
        if a in boots and b in boots:
            al = alpha_of(a, b)
            out["differences"][f"{a} - {b}"] = {
                k: {"diff": float(np.nanmean(boots[a][k] - boots[b][k])),
                    "lo": float(np.nanpercentile(boots[a][k] - boots[b][k], 100 * al / 2)),
                    "hi": float(np.nanpercentile(boots[a][k] - boots[b][k], 100 * (1 - al / 2))),
                    "level": 1 - al} for k in BOOT_KEYS}
    return out


def main() -> None:
    import warnings

    from scripts.clustering.posture import cache_dir, window_features
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    ap = base_parser(__doc__)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--posture-settings", default=None,
                    help="settings.json of the posture cache to use (default: the one posture.py writes by default)")
    args = ap.parse_args()
    cfg, out = load(args)
    rng = np.random.default_rng(args.seed)

    rows = label_table(cfg)
    r = rows.reset_index(drop=True).copy()
    parts = r["session"].str.split("/")
    r["cls"] = parts.str[0] + "/" + parts.str[1]
    r["sess"] = r["cls"] + "/" + r["sid"].astype(str)
    cols = base_columns(cfg, r)
    X = r[cols].to_numpy(float)
    r["consensus"] = consensus_target(cfg, r)
    # posture: the default posture.py settings unless a settings file is given
    pset = (json.loads(open(args.posture_settings, encoding="utf-8").read()) if args.posture_settings else
            {"weights": cfg["detector"]["weights"], "every": 1, "imgsz": 1280, "conf": 0.1, "kp_conf": 0.5,
             "min_iou": 0.3, "max_frames": 0})
    pset.pop("device", None)
    pdir = cache_dir(cfg, pset)
    po = window_features(pdir, r[["stem", "tid", "t"]], float(cfg["window"]["length_s"]))
    po_cols = [c for c in po.columns if po[c].notna().any()]
    orig = ~r["aug"].astype(bool).to_numpy()
    print(f"{len(r)} rows ({orig.sum()} original), {len(cols)} base columns; consensus for "
          f"{np.isfinite(r['consensus']).mean():.1%} of rows; posture cache {pdir} "
          f"({'found' if po_cols else 'MISSING'}: {len(po_cols)} columns, defined for {po.notna().any(axis=1)[orig].mean():.1%} "
          f"of original rows)", flush=True)
    if not po_cols:
        print("note: no posture cache, so P0/P1 are skipped; run `python -m scripts.clustering.posture` first", flush=True)
    XP = np.column_stack([X, po[po_cols].to_numpy(float)]) if po_cols else None
    rating = r["rating"].to_numpy(float)
    res = {"settings": {k: v for k, v in vars(args).items() if k != "set"}, "posture_cache": str(pdir),
           "posture_columns": po_cols, "tracks": {}}
    saved = None
    sp = out / "ablation" / "ablation_predictions.csv.gz"
    if sp.exists():
        saved = pd.read_csv(sp)
        saved = saved[(saved["split"] == "group") & (saved["arm"] == "base")]
    for track in ("experts", "self"):
        in_track = (r["labeler"] != SELF).to_numpy() if track == "experts" else (r["labeler"] == SELF).to_numpy()
        lv = levels_of(track)
        y_lv = to_levels(rating, lv)
        arms = {"O0": ("ridge", True, RIDGE_GRID, X, y_lv, False), "O1": ("ordinal", True, ORD_GRID, X, y_lv, False)}
        if track == "experts":
            arms.update({"C0": ("ridge", False, RIDGE_GRID, X, rating, False),
                         "C1": ("ridge", False, RIDGE_GRID, X, r["consensus"].to_numpy(float), False)})
        if XP is not None:
            arms.update({"P0": ("ridge", True, RIDGE_GRID, X, y_lv, True), "P1": ("ridge", True, RIDGE_GRID, XP, y_lv, True)})
        preds = {a: [] for a in arms}
        chosen = {a: {} for a in arms}
        for name, tr, te in folds(r, "group"):
            for arm, (kind, per_lab, grid, XX, y, originals_only) in arms.items():
                trm = tr & in_track & (orig if originals_only else True)
                tem = te & in_track
                p, ch = fit_predict(r, XX, y, trm, tem, kind, per_lab, grid, rng=rng)
                ok = tem & np.isfinite(p)
                preds[arm].append(entry_scores(r, ok, p[ok]))
                chosen[arm][name] = ch
            print(f"  {track} fold {name}: done ({', '.join(arms)})", flush=True)
        e_by_arm = {a: pd.concat(v, ignore_index=True) for a, v in preds.items()}
        common = set.intersection(*(set(e["entry_id"]) for e in e_by_arm.values()))
        if len(common) < 20:
            print(f"\n=== {track}: only {len(common)} entries scored by every arm (too few training rows per "
                  f"model); skipped", flush=True)
            res["tracks"][track] = {"skipped": f"{len(common)} entries scored by every arm"}
            continue
        if saved is not None:
            s = saved[saved["track"] == track].groupby("entry_id", as_index=False)["pred"].mean()
            ref = e_by_arm["O0"].drop(columns="pred").merge(s, on="entry_id", how="inner")
            if len(ref) >= 0.9 * len(e_by_arm["O0"]):
                e_by_arm["base (exp. 10)"] = ref
        n_cls = r.loc[in_track & orig, "cls"].nunique()
        W = rng.multinomial(n_cls, np.full(n_cls, 1 / n_cls), size=args.boot) if args.boot and n_cls >= 3 else np.zeros((0, n_cls))
        pairs = [("O1", "O0"), ("C1", "C0"), ("P1", "P0"), ("O0", "base (exp. 10)")]
        ev = evaluate(e_by_arm, track, W, pairs, lambda a, b: PRIMARY_ALPHA if b in ("O0", "C0", "P0") else 0.05)
        ev["strengths"] = chosen
        res["tracks"][track] = ev
        print(f"\n=== {track}: {next(iter(ev['arms'].values()))['entries']} entries scored by every arm")
        for arm, m in ev["arms"].items():
            print(f"  {arm:16s} QWK {m['qwk']:+.3f}  macro-F1 {m['macro_f1']:.3f}  balanced acc {m['balanced_acc']:.3f}  "
                  f"Spearman {m['spearman']:+.3f} (within labeler {m['spearman_within_labeler']:+.3f})  MAE {m['mae']:.3f}")
        for name, d in ev["differences"].items():
            q = d["qwk"]
            print(f"  {name:20s} QWK {q['diff']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}] ({q['level']:.2%})  "
                  f"macro-F1 {d['macro_f1']['diff']:+.3f}  balanced acc {d['balanced_acc']['diff']:+.3f}")
        for arm, ch in chosen.items():
            edge = [f for f, units in ch.items() for u, v in units.items()
                    if v["strength"] in ((RIDGE_GRID if arm != "O1" else ORD_GRID)[0], (RIDGE_GRID if arm != "O1" else ORD_GRID)[-1])]
            if edge:
                print(f"  note: {arm} strength at the edge of its grid in {len(edge)} fits")
    d = out / "training_variants"
    d.mkdir(parents=True, exist_ok=True)
    (d / "training_variants.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {d / 'training_variants.json'}")


if __name__ == "__main__":
    main()
