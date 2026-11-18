"""The feature vector as the per-labeler model receives it: layout, distributions, and how each column
relates to every labeler's ratings and to the model's predictions (experiment 08, §1).

  python -m scripts.analysis.feature_study [--config docs/experiments/06_full_pipeline_and_labels.json]
         [--model <config-hash prefix>] [--out outputs/experiments/08_pca]

Rows: one per (rating entry, linked track of the rated student), original recordings only, exactly
the rows train_models builds (label_rows). Columns: the 95 model inputs (27 trk.* per-student, 68
vec.* class context) and the 26 scene.* gate inputs.

Writes to --out:
  vector_stats.csv       per column: block, family, distribution (mean, SD, median, 5/95th
                         percentiles, share at the most common value, skew, NaN share), variance
                         shares between tracks and between sessions, and the most-correlated other
                         model input (redundancy)
  correlations.csv       Spearman rho of every model input with each labeler's ratings, all experts
                         pooled, the self-report, and the model's own-labeler prediction; raw and
                         within student (feature and rating both demeaned per student-session and
                         labeler, so only "when this student's feature moves, does the rating move" is left)
  feature_correlations.png  both as heatmaps
  deciles.png            mean rating per feature decile for the strongest within-student features
The model prediction comes from the run's final bundle (outputs/models/<hash>.joblib, trained on all
groups), so it is in-sample: it shows what the model responds to, not how well it generalises.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.run_pipeline import parse_sets
from scripts.train_models import label_rows, load_table
from src.core.config import load_config

EXPERTS = ["labeler_01", "labeler_02", "labeler_03", "labeler_04", "labeler_05"]
SELF = "self_labeling"


def family(col: str) -> str:
    name = col.split(".", 1)[1]
    for pre, fam in (("kg.", "motion (Kalman)"), ("rf.", "flow (residual)"), ("scb.", "behaviour (SCB)"),
                     ("occ_", "occupancy"), ("n_tracks", "count")):
        if name.startswith(pre):
            return fam
    return "scene"


def spearman_cols(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Spearman rho of every column of X with y (NaNs dropped pairwise)."""
    out = {}
    for c in X:
        m = np.isfinite(X[c].to_numpy(float)) & np.isfinite(y)
        if m.sum() < 30 or np.ptp(X[c].to_numpy(float)[m]) == 0 or np.ptp(y[m]) == 0:
            out[c] = np.nan
            continue
        a = pd.Series(X[c].to_numpy(float)[m]).rank().to_numpy()
        b = pd.Series(y[m]).rank().to_numpy()
        out[c] = float(np.corrcoef(a, b)[0, 1])
    return pd.Series(out)


def demean(values: pd.DataFrame | pd.Series, keys: pd.Series):
    return values - values.groupby(keys).transform("mean")


def variance_share(x: pd.Series, keys: pd.Series) -> float:
    """Share of x's variance between the groups given by keys (1 - within / total)."""
    ok = x.notna()
    x, keys = x[ok], keys[ok]
    total = x.var(ddof=0)
    if not total or not np.isfinite(total):
        return np.nan
    return float(1 - (x - x.groupby(keys).transform("mean")).var(ddof=0) / total)


def stats_table(rows: pd.DataFrame, model_cols: list[str], gate_cols: list[str]) -> pd.DataFrame:
    win = rows.drop_duplicates(["session", "t", "tid"])          # one row per (window, track)
    track = win["session"] + "#" + win["tid"].astype(str)
    ranks = win[model_cols].rank()
    corr = ranks.corr().abs().to_numpy().copy()
    np.fill_diagonal(corr, np.nan)
    out = []
    for c in model_cols + gate_cols:
        x = win[c].astype(float)
        mode_share = float(x.round(6).value_counts(normalize=True, dropna=True).iloc[0]) if x.notna().any() else np.nan
        rec = {"column": c, "block": c.split(".", 1)[0], "family": family(c),
               "mean": x.mean(), "sd": x.std(ddof=0), "median": x.median(), "p05": x.quantile(0.05),
               "p95": x.quantile(0.95), "share_at_mode": mode_share, "skew": x.skew(), "nan_share": x.isna().mean(),
               "between_track_share": variance_share(x, track), "between_session_share": variance_share(x, win["session"])}
        if c in model_cols:
            i = model_cols.index(c)
            j = int(np.nanargmax(corr[i])) if np.isfinite(corr[i]).any() else None
            rec["most_correlated"] = model_cols[j] if j is not None else ""
            rec["max_abs_rho"] = float(corr[i, j]) if j is not None else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def correlations(rows: pd.DataFrame, model_cols: list[str], pred: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    student = rows["group"] + "/" + rows["exp"].astype(str) + "/" + rows["sid"].astype(str) + "/" + rows["labeler"]
    targets = {lab: rows["labeler"] == lab for lab in EXPERTS}
    targets["all experts"] = rows["labeler"].isin(EXPERTS)
    targets["self-report"] = rows["labeler"] == SELF
    raw, within = {}, {}
    y = rows["rating"].astype(float)
    for name, m in targets.items():
        sub = rows[m]
        raw[name] = spearman_cols(sub[model_cols], y[m].to_numpy())
        within[name] = spearman_cols(demean(sub[model_cols], student[m]), demean(y[m], student[m]).to_numpy())
    m = targets["all experts"].to_numpy()
    p = pd.Series(pred, index=rows.index)
    raw["model (experts' rows)"] = spearman_cols(rows.loc[m, model_cols], p[m].to_numpy())
    within["model (experts' rows)"] = spearman_cols(demean(rows.loc[m, model_cols], student[m]),
                                                    demean(p[m], student[m]).to_numpy())
    return pd.DataFrame(raw), pd.DataFrame(within)


def heatmaps(raw: pd.DataFrame, within: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(raw)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.2 * n + 2.5), sharey=True)
    lim = max(0.1, float(np.nanmax(np.abs(np.r_[raw.to_numpy().ravel(), within.to_numpy().ravel()]))))
    for ax, (title, df) in zip(axes, (("raw", raw), ("within student (demeaned per student-session)", within))):
        im = ax.imshow(df.to_numpy(float), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(df.shape[1]), df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n), df.index, fontsize=6)
        ax.set_title(f"Spearman rho, {title}", fontsize=9)
        blocks = [i for i in range(1, n) if df.index[i].split(".")[0] != df.index[i - 1].split(".")[0]]
        for b in blocks:
            ax.axhline(b - 0.5, color="black", lw=1)
        ax.axvline(df.shape[1] - 1.5, color="black", lw=1)
    fig.colorbar(im, ax=axes, shrink=0.4, label="Spearman rho")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def deciles(rows: pd.DataFrame, cols: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, (len(cols) + 1) // 2, figsize=(4 * ((len(cols) + 1) // 2), 7), squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        q = pd.qcut(rows[c].rank(method="first"), 10, labels=False)
        for lab in EXPERTS + [SELF]:
            m = rows["labeler"] == lab
            curve = rows[m].groupby(q[m])["rating"].mean()
            ax.plot(curve.index + 1, curve.values, lw=2.2 if lab == SELF else 1, label=lab.replace("_labeling", ""),
                    color="black" if lab == SELF else None)
        ax.set_title(c, fontsize=8)
        ax.set_xlabel("feature decile")
        ax.set_ylabel("mean rating")
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def components(rows: pd.DataFrame, model_cols: list[str], n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Principal components of the standardised model inputs (label-free; fitted on the distinct
    window x track rows): explained variance, the four largest loadings, and each component's Spearman
    rho with the ratings (all experts and the self-report; raw and within student)."""
    win = rows.drop_duplicates(["session", "t", "tid"])
    X = win[model_cols].astype(float)
    med, mu = X.median(), None
    X = X.fillna(med)
    mu, sd = X.mean(), X.std(ddof=0).replace(0, 1)
    _, s, vt = np.linalg.svd(((X - mu) / sd).to_numpy(), full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    Z = ((rows[model_cols].astype(float).fillna(med) - mu) / sd).to_numpy() @ vt[:n].T
    pcs = pd.DataFrame(Z, columns=[f"PC{i + 1}" for i in range(n)], index=rows.index)
    student = rows["group"] + "/" + rows["exp"].astype(str) + "/" + rows["sid"].astype(str) + "/" + rows["labeler"]
    y = rows["rating"].astype(float)
    out = []
    for i in range(n):
        load = pd.Series(vt[i], index=model_cols)
        top = load.reindex(load.abs().sort_values(ascending=False).index)[:4]
        rec = {"component": f"PC{i + 1}", "variance_share": ev[i], "cumulative": ev[: i + 1].sum(),
               "top_loadings": ", ".join(f"{k} {v:+.2f}" for k, v in top.items())}
        for name, m in (("experts", rows["labeler"].isin(EXPERTS)), ("self", rows["labeler"] == SELF)):
            c = pcs.loc[m, [f"PC{i + 1}"]]
            rec[f"rho_{name}_raw"] = spearman_cols(c, y[m].to_numpy()).iloc[0]
            rec[f"rho_{name}_within"] = spearman_cols(demean(c, student[m]), demean(y[m], student[m]).to_numpy()).iloc[0]
        out.append(rec)
    loadings = pd.DataFrame(vt.T, index=model_cols, columns=[f"PC{i + 1}" for i in range(len(ev))])
    return pd.DataFrame(out), loadings, ev


CATEGORIES = [("student motion", "trk.kg."), ("student flow", "trk.rf."), ("student SCB", "trk.scb."),
              ("class motion", "vec.kg."), ("class flow", "vec.rf."), ("class SCB", "vec.scb."),
              ("class occupancy + count", ("vec.occ_", "vec.n_tracks"))]
CATEGORY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def pies(pcs: pd.DataFrame, loadings: pd.DataFrame, path: Path) -> None:
    """One pie per component: the share of the component (its squared loadings, which sum to 1)
    from each feature category; title with the variance share and the within-student rho."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = loadings.shape[1]
    ncol = 5
    fig, axes = plt.subplots((n + ncol - 1) // ncol, ncol, figsize=(3.2 * ncol, 3.4 * ((n + ncol - 1) // ncol) + 0.8))
    sq = loadings ** 2
    for ax, pc in zip(axes.ravel(), loadings.columns):
        shares = [sq.loc[[c for c in sq.index if c.startswith(pre)], pc].sum() for _, pre in CATEGORIES]
        wedges, _ = ax.pie(shares, colors=CATEGORY_COLORS, startangle=90, counterclock=False,
                           wedgeprops={"edgecolor": "white", "linewidth": 2})
        for w, sh in zip(wedges, shares):
            if sh >= 0.10:
                ang = np.deg2rad((w.theta1 + w.theta2) / 2)
                ax.text(0.68 * np.cos(ang), 0.68 * np.sin(ang), f"{sh:.0%}", ha="center", va="center",
                        fontsize=8, color="white" if CATEGORY_COLORS[wedges.index(w)] in ("#2a78d6", "#008300", "#6250d6") else "black")
        r = pcs.set_index("component").loc[pc]
        ax.set_title(f"{pc}: {r['variance_share']:.1%} of variance\nrho experts {r['rho_experts_within']:+.3f}, "
                     f"self {r['rho_self_within']:+.3f}", fontsize=8.5)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in CATEGORY_COLORS], [c for c, _ in CATEGORIES],
               loc="lower center", ncol=len(CATEGORIES), fontsize=9, frameon=False)
    fig.suptitle("What each principal component is made of: share of its squared loadings by feature category",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def dominant_category(loadings: pd.DataFrame) -> list[int]:
    """Index into CATEGORIES of the category holding the largest share of each component."""
    sq = loadings ** 2
    shares = np.array([[sq.loc[[c for c in sq.index if c.startswith(pre)], pc].sum() for _, pre in CATEGORIES]
                       for pc in loadings.columns])
    return shares.argmax(axis=1).tolist()


def scree(ev: np.ndarray, loadings: pd.DataFrame, path: Path) -> None:
    """Two panels on one component axis (no second y-scale): each component's share of the variance,
    coloured by its dominant feature category; and the cumulative share, with the trial's cuts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(ev)
    x = np.arange(1, n + 1)
    dom = dominant_category(loadings)
    cum = np.cumsum(ev)
    fig, (top, bot) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    top.bar(x, ev * 100, width=0.8, color=[CATEGORY_COLORS[d] for d in dom], edgecolor="white", linewidth=0.5)
    top.set_ylabel("share of variance (%)")
    top.set_title("Variance captured by each principal component (95 standardised model inputs), "
                  "coloured by the feature category that dominates it", fontsize=10)
    top.annotate("PC3: the student's own flow\n(tracks the experts most, rho +0.053)", xy=(3, ev[2] * 100),
                 xytext=(9, ev[0] * 100 * 0.75), fontsize=8.5, arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "#555"})
    bot.step(x, cum * 100, where="mid", color="#2a78d6", lw=2)
    cuts = {"5": 5, "10": 10, "20": 20, "90%": int(np.searchsorted(cum, 0.90) + 1), "95%": int(np.searchsorted(cum, 0.95) + 1)}
    for label, k in cuts.items():
        bot.plot([k], [cum[k - 1] * 100], "o", ms=8, color="#2a78d6", mec="white", mew=2)
        bot.annotate(f"{label}: {k} PCs, {cum[k - 1]:.0%}", (k, cum[k - 1] * 100), textcoords="offset points",
                     xytext=(6, -14), fontsize=8.5, color="#333")
    bot.set_ylabel("cumulative share (%)")
    bot.set_xlabel("principal component")
    bot.set_ylim(0, 102)
    for ax in (top, bot):
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    bot.set_xlim(0.3, n + 0.7)
    present = sorted(set(dom))
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[i]) for i in present],
               [CATEGORIES[i][0] for i in present], loc="lower center", ncol=len(present), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="docs/experiments/06_full_pipeline_and_labels.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--model", default="c5341cb6385e", help="config-hash prefix of a trained per-labeler bundle")
    ap.add_argument("--out", default="outputs/experiments/08_pca")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = label_rows(cfg, load_table(cfg))
    rows = rows[~rows["aug"]].reset_index(drop=True)
    model_cols = sorted(c for c in rows if c.startswith("trk.")) + sorted(c for c in rows if c.startswith("vec."))
    gate_cols = sorted(c for c in rows if c.startswith("scene."))

    found = sorted(Path("outputs/models").glob(f"{args.model}*.joblib"))
    if len(found) != 1:
        raise SystemExit(f"{args.model}: {len(found)} model bundles match in outputs/models")
    bundle = joblib.load(found[0])
    X = rows.reindex(columns=bundle["row_cols"]).to_numpy(float)
    G = bundle["gate"].predict_proba(rows.reindex(columns=bundle["gate_cols"]).to_numpy(float))
    preds = bundle["model"].predict(X, G)
    from src.labels.interp import to_rating
    pred = np.array([to_rating(preds[l][i]) if l in preds else np.nan for i, l in enumerate(rows["labeler"])])

    st = stats_table(rows, model_cols, gate_cols)
    st.to_csv(out / "vector_stats.csv", index=False)
    raw, within = correlations(rows, model_cols, pred)
    pd.concat({"raw": raw, "within_student": within}, axis=1).to_csv(out / "correlations.csv")
    heatmaps(raw, within, out / "feature_correlations.png")
    top = within["all experts"].abs().sort_values(ascending=False).index[:3].tolist()
    top += [c for c in within["self-report"].abs().sort_values(ascending=False).index if c not in top][:3]
    deciles(rows, top, out / "deciles.png")
    pcs, loadings, ev = components(rows, model_cols)
    pcs.to_csv(out / "components.csv", index=False)
    loadings.to_csv(out / "components_loadings.csv")
    pies(pcs, loadings.iloc[:, :20], out / "components_pies.png")
    scree(ev, loadings, out / "components_variance.png")
    print("\nprincipal components (variance, rho with experts / self, within student; top loadings):")
    for _, r in pcs.iterrows():
        print(f"  {r['component']:5s} {r['variance_share']:.3f} (cum {r['cumulative']:.2f})  "
              f"experts {r['rho_experts_within']:+.3f}  self {r['rho_self_within']:+.3f}  {r['top_loadings']}")

    print(f"rows {len(rows)} (originals), model inputs {len(model_cols)} "
          f"(trk {sum(c.startswith('trk.') for c in model_cols)}, vec {sum(c.startswith('vec.') for c in model_cols)}), "
          f"gate inputs {len(gate_cols)}; model {bundle['config_hash'][:12]}")
    for name, df in (("raw", raw), ("within student", within)):
        print(f"\nlargest |rho| {name}:")
        for col in ("all experts", "self-report", "model (experts' rows)"):
            s = df[col].dropna()
            top5 = s.reindex(s.abs().sort_values(ascending=False).index)[:5]
            print(f"  {col:22s} " + "  ".join(f"{k} {v:+.3f}" for k, v in top5.items()))
    print(f"\nwrote {out}/vector_stats.csv, correlations.csv, feature_correlations.png, deciles.png")


if __name__ == "__main__":
    main()
