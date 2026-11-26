"""The clustering experiment's arms, fixed in advance, on the same folds and models: task 10 of guide
15 §15.9 (the test of §15.6).

  python -m scripts.clustering.ablation [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--candidates <dir>] [--arms ...] [--primary sync+agglomerative] [--primary-self normalised]
         [--prototypes 8]
         [--proto-set subset] [--context classified] [--boot 2000] [--split both] [--workers 1]
         [--out outputs/experiments/10_clustering]

Folds run in parallel with --workers (2-4 on 16 GB: each worker holds the label rows) and are
resumable: each finished fold (every arm's metrics and predictions) is saved under
<out>/partial/ablation/<settings hash>/, so an interrupted run only redoes the missing folds.

Arms (each = the configured inputs, as train_models gives them, plus the columns named). Every arm is
defined before the run; nothing is chosen on test results:
  base                 experiment 09's model: no class-relative features (the baseline; called `null` in
                       the first full run, renamed: pandas reads the word "null" in a CSV as missing)
  normalised           base's inputs, each trk.* column standardised per student over their own
                       session (with an SD floor; common.TrackStandardiser)
  sync                 + option 0, per student: synchrony, lag, degree with the class, deviation
  agglomerative        + option 1, per student: group share, alone, in the largest group, distance
  hdbscan              + option 2, per student: the same, and membership strength
  prototype            + option 4, per student: conformity to a codebook fitted per fold (no labels)
  sync+agglomerative   the combinations, fixed in advance
  sync+hdbscan
  sync+prototype
  normalised+sync      normalised inputs + per-student synchrony
  normalised+sync+agglomerative   normalised inputs + per-student synchrony and clustering: does the
                       class-relative information add anything once the seat is removed?
  window               + every window-level feature (cluster.*: number of groups, fragmentation,
                       class synchrony, prototype distribution...), centred per recording, so they
                       carry only the moment-to-moment variation inside a recording and cannot stand
                       in for the scene or activity
  window_raw           (not in the default list) the same, not centred: for comparison with `window`
The per-student arms never include window-level features: whether the class's structure helps is
the `window` arm's question, separate from whether a student's place in it helps.

Splits (--split; both by default):
  group     leave one group out: the test students are never seen. PRIMARY for any claim about
            the features. --context sets the gate (classified by default, as train_models reports)
  activity  train on 4 of the 6 activities, test on the other 2, over all 15 pairs. The same
            students are on both sides, so it measures generalisation to new activities only. Oracle
            context (a gate cannot name an unseen activity; with the shared expert it does not
            change the rating). Needs mixture.expert_scope = shared

Protocol: augmented copies train and are never scored; the configured per-labeler model; each entry
scored once with rating_metrics (fold means in `ablation`), next to constant 3 and the training mean.

The test: paired differences with confidence intervals. Every held-out entry's prediction is kept
for every arm (in the activity split an entry is tested in 5 folds; its predictions are averaged).
For an arm against a reference (base, sync), the difference in Spearman (pooled over entries, on
ranks) and in MAE is bootstrapped over recordings (--boot resamples; each recording's entries stay
together, so dependence within a recording is respected): the 95% interval of the difference.
  Two tests are PRIMARY, both fixed before any result was seen:
    1 (experts)      --primary (sync+agglomerative) against base, group split, experts' labels,
                     Spearman: do the class-relative features help on the experts' ratings?
    2 (self-report)  --primary-self (normalised) against base, group split, self-report, Spearman:
                     does removing between-student (seat) differences recover the student's own
                     report? (The first full run found this in all 18 folds; this test confirms it.)
    With two primary tests, each uses a Bonferroni-adjusted 98.75% interval (two-sided alpha 0.025
    each), so the pair together keeps a 5% chance of any false claim. "Supported" if that interval
    lies above 0.
  everything else is EXPLORATORY (95% intervals, not corrected): report it, but do not claim it.

Writes to --out/ablation/ (tables as .csv and .json):
  <config hash>-<settings hash>.json   every fold's metrics per split, the settings, the intervals
  ablation             mean and SD over folds per split, arm, track, eval (heldout or train)
  ablation_intervals   per split, track, arm and reference: the differences, their 95% intervals,
                       and the Bonferroni-adjusted 98.75% intervals used for the primary tests
  ablation_predictions.csv.gz   every held-out entry's prediction per arm (what the intervals use)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.clustering.common import (PER_STUDENT, PER_WINDOW, SELF, base_columns, base_parser, candidate_columns,
                                       centre_window_features, check_split, fit_model, fold_context, fold_jobs,
                                       fold_of, label_table, latest_candidates, levels_of, load, load_candidates,
                                       own_ratings, params_hash, per_track_normalised, rows_key, score, shared,
                                       split_parser_arg, splits, subset_text, train_mask, with_fold_prototypes,
                                       write_table)
from src.core.config import config_hash
from src.labels.interp import to_rating

FAMILIES = {"sync": ".sync.", "agglomerative": ".agg.", "hdbscan": ".hdb.", "prototype": ".proto."}
DEFAULT_ARMS = ["base", "normalised", "sync", "agglomerative", "hdbscan", "prototype", "sync+agglomerative",
                "sync+hdbscan", "sync+prototype", "normalised+sync", "normalised+sync+agglomerative", "window"]
PRIMARY_ALPHA = 0.05 / 2          # two primary tests, Bonferroni: 98.75% intervals
ALL_ARMS = DEFAULT_ARMS + ["window_raw"]
METRICS = ["mae", "rmse", "spearman", "auc", "c_index"]


def arm_spec(arm: str, cands: list[str]) -> tuple[str, list[str]]:
    """(which inputs: "rows", "normed", "centred", "raw_window"; the candidate columns added)."""
    if arm == "base":
        return "rows", []
    if arm == "normalised":
        return "normed", []
    if arm in ("window", "window_raw"):
        return ("centred" if arm == "window" else "rows"), [c for c in cands if c.startswith(PER_WINDOW)]
    parts = arm.split("+")
    data = "normed" if parts[0] == "normalised" else "rows"
    fams = [FAMILIES[p] for p in parts if p in FAMILIES]
    if len(fams) != len([p for p in parts if p != "normalised"]):
        raise SystemExit(f"unknown arm {arm!r}; known: {ALL_ARMS}")
    return data, [c for c in cands if c.startswith(PER_STUDENT) and any(f in c for f in fams)]


def run_arm(cfg, r, cols, tr, te, G, tau) -> tuple[dict, pd.DataFrame]:
    """Fold metrics (held-out and on the training entries) and the held-out per-row predictions."""
    model = fit_model(cfg, r, cols, tr)
    X = r.reindex(columns=cols).to_numpy(float)
    lab = r["labeler"].to_numpy()
    lv = levels_of(cfg)
    fit_rows = train_mask(cfg, r, tr) & ~r["aug"].to_numpy(bool)       # what it trained on, originals only
    pred = own_ratings(model, X[te], G[te], lab[te])
    out = {f"{t}/heldout": m for t, m in score(r[te], pred, tau, lv).items()}
    out |= {f"{t}/train": m for t, m in
            score(r[fit_rows], own_ratings(model, X[fit_rows], G[fit_rows], lab[fit_rows]), tau, lv).items()}
    held = pd.DataFrame({"entry_id": r.loc[te, "entry_id"].to_numpy(), "recording": r.loc[te, "session"].to_numpy(),
                         "track": np.where(lab[te] == SELF, "self", "experts"),
                         "rating": r.loc[te, "rating"].to_numpy(float), "pred": pred})
    return out, held


def baselines(cfg, r, tr, te, tau) -> dict:
    out = {}
    lv = levels_of(cfg)
    mid = float(np.median(lv))                             # 3 on 1-5 and on LOW / MID / HIGH (1, 3, 5)
    is_self = r["labeler"].to_numpy() == SELF
    for track, in_track in (("experts", ~is_self), ("self", is_self)):
        sel = te & in_track
        if not sel.any():
            continue
        mean = float(to_rating(r["target"].to_numpy(float)[tr & in_track].mean()))
        for name, v in (("constant_3", mid), ("train_mean", mean)):
            out[f"{track}/heldout/{name}"] = score(r[sel], np.full(int(sel.sum()), v), tau, lv)[track]
    return out


def summarise(results: dict) -> pd.DataFrame:
    recs = []
    for arm, per_fold in results.items():
        keys = sorted({k for f in per_fold.values() for k in f})
        for k in keys:
            vals = [f[k] for f in per_fold.values() if k in f]
            rec = {"arm": arm, "track": k.split("/")[0], "eval": "/".join(k.split("/")[1:]), "folds": len(vals),
                   "n_entries": int(np.sum([v.get("n", 0) for v in vals]))}
            for m in METRICS:
                x = np.array([v.get(m, np.nan) for v in vals], float)
                rec[m] = float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
                rec[f"{m}_sd"] = float(np.nanstd(x, ddof=1)) if np.isfinite(x).sum() > 1 else np.nan
            recs.append(rec)
    return pd.DataFrame(recs)


def entry_table(held: pd.DataFrame) -> pd.DataFrame:
    """One row per (track, entry): the entry's rating, recording and mean prediction per arm (an entry
    seen by several tracks or tested in several folds counts once)."""
    t = held.groupby(["track", "entry_id", "arm"]).agg(rating=("rating", "first"), recording=("recording", "first"),
                                                      pred=("pred", "mean")).reset_index()
    wide = t.pivot_table(index=["track", "entry_id"], columns="arm", values="pred").reset_index()
    meta = t.drop_duplicates(["track", "entry_id"])[["track", "entry_id", "rating", "recording"]]
    return meta.merge(wide, on=["track", "entry_id"])


def boot_weights(recordings: np.ndarray, n_boot: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """(recording index per entry, (n_boot, R) counts of each recording in each cluster resample)."""
    rec_ids, inv = np.unique(recordings, return_inverse=True)
    W = rng.multinomial(len(rec_ids), np.full(len(rec_ids), 1 / len(rec_ids)), size=n_boot).astype(float)
    return inv, W


def pooled_stats(x: np.ndarray, y: np.ndarray, inv: np.ndarray, W: np.ndarray, R: int) -> tuple[np.ndarray, np.ndarray]:
    """Spearman (Pearson on the ranks of the full sample) and MAE per bootstrap resample, from
    per-recording sums: each resample is a weighted sum over recordings, so no data are copied."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    sums = np.zeros((R, 7))
    for j, v in enumerate((np.ones_like(rx), rx, ry, rx * rx, ry * ry, rx * ry, np.abs(x - y))):
        sums[:, j] = np.bincount(inv, weights=v, minlength=R)
    T = W @ sums
    n, sx, sy, sxx, syy, sxy, ae = T.T
    cov = sxy / n - (sx / n) * (sy / n)
    vx = np.maximum(sxx / n - (sx / n) ** 2, 1e-12)
    vy = np.maximum(syy / n - (sy / n) ** 2, 1e-12)
    return cov / np.sqrt(vx * vy), ae / n


def intervals(entries: pd.DataFrame, arms: list[str], refs: list[str], n_boot: int, rng, split: str) -> list[dict]:
    """Paired arm-minus-reference differences (Spearman up = better, MAE down = better) per track,
    on the entries both predict, with 95% intervals from a cluster bootstrap over recordings (the
    same resamples for the arm and the reference, so the difference is paired)."""
    recs = []
    for track, e in entries.groupby("track"):
        y_all = e["rating"].to_numpy(float)
        rec_all = e["recording"].to_numpy()
        for ref in refs:
            if ref not in e:
                continue
            for a in arms:
                if a == ref or a not in e:
                    continue
                pa, pr = e[a].to_numpy(float), e[ref].to_numpy(float)
                ok = np.isfinite(pa) & np.isfinite(pr) & np.isfinite(y_all)
                if ok.sum() < 20:                  # too few entries for a meaningful interval
                    continue
                inv, W = boot_weights(rec_all[ok], n_boot, rng)
                R = W.shape[1]
                one = np.ones((1, R))
                sp_a, mae_a = pooled_stats(pa[ok], y_all[ok], inv, W, R)
                sp_r, mae_r = pooled_stats(pr[ok], y_all[ok], inv, W, R)
                (sa,), (ma,) = pooled_stats(pa[ok], y_all[ok], inv, one, R)
                (sr,), (mr,) = pooled_stats(pr[ok], y_all[ok], inv, one, R)
                d_sp, d_mae = sp_a - sp_r, mae_a - mae_r
                recs.append({"split": split, "track": track, "arm": a, "against": ref,
                             "entries": int(ok.sum()), "recordings": int(R),
                             "spearman_arm": float(sa), "spearman_ref": float(sr), "spearman_diff": float(sa - sr),
                             "spearman_diff_lo": float(np.quantile(d_sp, 0.025)),
                             "spearman_diff_hi": float(np.quantile(d_sp, 0.975)),
                             "spearman_diff_lo_adj": float(np.quantile(d_sp, PRIMARY_ALPHA / 2)),
                             "spearman_diff_hi_adj": float(np.quantile(d_sp, 1 - PRIMARY_ALPHA / 2)),
                             "mae_diff": float(ma - mr),
                             "mae_diff_lo": float(np.quantile(d_mae, 0.025)),
                             "mae_diff_hi": float(np.quantile(d_mae, 0.975))})
    return recs


def ablation_fold(split: str, i: int, seed) -> dict:
    """One fold (worker process): every arm's fold metrics and held-out predictions, and the baselines."""
    cfg, a, out, proto_set = shared("cfg"), shared("args"), shared("out"), shared("proto_set")
    rows, windows, tau = shared("rows"), shared("windows"), shared("tau")
    context = "oracle" if split == "activity" else a["context"]
    g, tr, te = fold_of(split, i)
    r = with_fold_prototypes(cfg, rows, tr, proto_set, a["prototypes"], out) if a["need_proto"] else rows
    base, cands = base_columns(cfg, r), candidate_columns(r)
    data = {"rows": r}
    if any(arm_spec(arm, cands)[0] == "normed" for arm in a["arms"]):     # label-free, per session
        data["normed"] = per_track_normalised(cfg, r, base)
    if any(arm_spec(arm, cands)[0] == "centred" for arm in a["arms"]):    # label-free, per recording
        data["centred"] = centre_window_features(r)
    G = fold_context(cfg, r, windows, tr, context, split)
    metrics, columns, held = {}, {}, []
    for arm in a["arms"]:
        which, extra = arm_spec(arm, cands)
        cols = base + extra
        columns[arm] = len(cols)
        metrics[arm], h = run_arm(cfg, data[which], cols, tr, te, G, tau)
        held.append(h.assign(arm=arm, fold=g))
    return {"fold": g, "context": context, "metrics": metrics, "columns": columns,
            "baselines": baselines(cfg, r, tr, te, tau), "held": pd.concat(held, ignore_index=True)}


def main() -> None:
    ap = base_parser(__doc__, workers=True)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS, choices=ALL_ARMS)
    ap.add_argument("--primary", default="sync+agglomerative", choices=ALL_ARMS,
                    help="primary test 1: this arm against base (group split, experts' labels, Spearman)")
    ap.add_argument("--primary-self", default="normalised", choices=ALL_ARMS,
                    help="primary test 2: this arm against base (group split, self-report, Spearman)")
    ap.add_argument("--prototypes", type=int, default=8)
    ap.add_argument("--proto-set", default="subset")
    ap.add_argument("--context", choices=["classified", "oracle"], default="classified",
                    help="gate for the group split (the activity split always uses oracle)")
    ap.add_argument("--boot", type=int, default=2000, help="cluster-bootstrap resamples over recordings")
    split_parser_arg(ap)
    args = ap.parse_args()
    for must in ("base", "sync", args.primary, args.primary_self):
        if must not in args.arms:
            args.arms = [must] + args.arms
    cfg, out = load(args)
    for split in splits(args.split):
        check_split(cfg, split)
    rng = np.random.default_rng(cfg["seed"])
    tau = float(cfg["labels"]["tau_att"])
    cdir = latest_candidates(out, args.candidates)
    rows, windows = label_table(cfg, with_windows=True)
    rows = load_candidates(rows, cdir)
    proto_set = args.proto_set if (out / "descriptor_subset.json").exists() or args.proto_set != "subset" else "all"

    need_proto = any("prototype" in arm or arm.startswith("window") for arm in args.arms) and args.prototypes > 0
    a = {"arms": args.arms, "prototypes": args.prototypes, "context": args.context, "need_proto": need_proto}
    settings_folds = {"config": config_hash(cfg), "candidates": cdir.name, "args": a, "proto_set": proto_set,
                      "subset": subset_text(out), "rows": rows_key(cfg), "version": 2}
    by_split: dict[str, list] = {}
    for split, i, res in fold_jobs(ablation_fold, rows, splits(args.split), args.workers, out, "ablation",
                                   settings_folds, {"cfg": cfg, "args": a, "out": out, "proto_set": proto_set,
                                                    "windows": windows, "tau": tau}, cfg["seed"]):
        by_split.setdefault(split, []).append((i, res))
        print(f"  {split} fold {res['fold']}: " + ", ".join(f"{k} {v}" for k, v in res["columns"].items())
              + " columns", flush=True)
    per_split, tables, ints, preds = {}, [], [], []
    for split in splits(args.split):
        done = [res for _, res in sorted(by_split.get(split, []), key=lambda x: x[0])]
        results = {arm: {res["fold"]: res["metrics"][arm] for res in done} for arm in args.arms}
        base_res = {res["fold"]: res["baselines"] for res in done}
        columns = done[0]["columns"] if done else {}
        context = done[0]["context"] if done else None
        held = pd.concat([res["held"] for res in done], ignore_index=True)
        per_split[split] = {"context": context, "columns": columns, "arms": results, "baselines": base_res}
        table = summarise(results)
        base_tab = summarise({"baseline": base_res})
        base_tab["arm"] = base_tab["eval"].str.split("/").str[1]
        base_tab["eval"] = "heldout"
        tables.append(pd.concat([table, base_tab], ignore_index=True).assign(split=split, context=context))
        entries = entry_table(held)
        ints += intervals(entries, args.arms, ["base", "sync"], args.boot, rng, split)
        preds.append(held.assign(split=split))
    table = pd.concat(tables, ignore_index=True)
    table = table[["split", "context", "arm", "track", "eval"] + [c for c in table if c not in
                                                                  ("split", "context", "arm", "track", "eval")]]
    iv = pd.DataFrame(ints)
    group_vs_base = (iv["against"] == "base") & (iv["split"] == "group")
    p1 = group_vs_base & (iv["arm"] == args.primary) & (iv["track"] == "experts")
    p2 = group_vs_base & (iv["arm"] == args.primary_self) & (iv["track"] == "self")
    iv["primary"] = p1 | p2
    iv["status"] = np.where(p1, "PRIMARY 1 (experts)", np.where(p2, "PRIMARY 2 (self-report)", "exploratory"))
    iv["supported"] = iv["primary"] & (iv["spearman_diff_lo_adj"] > 0)

    settings = {"candidates": cdir.name, "arms": args.arms, "primary": args.primary,
                "primary_self": args.primary_self, "prototypes": args.prototypes,
                "proto_set": proto_set, "context": args.context, "splits": splits(args.split), "boot": args.boot}
    h = config_hash(cfg)
    adir = out / "ablation"
    adir.mkdir(parents=True, exist_ok=True)
    name = f"{h[:12]}-{params_hash(settings)}"
    (adir / f"{name}.json").write_text(json.dumps(
        {"config_hash": h, "settings": settings, "candidate_params": json.loads((cdir / "params.json").read_text()),
         "splits": per_split, "intervals": iv.to_dict("records")}, indent=2, default=float), encoding="utf-8")
    write_table(table, adir / "ablation")
    write_table(iv, adir / "ablation_intervals")
    pd.concat(preds, ignore_index=True).to_csv(adir / "ablation_predictions.csv.gz", index=False)

    show = table[table["eval"] == "heldout"]
    for split in splits(args.split):
        for track in ("experts", "self"):
            t = show[(show["split"] == split) & (show["track"] == track)].sort_values("spearman", ascending=False)
            if t.empty:
                continue
            print(f"\n{split} split, {track}: fold means over {t['folds'].max()} folds ({t['context'].iloc[0]} context)")
            print(f"  {'arm':20s} {'MAE':>7s} {'Spearman':>9s} {'AUC':>6s}")
            for _, r in t.iterrows():
                print(f"  {r['arm']:20s} {r['mae']:7.3f} {r['spearman']:+9.3f} {r['auc']:6.3f}")
    print(f"\nPRIMARY tests (group split, pooled Spearman against base, Bonferroni-adjusted 98.75% interval over "
          f"recordings, {args.boot} resamples):")
    for label, arm, track in (("1 (experts)", args.primary, "experts"), ("2 (self-report)", args.primary_self, "self")):
        pr = iv[iv["status"] == f"PRIMARY {label}"]
        if len(pr):
            r = pr.iloc[0]
            verdict = "SUPPORTED (interval above 0)" if r["supported"] else "not supported"
            print(f"  {label:16s} {arm:20s} {r['spearman_ref']:+.4f} -> {r['spearman_arm']:+.4f}: difference "
                  f"{r['spearman_diff']:+.4f} [{r['spearman_diff_lo_adj']:+.4f}, {r['spearman_diff_hi_adj']:+.4f}]  {verdict}")
        else:
            print(f"  {label:16s} {arm:20s} not computed (needs the group split and the {track} labels)")
    print("\nexploratory (difference and 95% interval; intervals not corrected for the number of comparisons):")
    for _, r in iv[~iv["primary"]].iterrows():
        print(f"  {r['split']:8s} {r['track']:7s} {r['arm']:20s} vs {r['against']:5s} Spearman {r['spearman_diff']:+.4f} "
              f"[{r['spearman_diff_lo']:+.4f}, {r['spearman_diff_hi']:+.4f}]  MAE {r['mae_diff']:+.4f} "
              f"[{r['mae_diff_lo']:+.4f}, {r['mae_diff_hi']:+.4f}]")
    print(f"\nwrote {adir}/{name}.json, ablation, ablation_intervals (.csv/.json), ablation_predictions.csv.gz")


if __name__ == "__main__":
    main()
