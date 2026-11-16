"""Compare two per-labeler results files, e.g. the mixture of experts against a single decoder.

  python -m scripts.compare_results outputs/results/<A>.json outputs/results/<B>.json   (or hash prefixes: c5341cb6385e f6dcb84aaa6b)
         [--names MoE single] [--context classified] [--prediction own_labeler] [--plot]
  python -m scripts.compare_results --latest      # newest MoE vs newest single-decoder run

For each track (expert entries, self entries): the summary metrics of A and B on the configured
variant, fold by fold (the folds are the same groups in both runs), and per activity (each
activity is scored in the fold that holds it out). Baselines (constant 3, training mean) are shown
for reference. Then the confusion matrices, summed over the held-out folds: true rating (rows)
against the prediction rounded to a whole rating (columns), as counts and as row shares (recall),
with per-rating precision and recall, and the engaged / not-engaged matrix at tau_att. --plot also
writes them as heatmaps to outputs/plots/compare_<A>_<B>.png. Both files must come from
engagement.labelers = "per_labeler"; confusion matrices need results written after 2026-11-16.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ("auc", "spearman", "balanced_acc", "macro_f1", "mae")
RATINGS = ["1", "2", "3", "4", "5"]
BINARY = ["not engaged", "engaged"]


def resolve(path: str, folder: str = "outputs/results") -> str:
    """A results file path, or a config-hash prefix (as train_models prints it) in `folder`."""
    if Path(path).exists():
        return path
    found = sorted(Path(folder).glob(f"{path}*.json"))
    if len(found) != 1:
        raise SystemExit(f"{path}: {'no' if not found else 'more than one'} results file matches in {folder}")
    return str(found[0])


def load(path: str) -> dict:
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    if r.get("mode") != "per_labeler":
        raise SystemExit(f"{path}: not a per-labeler results file")
    return r


def newest(folder: str, scope: str) -> str:
    """The most recent per-labeler results file in `folder` with mixture.expert_scope == scope."""
    found = []
    for f in Path(folder).glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if r.get("mode") == "per_labeler" and r["config"]["mixture"]["expert_scope"] == scope:
            found.append(f)
    if not found:
        raise SystemExit(f"no per-labeler results with expert_scope '{scope}' in {folder}")
    return str(max(found, key=lambda f: f.stat().st_mtime))


def confusion(r: dict, key: str, which: str = "confusion") -> np.ndarray | None:
    """A variant's confusion matrix summed over the held-out folds (None if not stored)."""
    folds = r["evaluation"][key]["folds"]
    if not all(which in f for f in folds):
        return None
    return np.sum([np.asarray(f[which]) for f in folds], axis=0)


def print_confusion(name: str, cm: np.ndarray, labels: list[str]) -> None:
    n = cm.sum()
    rows = cm.sum(1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros(cm.shape), where=rows > 0)
    prec = np.divide(np.diag(cm), cm.sum(0), out=np.full(len(cm), np.nan), where=cm.sum(0) > 0)
    rec = np.diag(share)
    print(f"\n{name} ({n} entries; rows true, columns predicted; count / row share)\n")
    print("| true \\ pred | " + " | ".join(labels) + " | entries | recall |")
    print("|---|" + "---|" * (len(labels) + 2))
    for i, lab in enumerate(labels):
        cells = " | ".join(f"{cm[i, j]} / {share[i, j]:.2f}" for j in range(len(labels)))
        print(f"| **{lab}** | {cells} | {rows[i, 0]} | {rec[i]:.2f} |")
    print("| precision | " + " | ".join("-" if np.isnan(p) else f"{p:.2f}" for p in prec) + " | | |")
    print("| predicted | " + " | ".join(str(c) for c in cm.sum(0)) + " | | |")


def plot_confusions(path: Path, panels: list[tuple[str, np.ndarray, list[str]]], cols: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.9 * rows), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, (title, cm, labels) in zip(axes.flat, panels):
        tot = cm.sum(1, keepdims=True)
        share = np.divide(cm, tot, out=np.zeros(cm.shape), where=tot > 0)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for (i, j), c in np.ndenumerate(cm):
            ax.text(j, i, f"{share[i, j]:.2f}\n({c})", ha="center", va="center", fontsize=7,
                    color="white" if share[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def row(name: str, values: dict) -> str:
    return f"| {name} | " + " | ".join("-" if v is None or not np.isfinite(v) else f"{v:.3f}"
                                       for v in (values.get(m) for m in METRICS)) + " |"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--latest", action="store_true",
                    help="compare the newest MoE (expert_scope per_experiment) and single-decoder (shared) "
                         "per-labeler results in --results")
    ap.add_argument("--results", default="outputs/results")
    ap.add_argument("--context", default="classified", choices=["classified", "oracle"])
    ap.add_argument("--prediction", default="own_labeler", choices=["own_labeler", "median", "majority"])
    ap.add_argument("--plot", action="store_true", help="also save the confusion matrices as heatmaps")
    args = ap.parse_args()
    if args.latest:
        args.a, args.b = (newest(args.results, scope) for scope in ("per_experiment", "shared"))
        args.names = ["MoE", "single"]
        print(f"MoE: {args.a}\nsingle: {args.b}")
    elif not (args.a and args.b):
        ap.error("give two results files (or config-hash prefixes), or --latest")
    else:
        args.a, args.b = resolve(args.a, args.results), resolve(args.b, args.results)
    runs = dict(zip(args.names, (load(args.a), load(args.b))))
    head = "| | " + " | ".join(METRICS) + " |\n|---|" + "---|" * len(METRICS)
    key = f"configured/{args.context}/{args.prediction}"
    first_run, panels = next(iter(runs.values())), []
    for track in ("experts", "self"):
        print(f"\n## {track} entries, {key}\n\n{head}")
        for name, r in runs.items():
            ev = r["evaluation"]
            print(row(name, {m: ev[f"{track}/{key}"]["summary"][m]["mean"] for m in METRICS}))
        first = next(iter(runs.values()))["evaluation"]
        for base in ("constant_3", "train_mean"):
            print(row(base, {m: first[f"{track}/{base}"]["summary"][m]["mean"] for m in METRICS}))

        print(f"\nBy fold (AUC; difference {args.names[0]} - {args.names[1]}):\n")
        folds = {n: {f["held_out"]: f for f in r["evaluation"][f"{track}/{key}"]["folds"]} for n, r in runs.items()}
        a, b = (folds[n] for n in args.names)
        print("| held-out group | " + " | ".join(args.names) + " | difference |\n|---|---|---|---|")
        for g in sorted(set(a) & set(b)):
            print(f"| {g} | {a[g]['auc']:.3f} | {b[g]['auc']:.3f} | {a[g]['auc'] - b[g]['auc']:+.3f} |")

        print(f"\nBy activity (AUC / Spearman):\n")
        print("| activity | " + " | ".join(args.names) + " |\n|---|---|---|")
        acts = sorted({c for f in a.values() for c in f.get("per_experiment", {})})
        for c in acts:
            cells = []
            for n in args.names:
                m = next((f["per_experiment"][c] for f in folds[n].values() if c in f.get("per_experiment", {})), {})
                cells.append(f"{m.get('auc', np.nan):.3f} / {m.get('spearman', np.nan):.3f}" if m else "-")
            print(f"| {c} | " + " | ".join(cells) + " |")

        print(f"\n### Confusion matrices, {track} entries")
        for n, r in list(runs.items()) + [("constant_3", first_run)]:
            k = f"{track}/constant_3" if n == "constant_3" else f"{track}/{key}"
            for which, labels, what in (("confusion", RATINGS, "ratings"), ("confusion_binary", BINARY, "binary")):
                cm = confusion(r, k, which)
                if cm is None:
                    print(f"\n{n}: no confusion matrices stored (results written before they were added; retrain)")
                    break
                title = f"{n}, {track}, {what}" + (f" at tau_att {r['config']['labels']['tau_att']}" if what == "binary" else "")
                print_confusion(title, cm, labels)
                panels.append((title, cm, labels))
    if args.plot and panels:
        out = Path("outputs/plots") / f"compare_{Path(args.a).stem[:12]}_{Path(args.b).stem[:12]}.png"
        plot_confusions(out, panels, cols=4)
        print(f"\nheatmaps: {out}")


if __name__ == "__main__":
    main()
