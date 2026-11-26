"""The choices between experiment 10's steps, read from the outputs (used by scripts/experiments/10_clustering/run.sh).

  python scripts/experiments/10_clustering/pick.py channels   channels with pass_sync true in signal_screen_channels.csv
  python scripts/experiments/10_clustering/pick.py taus       "<tau-desc> <tau-sync>" from stability.csv: desc:subset, agglomerative,
                                      usable, highest temporal ARI; and over sync:* (agglomerative, usable),
                                      the tau with the highest mean temporal ARI across channels
  python scripts/experiments/10_clustering/pick.py best       "<split> <arm>" per split: the arm the first ablation run chose for sync+best
Exits 1 with a message when there is nothing to pick.
"""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path("outputs/experiments/10_clustering")
what = sys.argv[1]

if what == "channels":
    ch = pd.read_csv(OUT / "signal_screen_channels.csv")
    ch = ch[ch["pass_sync"].astype(str) == "True"]["channel"].tolist()
    if not ch:
        sys.exit("no channel has pass_sync true in signal_screen_channels.csv")
    print(" ".join(ch))

elif what == "taus":
    s = pd.read_csv(OUT / "stability.csv")
    s = s[(s["method"] == "agglomerative") & (s["usable"].astype(str) == "True")]
    d = s[s["space"] == "desc:subset"].sort_values("temporal_ari", ascending=False)
    y = (s[s["space"].str.startswith("sync:")].groupby("param")["temporal_ari"].mean()
         .sort_values(ascending=False))
    print("descriptor taus (desc:subset):", file=sys.stderr)
    print(d[["param", "temporal_ari", "subsample_ari", "clusters_per_student"]].head(3).to_string(index=False), file=sys.stderr)
    print("synchrony taus (mean temporal ARI over sync:* channels):", file=sys.stderr)
    print(y.round(3).to_string(), file=sys.stderr)
    if d.empty or y.empty:
        sys.exit(f"no usable agglomerative setting for {'the descriptors' if d.empty else 'synchrony'}:"
                 f" send stability.csv ({OUT / 'stability.csv'})")
    print(f"{d['param'].iloc[0]:g} {y.index[0]:g}")

elif what == "best":
    runs = [p for p in (OUT / "ablation").glob("*-*.json")]
    runs = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in runs]
    runs = [(p, r) for p, r in runs if r["settings"]["best"] is None and "sync+best" in r["settings"]["arms"]]
    if not runs:
        sys.exit("no first ablation run (sync+best without --best) found")
    p, r = max(runs, key=lambda pr: pr[0].stat().st_mtime)
    best = {split: v["best"] for split, v in r["splits"].items()}
    print(f"first ablation run {p.name}: sync+best chose {best}", file=sys.stderr)
    for split, arm in best.items():          # one "<split> <arm>" line each: every split frozen on its own choice
        print(split, arm)
