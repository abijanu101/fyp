"""Run experiment 10's follow-up analyses that need the full data, one after the other, with one
resumable command (as run_all does for the experiment itself):

  python -m scripts.clustering.followups [--config docs/experiments/10_clustering.json] [--set k=v ...]
         [--only seat_check behaviour_profile] [--redo STEP ...] [--dry-run]
         [--out outputs/experiments/10_clustering_v2]

  seat_check         do the ratings depend on where a student sits (scripts/clustering/seat_check.py)
  behaviour_profile  which behaviours the labelers respond to, within a session and between students
                     (scripts/clustering/behaviour_profile.py)

Needs what experiment 10's run left: the label rows (cached), the feature and track caches, the
label archives, and <out>/ablation/ablation_predictions.csv.gz. A step that already finished with the
same command is skipped (marker in <out>/logs/done/); each step's output goes to the console and to
<out>/logs/<step>.log. If a step fails, the runner stops; fix the cause and run the same command
again. At the end it lists the folders to send back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.clustering.run_all import run_step, stamp

STEPS = {"seat_check": "seat_check", "behaviour_profile": "behaviour_profile"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="docs/experiments/10_clustering.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default="outputs/experiments/10_clustering_v2")
    ap.add_argument("--only", nargs="*", default=list(STEPS), choices=list(STEPS))
    ap.add_argument("--redo", nargs="*", default=[], choices=list(STEPS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    logs = out / "logs"
    (logs / "done").mkdir(parents=True, exist_ok=True)
    need = out / "ablation" / "ablation_predictions.csv.gz"
    if not need.exists():
        stamp(f"note: {need} is missing: seat_check's model analysis (E) will be skipped")
    base = ["--config", args.config, "--out", str(out)] + (["--set", *args.set] if args.set else [])
    stamp(f"experiment 10 follow-ups: {', '.join(args.only)}; out {out}")
    for name in args.only:                       # independent steps: no chained key between them
        run_step(name, [sys.executable, "-m", f"scripts.clustering.{STEPS[name]}"] + base, logs, set(args.redo),
                 args.dry_run, "")
    stamp("all done. Send back: " + ", ".join(str(out / n) for n in args.only))


if __name__ == "__main__":
    main()
