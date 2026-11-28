"""Run experiment 10's follow-up analyses that need the full data, one after the other, with one
resumable command (as run_all does for the experiment itself):

  python -m scripts.clustering.followups [--config docs/experiments/10_clustering.json] [--set k=v ...]
         [--only STEP ...] [--redo STEP ...] [--device 0] [--dry-run]
         [--out outputs/experiments/10_clustering_v2]

  seat_check         do the ratings depend on where a student sits (scripts/clustering/seat_check.py)
  behaviour_profile  which behaviours the labelers respond to, within a session and between students
                     (scripts/clustering/behaviour_profile.py)
  posture            body posture from the pose model on the cached tracks of the original recordings
                     (scripts/clustering/posture.py; GPU, ~1 h; written to the posture cache)
  training_variants  an ordinal model, training on the experts' consensus, and posture as inputs, each
                     against its reference (scripts/clustering/training_variants.py; ~30-40 min; needs
                     posture: it reruns when posture does)

Needs what experiment 10's run left: the label rows (cached), the feature and track caches, the
frames and label archives, and <out>/ablation/ablation_predictions.csv.gz. A step that already
finished with the same command is skipped (marker in <out>/logs/done/); each step's output goes to the
console and to <out>/logs/<step>.log. If a step fails, the runner stops; fix the cause and run the same
command again (posture also resumes per recording). At the end it lists the folders to send back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.clustering.run_all import run_step, stamp

STEPS = ["seat_check", "behaviour_profile", "posture", "training_variants"]
SEND = {"seat_check": "seat_check", "behaviour_profile": "behaviour_profile", "training_variants": "training_variants"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="docs/experiments/10_clustering.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default="outputs/experiments/10_clustering_v2")
    ap.add_argument("--only", nargs="*", default=STEPS, choices=STEPS)
    ap.add_argument("--redo", nargs="*", default=[], choices=STEPS)
    ap.add_argument("--device", default=None, help="posture's GPU (e.g. 0); default: ultralytics' choice")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    logs = out / "logs"
    (logs / "done").mkdir(parents=True, exist_ok=True)
    need = out / "ablation" / "ablation_predictions.csv.gz"
    if not need.exists():
        stamp(f"note: {need} is missing: seat_check's model analysis and training_variants' reference are skipped")
    base = ["--config", args.config, "--out", str(out)] + (["--set", *args.set] if args.set else [])
    stamp(f"experiment 10 follow-ups: {', '.join(args.only)}; out {out}")
    keys = {}
    for name in [s for s in STEPS if s in args.only]:
        extra = ["--device", args.device] if name == "posture" and args.device is not None else []
        upstream = keys.get("posture", "") if name == "training_variants" else ""
        keys[name] = run_step(name, [sys.executable, "-m", f"scripts.clustering.{name}"] + base + extra, logs,
                              set(args.redo), args.dry_run, upstream)
    send = [str(out / SEND[n]) for n in args.only if n in SEND]
    stamp("all done. Send back: " + ", ".join(send))


if __name__ == "__main__":
    main()
