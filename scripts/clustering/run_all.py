"""Run all of experiment 10 (steps 4-7 of scripts/clustering/README.md) with one resumable command.

  python -m scripts.clustering.run_all [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--workers N] [--label-workers 3] [--split both] [--channels ...] [--tau-desc X] [--tau-sync Y]
         [--redo STEP ...] [--dry-run] [--out outputs/experiments/10_clustering]

Needs the extraction caches (step 3; or experiment 09's, copied in). Runs, in order:

  signal_screen      which channels move together across the class (per recording, with a camera check)
  event_check        do students react together at big moments
  descriptor_subset  which descriptors to cluster on
  stability          which clustering thresholds give stable groups, beyond a null
  candidates         every candidate feature on the rating windows (JSON per recording)
  feature_screen     candidates against the ratings, training folds only, block null + BH
  importance         held-out permutation importance (--gate: the activity model too)
  ablation           the main test: fixed arms, paired 95% intervals, one primary comparison
  report             clustering_report.json, with a completeness check

The choices between steps are read from the outputs by fixed rules (no looking at results):
  channels   the rows of signal_screen_channels.csv with pass_sync True
  tau-desc   desc:subset, agglomerative, usable: the highest temporal excess over the null
  tau-sync   sync:*, agglomerative, usable: the highest temporal excess averaged over the channels
--channels, --tau-desc and --tau-sync override them (e.g. on a sample too small to test, where no
channel can pass). The choices are saved in <out>/run_choices.json.

Resumable at two levels: every step resumes inside itself (per recording or fold, see the README), and
this runner skips a step that already finished with the same command (marker in <out>/logs/done/). A
step whose command changed (other channels, other τ, other settings) runs again, as does every step
named in --redo. Each step's output goes to the console and to <out>/logs/<step>.log. If a step
fails, the runner stops there; fix the cause and run the same command again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

STEPS = ["signal_screen", "event_check", "descriptor_subset", "stability", "candidates", "feature_screen",
         "importance", "ablation", "report"]


def stamp(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def pick_channels(out: Path) -> list[str]:
    ch = pd.read_csv(out / "signal_screen_channels.csv")
    keep = ch[ch["pass_sync"].astype(str) == "True"]
    calm = keep[keep.get("pass_sync_low_motion", pd.Series(True, index=keep.index)).astype(str) != "True"]
    if len(calm):
        stamp(f"note: {', '.join(calm['channel'])} pass overall but not in the calm windows (camera?): kept")
    return keep["channel"].tolist()


def pick_taus(out: Path) -> tuple[float | None, float | None, list[str]]:
    s = pd.read_csv(out / "stability.csv")
    s = s[(s["method"] == "agglomerative") & (s["usable"].astype(str) == "True")]
    col = "temporal_excess" if "temporal_excess" in s else "temporal_ari"
    notes = []
    d = s[s["space"] == "desc:subset"]
    if d.empty:
        d = s[s["space"].str.startswith("desc:")]
    y = s[s["space"].str.startswith("sync:")].groupby("param")[col].mean()
    td = float(d.sort_values(col, ascending=False)["param"].iloc[0]) if len(d) else None
    ts = float(y.sort_values(ascending=False).index[0]) if len(y) else None
    if td is not None and td == d["param"].max():
        notes.append(f"tau-desc {td:g} is the largest tested: extend stability --taus-desc and rerun stability")
    if ts is not None and ts == y.index.max():
        notes.append(f"tau-sync {ts:g} is the largest tested: extend stability --taus-sync and rerun stability")
    return td, ts, notes


def run_step(name: str, cmd: list[str], logs: Path, redo: set[str], dry: bool, upstream: str) -> str:
    """Run one step unless it already finished with the same command after the same upstream steps;
    returns its key, which the next step's key includes (so anything after a rerun step reruns too,
    each resuming inside itself where its settings are unchanged)."""
    key = hashlib.sha256(json.dumps([cmd[1:], upstream]).encode()).hexdigest()[:16]
    marker = logs / "done" / name
    if name not in redo and marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("key") == key:
        stamp(f"{name}: done already (same command, same steps before it), skipped")
        return key
    stamp(f"=== {name}: {' '.join(cmd[2:])}")
    if dry:
        return key
    t0 = time.time()
    with open(logs / f"{name}.log", "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             text=True, encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        code = p.wait()
    if code != 0:
        stamp(f"FAILED {name} (exit {code}); see {logs / (name + '.log')}. Fix the cause and run the same command "
              f"again: finished steps are skipped and this one resumes where it stopped.")
        raise SystemExit(code)
    marker.write_text(json.dumps({"key": key, "cmd": cmd[2:], "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                  "minutes": round((time.time() - t0) / 60, 1)}), encoding="utf-8")
    stamp(f"{name}: finished in {(time.time() - t0) / 60:.1f} min")
    return key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="docs/experiments/09_vfoa.json")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides given to every step")
    ap.add_argument("--out", default="outputs/experiments/10_clustering")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="per-recording steps (default: CPU threads minus 2)")
    ap.add_argument("--label-workers", type=int, default=3,
                    help="fold workers of the label steps; each holds the label rows (~0.5 GB on the full data)")
    ap.add_argument("--split", choices=["both", "group", "activity"], default="both")
    ap.add_argument("--channels", nargs="*", default=None, help="override the channels chosen from signal_screen")
    ap.add_argument("--tau-desc", type=float, default=None, help="override the threshold chosen from stability")
    ap.add_argument("--tau-sync", type=float, default=None, help="override the threshold chosen from stability")
    ap.add_argument("--importance-repeats", type=int, default=2)
    ap.add_argument("--skip", nargs="*", default=[], choices=STEPS, help="steps not to run (e.g. importance)")
    ap.add_argument("--redo", nargs="*", default=[], choices=STEPS, help="steps to run again even if done")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    out = Path(args.out)
    logs = out / "logs"
    (logs / "done").mkdir(parents=True, exist_ok=True)
    base = ["--config", args.config, "--out", str(out)] + (["--set", *args.set] if args.set else [])
    py = [sys.executable, "-m"]
    redo, dry = set(args.redo), args.dry_run
    stamp(f"experiment 10: {args.config}, out {out}, workers {args.workers}, label workers {args.label_workers}")

    chain = {"key": ""}

    def step(name, *extra):
        if name in args.skip:
            stamp(f"{name}: skipped (--skip)")
            return
        chain["key"] = run_step(name, py + [f"scripts.clustering.{name}"] + base + list(extra), logs, redo, dry,
                                chain["key"])

    step("signal_screen", "--workers", str(args.workers))
    channels = args.channels
    if channels is None:
        channels = pick_channels(out) if (out / "signal_screen_channels.csv").exists() else ["<channels>"]
        if not channels:
            raise SystemExit("no channel passes the synchrony test (pass_sync), so there is nothing to cluster on in "
                             "synchrony space. That is itself a result; to run the rest anyway, give --channels.")
    stamp(f"channels: {' '.join(channels)}")
    step("event_check", "--workers", str(args.workers), "--channels", *channels)
    step("descriptor_subset")
    step("stability", "--workers", str(args.workers), "--channels", *channels)
    td, ts, notes = (None, None, [])
    if (out / "stability.csv").exists():
        td, ts, notes = pick_taus(out)
    td = args.tau_desc if args.tau_desc is not None else td
    ts = args.tau_sync if args.tau_sync is not None else ts
    for n in notes:
        stamp("note: " + n)
    if not dry and (td is None or ts is None):
        raise SystemExit("stability gave no usable agglomerative setting; give --tau-desc and --tau-sync")
    stamp(f"tau-desc {td}, tau-sync {ts}")
    choices = {"channels": channels, "tau_desc": td, "tau_sync": ts, "notes": notes,
               "overridden": {"channels": args.channels is not None, "tau_desc": args.tau_desc is not None,
                              "tau_sync": args.tau_sync is not None}}
    if not dry:
        (out / "run_choices.json").write_text(json.dumps(choices, indent=1), encoding="utf-8")
    step("candidates", "--workers", str(args.workers), "--channels", *channels,
         "--tau-desc", f"{td:g}" if td is not None else "<tau-desc>", "--tau-sync", f"{ts:g}" if ts is not None else "<tau-sync>")
    split = ["--split", args.split]
    step("feature_screen", "--workers", str(args.label_workers), *split)
    step("importance", "--workers", str(args.label_workers), "--repeats", str(args.importance_repeats), *split,
         *(["--gate"] if args.split != "activity" else []))
    step("ablation", "--workers", str(args.label_workers), *split)
    step("report")
    stamp(f"all done: {out / 'clustering_report.json'}")


if __name__ == "__main__":
    main()
