"""Gather every clustering result into one JSON file (guide 15 §15.9), to carry back from the machine
that ran them.

  python -m scripts.clustering.report [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--out outputs/experiments/10_clustering]

Reads what the other scripts wrote to --out (any that has not run is reported as missing) and writes
<out>/clustering_report.json:
  run          config hash, feature cache, sessions in the cache (originals, augmented copies)
  signals      screens A1-A3 per channel, and the seat-like descriptors (within-track share < 0.3)
  events       screen A4 per channel, each recording counted once, with and without camera-like events
  missingness  whether the NaN pattern alone predicts the recording
  descriptors  screen A5: the chosen subset
  stability    screen A6: the usable settings, most stable first
  candidates   B7: every candidate folder, its settings and session count
  screen       B8: the 30 strongest candidates on the experts' training folds, per split
  importance   B9: the per-labeler model per split, and the gate (group split)
  ablation     B10: every ablation run (its settings and the 95% intervals), the latest run's table,
               and ablation_check: whether the latest run covers every requested split and arm
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.clustering.common import base_parser, candidate_files, load
from scripts.run_pipeline import features_dir, tracks_dir
from src.core.config import config_hash


def records(path: Path, query=None, head: int | None = None):
    if not path.exists():
        return "missing"
    df = pd.read_json(path) if path.suffix == ".json" else pd.read_csv(path)
    if query is not None:
        df = query(df)
    if head:
        df = df.head(head)
    return json.loads(df.to_json(orient="records", double_precision=4))


def main() -> None:
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg, out = load(args)

    stems = sorted(p.stem for p in tracks_dir(cfg).glob("*.npz")) if tracks_dir(cfg).exists() else []
    rep: dict = {"run": {"config_hash": config_hash(cfg)[:12], "tracks_cache": str(tracks_dir(cfg)),
                         "features_cache": str(features_dir(cfg)),
                         "sessions": sum("." not in s for s in stems), "augmented_copies": sum("." in s for s in stems)}}
    rep["signals"] = {"channels": records(out / "signal_screen_channels.json"),
                      "seat_like_descriptors": records(out / "signal_screen_descriptors.json",
                                                       lambda d: d[d["within_track_share"] < 0.3])}
    rep["events"] = records(out / "event_check_channels.json")
    sub = out / "descriptor_subset.json"
    rep["descriptors"] = json.loads(sub.read_text(encoding="utf-8"))["subset"] if sub.exists() else "missing"
    rep["stability"] = records(out / "stability.json", lambda d: d[d["usable"]], head=15)
    rep["candidates"] = [{"folder": d.name, "settings": json.loads((d / "params.json").read_text(encoding="utf-8")),
                          "sessions": len(candidate_files(d))}
                         for d in sorted((out / "candidates").glob("*-*")) if (d / "params.json").exists()] or "missing"
    rep["screen"] = records(out / "feature_screen.json",
                            lambda d: d[d["track"] == "experts"].groupby("split", sort=False).head(30))
    miss = out / "feature_screen_missingness.json"
    rep["missingness"] = json.loads(miss.read_text(encoding="utf-8")) if miss.exists() else "missing"
    rep["importance"] = {"model": records(out / "importance_summary.json"),
                         "gate": records(out / "importance_gate.json")}
    abl = sorted((out / "ablation").glob("*-*.json"), key=lambda q: q.stat().st_mtime)
    runs = []
    for q in abl:
        d = json.loads(q.read_text(encoding="utf-8"))
        runs.append({"file": q.name, "settings": d["settings"],
                     "intervals": d.get("intervals"), "verdict_old_format": d.get("verdict")})
    rep["ablation"] = runs or "missing"
    if (out / "ablation" / "ablation.json").exists():
        rep["ablation_table_latest"] = records(out / "ablation" / "ablation.json",
                                               lambda d: d[d["eval"].isin(["heldout", "train"])])
    # completeness: the latest run must cover every split and arm it was asked for, with intervals
    problems = []
    if abl:
        d = json.loads(abl[-1].read_text(encoding="utf-8"))
        st = d["settings"]
        for split in st.get("splits", []):
            got = d.get("splits", {}).get(split)
            if not got:
                problems.append(f"split {split}: no results")
                continue
            lacking = [a for a in st.get("arms", []) if a not in got.get("arms", {})]
            if lacking:
                problems.append(f"split {split}: arms without results: {lacking}")
        if not d.get("intervals"):
            problems.append("no confidence intervals (an older ablation run?)")
        else:
            st_ = {r.get("status") for r in d["intervals"]}
            for label, need in (("PRIMARY 1 (experts)", "the group split and the experts' labels"),
                                ("PRIMARY 2 (self-report)", "the group split and the self-report labels")):
                if label not in st_:
                    problems.append(f"{label} was not computed (needs {need})")
    else:
        problems.append("no ablation run")
    rep["ablation_check"] = problems or "complete"
    path = out / "clustering_report.json"
    path.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    def missing(v):
        return v == "missing" or (isinstance(v, dict) and all(missing(x) for x in v.values()))

    print(f"{rep['run']['sessions']} sessions, {rep['run']['augmented_copies']} augmented copies in the cache")
    print("sections present:", ", ".join(k for k, v in rep.items() if not missing(v)))
    print("missing:", ", ".join(k for k, v in rep.items() if missing(v)) or "none")
    if rep["ablation_check"] != "complete":
        print("ABLATION INCOMPLETE:", "; ".join(rep["ablation_check"]))
    else:
        print("ablation: complete (every requested split and arm, with intervals and both primary tests)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
