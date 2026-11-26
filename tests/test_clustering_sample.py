"""Regression test on real data: every experiment 10 step, end to end, with real labels, on a small
sample of real recordings (at least 2 groups and 3 activities, with augmented copies). Experiment
10's full run crashed twice because the label steps first met real data there; this test runs them
first. It needs the sample's caches and labels on disk, so it is skipped unless FYP_SAMPLE_TEST=1:

  set FYP_SAMPLE_TEST=1                                    (PowerShell: $env:FYP_SAMPLE_TEST = "1")
  python -m unittest tests.test_clustering_sample -v

Settings (environment variables, all optional):
  FYP_SAMPLE_COPIES   augmented copies extracted for the sample (default 1)
  FYP_SAMPLE_SET      extra --set values used for the extraction, space-separated (e.g.
                      "ingestion.max_frames=300" for a quick sample of each recording's first minute)
  FYP_SAMPLE_SPLIT    both (default), group or activity: the activity split needs 3 activities with
                      labels, the group split 2 groups
  FYP_SAMPLE_OUT      scratch output folder (default tmp/exp10_sample_test)
Takes about 10-30 minutes on a laptop CPU, mostly building the label rows once.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

RUN = os.environ.get("FYP_SAMPLE_TEST") == "1"
COPIES = os.environ.get("FYP_SAMPLE_COPIES", "1")
OUT = Path(os.environ.get("FYP_SAMPLE_OUT", "tmp/exp10_sample_test"))
EXTRA = os.environ.get("FYP_SAMPLE_SET", "").split()
SPLIT = os.environ.get("FYP_SAMPLE_SPLIT", "both")
CFG = ["--config", "docs/experiments/09_vfoa.json", "--set", f"augmentation.copies={COPIES}", *EXTRA]


def run(module: str, *args: str) -> str:
    cmd = [sys.executable, "-m", f"scripts.clustering.{module}", *CFG, "--out", str(OUT), *args]
    p = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if p.returncode != 0:
        raise AssertionError(f"{module} failed:\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}")
    return p.stdout


@unittest.skipUnless(RUN, "set FYP_SAMPLE_TEST=1 (needs the sample's caches and labels)")
class TestExperiment10OnSample(unittest.TestCase):
    """The steps in order; each test depends on the ones before (unittest runs them alphabetically)."""

    def test_1_label_free_screens(self):
        out = run("signal_screen", "--workers", "2", "--max-windows", "40", "--surrogates", "10")
        self.assertIn("q (rec)", out)
        run("event_check")
        run("descriptor_subset")
        out = run("stability", "--workers", "2", "--max-windows", "15", "--boot", "3")
        self.assertIn("choice of tau", out)
        self.assertTrue((OUT / "stability.json").exists())

    def test_2_candidates_with_real_labels(self):
        out = run("candidates", "--workers", "2", "--channels", "rf.total", "kg.speed",
                  "--tau-desc", "1.0", "--tau-sync", "0.7", "--force")
        self.assertIn("student rows", out)
        dirs = sorted((OUT / "candidates").glob("*-*"))
        self.assertTrue(dirs)
        files = [p for p in dirs[-1].glob("*.json") if p.name != "params.json"]
        self.assertGreaterEqual(len(files), 2)             # at least the 2 groups the group split needs
        d = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertGreater(d["n_student_rows"], 0)

    def test_3_label_steps_both_splits(self):
        split = ["--split", SPLIT]
        out = run("feature_screen", "--perms", "50", "--prototypes", "4", "--workers", "2", *split)
        for name in (["activity", "group"] if SPLIT == "both" else [SPLIT]):
            self.assertIn(f"{name} split", out)
        run("importance", "--repeats", "1", "--prototypes", "4", "--workers", "2", *split,
            *(["--gate"] if SPLIT != "activity" else []))
        out = run("ablation", "--boot", "200", "--prototypes", "4", "--workers", "2", *split)
        self.assertIn("PRIMARY", out)
        iv = json.loads((OUT / "ablation" / "ablation_intervals.json").read_text(encoding="utf-8"))
        if SPLIT != "activity":                      # the primary test is on the group split
            self.assertEqual({r["status"] for r in iv if r["primary"]},
                             {"PRIMARY 1 (experts)", "PRIMARY 2 (self-report)"})
        # resuming: a second run loads every fold from partial/ instead of recomputing
        out = run("ablation", "--boot", "200", "--prototypes", "4", "--workers", "2", *split)
        self.assertIn("loaded from", out)

    def test_4_report_is_complete(self):
        out = run("report")
        self.assertIn("ablation: complete", out)


if __name__ == "__main__":
    unittest.main()
