"""Edge cases behind experiment 10's full-run failures, and the statistics added after it
(docs/run_reports/10_clustering.md; guide 15 §15.9)."""

import unittest

import numpy as np
import pandas as pd

from scripts.clustering.ablation import arm_spec, pooled_stats
from scripts.clustering.common import (TrackStandardiser, attach, bh_qvalues, centre_window_features,
                                       per_recording_test)
from scripts.clustering.feature_screen import rank_corr_with_null, shift_within
from src.cfmu.classrel import lag_behind_class, sync_features


class TestJoinEdgeCases(unittest.TestCase):
    def test_window_centres_under_half_a_millisecond_apart_do_not_duplicate_rows(self):
        # two labelers' entries 0.4 ms apart give two label windows whose rounded keys collide
        rows = pd.DataFrame({"stem": ["s", "s"], "t_key": [100.123, 100.456], "tid": [1, 1], "rating": [3, 4]})
        table = pd.DataFrame({"stem": ["s", "s", "s"], "t_centre": [100.1231, 100.1235, 100.456], "tid": [1, 1, 1],
                              "trk.cluster.sync.x.corr": [0.5, 0.5, 0.2]})
        out = attach(rows, table)
        self.assertEqual(len(out), 2)
        self.assertEqual(out["trk.cluster.sync.x.corr"].tolist(), [0.5, 0.2])


class TestShortTracks(unittest.TestCase):
    def test_lag_on_series_shorter_than_the_lag_range(self):
        # a student seen for 2 frames, the class for 2 frames, lag range 5: no crash, NaN or a value
        S = np.array([[1.0, 2.0], [2.0, 3.0], [0.5, 1.5]])
        lag, peak = lag_behind_class(S, max_lag=5, min_common=2)
        self.assertEqual(lag.shape, (3,))
        per, win = sync_features(S, max_lag=5, min_common=2)
        self.assertEqual(set(per), {"corr", "lag", "lag_corr", "degree"})


class TestStandardiserFloor(unittest.TestCase):
    def test_quiet_student_is_not_inflated(self):
        rng = np.random.default_rng(0)
        rows = []
        for tid, sd in ((1, 1.0), (2, 1.0), (3, 1.0), (4, 0.001)):            # student 4 barely moves
            rows += [{"tid": tid, "x": rng.normal(0, sd)} for _ in range(30)]
        t = pd.DataFrame(rows)
        quiet = pd.DataFrame({"tid": [4], "x": [t.loc[t.tid == 4, "x"].mean() + 0.01]})
        z_floor = TrackStandardiser(t, ["x"]).transform(quiet)[0, 0]
        z_none = TrackStandardiser(t, ["x"], floor=0).transform(quiet)[0, 0]
        self.assertGreater(abs(z_none), 5)        # without the floor, 0.01 of jitter is ~10 SDs
        self.assertLess(abs(z_floor), 0.1)        # with it, the typical student's scale applies


class TestStatistics(unittest.TestCase):
    def test_bh_matches_the_textbook_procedure(self):
        p = np.array([0.01, 0.04, 0.03, 0.20, np.nan])
        q = bh_qvalues(p)
        np.testing.assert_allclose(q[:4], [0.04, 0.0533333, 0.0533333, 0.20], rtol=1e-5)
        self.assertTrue(np.isnan(q[4]))

    def test_per_recording_test_needs_a_consistent_sign(self):
        self.assertLess(per_recording_test(pd.Series([0.1, 0.2, 0.15, 0.3, 0.05, 0.12])), 0.05)
        self.assertGreater(per_recording_test(pd.Series([0.1, -0.2, 0.15, -0.3, 0.05, -0.12])), 0.3)
        self.assertTrue(np.isnan(per_recording_test(pd.Series([0.1, 0.2]))))

    def test_shift_within_rotates_inside_each_student_only(self):
        keys = np.array(["a"] * 5 + ["b"] * 3 + ["c"])
        idx = shift_within(keys, 20, np.random.default_rng(0))
        for row in idx:
            self.assertEqual(sorted(row[:5]), [0, 1, 2, 3, 4])
            self.assertEqual(sorted(row[5:8]), [5, 6, 7])
            self.assertEqual(row[8], 8)                       # a student with one row stays put
            self.assertTrue((row[:5] != np.arange(5)).all())  # offset 1..len-1: every row moves

    def test_block_null_on_independent_autocorrelated_data(self):
        rng = np.random.default_rng(1)

        def ar(n):
            x = np.zeros(n)
            e = rng.normal(size=n)
            for i in range(1, n):
                x[i] = 0.95 * x[i - 1] + e[i]
            return x
        keys = np.repeat(np.arange(150), 40)
        y = np.concatenate([ar(40) for _ in range(150)])
        idx = shift_within(keys, 200, rng)
        fp = sum(rank_corr_with_null(pd.DataFrame({"x": np.concatenate([ar(40) for _ in range(150)])}), y, idx)["x"][2]
                 < 0.05 for _ in range(40))
        self.assertLessEqual(fp, 6)          # nominal 2 of 40; a naive iid test flags about half

    def test_pooled_spearman_equals_scipy_without_resampling(self):
        from scipy.stats import spearmanr

        rng = np.random.default_rng(2)
        x, y = rng.normal(size=300), rng.normal(size=300) + 0.3 * np.arange(300) / 300
        inv = rng.integers(0, 10, 300)
        sp, mae = pooled_stats(x, y, inv, np.ones((1, 10)), 10)
        self.assertAlmostEqual(sp[0], spearmanr(x, y).statistic, places=10)
        self.assertAlmostEqual(mae[0], np.mean(np.abs(x - y)), places=10)


def _square_or_fail(x, marker_dir):
    """Job for the pool tests: fails on 3, and leaves a marker file each time it really runs."""
    from pathlib import Path

    Path(marker_dir, f"ran_{x}_{len(list(Path(marker_dir).glob(f'ran_{x}_*')))}").touch()
    if x == 3:
        raise ValueError("job 3 fails")
    return x * x


class TestJobPool(unittest.TestCase):
    def _run(self, workers):
        import tempfile
        from pathlib import Path

        from scripts.clustering.common import map_jobs

        with tempfile.TemporaryDirectory() as d:
            cache, marks = Path(d) / "cache", Path(d) / "marks"
            marks.mkdir()
            jobs = [(x, str(marks)) for x in range(6)]
            names = [f"job{x}" for x in range(6)]
            got = {}
            with self.assertRaises(SystemExit) as cm:       # job 3 fails, the other five still finish
                for job, res in map_jobs(_square_or_fail, jobs, workers, names, cache):
                    got[job[0]] = res
            self.assertIn("1 of 6 jobs failed", str(cm.exception))
            self.assertEqual(got, {0: 0, 1: 1, 2: 4, 4: 16, 5: 25})
            self.assertEqual(len(list(cache.glob("*.pkl"))), 5)
            got2 = {}
            with self.assertRaises(SystemExit):            # rerun: only the failed job runs again
                for job, res in map_jobs(_square_or_fail, jobs, workers, names, cache):
                    got2[job[0]] = res
            self.assertEqual(got2, got)
            runs = {x: len(list(marks.glob(f"ran_{x}_*"))) for x in range(6)}
            self.assertEqual(runs, {0: 1, 1: 1, 2: 1, 3: 2, 4: 1, 5: 1})

    def test_failure_isolated_and_resume_serial(self):
        self._run(1)

    def test_failure_isolated_and_resume_parallel(self):
        self._run(2)


class TestRunAllMarkers(unittest.TestCase):
    def test_old_format_and_missing_markers_count_as_not_done(self):
        import json
        import tempfile
        from pathlib import Path

        from scripts.clustering.run_all import marker_key

        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "signal_screen"
            old.write_text("26 Nov 03:22\n", encoding="utf-8")          # what the first run's run.sh wrote
            self.assertIsNone(marker_key(old))
            self.assertIsNone(marker_key(Path(d) / "missing"))
            new = Path(d) / "stability"
            new.write_text(json.dumps({"key": "abc123"}), encoding="utf-8")
            self.assertEqual(marker_key(new), "abc123")


class TestArms(unittest.TestCase):
    def test_per_student_arms_never_include_window_features(self):
        cands = ["trk.cluster.sync.rf.total.corr", "trk.cluster.agg.desc.share", "cluster.agg.desc.n_clusters",
                 "cluster.sync.rf.total.mean_corr", "trk.cluster.hdb.desc.strength"]
        for arm in ("sync", "agglomerative", "hdbscan", "sync+agglomerative", "normalised+sync"):
            _, cols = arm_spec(arm, cands)
            self.assertTrue(cols and all(c.startswith("trk.cluster.") for c in cols), arm)
        which, cols = arm_spec("window", cands)
        self.assertEqual(which, "centred")
        self.assertTrue(all(c.startswith("cluster.") for c in cols))
        self.assertEqual(arm_spec("normalised+sync", cands)[0], "normed")
        which, cols = arm_spec("normalised+sync+agglomerative", cands)
        self.assertEqual(which, "normed")                          # normalised base inputs
        self.assertEqual(sorted(cols), ["trk.cluster.agg.desc.share", "trk.cluster.sync.rf.total.corr"])

    def test_centred_window_features_have_zero_mean_per_recording(self):
        rows = pd.DataFrame({"stem": ["a"] * 4 + ["b"] * 4, "t_key": [1, 1, 2, 3, 1, 2, 2, 3],
                             "cluster.x": [5.0, 5.0, 7.0, 9.0, 0.0, 1.0, 1.0, 2.0], "trk.y": range(8)})
        out = centre_window_features(rows)
        per_window = out.drop_duplicates(["stem", "t_key"]).groupby("stem")["cluster.x"].mean()
        np.testing.assert_allclose(per_window.to_numpy(), 0.0, atol=1e-12)
        self.assertEqual(out["trk.y"].tolist(), list(range(8)))        # per-student columns untouched


if __name__ == "__main__":
    unittest.main()
