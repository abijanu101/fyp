import unittest

import numpy as np

from src.core.types import Tracklet
from src.evaluation import krippendorff_alpha_ordinal, tracking_metrics
from src.tuning import beam_bisection_search, nested_logo_search, random_search, sample_params

nan = np.nan


class TestAlpha(unittest.TestCase):
    def test_krippendorff_reference_example(self):
        # Krippendorff (2011) "Computing Krippendorff's Alpha-Reliability": 4 coders x 12 units.
        coders = [[1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
                  [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
                  [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
                  [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan]]
        units = [list(u) for u in zip(*coders)]
        self.assertAlmostEqual(krippendorff_alpha_ordinal(units), 0.815, places=3)

    def test_perfect_agreement(self):
        self.assertAlmostEqual(krippendorff_alpha_ordinal([[1, 1], [2, 2, 2], [3, 3], [5, 5]]), 1.0)


def static_track(tid, box, t):
    t = np.asarray(t, float)
    return Tracklet(tid, t, np.tile(box, (len(t), 1)).astype(float), np.ones(len(t)), np.full((len(t), 2), nan))


class TestTrackingMetrics(unittest.TestCase):
    subjects = {"a": np.array([0, 0, 100, 100.0]), "b": np.array([200, 0, 300, 100.0])}

    def test_perfect_tracking(self):
        m = tracking_metrics([static_track(1, self.subjects["a"], range(10)),
                              static_track(2, self.subjects["b"], range(10))], self.subjects, 10, 0.3)
        self.assertAlmostEqual(m["idf1"], 1.0)
        self.assertEqual((m["coverage"], m["id_switches_per_subject"], m["tids_per_subject"]), (1.0, 0.0, 1.0))

    def test_id_switch_and_fragmentation(self):
        m = tracking_metrics([static_track(1, self.subjects["a"], range(6)),
                              static_track(3, self.subjects["a"], range(6, 10)),
                              static_track(2, self.subjects["b"], range(10))], self.subjects, 10, 0.3)
        self.assertEqual(m["id_switches_per_subject"], 0.5)
        self.assertEqual(m["tids_per_subject"], 1.5)
        self.assertAlmostEqual(m["idf1"], 2 * 16 / (2 * 16 + 4 + 4))   # track 3's 4 frames are IDFP and IDFN

    def test_missed_subject(self):
        m = tracking_metrics([static_track(1, self.subjects["a"], range(10))], self.subjects, 10, 0.3)
        self.assertEqual((m["coverage"], m["subjects_found"]), (0.5, 0.5))


class TestSearch(unittest.TestCase):
    space = {"c": [1, 2, 3], "x": {"low": 0.01, "high": 1.0, "log": True}, "n": {"low": 1, "high": 9, "int": True}}

    def test_sampling_is_seeded_and_in_range(self):
        a = sample_params(self.space, np.random.default_rng(1))
        self.assertEqual(a, sample_params(self.space, np.random.default_rng(1)))
        self.assertIn(a["c"], [1, 2, 3])
        self.assertTrue(0.01 <= a["x"] <= 1.0 and isinstance(a["n"], int))

    def test_baseline_is_trial_zero_and_best_first(self):
        trials = random_search(self.space, lambda p: {"score": -abs(p.get("x", 0.5) - 0.3)}, 20, seed=0,
                               baseline={"x": 0.9})
        self.assertEqual(next(t for t in trials if t["trial"] == 0)["params"], {"x": 0.9})
        self.assertEqual(trials[0]["score"], max(t["score"] for t in trials))

    def test_bisection_finds_the_optimum_and_respects_the_budget(self):
        space = {"a": {"low": 0.0, "high": 1.0}, "b": {"low": 1, "high": 100, "int": True, "log": True}}
        f = lambda p: {"score": -((p["a"] - 0.73) ** 2) - (np.log(p["b"]) - np.log(20)) ** 2 / 10}
        trials = beam_bisection_search(space, f, 60, {"a": 0.5, "b": 50}, beam=2)
        self.assertLessEqual(len(trials), 60)
        self.assertEqual(next(t for t in trials if t["trial"] == 0)["params"], {"a": 0.5, "b": 50})
        self.assertAlmostEqual(trials[0]["params"]["a"], 0.73, delta=0.03)
        self.assertTrue(15 <= trials[0]["params"]["b"] <= 27)

    def test_bisection_is_deterministic_and_holds_inactive_and_categorical_params(self):
        space = {"a": {"low": 0.0, "high": 1.0}, "z": {"low": 0.0, "high": 1.0}, "c": [1, 2]}
        seen = []
        f = lambda p: seen.append(p) or {"score": -abs(p["a"] - 0.2)}
        run = lambda: [t["params"] for t in beam_bisection_search(space, f, 12, {"a": 0.9, "z": 0.4, "c": 2},
                                                                 inactive={"z"})]
        self.assertEqual(run(), run())
        self.assertTrue(all(p["z"] == 0.4 and p["c"] == 2 for p in seen))

    def test_nested_search_never_scores_selection_on_outer_test(self):
        groups = np.array(["g1"] * 4 + ["g2"] * 4 + ["g3"] * 4)
        seen = []

        def score(params, tr, te):
            seen.append((set(groups[tr]), set(groups[te])))
            return params["c"]

        res = nested_logo_search(groups, {"c": [1, 2, 3]}, 5, 0, score, baseline={"c": 1})
        self.assertEqual(res["selected"], {"c": 3})
        for o in res["outer"]:
            inner = [s for s in seen if o["held_out"] not in s[0] | s[1]]
            self.assertTrue(inner)   # inner searches ran without the held-out group


if __name__ == "__main__":
    unittest.main()


class TestSharedFrames(unittest.TestCase):
    """scripts/parallel_runs.py feeds one pass of frames to many trackers (run_trackers); it must give
    each tracker exactly the tracks it gets alone (run_tracker), and the search's prefetch hook must
    not change the search."""

    def test_several_trackers_on_one_pass_match_each_run_alone(self):
        from pathlib import Path

        from scripts.tuning.tune_tracker import SessionDetections, run_tracker, run_trackers
        from src.core.types import Detections, SessionKey
        rng = np.random.default_rng(0)
        dets = []
        for f in range(60):                          # four people walking right, with jittered confidence
            xyxy = np.array([[50 + 120 * k + 3 * f, 100, 110 + 120 * k + 3 * f, 260] for k in range(4)], float)
            dets.append(Detections(xyxy + rng.normal(0, 1.5, xyxy.shape), rng.uniform(0.2, 0.9, 4)))
        sd = SessionDetections(SessionKey("g", "e", "c"), Path("none.zip"), Path("none.npz"),
                               np.arange(60) * 0.2, (480, 640), dets, None)
        specs = [("botsort", {"use_cmc": False, "use_embeddings": False, "track_high_thresh": 0.5}, 0.3),
                 ("botsort", {"use_cmc": False, "use_embeddings": False, "match_thresh": 0.6}, 0.5),
                 ("botsort", {"use_cmc": False, "use_embeddings": False, "max_lost_frames": 5}, 0.2)]
        shared = run_trackers(sd, specs)
        for (name, opts, conf), got in zip(specs, shared):
            alone = run_tracker(name, opts, sd, conf)
            self.assertEqual([(t.tid, t.t.tolist(), t.xyxy.tolist()) for t in got],
                             [(t.tid, t.t.tolist(), t.xyxy.tolist()) for t in alone])

    def test_prefetch_does_not_change_the_search(self):
        space = {"a": {"low": 0, "high": 1}, "b": {"low": 0, "high": 1}}
        f = lambda p: {"score": -(p["a"] - 0.3) ** 2 - (p["b"] - 0.8) ** 2}
        asked, seen = [], []
        plain = beam_bisection_search(space, f, 20, {"a": 0.5, "b": 0.5})
        hooked = beam_bisection_search(space, lambda p: (seen.append(p), f(p))[1], 20, {"a": 0.5, "b": 0.5},
                                       prefetch=asked.extend)
        self.assertEqual([t["params"] for t in plain], [t["params"] for t in hooked])
        self.assertTrue(asked and all(p in seen for p in asked))    # nothing prefetched is wasted
