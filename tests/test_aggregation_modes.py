"""Early vs late aggregation (config engagement.aggregate)."""

import unittest

import numpy as np
import pandas as pd

from src.core.config import load_config
from src.core.pipeline import Pipeline, to_track_frame
from src.core.types import SessionTrack, Tracklet
from src.experts import (EngagementRows, ExpertMixture, decision_threshold, early_rows, fit_predict, late_rows,
                         predict_windows, student_key, window_scores)
from src.experts.expert import SklearnExpert
from src.labels import WindowLabeler, parse_events
from src.core.clock import parse_clock
from src.core.types import Window

CLASSES = ["01", "02"]


class TestLateSettings(unittest.TestCase):
    def test_late_scores_are_cut_at_the_label_share(self):
        # late window score estimates the share engaged; the window label is share >= tau_share
        self.assertEqual(decision_threshold("late", mixture_tau=0.5, tau_share=0.6), 0.6)
        self.assertEqual(decision_threshold("early", mixture_tau=0.5, tau_share=0.6), 0.5)

    def test_window_mean_key_maps_to_the_student_feature(self):
        self.assertEqual(student_key("vec.kg.stillness.mean"), "trk.kg.stillness")
        self.assertEqual(student_key("vec.kg.stillness.std"), "vec.kg.stillness.std")
        self.assertEqual(student_key("vec.n_tracks"), "vec.n_tracks")

    def test_hand_weighted_baseline_reads_students_in_late_mode(self):
        from scripts.train_models import expert_factory
        cfg = load_config("configs/deliverable3.json")
        cfg["engagement"] = {"aggregate": "late", "late_context": False}
        cols = ["trk.kg.stillness", "trk.scb.write"]
        expert = expert_factory(cfg, cols, "hand_weighted")("02")
        self.assertEqual(sorted(cols[i] for i in expert.idx), cols)


class TestNobodyVisible(unittest.TestCase):
    def test_late_inference_without_any_student_falls_back_to_fill(self):
        windows = pd.DataFrame({"vec.a.mean": [np.nan, np.nan]})
        empty = pd.DataFrame(columns=["window", "tid"])      # what to_track_frame gives with no tracks
        X, _, w = late_rows(windows, empty, context=False, cols=["trk.a", "vec.a.mean"])
        self.assertEqual(X.shape, (0, 2))

        class NeverCalled:
            def predict_proba(self, X, g):
                raise AssertionError("no rows, nothing to predict")

        rows = EngagementRows(X, ["trk.a", "vec.a.mean"], w, np.zeros(0), np.zeros(0))
        p = predict_windows(NeverCalled(), rows, np.arange(2), np.full((2, 2), 0.5), fill=0.3)
        np.testing.assert_array_equal(p, [0.3, 0.3])


class TestRows(unittest.TestCase):
    def setUp(self):
        self.windows = pd.DataFrame({"vec.a.mean": [1.0, 2.0, 3.0], "vec.n_tracks": [2.0, 0.0, 1.0]})
        self.tracks = pd.DataFrame({"window": [0, 0, 2], "tid": [7, 8, 7], "trk.a": [0.5, 1.5, 3.0]})

    def test_late_rows_carry_window_context(self):
        X, cols, w = late_rows(self.windows, self.tracks, context=True)
        self.assertEqual(cols, ["trk.a", "vec.a.mean", "vec.n_tracks"])
        np.testing.assert_array_equal(w, [0, 0, 2])
        np.testing.assert_array_equal(X, [[0.5, 1.0, 2.0], [1.5, 1.0, 2.0], [3.0, 3.0, 1.0]])

    def test_late_rows_without_context_and_fixed_columns(self):
        X, cols, _ = late_rows(self.windows, self.tracks, context=False)
        self.assertEqual(cols, ["trk.a"])
        X2, _, _ = late_rows(self.windows, self.tracks, context=False, cols=["trk.a", "trk.missing"])
        self.assertTrue(np.isnan(X2[:, 1]).all())

    def test_window_scores_mean_rows_and_fill_empty_windows(self):
        p = window_scores(np.array([0.2, 0.6, 0.9]), np.array([0, 0, 2]), 3, fill=0.4)
        np.testing.assert_allclose(p, [0.4, 0.4, 0.9])   # window 0: mean(0.2, 0.6); window 1: no rows


class TestEarlyEqualsDirectMixture(unittest.TestCase):
    def test_early_rows_reproduce_the_window_level_mixture(self):
        rng = np.random.default_rng(0)
        n = 60
        df = pd.DataFrame({"vec.x": rng.normal(size=n), "vec.z": rng.normal(size=n)})
        exp = np.array(["01", "02"] * (n // 2))
        y = (df["vec.x"].to_numpy() + 0.3 * rng.normal(size=n) > 0).astype(float)
        X, cols, window = early_rows(df, ["vec.x", "vec.z"])
        rows = EngagementRows(X, cols, window, exp, y)
        make = lambda: ExpertMixture(CLASSES, lambda e: SklearnExpert("logreg", seed=0))
        g = np.tile([0.3, 0.7], (20, 1))
        tr, te = np.arange(40), np.arange(40, 60)
        direct = make().fit(X[tr], y[tr].astype(int), exp[tr]).predict_proba(X[te], g)
        np.testing.assert_allclose(fit_predict(make, rows, tr, te, g), direct)


class TestStudentLabels(unittest.TestCase):
    def test_student_labels_match_the_window_share(self):
        t0 = parse_clock("10:00:00:000000")
        tracks = lambda v: {"labeler_01": parse_events([{"datetime": "10:00:00:000000", "attention": str(v)}])}
        students = {"s1": tracks(4), "s2": tracks(2), "s3": tracks(5)}
        lab = WindowLabeler(tau_att=4, tau_share=0.5, min_labeled_students=1)
        w = Window(t0, t0 + 8)
        per_student = lab.student_labels(students, w)
        self.assertEqual(per_student, {"s1": 1.0, "s2": 0.0, "s3": 1.0})
        self.assertAlmostEqual(lab.label(students, w).share_engaged, np.mean(list(per_student.values())))


class TestFeaturePassRows(unittest.TestCase):
    def test_one_row_per_visible_track_per_window(self):
        cfg = load_config("configs/deliverable3.json")
        frame_t = np.arange(0, 20, 0.2)
        trs = []
        for i in range(3):
            n = 100 if i < 2 else 40                              # track 2 only visible for the first 8 s
            t = frame_t[:n]
            xyxy = np.tile([100.0 + 200 * i, 100, 150 + 200 * i, 200], (n, 1))
            trs.append(Tracklet(i, t, xyxy, np.ones(n), np.zeros((n, 2))))
        st = SessionTrack(None, frame_t, (1280, 720), trs, [{"motion": 0.0}] * len(frame_t),
                          {"fps": 5.0, "t_first": 0.0, "t_last": float(frame_t[-1])}, {})
        feats = Pipeline(cfg).features(st)
        per_window = pd.Series([r["window"] for r in feats.track_rows]).value_counts().sort_index()
        self.assertEqual(per_window.tolist(), [v["n_tracks"] for v in feats.vec])
        tf = to_track_frame(feats)
        self.assertIn("trk.kg.speed", tf.columns)
        self.assertEqual(set(tf["tid"]), {0.0, 1.0, 2.0})


if __name__ == "__main__":
    unittest.main()
