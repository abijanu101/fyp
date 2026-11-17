"""Per-labeler training targets and predictions (docs/misc/targets.md §7; src/labels/interp.py,
src/experts/per_labeler.py, rating_metrics, soft labels in SklearnExpert)."""

import unittest

import numpy as np
import pandas as pd

from src.core.types import Window
from src.evaluation import rating_metrics
from src.experts import ExpertMixture, PerLabelerModel, join_entries, window_curves
from src.experts.expert import SklearnExpert
from src.labels.interp import (interpolate, majority_rating, normalise, quantise, round_rating, to_rating,
                               window_mean)
from src.output import build_rating_graph

T, V = np.array([0.0, 10.0, 11.0, 30.0]), np.array([3.0, 2.0, 3.0, 4.0])


class TestInterpolation(unittest.TestCase):
    def test_step_holds_and_is_nan_before_the_first_entry(self):
        np.testing.assert_array_equal(interpolate(T, V, [-1, 0, 5, 10, 10.5, 40])[1:], [3, 3, 2, 2, 4])
        self.assertTrue(np.isnan(interpolate(T, V, -1.0)))

    def test_every_mode_stays_within_1_to_5(self):
        t = np.linspace(-1, 45, 400)
        for mode in ("step", "ramp", "linear", "pchip", "cubic", "gauss"):
            v = interpolate(T, V, t, mode)
            self.assertTrue(np.all((v[np.isfinite(v)] >= 1) & (v[np.isfinite(v)] <= 5)), mode)

    def test_window_mean_of_the_step(self):
        self.assertAlmostEqual(window_mean(T, V, 8, 12), 2.75)       # 3, 3, 2, 3 over four seconds

    def test_a_centred_window_blends_the_previous_value(self):
        # an entry 2 -> 3 at t=11 with an 8 s window: half the window still holds the old value
        self.assertAlmostEqual(window_mean(np.array([0.0, 11.0]), np.array([2.0, 3.0]), 7, 15), 2.5)


class TestRatings(unittest.TestCase):
    def test_normalise_and_back_clip_out_of_range_values(self):
        np.testing.assert_allclose(normalise([0, 1, 3, 5, 7]), [0, 0, 0.5, 1, 1])
        np.testing.assert_allclose(to_rating([-0.5, 0, 0.5, 1, 2]), [1, 1, 3, 5, 5])

    def test_halves_round_toward_3(self):
        np.testing.assert_array_equal(round_rating([2.5, 3.5, 2.49, 4.5, 1.5, 7]), [3, 3, 2, 4, 2, 5])
        np.testing.assert_array_equal(quantise([2.24, 2.25, 2.75, 3.25, 3.3], 0.5), [2, 2.5, 3, 3, 3.5])

    def test_majority_ties_and_resolution(self):
        np.testing.assert_array_equal(majority_rating([[3, 3, 2, 4, 2, 5], [2, 2, 4, 4, np.nan, np.nan]]), [3, 2])
        self.assertEqual(majority_rating([[2.4, 2.6, 2.5, 3.1, 4.0, 4.1]], 0.5)[0], 2.5)
        self.assertEqual(majority_rating([[2.4, 2.6, 2.5, 3.1, 4.0, 4.1]], 1.0)[0], 3.0)
        self.assertTrue(np.isnan(majority_rating([[np.nan, np.nan]])[0]))

    def test_rating_metrics(self):
        y = np.array([2, 3, 3, 4, 2, 5, 3, 1.0])
        perfect, constant = rating_metrics(y, y, 3), rating_metrics(y, np.full(8, 3.0), 3)
        self.assertEqual((perfect["mae"], perfect["accuracy"], perfect["auc"]), (0.0, 1.0, 1.0))
        self.assertEqual((constant["auc"], constant["balanced_acc"]), (0.5, 0.5))
        self.assertTrue(np.isnan(constant["spearman"]))

    def test_scores_per_perspective(self):
        import pandas as pd

        from scripts.train_models import score_with_perspectives
        rows = pd.DataFrame({"entry_id": ["a", "a", "b", "c", "d"], "rating": [2, 2, 3, 4, 3.0],
                             "labeler": ["l1", "l1", "l1", "l2", "l2"]})
        pred = np.array([2.0, 3.0, 3.0, 4.0, 2.0])
        folds = {}
        score_with_perspectives(folds, "k", "g1", rows, pred, 3)
        self.assertEqual(set(folds), {"k", "k/by_labeler/l1", "k/by_labeler/l2"})
        self.assertEqual(folds["k"][0]["n"], 4)                       # entry a counted once
        self.assertAlmostEqual(folds["k/by_labeler/l1"][0]["mae"], 0.25)   # a: 2.5 vs 2, b: exact
        self.assertAlmostEqual(folds["k/by_labeler/l2"][0]["mae"], 0.5)    # c exact, d off by 1

    def test_spread_metrics_decompose_the_squared_error(self):
        rng = np.random.default_rng(1)
        y = rng.integers(1, 6, 400).astype(float)
        for p in (np.clip(3 + 0.3 * (y - 3) + rng.normal(0, 0.4, 400), 1, 5), np.full(400, 2.8)):
            m = rating_metrics(y, p, 3)
            self.assertAlmostEqual(m["mse_bias2"] + m["mse_spread"] + m["mse_floor"], m["rmse"] ** 2)
        const = rating_metrics(y, np.full(400, 3.0), 3)
        self.assertEqual((const["pred_sd"], const["share_pred_3"]), (0.0, 1.0))
        self.assertTrue(np.isnan(const["calib_ratio"]))
        self.assertAlmostEqual(sum(const["pred_shares"]), 1.0)
        # the MSE-optimal spread for a given r: calib_ratio 1, spread term 0
        p = 3 + (y - y.mean()) * 0.5
        opt = y.mean() + np.corrcoef(p, y)[0, 1] * y.std() * (p - p.mean()) / p.std()
        m = rating_metrics(y, opt, 3)
        self.assertAlmostEqual(m["calib_ratio"], 1.0, places=6)
        self.assertAlmostEqual(m["mse_spread"], 0.0, places=9)


class TestSoftLabels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(1500, 3))
        self.truth = 1 / (1 + np.exp(-1.5 * self.X[:, 0]))
        self.y = np.clip(self.truth + rng.normal(0, 0.05, len(self.truth)), 0, 1)

    def test_logistic_regression_learns_the_soft_target(self):
        p = SklearnExpert("logreg").fit(self.X, self.y).predict_proba(self.X)
        self.assertAlmostEqual(p.mean(), self.y.mean(), places=2)
        self.assertGreater(np.corrcoef(p, self.truth)[0, 1], 0.99)

    def test_svm_and_knn_threshold_soft_labels_and_keep_the_sign(self):
        for name in ("svm", "knn"):
            p = SklearnExpert(name, max_train_rows=600).fit(self.X, self.y).predict_proba(self.X)
            self.assertGreater(np.corrcoef(p, self.truth)[0, 1], 0.5, name)

    def test_a_constant_soft_target_is_predicted_as_is(self):
        self.assertEqual(SklearnExpert("logreg").fit(self.X[:5], np.full(5, 0.4)).predict_proba(self.X[:2])[0], 0.4)


class TestPerLabeler(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 600
        self.X = rng.normal(size=(n, 2))
        self.exp = np.full(n, "01")
        self.labeler = np.array(["labeler_01", "labeler_02", "self_labeling"] * (n // 3))
        hi = 1 / (1 + np.exp(-2 * self.X[:, 0]))
        self.target = np.where(self.labeler == "labeler_02", 1 - hi, hi)    # labeler_02 disagrees
        mix = lambda: ExpertMixture(["01"], lambda e: SklearnExpert("logreg"), "shared", "soft")
        self.model = PerLabelerModel.fit(mix, self.X, self.target, self.exp, self.labeler)

    def test_each_labeler_model_learns_only_its_own_labels(self):
        p = self.model.predict(self.X, np.ones((len(self.X), 1)))
        self.assertGreater(np.corrcoef(p["labeler_01"], self.X[:, 0])[0, 1], 0.9)
        self.assertLess(np.corrcoef(p["labeler_02"], self.X[:, 0])[0, 1], -0.9)

    def test_a_pooled_model_answers_the_same_for_every_labeler(self):
        mix = lambda: ExpertMixture(["01"], lambda e: SklearnExpert("logreg"), "shared", "soft")
        pooled = PerLabelerModel.fit(mix, self.X, self.target, self.exp, self.labeler, pooled=True)
        p = pooled.predict(self.X, np.ones((len(self.X), 1)))
        self.assertEqual(set(p), {"labeler_01", "labeler_02", "self_labeling"})
        np.testing.assert_array_equal(p["labeler_01"], p["labeler_02"])
        # labeler_02 contradicts the other two: pooled, the majority view wins for everyone
        self.assertGreater(np.corrcoef(p["labeler_02"], self.X[:, 0])[0, 1], 0.5)

    def test_window_curves_in_both_modes(self):
        window = np.repeat(np.arange(len(self.X) // 4), 4)
        n_windows = window.max() + 2                                         # the last window has no rows
        g = np.ones((len(self.X), 1))
        med, curves = window_curves(self.model, self.X, g, window, n_windows, "per_labeler", 0.5)
        maj, _ = window_curves(self.model, self.X, g, window, n_windows, "majority", 0.5)
        self.assertEqual(set(curves), {"labeler_01", "labeler_02", "self_labeling"})
        stack = np.stack(list(curves.values()))
        np.testing.assert_allclose(med, np.median(stack, axis=0))
        self.assertTrue(np.all((maj >= 1) & (maj <= 5)) and np.all((med >= 1) & (med <= 5)))
        self.assertAlmostEqual(curves["labeler_01"][-1], to_rating(self.model.fills["labeler_01"]))

    def test_rating_graph_records(self):
        windows = [Window(0, 4), Window(2, 6)]
        rec = build_rating_graph(windows, np.array([[0.9, 0.1], [0.2, 0.8]]), ["01", "02"], np.array([3.2, 2.4]),
                                 {"labeler_01": np.array([3.0, 2.0]), "self_labeling": np.array([4.0, 3.0])},
                                 np.array([5, 6]), 3.0, "per_labeler")
        self.assertEqual([r["engaged"] for r in rec], ["Y", "N"])
        self.assertEqual(rec[1]["labeler_ratings"], {"labeler_01": 2.0, "self_labeling": 3.0})
        self.assertNotIn("tid", rec[0])


class TestJoinEntries(unittest.TestCase):
    def test_entries_join_their_students_linked_tracks_only(self):
        entries = pd.DataFrame({"sid": ["s1", "s1", "s2"], "labeler": ["labeler_01", "self_labeling", "labeler_01"],
                                "t": [10.0, 10.0, 20.0], "rating": [3.0, 4.0, 2.0], "target": [0.5, 0.75, 0.25]})
        feats = pd.DataFrame({"t_centre": [10.0, 10.0, 20.0], "tid": [7, 8, 9], "trk.x": [1.0, 2.0, 3.0]})
        rows = join_entries(entries, feats, {7: "s1", 9: "s3"})       # track 8 unlinked, s2 has no track
        self.assertEqual(len(rows), 2)
        self.assertTrue((rows["tid"] == 7).all() and (rows["sid"] == "s1").all())


if __name__ == "__main__":
    unittest.main()
