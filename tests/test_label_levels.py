"""Collapsed label levels (labels.collapse) and the rebalanced training subset (labels.train_keep)."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation import rating_metrics
from src.experts import train_keep_mask
from src.labels.interp import collapse_map, rating_levels, round_rating, round_to_levels

LMH = [[1, 2], [3], [4, 5]]


class TestCollapse(unittest.TestCase):
    def test_low_mid_high_sit_on_1_3_5(self):
        self.assertEqual(collapse_map(LMH), {1: 1.0, 2: 1.0, 3: 3.0, 4: 5.0, 5: 5.0})
        self.assertEqual(rating_levels(LMH).tolist(), [1.0, 3.0, 5.0])
        self.assertEqual(rating_levels().tolist(), [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertIsNone(collapse_map(None))

    def test_groups_must_split_1_to_5_in_order(self):
        for bad in ([[1, 2], [4, 5]], [[2, 1], [3], [4, 5]], [[1, 2, 3, 4, 5]], [[1, 2], [], [3, 4, 5]]):
            with self.assertRaises(ValueError):
                collapse_map(bad)

    def test_rounding_to_levels(self):
        x = np.linspace(0, 6, 1201)
        self.assertTrue((round_to_levels(x, rating_levels()) == round_rating(x)).all())    # 1-5: unchanged
        lv = rating_levels(LMH)
        self.assertEqual(round_to_levels(np.array([1.9, 2.0, 2.1, 4.0, 4.1]), lv).tolist(), [1, 3, 3, 3, 5])
        m = round_to_levels(np.array([[1.2, np.nan], [4.6, 3.1]]), lv)                     # any shape, NaN kept
        self.assertEqual(m.shape, (2, 2))
        self.assertTrue(np.isnan(m[0, 1]))
        self.assertEqual(float(round_to_levels(2.0, lv)), 3.0)

    def test_metrics_on_levels(self):
        lv = rating_levels(LMH)
        y = np.array([1, 1, 3, 3, 5, 5], float)
        p = np.array([1.8, 2.2, 3.0, 3.9, 4.1, 5.0])          # rounds to 1, 3, 3, 3, 5, 5
        m = rating_metrics(y, p, 3.0, lv)
        self.assertAlmostEqual(m["accuracy"], 5 / 6)
        self.assertAlmostEqual(m["mae_levels"], m["mae"] / 2)
        self.assertEqual(np.array(m["confusion"]).shape, (3, 3))
        self.assertEqual(len(m["pred_shares"]), 3)
        self.assertNotIn("mae_levels", rating_metrics(y, p, 3.0))  # 1-5: no second MAE


class TestTrainKeep(unittest.TestCase):
    def rows(self, n=2000):
        rng = np.random.default_rng(1)
        ratings = rng.choice([1.0, 3.0, 5.0], n, p=[0.3, 0.5, 0.2])
        ids = [f"e{i}" for i in range(n)]
        # every entry twice (a second track / augmented copy): kept or dropped together
        return pd.DataFrame({"entry_id": ids * 2, "rating": np.concatenate([ratings, ratings])})

    def test_no_setting_keeps_everything(self):
        self.assertTrue(train_keep_mask(self.rows(), None).all())

    def test_probabilities_per_level_and_whole_entries(self):
        rows = self.rows()
        keep = train_keep_mask(rows, {"3": 0.4, "5": 1.0}, seed=0)
        r = rows["rating"].to_numpy()
        self.assertTrue(keep[r == 1].all() and keep[r == 5].all())                # unnamed: 1.0
        self.assertAlmostEqual(keep[r == 3].mean(), 0.4, delta=0.05)
        half = len(rows) // 2
        self.assertTrue((keep[:half] == keep[half:]).all())                       # per entry, not per row
        self.assertTrue((keep == train_keep_mask(rows, {"3": 0.4}, seed=0)).all())  # seeded

    def test_bad_settings_fail(self):
        with self.assertRaises(ValueError):
            train_keep_mask(self.rows(), {"2": 0.5})            # no entry is 2 on the collapsed scale
        with self.assertRaises(ValueError):
            train_keep_mask(self.rows(), {"3": 1.5})


if __name__ == "__main__":
    unittest.main()
