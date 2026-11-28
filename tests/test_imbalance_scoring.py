"""scripts/analysis/imbalance_scoring.py with known answers: the mappings, the leakage guard, and
the metrics."""

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from scripts.analysis.imbalance_scoring import (class_metrics, map_fold, mapped, quantile_map, round_map,
                                                 to_levels)

LV = np.array([2, 3, 4, 5])


def fake(seed: int, signal: float, n_per: int = 600):
    """3 groups, 2 labelers with different class shares; scores = signal x the latent + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    shares = {"labeler_01": [0.45, 0.45, 0.08, 0.02], "labeler_03": [0.45, 0.33, 0.12, 0.10]}
    for g in ("group_01", "group_02", "group_03"):
        for lab, sh in shares.items():
            lat = rng.normal(size=n_per)
            cut = np.quantile(lat, np.cumsum(sh)[:-1])
            y = LV[np.searchsorted(cut, lat)]
            s = 3 + signal * lat + rng.normal(0, 1, n_per) * 0.3
            for i in range(n_per):
                rows.append({"entry_id": f"{g}/experiment_01/subject_{i % 15:02d}/{lab}/{i}", "grp": g,
                             "cls": f"{g}/experiment_0{i % 6 + 1}", "labeler": lab, "rating": float(y[i]),
                             "pred": float(s[i])})
    return pd.DataFrame(rows)


class TestImbalanceScoring(unittest.TestCase):
    def test_round_map_halves_toward_3(self):
        np.testing.assert_array_equal(round_map(np.array([2.5, 3.5, 2.4, 4.6, 1.0]), LV), [3, 3, 2, 5, 2])

    def test_quantile_map_keeps_order_and_shares(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=1000)
        c = quantile_map(s, np.array([0.4, 0.4, 0.15, 0.05]), LV)
        self.assertTrue(np.all(np.diff(c[np.argsort(s)]) >= 0))         # order kept
        np.testing.assert_allclose([np.mean(c == k) for k in LV], [0.4, 0.4, 0.15, 0.05], atol=0.002)

    def test_experts_rating_1_is_merged_into_2(self):
        np.testing.assert_array_equal(to_levels(np.array([1.0, 2, 5]), LV), [2, 2, 5])

    def test_cut_points_use_no_test_labels(self):
        d = fake(1, 1.0)
        a = map_fold(d, "group_02", LV, "quantile")
        d2 = d.copy()
        test = d2["grp"] == "group_02"
        d2.loc[test, "rating"] = np.random.default_rng(5).permutation(d2.loc[test, "rating"].to_numpy())
        np.testing.assert_array_equal(a, map_fold(d2, "group_02", LV, "quantile"))

    def test_per_labeler_shares_are_matched(self):
        d = fake(2, 1.0)
        c = mapped(d, LV, "quantile")
        for lab, sh in (("labeler_01", [0.45, 0.45, 0.08, 0.02]), ("labeler_03", [0.45, 0.33, 0.12, 0.10])):
            m = (d["labeler"] == lab).to_numpy()
            np.testing.assert_allclose([np.mean(c[m] == k) for k in LV], sh, atol=0.03)

    def test_informative_scores_beat_rounding_and_random(self):
        d = fake(3, 0.3)
        y = d["rating"].to_numpy()
        q = class_metrics(y, mapped(d, LV, "quantile"), LV)
        r = class_metrics(y, mapped(d, LV, "round"), LV)
        n = class_metrics(y, mapped(d, LV, "random", np.random.default_rng(0)), LV)
        self.assertGreater(q["macro_f1"], r["macro_f1"])
        self.assertGreater(q["balanced_acc"], r["balanced_acc"])
        self.assertGreater(q["qwk"], n["qwk"] + 0.3)

    def test_null_scores_give_random_level_agreement(self):
        d = fake(4, 0.0)
        y = d["rating"].to_numpy()
        q = class_metrics(y, mapped(d, LV, "quantile"), LV)
        self.assertLess(abs(q["qwk"]), 0.05)

    def test_metrics_against_hand_values(self):
        y = np.array([2, 2, 3, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5,
                      2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5], float)
        c = y.copy()
        c[::3] = 3
        m = class_metrics(y, c, LV)
        self.assertAlmostEqual(m["qwk"], cohen_kappa_score(y, c, weights="quadratic", labels=LV))
        per = [np.mean(np.abs(c[y == k] - k)) for k in LV]
        self.assertAlmostEqual(m["macro_mae"], float(np.mean(per)))
        self.assertAlmostEqual(m["balanced_acc"], float(np.mean([np.mean(c[y == k] == k) for k in LV])))
        self.assertAlmostEqual(m["mae"], float(np.mean(np.abs(c - y))))


if __name__ == "__main__":
    unittest.main()
