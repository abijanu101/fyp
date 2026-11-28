"""scripts/analysis/imbalance_variants.py: the share rules, the rank average, and the leakage guard."""

import unittest

import numpy as np
import pandas as pd

from scripts.analysis.imbalance_variants import cut, rank_average, shares_for

LV = np.array([2, 3, 4, 5])


def fake(seed: int, n: int = 400):
    rng = np.random.default_rng(seed)
    rows = []
    for g in ("group_01", "group_02", "group_03"):
        for lab in ("labeler_01", "labeler_03"):
            lat = rng.normal(size=n)
            y = LV[np.searchsorted(np.quantile(lat, [0.45, 0.88, 0.96]), lat)]
            for i in range(n):
                rows.append({"entry_id": f"{g}/e/s{i % 20}/{lab}/{i}", "grp": g, "cls": f"{g}/e{i % 6}",
                             "labeler": lab, "rating": float(y[i]), "pred": float(lat[i] + rng.normal(0, 1.5))})
    return pd.DataFrame(rows)


class TestImbalanceVariants(unittest.TestCase):
    def test_share_rules(self):
        base = np.array([0.45, 0.43, 0.08, 0.04])
        np.testing.assert_allclose(shares_for(base, "own"), base)
        np.testing.assert_allclose(shares_for(base, "equal"), [0.25] * 4)
        np.testing.assert_allclose(shares_for(base, "tempered"), (base + 0.25) / 2)
        self.assertAlmostEqual(shares_for(base, "tempered").sum(), 1.0)

    def test_tempered_and_equal_predict_the_rare_classes_more_often(self):
        d = fake(1)
        share5 = {m: np.mean(cut(d, LV, m) == 5) for m in ("own", "tempered", "equal")}
        self.assertLess(share5["own"], share5["tempered"])
        self.assertLess(share5["tempered"], share5["equal"])
        self.assertAlmostEqual(share5["equal"], 0.25, delta=0.02)

    def test_cut_points_use_no_test_labels(self):
        d = fake(2)
        a = cut(d, LV, "tempered")
        d2 = d.copy()
        t = d2["grp"] == "group_02"
        d2.loc[t, "rating"] = np.random.default_rng(3).permutation(d2.loc[t, "rating"].to_numpy())
        np.testing.assert_array_equal(a[t.to_numpy()], cut(d2, LV, "tempered")[t.to_numpy()])

    def test_rank_average_is_scale_free_and_aligned(self):
        d = fake(3)
        e = rank_average([d, d.assign(pred=d["pred"] * 100 + 7)])        # same order, another scale
        e = e.set_index("entry_id").loc[d["entry_id"]]
        r = d.groupby(["labeler", "grp"])["pred"].rank(pct=True).to_numpy()
        np.testing.assert_allclose(e["pred"].to_numpy(), r)


if __name__ == "__main__":
    unittest.main()
