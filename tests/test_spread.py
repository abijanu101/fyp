import unittest

import numpy as np

from src.cfmu import spread_decomposition


class TestSpread(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(20, 4))
        self.labels = rng.integers(0, 3, size=20)

    def test_within_plus_between_equals_total_population_variance(self):
        s = spread_decomposition(self.x, self.labels)
        np.testing.assert_allclose(s["total_sd"] ** 2, self.x.var(axis=0), rtol=1e-10)
        np.testing.assert_allclose(s["within_sd"] ** 2 + s["between_sd"] ** 2, s["total_sd"] ** 2)

    def test_single_cluster_has_no_between_spread(self):
        s = spread_decomposition(self.x, np.zeros(20), blocks={"all": [0, 1, 2, 3]})
        np.testing.assert_allclose(s["between_sd"], 0)
        self.assertEqual(s["between_share"]["all"], 0.0)

    def test_singletons_have_no_within_spread(self):
        s = spread_decomposition(self.x, np.arange(20), blocks={"a": [0, 1]})
        np.testing.assert_allclose(s["within_sd"], 0, atol=1e-12)
        self.assertAlmostEqual(s["between_share"]["a"], 1.0)

    def test_zero_total_spread_gives_zero_share(self):
        s = spread_decomposition(np.ones((5, 2)), np.array([0, 0, 1, 1, 2]), blocks={"b": [0, 1]})
        self.assertEqual(s["between_share"]["b"], 0.0)


if __name__ == "__main__":
    unittest.main()
