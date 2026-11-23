import unittest

import numpy as np

from src.cfmu.classrel import (PrototypeCodebook, agglomerative_labels, circular_shift, cluster_features,
                               correlation_distance, deviation_from_class, hdbscan_labels, lag_behind_class,
                               laplacian_score, noise_ratio, pairwise_corr, sync_features, sync_with_class)


def class_series(n=10, T=40, seed=0):
    """A shared class signal plus small per-student noise: every student moves with the class."""
    rng = np.random.default_rng(seed)
    base = np.sin(np.linspace(0, 4 * np.pi, T)) + 0.3 * rng.normal(size=T)
    return np.stack([base + 0.1 * rng.normal(size=T) for _ in range(n)]), base


class TestSynchrony(unittest.TestCase):
    def test_every_key_present_for_any_class_size(self):
        for n in (0, 1, 2, 3, 20):
            S = np.random.default_rng(n).normal(size=(n, 20))
            per, win = sync_features(S, max_lag=3)
            self.assertEqual(set(per), {"corr", "lag", "lag_corr", "degree"})
            self.assertEqual(set(win), {"mean_corr", "share_synced"})
            for v in per.values():
                self.assertEqual(len(v), n)
            if n < 3:
                self.assertTrue(all(np.isnan(v).all() for v in per.values()))
                self.assertTrue(all(np.isnan(v) for v in win.values()))

    def test_permutation_invariance(self):
        S, _ = class_series()
        S[2] = np.random.default_rng(5).normal(size=S.shape[1])
        perm = np.random.default_rng(1).permutation(len(S))
        a, wa = sync_features(S, max_lag=3)
        b, wb = sync_features(S[perm], max_lag=3)
        for k in a:
            np.testing.assert_allclose(a[k][perm], b[k], equal_nan=True)
        for k in wa:
            self.assertAlmostEqual(wa[k], wb[k])

    def test_student_shifted_by_k_frames_gets_lag_k(self):
        S, base = class_series(n=8, T=60)
        S[0] = np.r_[np.full(3, np.nan), base[:-3]]          # follows the class 3 frames late
        lag, peak = lag_behind_class(S, max_lag=5)
        self.assertEqual(lag[0], 3)
        self.assertGreater(peak[0], 0.9)
        self.assertTrue(np.all(np.abs(lag[1:]) <= 1))

    def test_synchronous_student_beats_noise_student(self):
        S, _ = class_series()
        S[0] = np.random.default_rng(9).normal(size=S.shape[1])
        sync = sync_with_class(S)
        self.assertLess(sync[0], 0.5)
        self.assertGreater(np.nanmin(sync[1:]), 0.9)

    def test_too_few_common_frames_gives_nan(self):
        S = np.random.default_rng(0).normal(size=(4, 20))
        S[0, 5:] = np.nan                                   # 5 frames, below MIN_COMMON
        C = pairwise_corr(S)
        self.assertTrue(np.isnan(C[0, 1:]).all())
        self.assertTrue(np.isnan(sync_with_class(S)[0]))

    def test_circular_shift_keeps_each_series_values(self):
        S, _ = class_series()
        R = circular_shift(S, np.random.default_rng(0))
        for a, b in zip(S, R):
            np.testing.assert_allclose(np.sort(a), np.sort(b))
        self.assertLess(np.nanmean(pairwise_corr(R)), np.nanmean(pairwise_corr(S)))


class TestDescriptorSpace(unittest.TestCase):
    def test_outlier_has_the_largest_deviation(self):
        X = np.random.default_rng(0).normal(scale=0.1, size=(12, 5))
        X[4] += 3
        dev = deviation_from_class(X)
        self.assertEqual(int(np.argmax(dev)), 4)
        self.assertTrue(np.isnan(deviation_from_class(X[:2])).all())

    def test_two_separated_groups_form_two_clusters(self):
        rng = np.random.default_rng(0)
        X = np.r_[rng.normal(0, 0.05, size=(6, 4)), rng.normal(3, 0.05, size=(6, 4))]
        labels = agglomerative_labels(1.0, X=X)
        self.assertEqual(len(np.unique(labels)), 2)
        per, win = cluster_features(labels, X=X)
        self.assertAlmostEqual(win["largest_share"], 0.5)
        self.assertGreater(win["fragmentation"], 0.95)
        np.testing.assert_allclose(per["share"], 0.5)

    def test_identical_students_have_zero_chaos(self):
        X = np.ones((8, 3))
        labels = agglomerative_labels(0.5, X=X)
        per, win = cluster_features(labels, X=X)
        self.assertEqual(win["n_clusters"], 1 / 8)
        self.assertAlmostEqual(win["chaos"], 0.0)
        np.testing.assert_allclose(per["centroid_dist"], 0.0)

    def test_cluster_features_nan_below_four_and_id_free(self):
        per, win = cluster_features(np.array([0, 0, 1]), X=np.zeros((3, 2)))
        self.assertTrue(all(np.isnan(v).all() for v in per.values()))
        a, wa = cluster_features(np.array([0, 0, 0, 1, 1]), X=np.arange(10.).reshape(5, 2))
        b, wb = cluster_features(np.array([7, 7, 7, 2, 2]), X=np.arange(10.).reshape(5, 2))
        for k in a:
            np.testing.assert_allclose(a[k], b[k])

    def test_synchrony_space_clustering_and_hdbscan_noise(self):
        S, _ = class_series(n=10)
        S[0] = np.random.default_rng(3).normal(size=S.shape[1]) * 3
        D = correlation_distance(pairwise_corr(S))
        labels = agglomerative_labels(0.5, D=D)
        per, _ = cluster_features(labels, D=D)
        self.assertEqual(per["singleton"][0], 1.0)
        self.assertEqual(per["in_largest"][1:].min(), 1.0)
        lab, strength = hdbscan_labels(D=D, min_cluster_size=3)
        self.assertEqual(len(lab), 10)
        self.assertEqual(len(strength), 10)


class TestPrototypes(unittest.TestCase):
    def test_conformity_follows_the_majority(self):
        rng = np.random.default_rng(0)
        pool = np.r_[rng.normal(0, 0.1, size=(200, 3)), rng.normal(4, 0.1, size=(200, 3))]
        cb = PrototypeCodebook(k=2, seed=0).fit(pool)
        window = np.r_[rng.normal(0, 0.1, size=(9, 3)), rng.normal(4, 0.1, size=(1, 3))]
        per, win = cb.window_features(window)
        self.assertLess(per["conformity"][9], 0.2)
        self.assertGreater(per["conformity"][:9].min(), 0.8)
        self.assertAlmostEqual(per["hard_share"][0], 8 / 9)
        self.assertEqual(set(win), {"entropy", "proto_0", "proto_1"})
        perm = rng.permutation(10)
        per2, win2 = cb.window_features(window[perm])
        np.testing.assert_allclose(per["conformity"][perm], per2["conformity"])
        self.assertAlmostEqual(win["entropy"], win2["entropy"])


class TestScreens(unittest.TestCase):
    def test_noise_ratio_separates_noise_from_slow_signal(self):
        rng = np.random.default_rng(0)
        self.assertLess(noise_ratio(rng.normal(size=400)), 1.3)
        self.assertGreater(noise_ratio(np.sin(np.linspace(0, 2 * np.pi, 40)) + 0.01 * rng.normal(size=40)), 5)

    def test_laplacian_score_prefers_the_structured_feature(self):
        rng = np.random.default_rng(0)
        g = np.repeat([0.0, 5.0], 50)
        X = np.c_[g + 0.1 * rng.normal(size=100), rng.normal(size=100)]
        s = laplacian_score(X, k=5)
        self.assertLess(s[0], s[1])


if __name__ == "__main__":
    unittest.main()
