"""scripts/clustering/feature_similarity.py on synthetic data with a known answer. Each confound the
design guards against gets data where only that confound links features and ratings, and its
reference has to catch it."""

import unittest

import numpy as np
import pandas as pd

from scripts.clustering.feature_similarity import (_weighted_pearson, analyse_track, complete_rows, entries, in_force,
                                                    neighbours, rank_corr, standardise)
from src.labels.attention import AttentionTrack


def ar(n, phi, rng):
    x = np.zeros(n)
    e = rng.normal(0, np.sqrt(1 - phi ** 2), n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def fake(kind: str, seed: int, dims: int = 8, length_s: float = 400, jitter_s: float = 0.0):
    """3 groups x 4 activities x 6 students, 2 experts rating each window every 4 s. Ratings: class-
    session and student offsets plus noise, and:
      signal  + the first feature
      drift   features and an unrelated rating component both drift slowly in time within a session
      moment  features and an unrelated rating component both follow the class's moment in time
      class   features carry only the class session
    jitter_s: the second expert's entries moved by up to this many seconds."""
    rng = np.random.default_rng(seed)
    steps = np.arange(0, length_s, 4.0)
    rows, X = [], []
    for g in range(3):
        for e in range(4):
            c = f"g{g}/{e}"
            emb, c_off = rng.normal(0, 3, dims), rng.normal(0, 0.4)
            m_feat = np.stack([ar(len(steps), 0.98, rng) for _ in range(dims)], 1)   # the class's moment
            m_rate = ar(len(steps), 0.98, rng)
            for s in range(6):
                s_off = rng.normal(0, 0.4)
                d_feat = np.stack([ar(len(steps), 0.98, rng) for _ in range(dims)], 1)
                d_rate = ar(len(steps), 0.98, rng)
                for i, t in enumerate(steps):
                    x = rng.normal(0, 1, dims)
                    true = 3 + c_off + s_off
                    if kind == "signal":
                        true += 0.8 * x[0]
                    elif kind == "drift":
                        x, true = d_feat[i] + 0.2 * x, true + 0.8 * d_rate[i]
                    elif kind == "moment":
                        x, true = m_feat[i] + 0.3 * x, true + 0.8 * m_rate[i]
                    elif kind == "class":
                        x = emb + 0.3 * x
                    for li, lab in enumerate(("labeler_01", "labeler_02")):
                        tt = 1000 + t + (rng.uniform(-jitter_s, jitter_s) if li and jitter_s else 0.0)
                        rows.append({"rating": float(np.clip(np.round(true + rng.normal(0, 0.6)), 1, 5)),
                                     "sess": f"{c}/{s}", "cls": c, "grp": f"g{g}", "stem": f"{c}/cam", "t": tt,
                                     "tid": s, "labeler": lab, "track": "experts", "exp": e})
                        X.append(x)
    X = np.asarray(X, np.float32)
    return pd.DataFrame(rows), (X - X.mean(0)) / X.std(0)


def run(kind, seed=1, boot=0, learned=False, **kw):
    return analyse_track(*fake(kind, seed, **kw), k=5, gap_s=20, boot=boot, rng=np.random.default_rng(seed),
                         learned=learned)


def rho(res, key):
    return res["agreement"][key]["spearman"]


class TestFeatureSimilarity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = {k: run(k) for k in ("null", "signal", "drift", "moment")}

    def test_null_features_find_nothing_beyond_the_references(self):
        r = self.res["null"]
        for nn, ref in (("nn_other_group", "random_other_group"), ("nn_other_group_same", "random_other_group_same"),
                        ("nn_within_paired", "random_within_matched"), ("nn_classmate", "random_classmate_matched")):
            n = min(r["agreement"][nn]["n"], r["agreement"][ref]["n"])
            tol = max(0.05, 3 * np.sqrt(2 / n))                  # 3 SE of a difference of two correlations
            self.assertLess(abs(rho(r, nn) - rho(r, ref)), tol, (nn, n, rho(r, nn), rho(r, ref)))
        for b in (b for b in r["by_distance"] if b["n"] > 1000):        # small bins are noisy
            self.assertLess(abs(b["excess"]), 0.05, b)

    def test_rating_features_are_found(self):
        r = self.res["signal"]
        self.assertGreater(rho(r, "nn_other_group"), rho(r, "random_other_group") + 0.2)
        self.assertGreater(rho(r, "nn_within_paired"), rho(r, "random_within_matched") + 0.1)
        self.assertGreater(rho(r, "nn_other_group_same"), rho(r, "random_other_group_same") + 0.2)
        self.assertGreater(rho(r, "knn_other_group"), rho(r, "nn_other_group"))   # averaging k neighbours helps
        for b in (b for b in r["by_distance"] if b["n"] > 30):
            self.assertGreater(b["excess"], 0.05, b)

    def test_time_drift_is_caught_by_the_time_matched_reference(self):
        # features and ratings both drift in time, unrelated: the nearest in features is near in time,
        # so it agrees in rating; a random entry at the same time gap agrees as much
        r = self.res["drift"]
        self.assertLess(abs(rho(r, "nn_within_paired") - rho(r, "random_within_matched")), 0.05,
                        (rho(r, "nn_within_paired"), rho(r, "random_within_matched")))

    def test_shared_moment_is_caught_by_the_classmate_reference(self):
        # classmates share the class's moment in features and in rating, unrelated otherwise: the
        # nearest is a classmate at about the same time, and a random classmate then agrees as much
        r = self.res["moment"]
        self.assertGreater(r["nn_any_is_classmate"], 0.5)
        self.assertGreater(rho(r, "nn_classmate"), 0.1)                  # classmates do agree...
        self.assertLess(abs(rho(r, "nn_classmate") - rho(r, "random_classmate_matched")), 0.07,
                        (rho(r, "nn_classmate"), rho(r, "random_classmate_matched")))   # ...through the moment
        # (other groups are not checked here: both moment processes drift so slowly that one seed has
        # few independent samples per class; the null and class tests cover other groups)

    def test_class_only_features_carry_nothing_to_new_groups(self):
        # 12 class sessions a seed, so one seed's value is noisy (SD ~0.1): averaged over 20 seeds
        d = [rho(x, "nn_other_group") - rho(x, "random_other_group") for x in
             (run("class", s, length_s=60) for s in range(100, 120))]
        self.assertLess(abs(np.mean(d)), 0.06, d)

    def test_in_force_reads_the_other_experts_step_functions(self):
        # DIPSER ratings are change events: a value holds until the labeler's next one
        tracks = {("s", "labeler_01"): AttentionTrack(np.array([10.0, 20.0]), np.array([2.0, 4.0])),
                  ("s", "labeler_02"): AttentionTrack(np.array([15.0]), np.array([3.0])),
                  ("s", "self_labeling"): AttentionTrack(np.array([0.0]), np.array([5.0]))}
        r = pd.DataFrame({"sess": ["s", "s", "s", "s"], "t": [12.0, 25.0, 16.0, 30.0],
                          "labeler": ["labeler_02", "labeler_02", "labeler_01", "self_labeling"]})
        f = in_force(r, tracks)
        i, v = f["pairs"]
        self.assertEqual(list(zip(i.tolist(), v.tolist())), [(0, 2.0), (1, 4.0), (2, 3.0), (3, 4.0), (3, 3.0)])
        np.testing.assert_allclose(f["mean"], [2.0, 4.0, 3.0, 3.5])   # self-report: every expert, not itself
        r2 = pd.DataFrame({"sess": ["s"], "t": [5.0], "labeler": ["labeler_02"]})
        self.assertTrue(np.isnan(in_force(r2, tracks)["mean"][0]))     # before labeler_01's first rating

    def test_one_row_per_entry(self):
        rows = pd.DataFrame({"entry_id": ["a", "a", "b", "c", "c", "c", "d"], "aug": [0, 0, 0, 0, 1, 0, 0],
                             "group": "g", "exp": "1", "sid": "s1", "labeler": "labeler_01",
                             "stem": ["cam1", "cam2", "cam1", "cam1", "cam1", "cam3", "cam2"]})
        for seed in range(5):
            r = entries(rows, seed=seed).set_index("entry_id")
            self.assertEqual(sorted(r.index), ["a", "b", "c", "d"])       # each entry once, copies dropped
            # the student-session keeps to the camera that sees most of its entries (cam1: a, b, c);
            # d, which cam1 does not see, comes from cam2
            self.assertEqual(r["stem"].to_dict(), {"a": "cam1", "b": "cam1", "c": "cam1", "d": "cam2"})

    def test_learned_weights_do_not_leak(self):
        # the weights are fitted on the other groups: on random features they must find nothing
        d = [rho(x, "nn_other_group") - rho(x, "random_other_group") for x in
             (run("null", s, length_s=80, learned=True) for s in range(20, 30))]
        self.assertLess(abs(np.mean(d)), 0.03, d)

    def test_learned_weights_find_signal_in_one_column(self):
        # the rating lives in 1 of 30 columns: unweighted, the 29 others decide who is near; learned,
        # the informative one does
        plain = run("signal", 4, length_s=160, dims=30)
        learned = run("signal", 4, length_s=160, dims=30, learned=True)
        self.assertGreater(rho(learned, "nn_other_group"), rho(plain, "nn_other_group") + 0.1)
        self.assertGreater(rho(learned, "nn_within_paired"), rho(plain, "nn_within_paired"))
        self.assertIn("learned_weights", learned)

    def test_weighted_pearson_without_resampling_is_spearman(self):
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=200), rng.normal(size=200) + 0.5 * np.arange(200) / 200
        rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
        self.assertAlmostEqual(float(_weighted_pearson(rx, ry, np.ones((1, 200)))[0]), rank_corr(x, y), places=10)

    def test_intervals_cover_on_class_only_data(self):
        # only the class links features and ratings; the interval on nearest - random (other groups)
        # must contain 0 in about 95% of seeds (one class session per resample unit, 12 of them)
        hits = []
        for s in range(40, 60):
            x = run("class", s, boot=200, length_s=60)
            c = x["comparisons"]["nn_other_group - random_other_group"]
            hits.append(c["lo"] <= 0 <= c["hi"])
        self.assertGreaterEqual(np.mean(hits), 0.85, hits)

    def test_verdicts(self):
        key = "nn_other_group - random_other_group"
        self.assertEqual(run("signal", 5, boot=200, length_s=60)["comparisons"][key]["verdict"], "firm")
        v = [run("null", s, boot=200, length_s=60)["comparisons"][key]["verdict"] for s in range(70, 80)]
        self.assertGreaterEqual(v.count("no difference"), 8, v)
        self.assertEqual(v.count("firm"), 0, v)

    def test_missing_values(self):
        nan = np.nan
        a = np.array([[1, 2, nan], [2, nan, nan], [3, 4, nan], [4, 5, 6.0]])   # column 3 mostly missing
        b = np.array([[1.0], [2], [nan], [4]])
        # at 30%: column 3 of a is dropped (75% missing); row 2 lacks a's column 2 (25% missing), row 3 b's column
        self.assertTrue(complete_rows({"a": a, "b": b}, 0.05).all())   # at 5% every patchy column goes, no row does
        self.assertEqual(complete_rows({"a": a, "b": b}, 0.3).tolist(), [True, False, False, True])
        self.assertEqual(complete_rows({"a": a}, 0.3).tolist(), [True, False, True, True])
        X, kept, dropped = standardise(a[[0, 2, 3]], ["x", "y", "z"], 0.3)
        self.assertEqual(kept, ["x", "y"])
        self.assertEqual(dropped["often_missing"], ["z"])
        self.assertFalse(np.isnan(X).any())
        np.testing.assert_allclose(X.mean(0), 0, atol=1e-6)

    def test_neighbour_exclusions(self):
        df, X = fake("class", 3, length_s=120)
        keys = {c: df[c].to_numpy() for c in ("sess", "stem", "grp", "labeler")} | {"t": df["t"].to_numpy(float)}
        nb = neighbours(X, keys, 4, 20.0)
        sess, grp, lab, t = keys["sess"], keys["grp"], keys["labeler"], keys["t"]
        self.assertTrue((grp[nb["knn_i"]] != grp[:, None]).all())
        self.assertTrue((sess[nb["any_i"]] != sess).all())
        self.assertTrue((lab[nb["same_i"]] == lab).all() and (grp[nb["same_i"]] != grp).all())
        w = nb["within_i"]
        self.assertTrue((sess[w] == sess).all() and (np.abs(t[w] - t) >= 20).all())


if __name__ == "__main__":
    unittest.main()
