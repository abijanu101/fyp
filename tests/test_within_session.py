"""scripts/clustering/within_session.py on synthetic data with a known answer: the checks its design
promises (no signal from shuffled features, no use of test labels, drift removed by detrending, a
planted within-session effect found, a between-student effect not mistaken for one)."""

import unittest

import numpy as np
import pandas as pd

from scripts.clustering.within_session import (_cumsums, detrended, evaluate, prepare, primary_verdict, run_split,
                                               within_score)

CFG = {"labels": {}, "seed": 0}


def fake(kind: str, seed: int, dims: int = 6, length_s: float = 240, step_s: float = 6.0):
    """3 groups x 3 activities x 6 students; 2 experts and the self-report rate every step_s s; one
    augmented copy of every row (features + small noise). Ratings: student offset + noise, and:
      within   + a latent that varies within the session, which feature 0 measures
      between  feature 1 measures the student offset only
      drift    the latent is a linear trend in time (per-session slope), seen by feature 0
      null     features are unrelated to the ratings"""
    rng = np.random.default_rng(seed)
    steps = np.arange(0, length_s, step_s)
    rows, X = [], []
    for g in range(3):
        for e in range(3):
            for s in range(6):
                s_off = rng.normal(0, 0.5)
                if kind == "drift":
                    lat = rng.normal(0, 1.5) * (steps - steps.mean()) / np.ptp(steps)
                else:
                    lat = np.zeros(len(steps))
                    for i in range(1, len(steps)):
                        lat[i] = 0.9 * lat[i - 1] + rng.normal(0, 0.45)
                for lab in ("labeler_01", "labeler_02", "self_labeling"):
                    for i, t in enumerate(steps):
                        x = rng.normal(0, 1, dims)
                        if kind in ("within", "drift"):
                            x[0] = lat[i] + 0.3 * rng.normal()
                        if kind in ("within", "between"):
                            x[1] = s_off + 0.1 * rng.normal()
                        true = 3 + s_off + (0.8 * lat[i] if kind in ("within", "drift") else 0)
                        rating = float(np.clip(np.round(true + rng.normal(0, 0.5)), 1, 5))
                        base = {"group": f"group_0{g + 1}", "exp": f"0{e + 1}", "sid": f"subject_{s + 1:02d}",
                                "labeler": lab, "t": 1000.0 + t, "rating": rating, "tid": s,
                                "entry_id": f"g{g}/e{e}/s{s}/{lab}/{t}", "stem": f"g{g}e{e}",
                                "session": f"g{g}/e{e}/cam"}
                        rows.append({**base, "aug": 0})
                        X.append(x)
                        rows.append({**base, "aug": 1})
                        X.append(x + 0.05 * rng.normal(size=dims))
    return prepare(pd.DataFrame(rows)), np.asarray(X)


def arms_for(X):
    return {"W0p": X, "W1": X, "T": np.column_stack([np.zeros(len(X)), np.zeros(len(X))])}


def score(kind, seed=1, track="experts", boot=0, shuffle=False):
    r, X = fake(kind, seed)
    if shuffle:                                       # features shuffled within each student-session
        X = X.copy()
        for _, i in r.groupby("sess").indices.items():
            X[i] = X[np.random.default_rng(seed).permutation(i)]
    inputs = {"W0p": X, "W1": X}
    pred, _ = run_split(CFG, r, inputs, "group", track, log=lambda *a: None)
    return r, pred, evaluate(pred, np.random.default_rng(0), boot)


class TestWithinSession(unittest.TestCase):
    def test_targets_and_eligibility(self):
        rows = pd.DataFrame({"group": "g", "exp": "1", "sid": "s", "labeler": ["labeler_01"] * 4 + ["labeler_02"] * 2,
                             "rating": [2, 4, 3, 3, 3, 4.0], "t": range(6), "aug": 0,
                             "entry_id": list("abcdef"), "tid": 1, "stem": "x", "session": "x"})
        r = prepare(rows)
        np.testing.assert_allclose(r["deviation"].to_numpy()[:4], [-1, 1, 0, 0])
        self.assertTrue(r["eligible"].iloc[0])               # 4 entries, 3 distinct
        self.assertFalse(r["eligible"].iloc[4])              # 2 entries: too few

    def test_planted_within_session_effect_is_found(self):
        _, _, res = score("within")
        a = res["arms"]
        self.assertGreater(a["W1"]["within"], 0.25)
        self.assertGreater(a["W1"]["detrended"], 0.2)

    def test_shuffled_features_score_nothing(self):
        _, _, res = score("within", shuffle=True)
        for arm in ("W0p", "W1"):
            self.assertLess(abs(res["arms"][arm]["within"]), 0.05, (arm, res["arms"][arm]["within"]))

    def test_null_features_score_nothing(self):
        _, _, res = score("null")
        for arm in ("W0p", "W1"):
            self.assertLess(abs(res["arms"][arm]["within"]), 0.05)

    def test_between_student_effect_is_not_within_session_skill(self):
        # feature 1 tells the students apart and so predicts absolute ratings; within a session it
        # is constant up to noise, so it must not score within sessions
        _, _, res = score("between")
        self.assertLess(abs(res["arms"]["W1"]["within"]), 0.05)
        self.assertLess(abs(res["arms"]["W0p"]["within"]), 0.05)

    def test_drift_is_removed_by_detrending(self):
        _, _, res = score("drift")
        self.assertGreater(res["arms"]["W1"]["within"], 0.2)              # looks like skill...
        self.assertLess(abs(res["arms"]["W1"]["detrended"]), 0.05)        # ...but it is the trend

    def test_test_labels_never_reach_the_fit(self):
        r, X = fake("within", 3)
        a, _ = run_split(CFG, r, {"W1": X}, "group", "experts", log=lambda *x: None)
        r2 = r.copy()
        test = (r2["group"] == "group_02").to_numpy()
        rng = np.random.default_rng(9)
        r2.loc[test, "rating"] = rng.permutation(r2.loc[test, "rating"].to_numpy())
        r2 = prepare(r2.drop(columns=["ls_mean", "deviation", "eligible"]))
        # eligibility of the test sessions may change; the training rows (and so the model) may not
        r2.loc[~test, "eligible"] = r.loc[~test, "eligible"]
        b, _ = run_split(CFG, r2, {"W1": X}, "group", "experts", log=lambda *x: None)
        m = a.merge(b, on="entry_id", suffixes=("_a", "_b"))
        g2 = m["entry_id"].str.startswith("g1/")                  # group_02's entries (index 1)
        self.assertGreater(g2.sum(), 50)
        np.testing.assert_allclose(m.loc[g2, "pred_a"], m.loc[g2, "pred_b"], rtol=1e-10, atol=1e-10)

    def test_primary_verdict_and_intervals(self):
        _, _, res = score("within", boot=200)
        self.assertIn("W1 - W0p", res["comparisons"])
        w = res["comparisons"]["W1 - W0p"]["within"]
        self.assertTrue("lo" in w and w["lo"] <= w["diff"] <= w["hi"])
        v = primary_verdict(res, "experts", ["labeler_01", "labeler_02"])
        self.assertIn(v["verdict"].split(" ")[0], ("supported", "not"))

    def test_detrended_and_cumsums(self):
        v = pd.Series([1.0, 2.0, 3.0, 10.0, 10.0, 10.0])
        t = pd.Series([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        k = pd.Series(["a", "a", "a", "b", "b", "b"])
        np.testing.assert_allclose(detrended(v, t, k), 0, atol=1e-12)
        S1, S2, N = _cumsums(np.array([[1.0, np.nan], [3.0, 2.0]]))
        np.testing.assert_allclose(S1[-1], [4.0, 2.0])
        np.testing.assert_allclose(N[-1], [2, 1])

    def test_within_score_ignores_session_levels(self):
        e = pd.DataFrame({"pred": [10, 11, 0, 1.0], "rating": [1, 2, 4, 5.0], "ls": ["a", "a", "b", "b"]})
        self.assertAlmostEqual(within_score(e), 1.0)          # pooled it would be negative


if __name__ == "__main__":
    unittest.main()
