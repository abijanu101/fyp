"""The posture signals, the ordinal model and the training-variants comparison, with known answers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

from scripts.clustering.posture import SIGNALS, posture, window_features
from scripts.clustering.training_variants import Ordinal, consensus_from_tracks, fit_predict
from src.labels.attention import AttentionTrack


def skeleton(scale: float = 1.0, **moves):
    """A seated person: shoulders level at y 200, elbows below, wrists at the desk, hips at y 330;
    face keypoints 0-4 placed above. moves: index -> (x, y) overrides."""
    xy = np.zeros((17, 2))
    xy[0], xy[1], xy[2], xy[3], xy[4] = (150, 120), (140, 110), (160, 110), (125, 115), (175, 115)
    xy[5], xy[6] = (200, 200), (100, 200)          # left shoulder on the image right, as COCO from the front
    xy[7], xy[8] = (215, 260), (85, 260)
    xy[9], xy[10] = (190, 300), (110, 300)
    xy[11], xy[12] = (180, 330), (120, 330)
    for k, v in moves.items():
        xy[int(k[1:])] = v
    return xy * scale, np.ones(17), np.array([60, 80, 240, 360.0]) * scale


class TestPosture(unittest.TestCase):
    def test_upright_seated_person(self):
        xy, c, box = skeleton()
        p = posture(xy, c, box)
        self.assertAlmostEqual(p["po.sho_tilt"], 0.0)
        self.assertAlmostEqual(p["po.sho_height"], (200 - 80) / 280)
        self.assertAlmostEqual(p["po.wrist_h"], -1.0)          # wrists 100 px below, shoulders 100 px wide
        self.assertEqual(p["po.hand_up"], 0.0)
        self.assertEqual(p["po.arms_crossed"], 0.0)
        self.assertAlmostEqual(p["po.torso"], 1.3)
        self.assertAlmostEqual(p["po.elbow_spread"], 1.3)
        self.assertEqual(p["po.visible"], 1.0)

    def test_hand_at_head_and_raised_hand(self):
        xy, c, box = skeleton(k9=(170, 150))                   # left wrist up, between the shoulders
        p = posture(xy, c, box)
        self.assertEqual(p["po.hand_up"], 1.0)
        self.assertEqual(p["po.hand_head"], 1.0)
        xy, c, box = skeleton(k9=(260, 150))                   # up, outside the shoulders: raised, not at the head
        p = posture(xy, c, box)
        self.assertEqual((p["po.hand_up"], p["po.hand_head"]), (1.0, 0.0))

    def test_arms_crossed(self):
        xy, c, box = skeleton(k9=(110, 270), k10=(190, 270))
        self.assertEqual(posture(xy, c, box)["po.arms_crossed"], 1.0)

    def test_scale_free(self):
        a = posture(*skeleton(1.0))
        b = posture(*skeleton(2.5))
        for k in SIGNALS:
            self.assertTrue(np.isclose(a[k], b[k], equal_nan=True), k)

    def test_face_keypoints_are_never_used(self):
        a = posture(*skeleton())
        xy, c, box = skeleton(k0=(0, 0), k1=(5, 5), k2=(9, 9), k3=(3, 3), k4=(7, 7))
        c[:5] = 0.0
        b = posture(xy, c, box)
        for k in SIGNALS:
            self.assertTrue(np.isclose(a[k], b[k], equal_nan=True), k)

    def test_no_shoulders_no_geometry(self):
        xy, c, box = skeleton()
        c[5] = 0.1
        p = posture(xy, c, box)
        self.assertTrue(np.isnan(p["po.wrist_h"]) and np.isnan(p["po.sho_height"]))
        self.assertEqual(p["po.matched"], 1.0)
        self.assertAlmostEqual(p["po.visible"], 7 / 8)

    def test_window_features(self):
        with tempfile.TemporaryDirectory() as d:
            t = np.arange(0, 10, 0.5)
            arr = {k: np.full(len(t), np.nan) for k in SIGNALS}
            arr["po.wrist_h"] = t.copy()                       # equals the time
            arr["po.hand_up"] = (t > 5).astype(float)
            np.savez(Path(d) / "s1.npz", tid=np.full(len(t), 7), t=t, **arr)
            rows = pd.DataFrame({"stem": ["s1", "s1", "s2"], "tid": [7, 8, 7], "t": [4.0, 4.0, 4.0]})
            f = window_features(Path(d), rows, 4.0)
            self.assertAlmostEqual(f.loc[0, "trk.po.wrist_h"], 4.0)            # mean of 2.0 ... 6.0
            self.assertAlmostEqual(f.loc[0, "trk.po.hand_up"], 2 / 9)          # 5.5 and 6.0 of 9 frames
            self.assertTrue(np.isnan(f.loc[1, "trk.po.wrist_h"]))              # another track
            self.assertTrue(np.isnan(f.loc[2, "trk.po.wrist_h"]))              # no cache for s2


class TestOrdinal(unittest.TestCase):
    def data(self, seed=0, n=3000):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 4))
        lat = X @ np.array([1.0, -0.5, 0.0, 0.3]) + rng.logistic(size=n)
        y = np.array([2, 3, 4, 5])[np.searchsorted([-1.0, 1.2, 2.5], lat)]
        return X, y

    def test_gradient_matches_numerical(self):
        X, y = self.data(n=300)
        Z = (X - X.mean(0)) / X.std(0)
        j = np.searchsorted(np.unique(y), y)
        rng = np.random.default_rng(3)
        for _ in range(3):
            p = np.concatenate([rng.normal(0, 0.5, 4), [rng.normal()], rng.normal(0, 0.5, 2)])
            loss, g = Ordinal.objective(p, Z, j, 4, 1e-2)
            num = approx_fprime(p, lambda q: Ordinal.objective(q, Z, j, 4, 1e-2)[0], 1e-7)
            np.testing.assert_allclose(g, num, rtol=1e-4, atol=1e-6)

    def test_fit_is_at_a_minimum_and_recovers_order(self):
        X, y = self.data()
        m = Ordinal(1e-3).fit(X, y)
        self.assertTrue(np.all(np.diff(m.theta) > 0))
        P = m.proba(X)
        np.testing.assert_allclose(P.sum(1), 1, atol=1e-9)
        from scipy.stats import spearmanr
        self.assertGreater(spearmanr(m.predict(X), y).statistic, 0.4)
        self.assertGreater(m.w[0], 0)
        self.assertLess(m.w[1], 0)


class TestComparison(unittest.TestCase):
    def frame(self, seed=1):
        rng = np.random.default_rng(seed)
        rows, X = [], []
        for g in ("group_01", "group_02", "group_03"):
            for e in range(4):
                for lab in ("labeler_01", "labeler_02"):
                    for i in range(60):
                        x = rng.normal(size=3)
                        rows.append({"entry_id": f"{g}/{e}/{lab}/{i}", "group": g, "cls": f"{g}/{e}", "labeler": lab,
                                     "aug": 0, "rating": float(np.clip(np.round(3 + x[0] + rng.normal(0, 0.7)), 2, 5))})
                        X.append(x)
        return pd.DataFrame(rows), np.array(X)

    def test_no_test_labels_reach_the_fit(self):
        r, X = self.frame()
        tr = (r["group"] != "group_02").to_numpy()
        te = ~tr
        for kind, grid in (("ridge", [1.0, 100.0]), ("ordinal", [1e-3, 1e-1])):
            y = r["rating"].to_numpy(float)
            a, _ = fit_predict(r, X, y, tr, te, kind, True, grid)
            y2 = y.copy()
            y2[te] = np.random.default_rng(5).permutation(y2[te])
            b, _ = fit_predict(r, X, y2, tr, te, kind, True, grid)
            np.testing.assert_allclose(a[te], b[te])

    def test_consensus(self):
        tracks = {("s", "labeler_01"): AttentionTrack(np.array([0.0, 10.0]), np.array([2.0, 4.0])),
                  ("s", "labeler_02"): AttentionTrack(np.array([5.0]), np.array([3.0])),
                  ("s", "self_labeling"): AttentionTrack(np.array([0.0]), np.array([5.0]))}
        ent = pd.DataFrame({"entry_id": ["a", "b", "c"], "sess": "s", "labeler": ["labeler_01", "labeler_02", "self_labeling"],
                            "t": [12.0, 6.0, 12.0]})
        v = consensus_from_tracks(ent, tracks)
        self.assertEqual(v, {"a": 3.5, "b": 2.5})              # all experts in force; the self-report not a target


if __name__ == "__main__":
    unittest.main()
