"""scripts/clustering/seat_check.py on synthetic classes with a known answer (18 class sessions of
15 students, 3 front cameras, 3 experts and the self-report, as in DIPSER's scope)."""

import unittest

import numpy as np
import pandas as pd

from scripts.clustering.seat_check import (agreement_with_and_without_seat, features_against_seat, joined,
                                           level_table, model_against_seat, neighbour_agreement, neighbour_pairs,
                                           pct_rank, seat_effect, seat_table)
from src.labels.attention import AttentionTrack

LABS = ["labeler_01", "labeler_02", "labeler_03", "self_labeling"]


def fake(kind: str, seed: int, n_cls: int = 18, n_stu: int = 15):
    """Boxes from each student's true nearness to the front r in [0, 1] (bigger and lower when nearer,
    each class with its own camera zoom); levels per (labeler, student) = class offset + student
    trait shared by every labeler + labeler offset + noise, and:
      planted   + 0.8 r for the experts only (an observer effect)
      zoom      classes with a larger zoom also rate higher (a between-class confound only)
      seat_agree / trait_agree  the experts agree only through r, or only through a trait unrelated to r"""
    rng = np.random.default_rng(seed)
    boxes, rows = {}, []
    for c in range(n_cls):
        cls = f"group_0{c % 3 + 1}/experiment_0{c // 3 + 1}"
        zoom = rng.uniform(0.6, 1.6)
        c_off = rng.normal(0, 0.3) + (0.8 * (zoom - 1.1) if kind == "zoom" else 0)
        r = rng.uniform(0, 1, n_stu)
        trait = rng.normal(0, 0.5, n_stu)
        for cam in ("context_cam_01", "context_cam_02", "context_cam_03"):
            b = {}
            for i in range(n_stu):
                w = zoom * (60 + 80 * r[i]) * rng.uniform(0.9, 1.1)
                x0, y1 = rng.uniform(0, 1100), 300 + 300 * r[i] + rng.normal(0, 15)
                b[f"subject_{i + 1:02d}"] = np.array([x0, y1 - w, x0 + w, y1])
            boxes[(cls, cam)] = b
        for lab in LABS:
            l_off = rng.normal(0, 0.3)
            for i in range(n_stu):
                v = 3 + c_off + l_off + rng.normal(0, 0.3)
                if kind == "seat_agree":
                    v += (r[i] if lab != "self_labeling" else 0)
                elif kind == "trait_agree":
                    v += trait[i]
                else:
                    v += trait[i]                         # a strong student trait, shared by every labeler
                    if kind == "planted" and lab != "self_labeling":
                        v += 0.8 * r[i]
                rows.append({"cls": cls, "sid": f"subject_{i + 1:02d}", "labeler": lab, "level": v,
                             "entries_mean": v + rng.normal(0, 0.1), "n_entries": 10})
    return boxes, pd.DataFrame(rows)


def effect(kind, seed, track="experts", perms=300, measure="front"):
    boxes, levels = fake(kind, seed)
    _, seat = seat_table(boxes)
    lv = levels[(levels["labeler"] != "self_labeling") if track == "experts" else (levels["labeler"] == "self_labeling")]
    return seat_effect(joined(lv, seat), measure, perms, 0, np.random.default_rng(seed))


class TestSeatCheck(unittest.TestCase):
    def test_ranks_and_seat_table(self):
        np.testing.assert_allclose(pct_rank(np.array([3.0, 1.0, 2.0])), [1.0, 0.0, 0.5])
        boxes = {("c", "cam1"): {"a": np.array([0, 0, 10, 10.0]), "b": np.array([0, 0, 30, 30.0]),
                                 "c": np.array([0, 0, 20, 20.0])},
                 ("c", "cam2"): {"a": np.array([0, 0, 10, 10.0]), "b": np.array([0, 0, 20, 20.0]),
                                 "c": np.array([0, 0, 30, 30.0])}}
        per_cam, seat = seat_table(boxes)
        s = seat.set_index("sid")
        self.assertEqual(s.loc["a", "front"], 0.0)                      # smallest in both cameras
        self.assertEqual(s.loc["b", "front"], 0.75)                     # largest in one, middle in the other
        self.assertEqual(int(s.loc["a", "n_cameras"]), 2)
        self.assertEqual(len(per_cam), 6)

    def test_level_is_time_weighted_over_the_class_session(self):
        tr = {("c", "a", "labeler_01"): AttentionTrack(np.array([0.0, 10.0]), np.array([2.0, 4.0])),
              ("c", "b", "labeler_01"): AttentionTrack(np.array([0.0, 20.0]), np.array([3.0, 3.0]))}
        lv = level_table(tr).set_index("sid")
        self.assertAlmostEqual(lv.loc["a", "level"], 3.0)               # 2 for 10 s, 4 for 10 s of 0..20
        self.assertAlmostEqual(lv.loc["a", "entries_mean"], 4.0)        # the session-start entry excluded
        self.assertEqual(int(lv.loc["a", "n_entries"]), 1)

    def test_no_false_positives_with_strong_shared_student_traits(self):
        # every labeler shares a strong student trait, unrelated to seat: permuting rows rather than
        # students would make these 'significant' far too often
        p = [effect("null", s)["p_perm"] for s in range(40)]
        self.assertLessEqual(np.mean(np.array(p) < 0.05), 0.125, p)

    def test_planted_observer_effect_is_found_for_experts_not_self(self):
        e, s = effect("planted", 1), effect("planted", 1, track="self")
        self.assertGreater(e["spearman"], 0.2)
        self.assertLess(e["p_perm"], 0.01)
        self.assertGreater(e["front_minus_back_points"], 0.2)
        self.assertGreater(s["p_perm"], 0.01)

    def test_class_zoom_and_level_do_not_create_an_effect(self):
        p = [effect("zoom", s)["p_perm"] for s in range(20)]
        self.assertLessEqual(np.mean(np.array(p) < 0.05), 0.15, p)

    def test_share_of_expert_agreement_explained_by_seat(self):
        boxes, lv = fake("seat_agree", 3)
        a = agreement_with_and_without_seat(lv, seat_table(boxes)[1])
        self.assertGreater(a["agreement"], 0.3)
        self.assertGreater(a["share_explained_by_seat"], 0.7)
        boxes, lv = fake("trait_agree", 3)
        a = agreement_with_and_without_seat(lv, seat_table(boxes)[1])
        self.assertGreater(a["agreement"], 0.3)
        self.assertLess(abs(a["share_explained_by_seat"]), 0.15)

    def test_neighbour_agreement(self):
        def spatial(seed, smooth):
            rng = np.random.default_rng(seed)
            boxes, rows = {}, []
            for c in range(18):
                cls = f"g/{c}"
                xy = rng.uniform(0, 1000, (15, 2))                # the same seat in every camera
                for cam in ("c1", "c2", "c3"):
                    j = xy + rng.normal(0, 20, xy.shape)
                    boxes[(cls, cam)] = {f"s{i:02d}": np.array([j[i, 0], j[i, 1], j[i, 0] + 80, j[i, 1] + 80])
                                         for i in range(15)}
                for lab in ("labeler_01", "labeler_02"):
                    for i in range(15):
                        v = (np.sin(xy[i, 0] / 200) + np.cos(xy[i, 1] / 200)) if smooth else 0.0
                        rows.append({"cls": cls, "sid": f"s{i:02d}", "labeler": lab,
                                     "level": 3 + 0.6 * v + rng.normal(0, 0.5)})
            return boxes, pd.DataFrame(rows)
        b, lv = spatial(1, True)
        h = neighbour_agreement(b, lv, 200, np.random.default_rng(0))
        self.assertGreater(h["excess"], 0.15)
        self.assertLess(h["p_perm"], 0.01)
        p = [neighbour_agreement(*spatial(s, False), 100, np.random.default_rng(s))["p_perm"] for s in range(20)]
        self.assertLessEqual(np.mean(np.array(p) < 0.05), 0.15, p)
        pairs = neighbour_pairs(b)
        self.assertTrue((pairs["a"] != pairs["b"]).all())

    def test_models_and_features_are_read_in_the_real_formats(self):
        n = 12
        sids = [f"subject_{i:02d}" for i in range(1, n + 1)]
        seat = pd.DataFrame({"cls": "group_01/experiment_03", "sid": sids, "front": np.linspace(0, 1, n),
                             "low": np.linspace(0, 1, n), "central": 0.5, "n_cameras": 3})
        pred = pd.DataFrame({"entry_id": [f"group_01/experiment_03/{sid}/labeler_01/{t}" for sid in sids for t in range(4)],
                             "split": "group", "arm": "base", "track": "experts", "rating": 3.0})
        pred["pred"] = pred["entry_id"].str.split("/").str[2].str[-2:].astype(float)   # rises with the student
        m = model_against_seat(pred, seat)
        self.assertAlmostEqual(m["base|experts"]["front"], 1.0)
        rows = pd.DataFrame({"aug": 0, "group": "group_01", "session": "group_01/experiment_03/context_cam_01",
                             "sid": [sid for sid in sids for _ in range(3)],
                             "trk.rf.mag_00": [float(i) for i in range(n) for _ in range(3)]})
        f = features_against_seat(rows, seat, ["trk.rf.mag_00"])
        self.assertAlmostEqual(f["trk.rf.mag_00"], 1.0)

if __name__ == "__main__":
    unittest.main()
