"""Label agreement analysis (scripts/labels/label_agreement.py) on synthetic labelers."""

import unittest

import numpy as np

from scripts.labels.label_agreement import (LABELERS, Student, ceiling_rows, coarse, fast_alpha, others, stack_pairs,
                                     unit_values)
from src.evaluation import krippendorff_alpha_ordinal
from src.labels.interp import interpolate


def student(tracks: dict, t_last: float = 120.0, group: str = "group_01", exp: str = "01") -> Student:
    s = Student(group, exp, f"experiment_{exp}", "subject_01", 0.0, t_last,
                {k: (np.asarray(t, float), np.asarray(v, float)) for k, (t, v) in tracks.items()})
    s.start = min(t[0] for t, _ in s.tracks.values())
    return s


A = ([0.0, 20.0, 50.0, 80.0], [3.0, 2.0, 4.0, 3.0])


class TestAlpha(unittest.TestCase):
    def test_fast_alpha_matches_the_reference_implementation(self):
        rng = np.random.default_rng(0)
        v = rng.integers(1, 6, size=(300, 4)).astype(float)
        v[rng.random(v.shape) < 0.3] = np.nan
        v[:150, 1] = v[:150, 0]                                  # some agreement
        self.assertAlmostEqual(fast_alpha(v)[0], krippendorff_alpha_ordinal(v.tolist()), places=10)

    def test_identical_labelers_agree_perfectly(self):
        s = student({"labeler_01": A, "labeler_02": A})
        v = unit_values(s, "second", "step", 4.0, 0.5)
        self.assertAlmostEqual(fast_alpha(v)[0], 1.0)


class TestCeiling(unittest.TestCase):
    def test_identical_labelers_predict_each_other_exactly(self):
        df = ceiling_rows([student({"labeler_01": A, "labeler_02": A})])
        y, p, _, _ = stack_pairs(df[df["kind"] == "entry"], "step|0")
        np.testing.assert_array_equal(y, p)

    def test_a_late_labeler_is_recovered_by_the_matching_lag(self):
        late = ([0.0, 22.0, 52.0, 82.0], A[1])                   # the same ratings, 2 s later
        df = ceiling_rows([student({"labeler_01": A, "labeler_02": late})])
        ent = df[(df["kind"] == "entry") & (df["target"] == "labeler_01")]
        self.assertGreater(np.abs(ent["labeler_02|step|0"] - ent["y"]).sum(), 0)       # misses at lag 0
        np.testing.assert_array_equal(ent["labeler_02|step|2"], ent["y"])               # exact at +2 s
        np.testing.assert_array_equal(ent["labeler_02|tol2"], ent["y"])                 # and within +-2 s

    def test_run_midpoints_are_unaffected_by_a_small_shift(self):
        late = ([0.0, 22.0, 52.0, 82.0], A[1])
        df = ceiling_rows([student({"labeler_01": A, "labeler_02": late})])
        mid = df[(df["kind"] == "midpoint") & (df["target"] == "labeler_01")]
        np.testing.assert_array_equal(mid["labeler_02|step|0"], mid["y"])

    def test_gauss_puts_half_the_jump_at_the_entry(self):
        self.assertAlmostEqual(float(interpolate(np.array([0.0, 10.0]), np.array([2.0, 4.0]), 10.0, "gauss")), 3.0)

    def test_session_start_is_excluded_and_self_consistency_columns(self):
        df = ceiling_rows([student({"labeler_01": A, "labeler_02": A})])
        ent = df[(df["kind"] == "entry") & (df["target"] == "labeler_01")]
        np.testing.assert_array_equal(ent["t"], [20.0, 50.0, 80.0])                     # t = 0 is the start
        np.testing.assert_array_equal(ent["own_prev"], [3.0, 2.0, 4.0])
        # mean over the other runs, time weighted: the entry at 20 s (2 for 30 s) leaves 3x20, 4x30, 3x40
        self.assertAlmostEqual(ent["own_loro"].iloc[0], (3 * 20 + 4 * 30 + 3 * 40) / 90)

    def test_median_and_mean_of_other_experts_skip_the_target_and_self(self):
        s = student({"labeler_01": A, "labeler_02": ([0.0], [2.0]), "labeler_03": ([0.0], [4.0]),
                     "self_labeling": ([0.0], [5.0])})
        ent = ceiling_rows([s])
        ent = ent[(ent["kind"] == "entry") & (ent["target"] == "labeler_01")]
        np.testing.assert_array_equal(others(ent, "step|0", "mean"), [3.0, 3.0, 3.0])


class TestCoarse(unittest.TestCase):
    def test_coarse_means_agree_when_labelers_differ_only_in_timing(self):
        rng = np.random.default_rng(2)
        studs = []
        for i in range(30):
            base = float(rng.integers(2, 5))
            t = [0.0, 30.0, 60.0, 90.0]
            v = [base, base + 1 if base < 5 else base, base, base]
            late = [0.0, 33.0, 63.0, 93.0]
            s = student({"labeler_01": (t, v), "labeler_02": (late, v), "labeler_03": (t, v)}, t_last=300.0)
            s.sid = f"subject_{i:02d}"
            studs.append(s)
        res = coarse(studs)["student_x_activity"]["overall"]
        self.assertGreater(res["experts_pairwise"]["spearman"], 0.95)
        self.assertEqual(len(LABELERS), 6)


if __name__ == "__main__":
    unittest.main()
