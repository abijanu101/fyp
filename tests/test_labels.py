import unittest

import numpy as np

from src.core.clock import parse_clock
from src.core.types import Window
from src.labels import WindowLabeler, parse_events

T0 = parse_clock("10:00:00:000000")


def events(*pairs):
    """pairs of (seconds after T0, attention or None for an emotion-only entry)."""
    out = []
    for s, a in pairs:
        e = {"datetime": f"10:00:{s:02d}:000000", "emotion": "1"}
        if a is not None:
            e["attention"] = str(a)
        out.append(e)
    return out


class TestAttentionTrack(unittest.TestCase):
    def test_clock(self):
        self.assertAlmostEqual(parse_clock("10:40:43:047895"), 10 * 3600 + 40 * 60 + 43.047895)
        self.assertAlmostEqual(parse_clock("10_40_43_4"), 10 * 3600 + 40 * 60 + 43.4)

    def test_emotion_only_entries_do_not_change_attention(self):
        tr = parse_events(events((0, 3), (5, None), (10, 5)))
        self.assertEqual(tr.value.tolist(), [3, 5])

    def test_time_weighted_mean(self):
        tr = parse_events(events((0, 2), (4, 4)))
        self.assertAlmostEqual(tr.mean_over(T0, T0 + 8), 3.0)       # 4 s at 2, 4 s at 4
        self.assertAlmostEqual(tr.mean_over(T0 + 6, T0 + 10), 4.0)  # last value holds

    def test_unlabeled_before_first_entry(self):
        tr = parse_events(events((5, 3)))
        self.assertTrue(np.isnan(tr.mean_over(T0, T0 + 8)))

    def test_out_of_range_values_dropped(self):
        tr = parse_events(events((0, 3), (5, 7), (10, 2)))
        self.assertEqual(tr.value.tolist(), [3, 2])

    def test_offset(self):
        tr = parse_events(events((0, 3)), offset_s=2.0)
        self.assertAlmostEqual(tr.t[0], T0 + 2.0)


class TestWindowLabeler(unittest.TestCase):
    def student(self, *labeler_values):
        return {f"labeler_{i:02d}": parse_events(events((0, v))) for i, v in enumerate(labeler_values)}

    def test_median_consensus_and_share(self):
        students = {"s1": self.student(5, 4, 1),   # median 4 -> engaged
                    "s2": self.student(2, 2, 5),   # median 2 -> not engaged
                    "s3": self.student(4, 4, 4)}   # engaged
        lab = WindowLabeler(tau_att=4, tau_share=0.5, min_labeled_students=3).label(students, Window(T0, T0 + 8))
        self.assertEqual((lab.engaged, lab.n_labeled), (1, 3))
        self.assertAlmostEqual(lab.share_engaged, 2 / 3)

    def test_too_few_students_dropped(self):
        lab = WindowLabeler(min_labeled_students=3).label({"s1": self.student(5)}, Window(T0, T0 + 8))
        self.assertIsNone(lab.engaged)
        self.assertEqual(lab.n_labeled, 1)

    def test_exclude_self(self):
        s = {"labeler_01": parse_events(events((0, 2))), "self_labeling": parse_events(events((0, 5)))}
        w = Window(T0, T0 + 8)
        self.assertAlmostEqual(WindowLabeler(exclude_self=False).consensus(s, w), 3.5)
        self.assertAlmostEqual(WindowLabeler(exclude_self=True).consensus(s, w), 2.0)


if __name__ == "__main__":
    unittest.main()
