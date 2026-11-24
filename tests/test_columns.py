"""Model input columns (engagement.columns, engagement.class_relative; src/experts/columns.py)."""

import unittest

import numpy as np
import pandas as pd

from src.experts.columns import REL_COLUMNS, add_class_relative, model_columns

AVAILABLE = ["tid", "trk.kg.speed", "trk.rf.mag_00", "trk.hp.yaw.mean", "trk.hp.head_height.mean", *REL_COLUMNS]
VEC = ["vec.kg.speed.mean", "vec.hp.yaw.mean.mean", "vec.n_tracks"]


def cfg(**eng):
    return {"engagement": {"use_class_context": False, **eng}}


class TestModelColumns(unittest.TestCase):
    def test_default_is_every_track_column(self):
        self.assertEqual(model_columns(cfg(), AVAILABLE, VEC),
                         ["trk.hp.head_height.mean", "trk.hp.yaw.mean", "trk.kg.speed", "trk.rf.mag_00"])
        self.assertEqual(model_columns(cfg(use_class_context=True), AVAILABLE, VEC)[-3:], VEC)

    def test_relative_and_filters(self):
        self.assertEqual(model_columns(cfg(class_relative=True), AVAILABLE, VEC)[-4:], list(REL_COLUMNS))
        base = model_columns(cfg(class_relative=True, use_class_context=True,
                                 columns={"exclude": ["trk.hp.", "vec.hp.", "rel."]}), AVAILABLE, VEC)
        self.assertEqual(base, ["trk.kg.speed", "trk.rf.mag_00", "vec.kg.speed.mean", "vec.n_tracks"])
        alone = model_columns(cfg(use_class_context=True, columns={"include": ["trk.hp.", "vec.hp."]}), AVAILABLE, VEC)
        self.assertEqual(alone, ["trk.hp.head_height.mean", "trk.hp.yaw.mean", "vec.hp.yaw.mean.mean"])
        with self.assertRaises(ValueError):
            model_columns(cfg(columns={"include": ["nothing."]}), AVAILABLE, VEC)


class TestClassRelative(unittest.TestCase):
    def test_against_the_windows_other_students(self):
        rows = pd.DataFrame({"window": [0, 0, 0, 0, 1, 1],
                             "trk.hp.yaw.mean": [0.4, 0.2, -0.3, np.nan, 0.1, np.nan],
                             "trk.hp.head_height.mean": [0.5, 0.3, 0.1, 0.2, 0.0, 0.4]})
        out = add_class_relative(rows, "window")
        np.testing.assert_allclose(out["rel.hp.yaw_dev"][:3], [0.2, 0.0, -0.5])        # median of window 0: 0.2
        np.testing.assert_allclose(out["rel.hp.yaw_absdev"][:3], [0.2, 0.0, 0.5])
        np.testing.assert_allclose(out["rel.hp.height_dev"][:4], [0.25, 0.05, -0.15, -0.05])  # median 0.25
        # same sign among the *other* students with a yaw: +, +, - in window 0
        np.testing.assert_allclose(out["rel.hp.same_dir"][:3], [0.5, 0.5, 0.0])
        self.assertTrue(np.isnan(out["rel.hp.same_dir"][3]))                           # no yaw of its own
        self.assertTrue(np.isnan(out["rel.hp.same_dir"][4]))                           # nobody else with a yaw
        self.assertEqual(float(out["rel.hp.yaw_dev"][4]), 0.0)

    def test_without_head_pose_columns(self):
        out = add_class_relative(pd.DataFrame({"window": [0, 0], "trk.kg.speed": [1.0, 2.0]}), "window")
        self.assertTrue(out[list(REL_COLUMNS)].isna().all().all())


if __name__ == "__main__":
    unittest.main()
