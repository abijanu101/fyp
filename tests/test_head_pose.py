"""Head-pose unit (src/bfmu/head_pose.py): keypoint geometry, pose-to-track matching, window
features. No model weights needed."""

import unittest

import numpy as np

from src.bfmu import FEATURE_EXTRACTORS
from src.bfmu.head_pose import head_geometry, match_poses
from src.core.types import Tracklet


def person(nose_dx=0.0, nose_y=40.0, ears=True, conf=0.9):
    """COCO keypoints of an upright person: shoulders at y 100, 80 px apart, centred on x 100."""
    xy = np.zeros((17, 2))
    c = np.zeros(17)
    xy[0] = (100 + nose_dx, nose_y)                 # nose
    xy[1], xy[2] = (92, 35), (108, 35)              # eyes
    xy[3], xy[4] = (85, 40), (115, 40)              # ears (30 px apart)
    xy[5], xy[6] = (60, 100), (140, 100)            # shoulders (80 px apart)
    c[[0, 1, 2, 5, 6]] = conf
    c[[3, 4]] = conf if ears else 0.1
    return xy, c


class TestHeadGeometry(unittest.TestCase):
    def test_facing_forward(self):
        g = head_geometry(*person())
        self.assertAlmostEqual(g["hp.yaw"], 0.0)
        self.assertAlmostEqual(g["hp.head_height"], 60 / 80)
        self.assertEqual(g["hp.matched"], 1.0)
        self.assertAlmostEqual(g["hp.face_conf"], 0.9)

    def test_turned_and_lowered(self):
        left = head_geometry(*person(nose_dx=-9))
        self.assertLess(left["hp.yaw"], 0)
        self.assertAlmostEqual(left["hp.yaw"], -9 / 30)            # ears are the reference pair
        low = head_geometry(*person(nose_y=80))
        self.assertLess(low["hp.head_height"], head_geometry(*person())["hp.head_height"])

    def test_shoulders_replace_missing_ears(self):
        g = head_geometry(*person(nose_dx=8, ears=False))
        self.assertAlmostEqual(g["hp.yaw"], 8 / 80)

    def test_missing_nose_gives_nan(self):
        xy, c = person()
        c[0] = 0.1
        g = head_geometry(xy, c)
        self.assertTrue(np.isnan(g["hp.yaw"]) and np.isnan(g["hp.head_height"]))
        self.assertEqual(g["hp.matched"], 1.0)


class TestMatching(unittest.TestCase):
    def test_one_to_one_by_iou(self):
        tracks = np.array([[0, 0, 10, 10], [20, 0, 30, 10.0]])
        poses = np.array([[21, 0, 31, 10], [0, 1, 10, 11], [100, 100, 110, 110.0]])
        self.assertEqual(match_poses(poses, tracks, 0.3), {0: 1, 1: 0})
        self.assertEqual(match_poses(np.empty((0, 4)), tracks, 0.3), {})


class TestUnit(unittest.TestCase):
    def test_tracks_mode_reads_keypoints_from_the_tracks(self):
        from src.core.types import Frame, TrackState
        unit = FEATURE_EXTRACTORS.build("head_pose", mode="tracks")
        xy, c = person(nose_dx=-9)
        tracks = [TrackState(1, np.zeros(4), 0.9, None, xy, c), TrackState(2, np.zeros(4), 0.9)]
        out = unit.frame_signals(Frame(0, 0.0, np.zeros((4, 4, 3), np.uint8)), tracks)
        self.assertAlmostEqual(out[1]["hp.yaw"], -9 / 30)
        self.assertEqual((out[1]["hp.matched"], out[2]["hp.matched"]), (1.0, 0.0))
        self.assertTrue(np.isnan(out[2]["hp.yaw"]))
        self.assertIsNone(unit._model)                             # no model of its own

    def test_tracker_hands_each_track_its_detections_keypoints(self):
        from src.core.types import Detections
        kp = np.arange(2 * 17 * 2, dtype=float).reshape(2, 17, 2)
        d = Detections(np.array([[0, 0, 10, 10], [20, 0, 30, 10.0]]), np.array([0.9, 0.8]), kp, np.ones((2, 17)))
        kept = d.take(np.array([False, True]))
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept.keypoints[0], kp[1]))

    def test_registered_and_window_features(self):
        unit = FEATURE_EXTRACTORS.build("head_pose")
        self.assertTrue(unit.uses_frames)
        n = 10
        tr = Tracklet(1, np.arange(n) / 5.0, np.zeros((n, 4)), np.ones(n), np.zeros((n, 2)),
                      {"hp.yaw": np.r_[np.full(5, 0.2), np.full(5, -0.2)],
                       "hp.head_height": np.r_[np.full(4, 0.05), np.full(6, 0.3)],
                       "hp.face_conf": np.full(n, 0.8), "hp.ear_asym": np.zeros(n),
                       "hp.matched": np.r_[np.ones(8), np.zeros(2)]})
        f = unit.window_features(tr, 5.0)
        self.assertEqual(set(f), set(unit.keys))
        self.assertAlmostEqual(f["hp.yaw.mean"], 0.0)
        self.assertAlmostEqual(f["hp.yaw.abs"], 0.2)
        self.assertAlmostEqual(f["hp.head_low"], 0.4)              # 4 of 10 frames below 0.1
        self.assertAlmostEqual(f["hp.matched"], 0.8)
        empty = Tracklet(2, np.arange(3.0), np.zeros((3, 4)), np.ones(3), np.zeros((3, 2)), {})
        self.assertTrue(all(np.isnan(v) for v in unit.window_features(empty, 5.0).values()))


if __name__ == "__main__":
    unittest.main()
