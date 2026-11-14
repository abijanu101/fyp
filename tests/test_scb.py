import fractions
import tempfile
import unittest

import numpy as np

from src.bfmu import FEATURE_EXTRACTORS
from src.bfmu.scb import normalize_class
from src.bfmu.yolov7 import YOLOV7_REPO, load_yolov7
from src.core.config import load_config
from src.core.pipeline import Pipeline, tracking_hash
from src.core.types import Tracklet

SPEC = [{"weights": "unused.pt", "classes": ["hand_raising", "read", "write"]}]


class TestScbExtractor(unittest.TestCase):
    def setUp(self):
        self.ex = FEATURE_EXTRACTORS.build("scb", models=SPEC, mode="flags", min_overlap=0.5)
        self.soft = FEATURE_EXTRACTORS.build("scb", models=SPEC, mode="soft", min_overlap=0.5)

    def test_keys_known_without_loading_weights(self):
        self.assertEqual(self.ex.keys, ("scb.hand_raising", "scb.read", "scb.write"))
        self.assertEqual(self.soft.keys, ("scb.hand_raising", "scb.hand_raising.max", "scb.read", "scb.read.max",
                                          "scb.write", "scb.write.max", "scb.m0.obj", "scb.m0.obj.max"))
        self.assertIsNone(self.ex._detectors)

    def test_soft_scores_take_max_of_candidates_per_track(self):
        tracks = np.array([[0, 0, 100, 200], [120, 0, 220, 200.0]])
        cands = np.array([[10, 10, 60, 80],      # track 0
                          [20, 20, 70, 90],      # track 0, higher read score
                          [130, 30, 200, 120],   # track 1
                          [400, 0, 450, 50.0]])  # nobody
        scores = np.array([[0.1, 0.5, 0.0, 0.6],
                           [0.2, 0.7, 0.3, 0.8],
                           [0.4, 0.0, 0.1, 0.5],
                           [0.9, 0.9, 0.9, 0.9]])
        got = self.soft.soft_scores(cands, scores, tracks)
        np.testing.assert_allclose(got, [[0.2, 0.7, 0.3, 0.8], [0.4, 0.0, 0.1, 0.5]])

    def test_soft_candidate_goes_to_one_track_only(self):
        tracks = np.array([[0, 0, 100, 200], [0, 0, 100, 200.0]])           # duplicate tracks
        got = self.soft.soft_scores(np.array([[10, 10, 50, 50.0]]), np.array([[1.0]]), tracks)
        self.assertEqual(got.sum(), 1.0)

    def test_soft_window_features_mean_and_peak(self):
        tr = Tracklet(1, np.arange(4.0), np.zeros((4, 4)), np.ones(4), np.zeros((4, 2)),
                      {"scb.hand_raising": np.array([0, 0, 0.8, 0.0])})
        f = self.soft.window_features(tr, 5.0)
        self.assertAlmostEqual(f["scb.hand_raising"], 0.2)
        self.assertAlmostEqual(f["scb.hand_raising.max"], 0.8)

    def test_normalize_class(self):
        self.assertEqual([normalize_class(n) for n in ["hand-raising", "Read", "Using Phone"]],
                         ["hand_raising", "read", "using_phone"])

    def test_behaviour_box_assigned_to_containing_track(self):
        tracks = np.array([[0, 0, 100, 200], [120, 0, 220, 200.0]])
        behaviour = np.array([[130, 40, 200, 120],        # inside track 2
                              [400, 0, 450, 50.0]])       # nobody
        self.assertEqual(self.ex.match(behaviour, tracks, [7, 9]), [9])

    def test_one_box_flags_one_track(self):
        tracks = np.array([[0, 0, 100, 200], [0, 0, 100, 200.0]])   # duplicate tracks on one student
        self.assertEqual(len(self.ex.match(np.array([[10, 10, 50, 50.0]]), tracks, [1, 2])), 1)

    def test_class_conf_overrides_default(self):
        ex = FEATURE_EXTRACTORS.build("scb", mode="flags",
                                      models=[{**SPEC[0], "conf": 0.25, "class_conf": {"hand_raising": 0.6}}])
        self.assertEqual(ex.thresholds, [[0.6, 0.25, 0.25]])

    def test_window_feature_is_share_of_flagged_frames(self):
        t = np.arange(10.0)
        tr = Tracklet(1, t, np.zeros((10, 4)), np.ones(10), np.zeros((10, 2)),
                      {"scb.write": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0.0])})
        f = self.ex.window_features(tr, 5.0)
        self.assertAlmostEqual(f["scb.write"], 0.3)
        self.assertTrue(np.isnan(f["scb.read"]))


class TestScbInPipeline(unittest.TestCase):
    def test_pipeline_builds_and_scb_is_frame_level(self):
        cfg = load_config("configs/deliverable3.json")
        pipe = Pipeline(cfg)
        self.assertIn("ScbBehaviour", [type(e).__name__ for e in pipe.extractors])
        changed = load_config("configs/deliverable3.json")
        changed["bfmu"][2]["min_overlap"] = 0.6
        self.assertNotEqual(tracking_hash(cfg), tracking_hash(changed))   # needs a new frame pass


@unittest.skipUnless((YOLOV7_REPO / "models" / "yolo.py").is_file(), "lib/yolov7 not cloned")
class TestSafeLoader(unittest.TestCase):
    def test_refuses_unexpected_globals(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/bad.pt"
            torch.save({"model": fractions.Fraction(1, 3)}, path)
            with self.assertRaises(RuntimeError):
                load_yolov7(path)


if __name__ == "__main__":
    unittest.main()
