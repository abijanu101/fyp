"""Pure-numpy stages: windows, tracklets, association, BFMU geometry, aggregation, config, pipeline wiring."""

import unittest

import numpy as np

from src.aggregation import WindowAggregator, make_windows
from src.bfmu import FEATURE_EXTRACTORS
from src.cfmu.extractor import NullClusterExtractor
import tempfile

from src.core.config import config_hash, load_config
from src.core.pipeline import Pipeline, feature_hash, tracking_hash
from src.core.store import load_track, save_track
from src.core.types import SessionKey, SessionTrack, TrackState, Tracklet, Window
from src.output import build_graph
from src.tracking import TrackletBuilder, associate_sids, filter_visible


def moving_tracklet(tid, n=20, dt=0.4, vx=10.0, x0=100.0, y0=100.0):
    t = np.arange(n) * dt
    x = x0 + vx * t
    xyxy = np.stack([x, np.full(n, y0), x + 50, np.full(n, y0 + 100)], 1)
    return Tracklet(tid, t, xyxy, np.ones(n), np.full((n, 2), np.nan))


class TestWindows(unittest.TestCase):
    def test_overlap(self):
        w = make_windows(0, 16, 8, 0.5)
        self.assertEqual([(x.t_start, x.t_end) for x in w], [(0, 8), (4, 12), (8, 16)])

    def test_too_short(self):
        self.assertEqual(make_windows(0, 5, 8, 0.5), [])


class TestTracklets(unittest.TestCase):
    def build(self, times):
        b = TrackletBuilder()
        for t in times:
            b.add(t, [TrackState(7, np.array([0, 0, 10, 10.0]), 0.9)], {7: {"sig": t}})
        (tr,) = b.build()
        return tr

    def test_split_where_tracker_lost_the_track(self):
        frame_t = np.arange(0, 4.0, 0.2)                   # 5 fps, every frame decoded
        times = [0, 0.2, 0.4, 0.6, 2.4, 2.6]               # absent from 8 existing frames = 1.6 s
        tr = self.build(times)
        self.assertEqual(tr.signals["sig"].tolist(), times)
        pieces = filter_visible([tr], frame_t, 5.0, min_visible_s=0.0, max_gap_s=1.0)
        self.assertEqual([len(p.t) for p in pieces], [4, 2])
        self.assertEqual([len(p.t) for p in filter_visible([tr], frame_t, 5.0, 0.7, 1.0)], [4])

    def test_camera_frame_drop_is_not_a_gap(self):
        frame_t = np.array([0, 0.2, 0.4, 0.6, 3.6, 3.8])  # camera dropped 3 s of frames
        tr = self.build(frame_t)                           # tracker kept the same TID throughout
        self.assertEqual([len(p.t) for p in filter_visible([tr], frame_t, 5.0, 0.0, 1.0)], [6])


class TestAssociation(unittest.TestCase):
    def test_majority_vote(self):
        subjects = {"subject_01": np.array([100, 100, 150, 200.0]), "subject_02": np.array([400, 100, 450, 200.0])}
        a = moving_tracklet(1, vx=0.0, x0=100)
        b = moving_tracklet(2, vx=0.0, x0=400)
        far = moving_tracklet(3, vx=0.0, x0=900)
        self.assertEqual(associate_sids([a, b, far], subjects), {1: "subject_01", 2: "subject_02"})

    def test_walker_passing_a_student_stays_unassociated(self):
        subjects = {"subject_01": np.array([100, 100, 150, 200.0])}
        student = moving_tracklet(1, vx=0.0, x0=100)
        walker = moving_tracklet(9, n=40, vx=50.0, x0=0)   # sweeps across the student's box for a few frames
        self.assertEqual(associate_sids([student, walker], subjects), {1: "subject_01"})
        self.assertIn(9, associate_sids([walker], subjects, min_match_share=0.0))   # the old behaviour


class TestKalmanGeometry(unittest.TestCase):
    def test_finite_difference_speed(self):
        kg = FEATURE_EXTRACTORS.build("kalman_geometry", use_kalman=True)
        tr = moving_tracklet(1, vx=10.0)
        f = kg.window_features(tr, fps=2.5)
        diag = np.hypot(50, 100)
        self.assertAlmostEqual(f["kg.speed"], 10.0 / diag, places=6)
        self.assertEqual(f["kg.stillness"], 0.0)
        self.assertAlmostEqual(f["kg.aspect_trend"], 0.0)

    def test_kalman_velocity_preferred_when_present(self):
        kg = FEATURE_EXTRACTORS.build("kalman_geometry")
        tr = moving_tracklet(1, vx=10.0)
        tr.velocity[:] = [0.0, 0.0]          # tracker says still
        self.assertEqual(kg.window_features(tr, fps=2.5)["kg.stillness"], 1.0)


class TestAggregation(unittest.TestCase):
    def test_vec_keys_stable_and_nan_when_empty(self):
        agg = WindowAggregator([FEATURE_EXTRACTORS.build("kalman_geometry")], NullClusterExtractor(),
                               thresholds={"kg.stillness": 0.5})
        full = agg.aggregate(Window(0, 8), [moving_tracklet(1), moving_tracklet(2, vx=0.0)], 2.5, (1280, 720))
        empty = agg.aggregate(Window(100, 108), [moving_tracklet(1)], 2.5, (1280, 720))
        self.assertEqual(sorted(full), sorted(empty))
        self.assertEqual(full["n_tracks"], 2.0)
        self.assertEqual(full["kg.stillness.frac"], 0.5)
        self.assertTrue(np.isnan(empty["kg.speed.mean"]))
        self.assertAlmostEqual(sum(v for k, v in full.items() if k.startswith("occ_")), 1.0)


class TestConfigAndWiring(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config("configs/deliverable3.json")

    def test_hash_excludes_output_dir_and_tuning(self):
        other = load_config("configs/deliverable3.json", {"output_dir": "elsewhere", "tuning.seed": 7})
        self.assertEqual(config_hash(self.cfg), config_hash(other))
        relabeled = load_config("configs/deliverable3.json", {"labels.tau_att": 3})
        self.assertNotEqual(config_hash(self.cfg), config_hash(relabeled))

    def test_cache_keys_split_frame_pass_from_feature_pass(self):
        variant = lambda o: load_config("configs/deliverable3.json", o)
        relabeled, rewindowed = variant({"labels.tau_att": 3}), variant({"window.length_s": 10})
        still, flow = variant({}), variant({})
        still["bfmu"][0]["still_threshold"] = 0.1                         # window-level BFMU setting
        flow["bfmu"][1]["grid"] = 3                                       # pixel-level BFMU setting
        retracked = variant({"tracker.name": "strongsort"})
        same, base_t, base_f = tracking_hash, tracking_hash(self.cfg), feature_hash(self.cfg)
        self.assertEqual(feature_hash(relabeled), base_f)                  # labels never touch features
        for v in (rewindowed, still):                                     # feature pass only
            self.assertEqual(same(v), base_t)
            self.assertNotEqual(feature_hash(v), base_f)
        for v in (flow, retracked):                                       # needs a new frame pass
            self.assertNotEqual(same(v), base_t)
            self.assertNotEqual(feature_hash(v), base_f)

    def test_pipeline_builds_without_gpu(self):
        pipe = Pipeline(self.cfg)
        self.assertEqual([type(e).__name__ for e in pipe.extractors], ["KalmanGeometry", "ResidualFlow", "ScbBehaviour"])

    def test_track_store_round_trip(self):
        a = moving_tracklet(1)
        a.velocity[:] = [1.5, -2.0]
        a.signals = {"rf.mag_00": np.linspace(0, 1, len(a.t))}
        b = moving_tracklet(4, n=5)                                       # no signals at all
        st = SessionTrack(SessionKey("group_01", "experiment_02", "context_cam_03"), np.arange(30) * 0.4,
                          (1280, 720), [a, b], [{"motion": 0.1, "val": 0.5}] * 30,
                          {"fps": 2.5, "t_first": 0.0}, {"detection": 1.0})
        with tempfile.TemporaryDirectory() as d:
            save_track(st, f"{d}/s.npz")
            got = load_track(f"{d}/s.npz")
        self.assertEqual((got.key, got.frame_wh, got.timing, got.scene_rows), (st.key, st.frame_wh, st.timing, st.scene_rows))
        self.assertEqual([t.tid for t in got.tracklets], [1, 4])
        for x, y in zip(got.tracklets, st.tracklets):
            for f in ("t", "xyxy", "conf", "velocity"):
                np.testing.assert_array_equal(getattr(x, f), getattr(y, f))
        np.testing.assert_array_equal(got.tracklets[0].signals["rf.mag_00"], a.signals["rf.mag_00"])
        self.assertTrue(np.isnan(got.tracklets[1].signals["rf.mag_00"]).all())

    def test_feature_pass_runs_from_a_track_without_video(self):
        frame_t = np.arange(0, 20, 0.2)
        trs = [moving_tracklet(i, n=100, dt=0.2, vx=0.0, x0=100 + 200 * i) for i in range(3)]
        st = SessionTrack(None, frame_t, (1280, 720), trs, [{"motion": 0.0}] * len(frame_t),
                          {"fps": 5.0, "t_first": 0.0, "t_last": float(frame_t[-1])}, {})
        feats = Pipeline(self.cfg).features(st)
        self.assertEqual(len(feats.windows), 3)                             # 8 s windows, 50% overlap, ~20 s
        self.assertEqual([v["n_tracks"] for v in feats.vec], [3.0, 3.0, 3.0])
        self.assertIn("motion.mean", feats.scene[0])

    def test_graph_has_no_identity_fields(self):
        rec = build_graph([Window(0, 8)], np.array([[0.2, 0.8]]), ["01", "02"], np.array([0.7]),
                          np.array([12]), tau=0.5)[0]
        self.assertEqual(set(rec), {"t_start", "t_end", "predicted_context", "layout_family", "engaged",
                                    "engagement_prob", "n_visible_tracks"})
        self.assertEqual(rec["predicted_context"]["label"], "02")


if __name__ == "__main__":
    unittest.main()
