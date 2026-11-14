"""Tune detector confidence + tracker parameters against the manual SID boxes (design §4, §11).

  python -m scripts.tune_tracker [--profile quick|full|full_reid_cmc|full_strongsort] [--tracker NAME]
                                 [--trials N] [--check] [--set key=value ...]

How it works
  1. Once per session, frames are decoded and the detector runs at a low confidence floor; the
     detections, frame timestamps and frame size are cached (outputs/cache/detections/).
  2. If any trial can use ReID, every floor-level detection is also embedded once with the tracker's
     own ReID model (BoxMOT 26 default `osnet-x0-25-msmt17`, shared by BoT-SORT and StrongSORT)
     and cached next to the detections (<session>.reid.npz, float32). Embeddings depend only on the
     frame and the box, never on tracker settings, so trials reuse them.
  3. Each trial samples parameters from tuning.tracker.space[<tracker>] (trial 0 = current config),
     re-runs only the tracker on the cached detections filtered at the trial's `det_conf` (with the
     cached embeddings when it uses ReID), and scores the raw tracks against the manual boxes
     (evaluation.tracking_metrics). The objective is averaged over sessions; a per-group breakdown
     shows whether the winner generalises.
  4. Results: <output_dir>/tuning/tracker_<tracker>_<profile>.csv (all trials) and .json (best
     params as a config patch for `detector.conf` and `tracker.options`).

What a trial reads
  Frames are decoded only when the trial turns camera-motion compensation (use_cmc) on: CMC is the
  only part that still needs pixels, and it cannot be cached (it masks the trial's own detections
  when estimating camera motion). Otherwise the tracker gets a blank frame of the right size.
  --check verifies, on the profile's first session with its trial-0 settings, that this fast path
  (blank frames and/or cached embeddings) gives exactly the tracks of the tracker's own path
  (decoded frames, its own ReID); it exits nonzero otherwise, which blocks a chained run.

Profiles (tuning.tracker.profiles) may name their `tracker` and pin parameters in `fixed`.

Resuming
  Every (trial settings, session) result is appended to <stem>.sessions.jsonl as soon as it is
  computed. Re-running the same command resumes: the seeded search proposes the same trials in the
  same order, and results already on disk are reused, so at most one session's work is lost when a
  run is interrupted. The file's first line fingerprints everything that affects scores (tracker,
  profile, space, detector, association, seed, ReID model); a mismatch stops the run rather than
  mixing results. The trial count is not part of it: raising --trials (or n_trials) continues the
  same seeded sequence. --fresh moves an existing results file aside and starts over.

Tuning on all groups makes tracking metrics in-sample; the per-group breakdown is reported so a
winner that only helps one room is visible.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_pipeline import parse_sets, session_stem
from src.core.config import config_hash, load_config
from src.core.pipeline import Pipeline
from src.core.types import Detections, Frame, SessionKey
from src.datasets import DipserIndex
from src.evaluation import tracking_metrics
from src.ingestion import ZipFrameSource
from src.tracking import TRACKERS, TrackletBuilder
from src.tuning import random_search

REID_MODEL = "osnet-x0-25-msmt17"     # BoxMOT 26 default ReID model for both BoT-SORT and StrongSORT


class SessionDetections:
    """Cached floor-confidence detections (and optionally ReID embeddings) for one session."""

    def __init__(self, key: SessionKey, zip_path: Path, cache: Path, frame_t: np.ndarray,
                 frame_hw: tuple[int, int], dets: list[Detections], max_frames: int | None):
        self.key, self.zip_path, self.cache, self.frame_t, self.frame_hw = key, zip_path, cache, frame_t, frame_hw
        self.dets, self.max_frames = dets, max_frames

    @classmethod
    def load_or_detect(cls, cfg: dict, index: DipserIndex, key: SessionKey, max_frames: int | None,
                       floor: float, detector_factory) -> "SessionDetections":
        """detector_factory() returns the shared detector; it is only called on a cache miss."""
        det_cfg = {**cfg["detector"], "conf": floor}
        cache = (Path(cfg["output_dir"]) / "cache" / "detections" /
                 config_hash({"detector": det_cfg, "max_frames": max_frames, "v": 2})[:12] /
                 f"{session_stem(key)}.npz")
        zip_path = index.frames_zip(key)
        if not cache.exists():
            detector = detector_factory()
            t, dets, hw = [], [], (0, 0)
            for f in ZipFrameSource(zip_path, max_frames=max_frames):
                t.append(f.t)
                dets.append(detector.detect(f))
                hw = f.image.shape[:2]
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp.npz")             # atomic write: an interrupt never leaves a bad cache
            np.savez_compressed(tmp, frame_t=np.array(t), frame_hw=np.array(hw),
                                counts=np.array([len(d) for d in dets]),
                                xyxy=np.concatenate([d.xyxy for d in dets]) if dets else np.zeros((0, 4)),
                                conf=np.concatenate([d.conf for d in dets]) if dets else np.zeros(0))
            tmp.replace(cache)
        with np.load(cache) as z:      # read each array ONCE: every z[...] access re-decompresses it
            frame_t, frame_hw, counts = z["frame_t"], z["frame_hw"], z["counts"]
            xyxy, conf = z["xyxy"], z["conf"]
        bounds = np.cumsum(np.concatenate([[0], counts]))
        dets = [Detections(xyxy[a:b], conf[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
        return cls(key, zip_path, cache, frame_t, tuple(int(v) for v in frame_hw), dets, max_frames)

    def position(self, frame: Frame) -> int:
        """Row of this frame in the cache (by timestamp, so a frame that fails to decode cannot shift
        the alignment)."""
        return int(np.searchsorted(self.frame_t, frame.t))

    def frames(self, pixels: bool):
        """Decoded frames, or blank frames with the same timestamps when the tracker ignores pixels."""
        if pixels:
            yield from ZipFrameSource(self.zip_path, max_frames=self.max_frames)
            return
        blank = np.zeros((*self.frame_hw, 3), np.uint8)
        for i, t in enumerate(self.frame_t):
            yield Frame(i, float(t), blank)

    # -- ReID embeddings, cached once per session
    @property
    def reid_path(self) -> Path:
        return self.cache.with_name(self.cache.stem + f".reid-{REID_MODEL}.npz")

    def ensure_embeddings(self, encoder_factory) -> None:
        if self.reid_path.exists():
            return
        encoder = encoder_factory()
        parts = {}
        for f in ZipFrameSource(self.zip_path, max_frames=self.max_frames):
            i = self.position(f)
            if len(self.dets[i]):
                parts[i] = encoder.embed(f.image, self.dets[i])
        dim = next(iter(parts.values())).shape[1] if parts else 0
        embs = np.concatenate([parts.get(i, np.zeros((len(d), dim), np.float32)) for i, d in enumerate(self.dets)])
        tmp = self.reid_path.with_suffix(".tmp.npz")
        np.savez(tmp, embs=embs.astype(np.float32))       # embeddings do not compress; keep full precision
        tmp.replace(self.reid_path)

    def embeddings(self) -> list[np.ndarray]:
        """Per frame, the embeddings of that frame's floor-level detections (loaded on demand, so only
        one session's embeddings are in memory at a time)."""
        with np.load(self.reid_path) as z:
            embs = z["embs"]
        bounds = np.cumsum([0] + [len(d) for d in self.dets])
        return [embs[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def needs_pixels(options: dict) -> bool:
    """Only camera-motion compensation reads the image once embeddings are cached (BoxMOT default: on)."""
    return bool(options.get("use_cmc", True))


def needs_embeddings(tracker: str, options: dict) -> bool:
    return tracker == "strongsort" or bool(options.get("use_embeddings", True))


def run_tracker(name: str, options: dict, sd: SessionDetections, det_conf: float, fast: bool = True):
    """fast=True: blank frames unless CMC is on, cached embeddings when ReID is on (what trials use).
    fast=False: the tracker's own path, decoded frames and its own ReID (the --check reference)."""
    tracker = TRACKERS.build(name, **options)
    embs = sd.embeddings() if fast and needs_embeddings(name, options) else None
    builder = TrackletBuilder()
    for frame in sd.frames(needs_pixels(options) or not fast):
        i = sd.position(frame)
        d = sd.dets[i]
        keep = d.conf >= det_conf
        kept = Detections(d.xyxy[keep], d.conf[keep])
        states = (tracker.update_with_embeddings(frame, kept, embs[i][keep]) if embs is not None
                  else tracker.update(frame, kept))
        builder.add(frame.t, states, {})
    return builder.build()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--tracker")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--check", "--check-blank", dest="check", action="store_true",
                    help="verify the fast path reproduces the tracker's own tracks on the first session, then exit")
    ap.add_argument("--fresh", action="store_true", help="move an existing results file aside and start over")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    tcfg = cfg["tuning"]["tracker"]
    prof = tcfg["profiles"][args.profile]
    name = args.tracker or prof.get("tracker") or cfg["tracker"]["name"]
    if prof.get("tracker", name) != name:
        raise SystemExit(f"profile '{args.profile}' is for {prof['tracker']}, not {name}")
    # Current config options only apply when tuning the configured tracker; another tracker starts
    # from BoxMOT defaults (BoT-SORT options are not valid StrongSORT options and vice versa).
    base_options = cfg["tracker"].get("options", {}) if name == cfg["tracker"]["name"] else {}
    fixed = prof.get("fixed", {})
    space = {k: v for k, v in tcfg["space"][name].items() if k not in fixed}
    index = DipserIndex.from_config(cfg)
    min_iou = cfg["association"]["min_iou"]

    keys = index.sessions() if prof["sessions"] == "all" else [SessionKey(*s.split("/")) for s in prof["sessions"]]
    if args.check:
        keys = keys[:1]
    det_pipe = Pipeline({**cfg, "detector": {**cfg["detector"], "conf": tcfg["det_conf_floor"]}})
    shared_detector = lambda: det_pipe.detector              # built once, on the first cache miss
    sessions = []
    for i, key in enumerate(keys, 1):
        sd = SessionDetections.load_or_detect(cfg, index, key, prof["max_frames"], tcfg["det_conf_floor"],
                                              shared_detector)
        sessions.append((sd, index.annotations(key)))
        print(f"[{i}/{len(keys)}] {key}: {len(sd.dets)} frames, {sum(map(len, sd.dets))} detections", flush=True)

    def options_for(params: dict) -> tuple[dict, float]:
        params = {**dict(params), **fixed}
        det_conf = params.pop("det_conf", cfg["detector"]["conf"])
        return {**base_options, "det_thresh": det_conf, **params}, det_conf

    baseline = {"det_conf": cfg["detector"]["conf"],
                **{k: v for k, v in base_options.items() if k in space}}

    # Embed every session once if any trial can use ReID.
    if needs_embeddings(name, options_for(baseline)[0]) or "use_embeddings" in space:
        encoder = {}

        def encoder_factory():
            if "e" not in encoder:
                encoder["e"] = TRACKERS.build(name, use_embeddings=True) if name == "botsort" else TRACKERS.build(name)
                model = encoder["e"].tracker._reid_config.model
                if model != REID_MODEL:
                    raise SystemExit(f"tracker ReID model {model!r} != cached model {REID_MODEL!r}")
            return encoder["e"]
        for i, (sd, _) in enumerate(sessions, 1):
            if not sd.reid_path.exists():
                print(f"embedding [{i}/{len(sessions)}] {sd.key}", flush=True)
            sd.ensure_embeddings(encoder_factory)

    if args.check:
        sd, _ = sessions[0]
        opts, det_conf = options_for(baseline)
        runs = [[(tr.tid, tr.t.tolist(), tr.xyxy.round(3).tolist())
                 for tr in run_tracker(name, opts, sd, det_conf, fast=f)] for f in (True, False)]
        a, b = runs
        print(f"fast path (pixels={needs_pixels(opts)}, cached ReID={needs_embeddings(name, opts)}) "
              f"identical to tracker's own path: {a == b} ({len(a)} vs {len(b)} tracks)")
        raise SystemExit(0 if a == b else 1)   # nonzero blocks a chained full run

    # -- resumable results: one JSON line per (trial settings, session)
    out = Path(cfg["output_dir"]) / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"tracker_{name}_{args.profile}"
    results_path = out / f"{stem}.sessions.jsonl"
    fingerprint = config_hash({"tracker": name, "profile": {k: v for k, v in prof.items() if k != "n_trials"}, "space": space, "base_options": base_options,
                               "default_det_conf": cfg["detector"]["conf"], "detector": cfg["detector"],
                               "floor": tcfg["det_conf_floor"], "min_iou": min_iou, "seed": cfg["tuning"]["seed"],
                               "reid": REID_MODEL, "v": 1})
    if args.fresh and results_path.exists():
        aside = results_path.with_name(f"{results_path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
        results_path.rename(aside)
        print(f"moved previous results to {aside}")
    done: dict[tuple[str, str], dict] = {}
    if results_path.exists():
        lines = results_path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0]) if lines else {}
        if header.get("fingerprint") != fingerprint:
            raise SystemExit(f"{results_path} was produced under different settings; "
                             f"move it away or pass --fresh to start over")
        for line in lines[1:]:
            if line.strip():
                r = json.loads(line)
                done[r["params_key"], r["session"]] = r["metrics"]
        print(f"resuming: {len(done)} session results already in {results_path.name}", flush=True)
    else:
        results_path.write_text(json.dumps({"fingerprint": fingerprint, "tracker": name,
                                            "profile": args.profile}) + "\n", encoding="utf-8")

    def objective(params: dict) -> dict:
        options, det_conf = options_for(params)
        key = json.dumps(params, sort_keys=True)
        per, start, trial_no = [], time.monotonic(), objective.calls
        objective.calls += 1
        for n, (sd, subj) in enumerate(sessions, 1):
            if (key, str(sd.key)) in done:
                m = done[key, str(sd.key)]
            else:
                m = tracking_metrics(run_tracker(name, options, sd, det_conf), subj, len(sd.dets), min_iou)
                done[key, str(sd.key)] = m
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"params_key": key, "session": str(sd.key), "metrics": m}) + "\n")
            per.append(m)
            print(f"\r  trial {trial_no}: {n}/{len(sessions)} sessions, {(time.monotonic() - start) / 60:.1f} min",
                  end="", flush=True)
        print("\r" + " " * 60 + "\r", end="", flush=True)
        mean = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        groups = sorted({sd.key.group for sd, _ in sessions})
        by_group = {g: float(np.mean([p[tcfg["objective"]] for p, (sd, _) in zip(per, sessions) if sd.key.group == g]))
                    for g in groups}
        return {"score": mean[tcfg["objective"]], **mean, "by_group": by_group}
    objective.calls = 0

    def log(r: dict) -> None:
        groups = " ".join(f"{g[-2:]}:{v:.3f}" for g, v in r["by_group"].items())
        print(f"trial {r['trial']:3d}  {tcfg['objective']}={r['score']:.3f} [{groups}]  cov={r['coverage']:.2f}  "
              f"sw/subj={r['id_switches_per_subject']:.2f}  tids/subj={r['tids_per_subject']:.2f}  {r['params']}",
              flush=True)

    trials = random_search(space, objective, args.trials or prof["n_trials"], cfg["tuning"]["seed"],
                           baseline=baseline, on_trial=log)

    pd.DataFrame([{**{k: v for k, v in t.items() if k not in ("params", "by_group")},
                   **{f"group.{g}": v for g, v in t["by_group"].items()}, **t["params"]} for t in trials]) \
        .to_csv(out / f"{stem}.csv", index=False)
    best_opts, det_conf = options_for(trials[0]["params"])
    base = next(t for t in trials if t["trial"] == 0)
    summary = {"tracker": name, "profile": args.profile, "n_sessions": len(sessions),
               "max_frames": prof["max_frames"], "fixed": fixed, "objective": tcfg["objective"],
               "baseline": {"score": base["score"], "by_group": base["by_group"]},
               "best": {k: v for k, v in trials[0].items() if k != "params"},
               "config_patch": {"detector.conf": det_conf, "tracker.options": best_opts}}
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nbaseline {base['score']:.3f} {base['by_group']} -> best {trials[0]['score']:.3f} {trials[0]['by_group']}")
    print("config patch:", json.dumps(summary["config_patch"]))


if __name__ == "__main__":
    main()
