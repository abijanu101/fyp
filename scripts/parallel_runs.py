"""Run the long jobs in parallel, one search (or decoder variant) per thread, sharing what they read.

  python -m scripts.parallel_runs tracker-scans --augment 1 [--profiles full full_reid_cmc full_strongsort]
                                 [--trials 26 34 22] [--workers 5] [--gpu-workers 2] [--beam 2]
                                 [--det-conf-from JSON] [--start JSON] [--tag TAG]
  python -m scripts.parallel_runs extract --set augmentation.copies=8 [--workers 3]

extract
  `run_pipeline extract --reuse` with several recordings at once: one worker process per recording,
  each holding its own pipeline (detector, ReID, SCB on the GPU). Jobs run in run_pipeline's order
  (originals, then every session's copy 1, then copy 2, ...), so the copies finish in layers and
  training at copies=k can start once layer k is done. Finished recordings are skipped (the
  window .csv is written last), so it is resumable and raising `copies` only extracts the new ones.

tracker-scans
  The augmented tracker scans (docs/experiments/03_tracking.md, T2), all at once:
  1. prepare  every session's floor-level detections and, where a scan can use ReID, its embeddings,
              in GPU worker processes: each session's frames are decoded and augmented ONCE for both
              (tune_tracker would read them twice, one session at a time).
  2. search   one thread per scan runs that scan's beam bisection (scripts/tuning/tune_tracker.py, Scan).
              Their trials go to one pool of worker processes that works session by session: a
              worker takes ALL pending trials of ALL scans for one session, decodes and augments
              each frame once and feeds it to every one of those trackers (run_trackers). Every
              trial still needs every session, so the pool also spreads each trial over the cores.
              The search's prefetch hook hands each level's probe trials over in one go, so they
              share frames too. Results go to each scan's own resumable .sessions.jsonl as they
              arrive, and the scan files are exactly those `tune_tracker --augment` writes: rerunning
              resumes, and either tool can finish a scan the other started.

Why: with camera-motion compensation on, every trial reads every frame; decoding and augmenting a
frame costs about 120 ms, a tracker update about 8 ms, so feeding one decoded frame to 40 trials
instead of decoding it 40 times cuts the frame work about tenfold, on top of using all cores.
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from scripts.run_pipeline import parse_sets
from src.core.config import load_config

# ----------------------------------------------------------------------------- worker functions
# (top level: worker processes import this module)


def prepare_session(cfg: dict, key_str: str, max_frames, floor: float, aug_params, aug_tag, reid_tracker):
    """Detections at the floor and, if reid_tracker is given, ReID embeddings for one session, from one
    pass of decoded (and augmented) frames. Writes the caches tune_tracker reads; skips what exists."""
    from scripts.tuning.tune_tracker import REID_MODEL, SessionDetections
    from src.core.pipeline import Pipeline
    from src.core.types import SessionKey
    from src.datasets import DipserIndex
    from src.tracking import TRACKERS

    key = SessionKey(*key_str.split("/"))
    zip_path = DipserIndex.from_config(cfg).frames_zip(key)
    cache = SessionDetections.cache_path(cfg, key, max_frames, floor, aug_tag)
    reid_path = cache.with_name(cache.stem + f".reid-{REID_MODEL}.npz")
    need_det, need_reid = not cache.exists(), bool(reid_tracker) and not reid_path.exists()
    if not (need_det or need_reid):
        return key_str, "cached"
    start = time.monotonic()
    detector = Pipeline({**cfg, "detector": {**cfg["detector"], "conf": floor}}).detector if need_det else None
    known = None if need_det else SessionDetections.from_cache(key, zip_path, cache, max_frames, aug_params)
    encoder = None
    if need_reid:
        encoder = TRACKERS.build(reid_tracker, use_embeddings=True) if reid_tracker == "botsort" \
            else TRACKERS.build(reid_tracker)
        if encoder.tracker._reid_config.model != REID_MODEL:
            raise RuntimeError(f"tracker ReID model != cached model {REID_MODEL!r}")
    t, dets, parts, hw = [], [], {}, (0, 0)
    for f in SessionDetections.source(zip_path, max_frames, aug_params):
        if need_det:
            d = detector.detect(f)
            i = len(dets)
            t.append(f.t)
            dets.append(d)
            hw = f.image.shape[:2]
        else:
            i = known.position(f)
            d = known.dets[i]
        if need_reid and len(d):
            parts[i] = encoder.embed(f.image, d)
    if need_det:
        SessionDetections.save(cache, t, dets, hw)
        known = SessionDetections.from_cache(key, zip_path, cache, max_frames, aug_params)
    if need_reid:
        SessionDetections.save_embeddings(reid_path, known.dets, parts)
    return key_str, f"{'detections' if need_det else ''}{' + ' if need_det and need_reid else ''}" \
                    f"{'embeddings' if need_reid else ''} in {time.monotonic() - start:.0f}s"


def track_session(spec: dict, requests: list[tuple[str, dict, float]]) -> list[dict]:
    """Every requested (tracker, options, det_conf) on one session, from one pass of its frames;
    returns their tracking metrics in order."""
    from pathlib import Path

    from scripts.tuning.tune_tracker import SessionDetections, run_trackers
    from src.core.types import SessionKey
    from src.evaluation import tracking_metrics

    sd = SessionDetections.from_cache(SessionKey(*spec["key"].split("/")), Path(spec["zip"]), Path(spec["cache"]),
                                      spec["max_frames"], spec["augment"])
    return [tracking_metrics(tracks, spec["subjects"], len(sd.dets), spec["min_iou"])
            for tracks in run_trackers(sd, requests)]


_PIPE = None


def _extract_init(cfg: dict, threads: int) -> None:
    """Worker start-up: one pipeline per process, a fair share of the CPU threads."""
    global _PIPE
    import cv2
    import torch

    from src.core.pipeline import Pipeline
    cv2.setNumThreads(threads)
    torch.set_num_threads(threads)
    _PIPE = Pipeline(cfg)


def extract_job(cfg: dict, key_str: str, copy) -> str:
    """One recording (or augmented copy) through run_pipeline's extract_one, with --reuse."""
    from scripts.run_pipeline import extract_one
    from src.core.types import SessionKey
    from src.datasets import DipserIndex

    return extract_one(cfg, _PIPE, DipserIndex.from_config(cfg), SessionKey(*key_str.split("/")), copy, True)


# ----------------------------------------------------------------------------- the session pool

class SessionPool:
    """Trials of several scans, scheduled session by session. submit() queues (scan, item) work and
    returns at once; run() also waits. A free worker takes the session with the most pending work
    and runs all of it in one pass of frames. Work added within `linger` seconds of the last addition
    to a session waits for it, so scans submitting at about the same time share a pass."""

    def __init__(self, workers: int, linger: float = 3.0):
        self.ex = ProcessPoolExecutor(workers)
        self.slots, self.linger = workers, linger
        self.cond = threading.Condition()
        self.pending: dict[str, dict] = {}          # session id -> {"spec", "reqs", "t"}
        self.running: set[str] = set()
        self.events: dict[tuple, threading.Event] = {}
        self.errors: list[BaseException] = []
        self.passes = self.items = 0
        threading.Thread(target=self._dispatch, daemon=True).start()

    @staticmethod
    def spec(scan, i: int) -> dict:
        sd, subjects = scan.sessions[i]
        return {"key": str(sd.key), "zip": str(sd.zip_path), "cache": str(sd.cache), "max_frames": sd.max_frames,
                "augment": sd.augment, "subjects": subjects, "min_iou": scan.min_iou}

    def submit(self, scan, items) -> list[threading.Event]:
        out = []
        with self.cond:
            for key, options, det_conf, i in items:
                rid = (scan.label, key, i)
                if rid in self.events:
                    out.append(self.events[rid])
                    continue
                if (key, str(scan.sessions[i][0].key)) in scan.done:
                    continue
                ev = self.events[rid] = threading.Event()
                out.append(ev)
                spec = self.spec(scan, i)
                entry = self.pending.setdefault(spec["cache"], {"spec": spec, "reqs": [], "t": 0.0})
                entry["reqs"].append((scan, key, options, det_conf, i, ev))
                entry["t"] = time.monotonic()
            self.cond.notify_all()
        return out

    def run(self, scan, items) -> None:
        for ev in self.submit(scan, items):
            ev.wait()
        if self.errors:
            raise RuntimeError("a worker failed") from self.errors[0]

    def prefetch(self, scan):
        return lambda points: self.submit(scan, [it for p in points for it in scan.missing(p)])

    def _dispatch(self) -> None:
        while True:
            with self.cond:
                while True:
                    now = time.monotonic()
                    ready = [s for s, e in self.pending.items() if s not in self.running and now - e["t"] >= self.linger]
                    if self.slots > 0 and ready:
                        break
                    self.cond.wait(timeout=0.5)
                sid = max(ready, key=lambda s: len(self.pending[s]["reqs"]))
                entry = self.pending.pop(sid)
                self.running.add(sid)
                self.slots -= 1
                self.passes += 1
                self.items += len(entry["reqs"])
            reqs = entry["reqs"]
            fut = self.ex.submit(track_session, entry["spec"], [(r[0].name, r[2], r[3]) for r in reqs])
            fut.add_done_callback(lambda f, sid=sid, reqs=reqs: self._done(sid, reqs, f))

    def _done(self, sid: str, reqs: list, fut) -> None:
        try:
            metrics = fut.result()
        except BaseException as e:                  # surface in the waiting search threads
            self.errors.append(e)
            metrics = [None] * len(reqs)
        with self.cond:
            for (scan, key, _, _, i, ev), m in zip(reqs, metrics):
                if m is not None:
                    scan.record(key, i, m)
                ev.set()
            self.running.discard(sid)
            self.slots += 1
            self.cond.notify_all()


# ----------------------------------------------------------------------------- tracker-scans

def tracker_scans(args) -> None:
    from scripts.tuning.tune_tracker import Scan, needs_embeddings, profile_setup
    from src.core.types import SessionKey
    from src.datasets import DipserIndex
    from src.ingestion import aug_tag, sample_augmentation

    from scripts.tuning.tune_tracker import apply_det_conf

    cfg = load_config(args.config, parse_sets(args.set))
    fix = apply_det_conf(cfg, args.det_conf_from) or args.fix_det_conf
    tcfg = cfg["tuning"]["tracker"]
    trials = args.trials or [None] * len(args.profiles)
    if len(trials) != len(args.profiles):
        raise SystemExit("--trials needs one value per profile")
    tag = aug_tag(cfg["augmentation"], args.augment) if args.augment else None

    # 1. prepare: detections (+ embeddings) per session, one decode per session for both
    jobs = {}
    for profile in args.profiles:
        prof = tcfg["profiles"][profile]
        name = prof.get("tracker") or cfg["tracker"]["name"]
        base, _, space = profile_setup(cfg, name, prof)
        reid = needs_embeddings(name, {**base, **prof.get("fixed", {})}) or "use_embeddings" in space
        index = DipserIndex.from_config(cfg)
        keys = index.sessions() if prof["sessions"] == "all" else [SessionKey(*s.split("/")) for s in prof["sessions"]]
        for key in keys:
            j = jobs.setdefault((str(key), prof["max_frames"]), {"reid": None})
            if reid and j["reid"] is None:
                j["reid"] = name
    print(f"prepare: {len(jobs)} sessions on {args.gpu_workers} GPU workers", flush=True)
    start = time.monotonic()
    with ProcessPoolExecutor(args.gpu_workers) as ex:
        futs = [ex.submit(prepare_session, cfg, k, mf, tcfg["det_conf_floor"],
                          sample_augmentation(cfg["augmentation"], k, args.augment) if args.augment else None,
                          tag, j["reid"]) for (k, mf), j in jobs.items()]
        for n, f in enumerate(as_completed(futs), 1):
            k, what = f.result()
            print(f"  [{n}/{len(futs)}] {k}: {what}  ({(time.monotonic() - start) / 60:.1f} min)", flush=True)

    # 2. search: one thread per scan, one shared pool of tracking workers
    scans = [Scan(cfg, p, None, t, args.augment, beam=args.beam, start=args.start, fix_det_conf=fix, tag=args.tag)
             for p, t in zip(args.profiles, trials)]
    pool = SessionPool(args.workers)
    results, threads = {}, []

    def run(scan):
        try:
            results[scan.label] = scan.finish(scan.search(runner=pool.run, prefetch=pool.prefetch(scan)))
        except BaseException as e:
            results[scan.label] = e
            raise

    for scan in scans:
        th = threading.Thread(target=run, args=(scan,), name=scan.label)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    print(f"\nshared passes: {pool.passes} session passes served {pool.items} (trial, session) items "
          f"({pool.items / max(pool.passes, 1):.1f} per pass)")
    for label, r in results.items():
        print(f"{label}: {'FAILED: ' + repr(r) if isinstance(r, BaseException) else 'best ' + format(r['best']['score'], '.3f')}")


def extract(args) -> None:
    import json
    import os

    from scripts.run_pipeline import extract_jobs, features_dir, session_stem, tracks_dir
    from src.datasets import DipserIndex
    from src.ingestion import aug_tag

    cfg = load_config(args.config, parse_sets(args.set))
    out, aug = features_dir(cfg), cfg.get("augmentation", {})
    for d in (out, tracks_dir(cfg)):
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def done(key, copy) -> bool:
        stem = session_stem(key) + ("" if copy is None else f".{aug_tag(aug, copy)}")
        return (out / f"{stem}.csv").exists() and (out / f"{stem}.tracks.csv").exists()

    jobs = extract_jobs(cfg, DipserIndex.from_config(cfg).sessions())
    todo = [(k, c) for k, c in jobs if not done(k, c)]
    print(f"extract: {len(jobs) - len(todo)} of {len(jobs)} recordings already done, {len(todo)} to go "
          f"on {args.workers} workers -> {out}", flush=True)
    if not todo:
        return
    threads = max(1, (os.cpu_count() or 1) // args.workers)
    start = time.monotonic()
    with ProcessPoolExecutor(args.workers, initializer=_extract_init, initargs=(cfg, threads)) as ex:
        futs = [ex.submit(extract_job, cfg, str(k), c) for k, c in todo]
        for n, f in enumerate(as_completed(futs), 1):
            el = time.monotonic() - start
            print(f"[{n}/{len(todo)}] {f.result()}  ({el / 3600:.2f} h, ~{el / n * (len(todo) - n) / 3600:.1f} h left)",
                  flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tracker-scans", help="several tracker scans at once, sharing decoded frames")
    t.add_argument("--profiles", nargs="+", default=["full", "full_reid_cmc", "full_strongsort"])
    t.add_argument("--trials", nargs="+", type=int, help="one budget per profile (default: each profile's n_trials)")
    t.add_argument("--augment", type=int, metavar="COPY", help="augmented copy COPY of every session (T2: 1)")
    t.add_argument("--workers", type=int, default=5, help="tracking worker processes (CPU)")
    t.add_argument("--gpu-workers", type=int, default=2, help="prepare-stage worker processes (GPU)")
    t.add_argument("--beam", type=int, default=2)
    t.add_argument("--fix-det-conf", action="store_true", help="keep the detection threshold at detector.conf")
    t.add_argument("--det-conf-from", metavar="JSON",
                   help="detector.conf from a detector scan's result (D2: outputs/tuning/detector_search_augmented.json), fixed")
    t.add_argument("--start", help="bisection search: start from an earlier scan's .json (its best trial), as tune_tracker")
    t.add_argument("--tag", help="suffix for the result files, as tune_tracker --tag (T3: pose)")
    e = sub.add_parser("extract", help="run_pipeline extract --reuse, several recordings at once")
    e.add_argument("--workers", type=int, default=3, help="worker processes, each with its own pipeline on the GPU")
    for p in (t, e):
        p.add_argument("--config", default="configs/deliverable3.json")
        p.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    if args.command == "tracker-scans":
        tracker_scans(args)
    elif args.command == "extract":
        extract(args)


if __name__ == "__main__":
    main()
