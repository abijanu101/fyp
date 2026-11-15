"""Tune detector confidence + tracker parameters against the manual SID boxes (design §4, §11).

  python -m scripts.tune_tracker [--profile quick|full|full_reid_cmc|full_strongsort] [--tracker NAME]
                                 [--trials N] [--check] [--fresh] [--set key=value ...]
                                 [--randomize] [--beam B] [--start earlier_scan.json] [--augment [COPY]]

Search (default: beam bisection, src/tuning/search.py beam_bisection_search)
  Binary search per numeric parameter, k branches per node (k = numeric parameters that matter
  under the profile's fixed settings), a beam of B nodes as the bound; deterministic. Each level
  costs up to (2k + 1) x B trials. It starts from the best trial of the profile's random search on
  the original data (tracker_<tracker>_<profile>.json) when that exists, or from --start (any
  earlier scan's .json), else from the profile's baseline. --randomize uses
  the seeded random search instead (the original method; resume earlier random scans with it).
  Both read and extend the same <stem>.sessions.jsonl, so results are shared between methods.

Augmented data
  --augment [COPY] runs the same profile on augmented copy COPY (default 1) of every session,
  augmented on the fly (config `augmentation`; boxes mirrored where flipped). It has its own result files
  (<stem>_<aug tag>...) and its own detection cache.

How it works
  1. Once per session, frames are decoded and the detector runs at a low confidence floor; the
     detections, frame timestamps and frame size are cached (outputs/cache/detections/).
  2. If any trial can use ReID, every floor-level detection is also embedded once with the tracker's
     own ReID model (BoxMOT 26 default `osnet-x0-25-msmt17`, shared by BoT-SORT and StrongSORT)
     and cached next to the detections (<session>.reid-<model>.npz, float32). Embeddings depend only on the
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
from src.ingestion import AugmentedZipFrameSource, aug_tag, hflip_boxes, sample_augmentation
from src.tuning import beam_bisection_search, random_search

REID_MODEL = "osnet-x0-25-msmt17"     # BoxMOT 26 default ReID model for both BoT-SORT and StrongSORT


class SessionDetections:
    """Cached floor-confidence detections (and optionally ReID embeddings) for one session."""

    def __init__(self, key: SessionKey, zip_path: Path, cache: Path, frame_t: np.ndarray,
                 frame_hw: tuple[int, int], dets: list[Detections], max_frames: int | None,
                 augment: dict | None = None):
        self.key, self.zip_path, self.cache, self.frame_t, self.frame_hw = key, zip_path, cache, frame_t, frame_hw
        self.dets, self.max_frames, self.augment = dets, max_frames, augment

    @staticmethod
    def source(zip_path: Path, max_frames: int | None, augment: dict | None):
        """The session's frames: as recorded, or augmented as they are read (never written to disk)."""
        if augment is None:
            return ZipFrameSource(zip_path, max_frames=max_frames)
        return AugmentedZipFrameSource(zip_path, augment, max_frames=max_frames)

    @classmethod
    def load_or_detect(cls, cfg: dict, index: DipserIndex, key: SessionKey, max_frames: int | None,
                       floor: float, detector_factory, augment: dict | None = None,
                       augment_tag: str | None = None) -> "SessionDetections":
        """detector_factory() returns the shared detector; it is only called on a cache miss.
        augment / augment_tag: an augmented copy's parameters and tag; its detections are cached
        separately (the original data's cache key is unchanged)."""
        det_cfg = {**cfg["detector"], "conf": floor}
        source = {"augment": augment_tag} if augment is not None else {}
        cache = (Path(cfg["output_dir"]) / "cache" / "detections" /
                 config_hash({"detector": det_cfg, "max_frames": max_frames, "v": 2, **source})[:12] /
                 f"{session_stem(key)}.npz")
        zip_path = index.frames_zip(key)
        if not cache.exists():
            detector = detector_factory()
            t, dets, hw = [], [], (0, 0)
            for f in cls.source(zip_path, max_frames, augment):
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
        return cls(key, zip_path, cache, frame_t, tuple(int(v) for v in frame_hw), dets, max_frames, augment)

    def position(self, frame: Frame) -> int:
        """Row of this frame in the cache (by timestamp, so a frame that fails to decode cannot shift
        the alignment)."""
        return int(np.searchsorted(self.frame_t, frame.t))

    def frames(self, pixels: bool):
        """Decoded frames, or blank frames with the same timestamps when the tracker ignores pixels."""
        if pixels:
            yield from self.source(self.zip_path, self.max_frames, self.augment)
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
        for f in self.source(self.zip_path, self.max_frames, self.augment):
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


def profile_setup(cfg: dict, name: str, prof: dict) -> tuple[dict, dict, dict]:
    """(base_options, fixed, space) for a profile. Trial 0 runs base_options. A profile may give its
    own `base_options` (e.g. the best known values for a tracker other than the configured one);
    otherwise the configured tracker starts from tracker.options and another tracker from BoxMOT's
    defaults (BoT-SORT options are not valid StrongSORT options and vice versa)."""
    if "base_options" in prof:
        base_options = dict(prof["base_options"])
    else:
        base_options = dict(cfg["tracker"].get("options", {})) if name == cfg["tracker"]["name"] else {}
    base_options.pop("det_thresh", None)                     # unused by BoxMOT's BoT-SORT and StrongSORT
    fixed = prof.get("fixed", {})
    space = {k: v for k, v in cfg["tuning"]["tracker"]["space"][name].items() if k not in fixed}
    return base_options, fixed, space


def run_fingerprint(cfg: dict, name: str, prof: dict, dataset_tag: str | None = None) -> str:
    """Everything that affects a profile's scores; a resumable log only resumes under the same one.
    n_trials is left out so raising --trials continues the same seeded sequence. dataset_tag: the
    augmented copy the scores come from (None = the original data, whose fingerprints are unchanged)."""
    base_options, _, space = profile_setup(cfg, name, prof)
    tcfg = cfg["tuning"]["tracker"]
    return config_hash({"tracker": name,
                        "profile": {k: v for k, v in prof.items() if k not in ("n_trials", "start_from")},
                        "space": space, "base_options": base_options,
                        "default_det_conf": cfg["detector"]["conf"], "detector": cfg["detector"],
                        "floor": tcfg["det_conf_floor"], "min_iou": cfg["association"]["min_iou"],
                        "seed": cfg["tuning"]["seed"], "reid": REID_MODEL, "v": 2,
                        **({"dataset": dataset_tag} if dataset_tag else {})})


def inactive_params(name: str, base_options: dict, fixed: dict) -> set[str]:
    """Parameters with no effect under the profile's fixed settings: searching them wastes trials."""
    if name == "botsort" and not {**base_options, **fixed}.get("use_embeddings", True):
        return {"proximity_thresh", "appearance_thresh"}       # appearance gating, ReID only
    return set()


def start_from(path: str, space: dict) -> dict:
    """Starting point from an earlier scan's .json (its best trial, as a config patch)."""
    patch = json.loads(Path(path).read_text(encoding="utf-8"))["config_patch"]
    return {"det_conf": patch["detector.conf"], **{k: v for k, v in patch["tracker.options"].items() if k in space}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--tracker")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--check", "--check-blank", dest="check", action="store_true",
                    help="verify the fast path reproduces the tracker's own tracks on the first session, then exit")
    ap.add_argument("--fresh", action="store_true", help="move an existing results file aside and start over")
    ap.add_argument("--randomize", action="store_true",
                    help="seeded random search (the earlier method) instead of the beam bisection search")
    ap.add_argument("--beam", type=int, default=2, help="bisection search: nodes kept per level")
    ap.add_argument("--start", help="bisection search: start from an earlier scan's .json (its best trial)")
    ap.add_argument("--augment", nargs="?", type=int, const=1, metavar="COPY",
                    help="run on augmented copy COPY (default 1) of every session, augmented on the fly")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    tcfg = cfg["tuning"]["tracker"]
    prof = tcfg["profiles"][args.profile]
    name = args.tracker or prof.get("tracker") or cfg["tracker"]["name"]
    if prof.get("tracker", name) != name:
        raise SystemExit(f"profile '{args.profile}' is for {prof['tracker']}, not {name}")
    base_options, fixed, space = profile_setup(cfg, name, prof)
    index = DipserIndex.from_config(cfg)
    min_iou = cfg["association"]["min_iou"]

    keys = index.sessions() if prof["sessions"] == "all" else [SessionKey(*s.split("/")) for s in prof["sessions"]]
    dataset_tag = aug_tag(cfg["augmentation"], args.augment) if args.augment else None
    params_of = lambda key: sample_augmentation(cfg["augmentation"], str(key), args.augment) if args.augment else None
    if args.augment:
        print(f"augmented copy {args.augment} ({dataset_tag}), augmented on the fly; boxes mirrored where flipped",
              flush=True)
    if args.check:
        keys = keys[:1]
    det_pipe = Pipeline({**cfg, "detector": {**cfg["detector"], "conf": tcfg["det_conf_floor"]}})
    shared_detector = lambda: det_pipe.detector              # built once, on the first cache miss
    sessions = []
    for i, key in enumerate(keys, 1):
        p = params_of(key)
        sd = SessionDetections.load_or_detect(cfg, index, key, prof["max_frames"], tcfg["det_conf_floor"],
                                              shared_detector, p, dataset_tag)
        subjects = index.annotations(key)
        if p is not None and p["hflip"]:                     # a mirrored copy: mirror the manual boxes too
            subjects = {s: hflip_boxes(b[None], sd.frame_hw[1])[0] for s, b in subjects.items()}
        sessions.append((sd, subjects))
        print(f"[{i}/{len(keys)}] {key}: {len(sd.dets)} frames, {sum(map(len, sd.dets))} detections", flush=True)

    def options_for(params: dict) -> tuple[dict, float]:
        """Tracker options for a trial. det_conf filters the cached detections (it stands in for
        detector.conf); it is not a tracker option (BoxMOT's det_thresh is unused by both trackers)."""
        params = {**dict(params), **fixed}
        det_conf = params.pop("det_conf", cfg["detector"]["conf"])
        return {**base_options, **params}, det_conf

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
    stem = f"tracker_{name}_{args.profile}" + (f"_{dataset_tag}" if dataset_tag else "")
    results_path = out / f"{stem}.sessions.jsonl"      # shared by both search methods (a result cache)
    fingerprint = run_fingerprint(cfg, name, prof, dataset_tag)
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

    n_trials = args.trials or prof["n_trials"]
    if args.randomize:
        method = "random"
        trials = random_search(space, objective, n_trials, cfg["tuning"]["seed"], baseline=baseline, on_trial=log)
    else:
        method = "bisect"
        # Start from the winner of a random search on the original data: this profile's own, or the
        # one named by the profile's `start_from` (e.g. BoT-SORT + ReID starts from vanilla BoT-SORT).
        start = args.start or out / f"tracker_{name}_{prof.get('start_from', args.profile)}.json"
        if Path(start).exists():
            baseline = {**baseline, **start_from(start, space)}
            print(f"starting point: best trial of {Path(start).name}", flush=True)
        elif args.start:
            raise SystemExit(f"--start {args.start}: not found")
        else:
            print(f"starting point: the profile's baseline (no {Path(start).name} yet)", flush=True)
        skip = inactive_params(name, base_options, fixed)
        k = sum(isinstance(v, dict) for p, v in space.items() if p not in skip)
        print(f"beam bisection: {k} numeric parameters, beam {args.beam}: up to {(2 * k + 1) * args.beam} trials "
              f"per level, budget {n_trials}" + (f"; held fixed (inactive): {sorted(skip)}" if skip else ""), flush=True)
        trials = beam_bisection_search(space, objective, n_trials, baseline, beam=args.beam, inactive=skip,
                                       on_trial=log)
    stem = stem + ("" if method == "random" else "_bisect")    # random keeps the original file names

    pd.DataFrame([{**{k: v for k, v in t.items() if k not in ("params", "by_group")},
                   **{f"group.{g}": v for g, v in t["by_group"].items()}, **t["params"]} for t in trials]) \
        .to_csv(out / f"{stem}.csv", index=False)
    best_opts, det_conf = options_for(trials[0]["params"])
    base = next(t for t in trials if t["trial"] == 0)
    summary = {"tracker": name, "profile": args.profile, "search": method, "dataset": dataset_tag or "original",
               "n_sessions": len(sessions),
               "max_frames": prof["max_frames"], "fixed": fixed, "objective": tcfg["objective"],
               "baseline": {"score": base["score"], "by_group": base["by_group"]},
               "best": {k: v for k, v in trials[0].items() if k != "params"},
               "config_patch": {"detector.conf": det_conf, "tracker.options": best_opts}}
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nbaseline {base['score']:.3f} {base['by_group']} -> best {trials[0]['score']:.3f} {trials[0]['by_group']}")
    print("config patch:", json.dumps(summary["config_patch"]))


if __name__ == "__main__":
    main()
