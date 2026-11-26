"""Shared loading for the clustering scripts (guide 15 §15.9): cached sessions, per-frame channel
series per window, per-track standardisation (R6), descriptor sets, candidate tables and the
prototype codebook per fold. Nothing here re-runs the frame pass: everything reads the caches
`run_pipeline extract` wrote (tracks/<tracking hash>/, features/<feature hash>/).
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_pipeline import features_dir, parse_sets, tracks_dir
from src.aggregation import WindowAggregator, frames_in
from src.cfmu.classrel import PrototypeCodebook
from src.cfmu.extractor import NullClusterExtractor
from src.core.config import config_hash, load_config
from src.core.store import load_track
from src.core.types import SessionKey, Tracklet, Window
from src.tracking import filter_visible

DEFAULT_CONFIG = "docs/experiments/09_vfoa.json"      # head direction is the natural input (guide 15)
DEFAULT_OUT = "outputs/experiments/10_clustering"
DEFAULT_CHANNELS = ["hp.yaw", "hp.head_height", "rf.total", "kg.speed"]
EXPERTS = ["labeler_01", "labeler_02", "labeler_03", "labeler_04", "labeler_05"]
SELF = "self_labeling"
PER_STUDENT, PER_WINDOW = "trk.cluster.", "cluster."    # the column names guide 15 §15.7 fixes

# windows where a student or a frame has no finite value are expected (occlusion, gaps) and give NaN
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)


# ------------------------------------------------------------------ arguments and config
def base_parser(doc: str, workers: bool = False) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, JSON values")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="session stems to use (group_01__experiment_03__context_cam_01 ...); default every cached one")
    if workers:
        ap.add_argument("--workers", type=int, default=1,
                        help="sessions processed in parallel (processes; CPU only, the GPU is not used here)")
    return ap


def map_sessions(fn, jobs: list[tuple], workers: int):
    """fn(*job) for every job, in worker processes when workers > 1; yields (job, result) as they
    finish, so progress shows on long runs. fn must be a module-level function (picklable)."""
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            yield job, fn(*job)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *job): job for job in jobs}
        for f in as_completed(futures):
            yield futures[f], f.result()


def write_table(df: pd.DataFrame, path: Path) -> None:
    """A result table as CSV (spreadsheets) and JSON (records; NaN -> null), same stem."""
    path = Path(path)
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".json").write_text(df.to_json(orient="records", indent=1, double_precision=6), encoding="utf-8")


def load(args) -> tuple[dict, Path]:
    cfg = load_config(args.config, parse_sets(args.set))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return cfg, out


def cached_stems(cfg: dict, only: list[str] | None = None, originals: bool = True) -> list[str]:
    """Stems with both a frame-pass cache (.npz) and a window table (.csv). Augmented copies
    (stems with a '.aug' tag) are left out unless originals is False."""
    npz = {p.stem for p in tracks_dir(cfg).glob("*.npz") if not p.name.endswith(".tmp.npz")}
    csv = {p.stem for p in features_dir(cfg).glob("*.csv") if "." not in p.stem}
    csv |= {p.name[:-4] for p in features_dir(cfg).glob("*.csv")
            if ".aug" in p.name and not p.name.endswith((".tracks.csv", ".labelwin.csv"))}
    stems = sorted(npz & csv)
    if originals:
        stems = [s for s in stems if "." not in s]
    if only:
        stems = [s for s in stems if s in set(only)]
    if not stems:
        raise SystemExit(f"no cached sessions in {tracks_dir(cfg)} and {features_dir(cfg)}; run "
                         f"`python -m scripts.run_pipeline extract --config <this config>` first")
    return stems


def stem_of(session: str) -> str:
    """group_01/experiment_03/context_cam_01#aug1-xxxx -> group_01__experiment_03__context_cam_01.aug1-xxxx"""
    return session.replace("/", "__").replace("#", ".")


def params_hash(params: dict) -> str:
    return config_hash(params)[:8]


# ------------------------------------------------------------------ one session
@dataclass
class SessionData:
    stem: str
    key: SessionKey
    frame_t: np.ndarray
    fps: float
    tracklets: list[Tracklet]          # filtered exactly as the feature pass filters them
    windows: list[Window]              # the regular grid of the cached window table
    tracks: pd.DataFrame               # <stem>.tracks.csv: window, tid, trk.* per visible student
    aggregator: WindowAggregator

    def visible(self, w: Window) -> list[Tracklet]:
        """The window's visible students, sliced to it (the aggregator's own rule, min_coverage)."""
        return self.aggregator.visible(w, self.tracklets)

    def labelwin(self, cfg: dict) -> pd.DataFrame | None:
        """<stem>.labelwin.csv (windows centred on rating entries), if train_models built it."""
        p = features_dir(cfg) / f"{self.stem}.labelwin.csv"
        return pd.read_csv(p) if p.exists() else None


def load_session(cfg: dict, stem: str) -> SessionData:
    st = load_track(tracks_dir(cfg) / f"{stem}.npz")
    fps = st.timing.get("fps") or 1.0
    tracklets = filter_visible(st.tracklets, st.frame_t, fps, **cfg["tracklets"])
    df = pd.read_csv(features_dir(cfg) / f"{stem}.csv")
    windows = [Window(a, b) for a, b in zip(df.t_start, df.t_end)]
    tp = features_dir(cfg) / f"{stem}.tracks.csv"
    tracks = pd.read_csv(tp) if tp.exists() else pd.DataFrame(columns=["window", "tid"])
    agg = WindowAggregator([], NullClusterExtractor(), **cfg["aggregation"])
    return SessionData(stem, SessionKey(*stem.split(".")[0].split("__")), np.asarray(st.frame_t, float), fps,
                       tracklets, windows, tracks, agg)


# ------------------------------------------------------------------ per-frame channels (synchrony space)
def channel(tr: Tracklet, name: str) -> np.ndarray:
    """One per-frame channel of a tracklet. Derived channels: kg.speed (Kalman centre speed, px per
    frame), rf.total (mean of the residual-flow cells); anything else is a stored signal."""
    if name == "kg.speed":
        return np.hypot(tr.velocity[:, 0], tr.velocity[:, 1])
    if name == "rf.total":
        cells = [v for k, v in tr.signals.items() if k.startswith("rf.mag_")]
        if not cells:
            return np.full(len(tr.t), np.nan)
        stack = np.stack(cells)
        n = np.isfinite(stack).sum(axis=0)
        return np.where(n > 0, np.nansum(stack, axis=0) / np.maximum(n, 1), np.nan)
    return np.asarray(tr.signals.get(name, np.full(len(tr.t), np.nan)), float)


def available_channels(sd: SessionData) -> list[str]:
    keys = sorted({k for tr in sd.tracklets for k in tr.signals})
    derived = ["kg.speed"] + (["rf.total"] if any(k.startswith("rf.mag_") for k in keys) else [])
    return derived + [k for k in keys if not k.startswith("rf.mag_") and k != "hp.matched"]


def series_matrix(sd: SessionData, w: Window, tracks: list[Tracklet], name: str) -> np.ndarray:
    """(n, T) the channel of every visible student on the window's decoded frames; NaN where the
    student is missing. All tracks of a camera share the frame timestamps (guide 15 §15.2)."""
    ft = sd.frame_t[frames_in(sd.frame_t, w)]
    S = np.full((len(tracks), len(ft)), np.nan)
    if not len(ft):
        return S
    for i, tr in enumerate(tracks):
        j = np.clip(np.searchsorted(ft, tr.t), 0, len(ft) - 1)
        ok = np.abs(ft[j] - tr.t) < 0.5 / sd.fps
        S[i, j[ok]] = channel(tr, name)[ok]
    return S


# ------------------------------------------------------------------ descriptors (descriptor space)
def descriptor_sets(cols: list[str], out: Path | None = None) -> dict[str, list[str]]:
    """Named descriptor subsets of the trk.* columns, plus `subset` from descriptor_subset.py."""
    trk = sorted(c for c in cols if c.startswith("trk.") and not c.startswith(PER_STUDENT))
    sets = {"all": trk,
            "head": [c for c in trk if c.startswith("trk.hp.")],
            "motion": [c for c in trk if c.startswith("trk.kg.")],
            "flow": [c for c in trk if c.startswith("trk.rf.")],
            "scb": [c for c in trk if c.startswith("trk.scb.")]}
    if out is not None and (out / "descriptor_subset.json").exists():
        chosen = json.loads((out / "descriptor_subset.json").read_text(encoding="utf-8"))["subset"]
        sets["subset"] = [c for c in chosen if c in trk]
    return {k: v for k, v in sets.items() if v}


class TrackStandardiser:
    """Per-track standardisation over the session (R6): every descriptor minus the track's own
    session mean, over its own session SD; session-wide statistics where a track has fewer than
    `min_rows` windows or a flat descriptor. Fitted on the regular-grid rows (tracks.csv)."""

    def __init__(self, tracks: pd.DataFrame, cols: list[str], min_rows: int = 5):
        t = tracks.reindex(columns=["tid"] + cols)
        self.cols = cols
        self.glob_mu = t[cols].mean()
        self.glob_sd = t[cols].std(ddof=0).replace(0, np.nan)
        g = t.groupby("tid")[cols]
        n = g.size()
        ok = n[n >= min_rows].index
        self.mu = g.mean().loc[ok]
        self.sd = g.std(ddof=0).loc[ok].where(lambda s: s > 1e-12)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.cols).to_numpy(float)
        tid = df["tid"].to_numpy()
        mu = self.mu.reindex(tid).to_numpy(float)
        sd = self.sd.reindex(tid).to_numpy(float)
        mu = np.where(np.isfinite(mu), mu, self.glob_mu.to_numpy(float))
        sd = np.where(np.isfinite(sd), sd, self.glob_sd.to_numpy(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (X - mu) / sd


# ------------------------------------------------------------------ label rows and candidates
def label_table(cfg: dict, with_windows: bool = False):
    """The training rows train_models builds (one per rating entry and linked track), augmented
    copies included (flag `aug`), with `stem` and `student` (student-session-labeler) keys. With
    with_windows, also the regular-grid window table the gate trains on."""
    from scripts.train_models import label_rows, load_table, with_augmented

    df = with_augmented(cfg, load_table(cfg))
    rows = label_rows(cfg, df)
    rows = rows.assign(stem=rows["session"].map(stem_of), t_key=rows["t"].astype(float).round(3))
    rows["student"] = rows["group"] + "/" + rows["exp"].astype(str) + "/" + rows["sid"].astype(str) + "/" + rows["labeler"]
    rows = rows.reset_index(drop=True)
    return (rows, df) if with_windows else rows


def latest_candidates(out: Path, given: str | None) -> Path:
    if given:
        return Path(given)
    dirs = sorted((out / "candidates").glob("*-*"), key=lambda p: p.stat().st_mtime)
    if not dirs:
        raise SystemExit(f"no candidate tables under {out / 'candidates'}; run `python -m scripts.clustering.candidates` first")
    return dirs[-1]


def base_columns(cfg: dict, rows: pd.DataFrame) -> list[str]:
    """The model inputs of the configured run, without any candidate: exactly what train_models
    gives the per-labeler model (src/experts/columns.model_columns, experiment 09): the trk.*
    per-student columns, the rel.hp.* class-relative head direction when engagement.class_relative,
    the class context when engagement.use_class_context, filtered by engagement.columns. The
    candidates (trk.cluster.*, cluster.*) are kept out here and added per arm."""
    from src.experts.columns import model_columns

    available = [c for c in rows if not c.startswith((PER_STUDENT, PER_WINDOW))]
    return model_columns(cfg, available, sorted(c for c in rows if c.startswith("vec.")))


def train_mask(cfg: dict, rows: pd.DataFrame, train: np.ndarray) -> np.ndarray:
    """The training rows a model may fit on: `train` and labels.train_keep (experiment 09's
    rebalancing, drawn per entry and seeded). Held-out rows are always scored in full."""
    from src.experts import train_keep_mask

    return train & train_keep_mask(rows, cfg["labels"].get("train_keep"), cfg["seed"])


def levels_of(cfg: dict) -> np.ndarray:
    """The rating levels the metrics round to: 1-5, or labels.collapse's levels (e.g. 1, 3, 5)."""
    from src.labels.interp import rating_levels

    return rating_levels(cfg["labels"].get("collapse"))


# ------------------------------------------------------------------ folds (train_models' protocol)
SPLITS = ("activity", "group")


def splits(arg: str) -> list[str]:
    """--split activity | group | both."""
    return list(SPLITS) if arg == "both" else [arg]


def split_parser_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--split", choices=["both", *SPLITS], default="both",
                    help="activity: train on 4 activities, test on the other 2, over all 15 pairs; "
                         "group: leave one group out (unseen students); both (default): report each")


def check_split(cfg: dict, split: str) -> None:
    """An activity split tests on activities the model never saw: that needs one expert shared by
    every activity (mixture.expert_scope = shared, as in experiments 06-09). Per-activity experts
    would have no expert for a held-out activity."""
    if split == "activity" and cfg["mixture"]["expert_scope"] != "shared":
        raise SystemExit("--split activity needs mixture.expert_scope=shared (a per-activity expert has nothing "
                         "to predict a held-out activity with); add --set mixture.expert_scope=shared or use --split group")


def folds(rows: pd.DataFrame, split: str = "group"):
    """(fold name, training mask incl. augmented copies, test mask of original rows only).

    group     leave one group out, exactly as train_models.main_per_labeler splits: the test
              students are never seen in training
    activity  every way of holding out 2 of the activities (15 folds for 6): train on the other
              activities, of every group. The same students and seats are in train and test
              (each group records every activity), so this tests generalisation to new activities,
              not to new students; compare with the group split to see how much that helps."""
    from itertools import combinations

    aug = rows["aug"].to_numpy(bool)
    if split == "group":
        group = rows["group"].to_numpy()
        groups = sorted(set(group.tolist()))
        if len(groups) < 2:
            raise SystemExit(f"leave-one-group-out needs at least two groups with label rows; have {groups}. "
                             f"Extract sessions of more groups first.")
        for g in groups:
            yield g, group != g, (group == g) & ~aug
    elif split == "activity":
        exp = rows["exp"].astype(str).to_numpy()
        exps = sorted(set(exp.tolist()))
        if len(exps) < 3:
            raise SystemExit(f"the activity split holds out 2 activities and needs at least 3 with label rows; have {exps}")
        for pair in combinations(exps, 2):
            test = np.isin(exp, pair)
            yield f"test_{pair[0]}+{pair[1]}", ~test, test & ~aug
    else:
        raise SystemExit(f"unknown split {split!r}")


def fold_context(cfg: dict, rows: pd.DataFrame, windows: pd.DataFrame | None, train_rows: np.ndarray,
                 mode: str, split: str = "group") -> np.ndarray:
    """Gate probabilities for every row in one fold: `oracle` (the true activity, isolates the
    experts' inputs) or `classified` (the configured gate trained on the grid windows of the
    training sessions, as train_models does). The activity split always uses oracle: a gate cannot
    name an activity it never saw, and with a shared expert the context does not change the rating."""
    from src.spu import GATES, context_probs

    classes = cfg["classes"]
    if mode == "oracle" or split == "activity":
        return context_probs("oracle", classes, rows["exp"].astype(str).to_numpy())
    scene = sorted(c for c in windows if c.startswith("scene."))
    vec = sorted(c for c in windows if c.startswith("vec."))
    gate_cols = scene + (vec if cfg["gate_uses_vec"] else [])
    held_out = set(rows.loc[~train_rows, "group"])
    tr = ~windows["group"].isin(held_out).to_numpy()
    gate = GATES.build(cfg["gate"]["name"], classes=classes, seed=cfg["seed"], **cfg["gate"].get("params", {}))
    gate.fit(windows.loc[tr, gate_cols].to_numpy(float), windows.loc[tr, "exp"].to_numpy())
    return gate.predict_proba(rows.reindex(columns=gate_cols).to_numpy(float))


def fit_model(cfg: dict, rows: pd.DataFrame, cols: list[str], train: np.ndarray):
    """The configured per-labeler model (expert, mixture, pooled or not) on the training rows."""
    from scripts.train_models import is_pooled, per_labeler_factory
    from src.experts import PerLabelerModel

    X = rows.reindex(columns=cols).to_numpy(float)
    train = train_mask(cfg, rows, train)                  # labels.train_keep, as train_models
    return PerLabelerModel.fit(per_labeler_factory(cfg, cols, cfg["mixture"]["expert_scope"]), X[train],
                               rows["target"].to_numpy(float)[train], rows["exp"].to_numpy()[train],
                               rows["labeler"].to_numpy()[train], pooled=is_pooled(cfg))


def own_ratings(model, X: np.ndarray, G: np.ndarray, labeler: np.ndarray) -> np.ndarray:
    """Each row's prediction by its own labeler's model, as a rating (1-5)."""
    from src.labels.interp import to_rating

    preds = model.predict(X, G)
    return np.array([to_rating(preds[l][i]) if l in preds else np.nan for i, l in enumerate(labeler)])


def score(rows: pd.DataFrame, pred: np.ndarray, tau_att: float, levels: np.ndarray | None = None) -> dict[str, dict]:
    """rating_metrics per track (experts, self), each entry counted once (train_models.per_entry);
    levels: levels_of(cfg), so collapsed labels (labels.collapse) are scored as train_models scores them."""
    from scripts.train_models import per_entry
    from src.evaluation import rating_metrics

    is_self = rows["labeler"].to_numpy() == SELF
    out = {}
    for track, sel in (("experts", ~is_self), ("self", is_self)):
        if sel.any():
            out[track] = rating_metrics(*per_entry(rows[sel], pred[sel]), tau_att, levels)
    return out


def per_track_normalised(cfg: dict, rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """rows with every trk.* column of `cols` standardised per student-track over its own session
    (TrackStandardiser, fitted on the session's regular-grid rows in <stem>.tracks.csv). Uses no
    labels and only each session's own features, so it is the same transform at inference; what is
    left is how a student's window differs from their own usual, not their seat or distance."""
    out = rows.copy()
    trk = [c for c in cols if c.startswith("trk.") and c in out]
    if not trk:
        return out
    pos = [out.columns.get_loc(c) for c in trk]
    for stem, idx in rows.groupby("stem").indices.items():
        tp = features_dir(cfg) / f"{stem}.tracks.csv"
        grid = pd.read_csv(tp) if tp.exists() else rows.iloc[idx]
        out.iloc[idx, pos] = TrackStandardiser(grid, trk).transform(rows.iloc[idx])
    return out


def with_fold_prototypes(cfg: dict, rows: pd.DataFrame, train: np.ndarray, set_name: str, k: int,
                         out: Path | None) -> pd.DataFrame:
    """rows with this fold's prototype columns: the codebook fitted on the training rows' sessions
    (augmented copies included), applied to every session."""
    stems = sorted(rows["stem"].unique())
    fit_stems = sorted(rows.loc[train, "stem"].unique())
    table = prototype_table(cfg, stems, fit_stems, set_name, k, cfg["seed"], out)
    keep = [c for c in rows if ".proto." not in c]
    return attach(rows[keep], table)


def candidate_dir(cfg: dict, out: Path, params: dict) -> Path:
    """Candidates are cached per feature hash and candidate settings, like the pipeline's caches."""
    return out / "candidates" / f"{features_dir(cfg).name}-{params_hash(params)}"


def attach(rows: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Join per-(stem, window centre, tid) feature columns onto the label rows."""
    t = table.assign(t_key=table["t_centre"].astype(float).round(3)).drop(columns="t_centre")
    t["tid"] = t["tid"].astype(int)
    # rating windows whose centres are < 0.5 ms apart (two labelers' entries at the same moment) share
    # a key after rounding; they cover the same frames, so keep one, or the left join duplicates rows
    t = t.drop_duplicates(["stem", "t_key", "tid"])
    r = rows.assign(tid=rows["tid"].astype(int))
    return r.merge(t, on=["stem", "t_key", "tid"], how="left")


def _clean(v: float):
    """6 significant digits (the features are far noisier than that); NaN -> null."""
    return float(f"{v:.6g}") if np.isfinite(v) else None


def candidates_to_json(df: pd.DataFrame, stem: str, meta: dict) -> dict:
    """One session's candidates as JSON. The feature names are listed once
    (per_student_features, per_window_features); each window centre then holds its window-level
    values once and one value list per visible student, in those orders (NaN -> null):
      {"windows": [{"t_centre": 39417.04, "window": [v, ...],
                    "students": [{"tid": 3, "values": [v, ...]}, ...]}, ...]}"""
    per = [c for c in df if c.startswith(PER_STUDENT)]
    win = [c for c in df if c.startswith(PER_WINDOW)]
    windows = []
    for _, g in (df.groupby("t_centre", sort=True) if len(df) else []):
        P = g[per].to_numpy(float)
        windows.append({"t_centre": float(g["t_centre"].iloc[0]),
                        "window": [_clean(v) for v in g[win].iloc[0].to_numpy(float)],
                        "students": [{"tid": int(t), "values": [_clean(v) for v in row]}
                                     for t, row in zip(g["tid"].to_numpy(), P)]})
    return {"session": stem, **meta, "per_student_features": [c[len(PER_STUDENT):] for c in per],
            "per_window_features": [c[len(PER_WINDOW):] for c in win], "n_windows": len(windows),
            "n_student_rows": int(len(df)), "windows": windows}


def candidates_from_json(d: dict) -> pd.DataFrame:
    per = [PER_STUDENT + k for k in d["per_student_features"]]
    win = [PER_WINDOW + k for k in d["per_window_features"]]
    stem, tc, tid, P, W = [], [], [], [], []
    for w in d["windows"]:
        for s in w["students"]:
            stem.append(d["session"])
            tc.append(w["t_centre"])
            tid.append(s["tid"])
            P.append(s["values"])
            W.append(w["window"])
    df = pd.DataFrame({"stem": stem, "t_centre": tc, "tid": tid})
    to = lambda rows, cols: pd.DataFrame(np.array(rows, dtype=float).reshape(len(rows), len(cols)), columns=cols)
    return pd.concat([df, to(P, per), to(W, win)], axis=1)


def read_candidates(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        return candidates_from_json(json.loads(path.read_text(encoding="utf-8")))
    return pd.read_csv(path)


def candidate_files(cand_dir: Path) -> dict[str, Path]:
    """{stem: file}, JSON preferred where a session has both."""
    files = {}
    for p in sorted(cand_dir.glob("*.csv")) + sorted(cand_dir.glob("*.json")):
        if p.name == "params.json" or ".tmp." in p.name:
            continue
        files[p.name[: -len(p.suffix)]] = p
    return files


def load_candidates(rows: pd.DataFrame, cand_dir: Path) -> pd.DataFrame:
    parts = [read_candidates(p) for p in candidate_files(cand_dir).values()]
    if not parts:
        raise SystemExit(f"no candidate tables in {cand_dir}; run `python -m scripts.clustering.candidates` first")
    return attach(rows, pd.concat(parts, ignore_index=True))


def candidate_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df if c.startswith((PER_STUDENT, PER_WINDOW)))


def prototype_table(cfg: dict, stems: list[str], fit_stems: list[str], set_name: str, k: int,
                    seed: int, out: Path | None = None) -> pd.DataFrame:
    """Option 4: a k-prototype codebook fitted (no labels) on the label-window descriptors of
    `fit_stems` (the training groups of one fold, augmented copies included), applied to every
    centred window of `stems`. Returns one row per (stem, t_centre, tid) with trk.cluster.proto.*
    and cluster.proto.* columns."""
    lw, std = {}, {}
    for s in sorted(set(stems) | set(fit_stems)):
        p = features_dir(cfg) / f"{s}.labelwin.csv"
        if not p.exists():
            continue
        lw[s] = pd.read_csv(p)
        tp = features_dir(cfg) / f"{s}.tracks.csv"
        grid = pd.read_csv(tp) if tp.exists() else lw[s]
        cols = descriptor_sets(list(grid.columns), out)[set_name]
        std[s] = TrackStandardiser(grid, cols)
    pool = [std[s].transform(lw[s]) for s in fit_stems if s in lw]
    if not pool:
        raise SystemExit("prototype codebook: no label-window tables for the fitting sessions")
    cb = PrototypeCodebook(k=k, seed=seed).fit(np.vstack(pool))
    out_rows = []
    for s in stems:
        if s not in lw:
            continue
        df = lw[s]
        X = std[s].transform(df)
        for c, idx in df.groupby(df["t_centre"].round(3)).indices.items():
            per, win = cb.window_features(X[idx])
            for j, i in enumerate(idx):
                rec = {"stem": s, "t_centre": float(df["t_centre"].iloc[i]), "tid": int(df["tid"].iloc[i])}
                rec |= {f"{PER_STUDENT}proto.{set_name}.{n}": float(v[j]) for n, v in per.items()}
                rec |= {f"{PER_WINDOW}proto.{set_name}.{n}": float(v) for n, v in win.items()}
                out_rows.append(rec)
    return pd.DataFrame(out_rows)


def demean(values, keys: pd.Series):
    return values - values.groupby(keys).transform("mean")


def plot_ready():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
