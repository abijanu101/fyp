"""Which feature columns a per-student model sees (experiment 09 §2), and the class-relative head
direction columns (`rel.*`), computed at training time from the per-track rows of one window.

Config (both absent = every trk.* column, plus vec.* when engagement.use_class_context is on; hashes of
configs without them are unchanged):

  engagement.class_relative   true: add the rel.hp.* columns below
  engagement.columns          {"include": [prefix, ...], "exclude": [prefix, ...]}: keep a column if it
                              starts with any include prefix (no include list = keep all), then drop it
                              if it starts with any exclude prefix. E.g. experiment 09's arms:
                                08's 27 inputs     {"exclude": ["trk.hp.", "vec.hp.", "rel."]}
                                head pose alone    {"include": ["trk.hp.", "vec.hp.", "rel."]}
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.types import TRACK_PREFIX, VEC_PREFIX

REL_PREFIX = "rel."
YAW, HEIGHT = TRACK_PREFIX + "hp.yaw.mean", TRACK_PREFIX + "hp.head_height.mean"
REL_COLUMNS = ("rel.hp.yaw_dev", "rel.hp.yaw_absdev", "rel.hp.height_dev", "rel.hp.same_dir")


def add_class_relative(rows: pd.DataFrame, by: str) -> pd.DataFrame:
    """rows: one row per (window, visible student) with the trk.hp.* columns; `by` names the window
    column ("t_centre" for label windows, "window" for the regular grid). Adds, per row:

      rel.hp.yaw_dev      the student's mean yaw minus the median over the window's students
      rel.hp.yaw_absdev   its absolute value
      rel.hp.height_dev   the student's head height minus the window's median
      rel.hp.same_dir     share of the *other* students in the window whose yaw has the same sign
                          (NaN when no other student has a yaw)

    Students without a head-pose value get NaN (imputed downstream like any missing feature)."""
    out = rows.copy()
    if YAW not in out or HEIGHT not in out:
        for c in REL_COLUMNS:
            out[c] = np.nan
        return out
    g = out.groupby(by, sort=False)
    out["rel.hp.yaw_dev"] = out[YAW] - g[YAW].transform("median")
    out["rel.hp.yaw_absdev"] = out["rel.hp.yaw_dev"].abs()
    out["rel.hp.height_dev"] = out[HEIGHT] - g[HEIGHT].transform("median")
    sign = np.sign(out[YAW])
    pos = (sign > 0).astype(float).where(sign.notna())
    neg = (sign < 0).astype(float).where(sign.notna())
    n = sign.notna().astype(float)
    tot_n, tot_pos, tot_neg = (s.groupby(out[by]).transform("sum") for s in (n, pos.fillna(0), neg.fillna(0)))
    others = tot_n - n                                     # students with a yaw, excluding this one
    same = np.where(sign > 0, tot_pos - 1, np.where(sign < 0, tot_neg - 1, np.nan))
    out["rel.hp.same_dir"] = np.where((others > 0) & sign.notna(), same / others.where(others > 0), np.nan)
    return out


def model_columns(cfg: dict, available: list[str], vec_cols: list[str]) -> list[str]:
    """The model's input columns, in a fixed order: trk.*, then rel.* (engagement.class_relative), then
    vec.* (engagement.use_class_context), filtered by engagement.columns."""
    from ..core.config import use_class_context

    eng = cfg["engagement"]
    cols = sorted(c for c in available if c.startswith(TRACK_PREFIX))
    if eng.get("class_relative"):
        cols += [c for c in REL_COLUMNS if c in available]
    if use_class_context(cfg):
        cols += [c for c in vec_cols if c.startswith(VEC_PREFIX)]
    spec = eng.get("columns") or {}
    include, exclude = tuple(spec.get("include") or ()), tuple(spec.get("exclude") or ())
    if include:
        cols = [c for c in cols if c.startswith(include)]
    if exclude:
        cols = [c for c in cols if not c.startswith(exclude)]
    if not cols:
        raise ValueError(f"engagement.columns leaves no input column: {spec!r}")
    return cols
