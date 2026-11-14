"""Seeded random search over config-defined spaces.

Space format (JSON):  {"param": [choice, ...]}                         categorical
                      {"param": {"low": a, "high": b, "log": false, "int": false}}   range
Trial 0 always evaluates `baseline` (the current config), so every search reports its gain over it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..spu.splits import logo_splits


def sample_params(space: dict, rng: np.random.Generator) -> dict:
    out = {}
    for name, spec in sorted(space.items()):
        if isinstance(spec, list):
            out[name] = spec[int(rng.integers(len(spec)))]
            continue
        lo, hi = float(spec["low"]), float(spec["high"])
        v = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if spec.get("log") else float(rng.uniform(lo, hi))
        out[name] = int(round(v)) if spec.get("int") else round(v, 4)
    return out


def random_search(space: dict, objective: Callable[[dict], dict], n_trials: int, seed: int = 0,
                  baseline: dict | None = None, on_trial: Callable[[dict], None] | None = None) -> list[dict]:
    """objective(params) -> {"score": float, ...}. Returns trials sorted by score, best first."""
    rng = np.random.default_rng(seed)
    trials = []
    for i in range(n_trials):
        params = dict(baseline or {}) if i == 0 else sample_params(space, rng)
        result = {"trial": i, "params": params, **objective(params)}
        trials.append(result)
        if on_trial:
            on_trial(result)
    return sorted(trials, key=lambda r: -np.nan_to_num(r["score"], nan=-np.inf))


def nested_logo_search(groups: np.ndarray, space: dict, n_trials: int, seed: int,
                       score: Callable[[dict, np.ndarray, np.ndarray], float],
                       baseline: dict | None = None) -> dict:
    """Hyperparameter search that does not select on its own test fold.

    score(params, train_idx, test_idx) -> float (higher is better). For each outer LOGO fold, the
    search runs inner LOGO on the outer-train groups only; the winner is refit and scored once on
    the outer test group. The `selected` params come from a final search over all groups and are
    the ones to put in the config; `outer` scores are the unbiased estimate of their quality.
    """
    def search(idx: np.ndarray) -> dict:
        def objective(params):
            s = [score(params, idx[tr], idx[te]) for _, tr, te in logo_splits(groups[idx])]
            return {"score": float(np.nanmean(s)), "fold_scores": s}
        return random_search(space, objective, n_trials, seed, baseline)[0]

    outer = []
    for g, tr, te in logo_splits(groups):
        best = search(tr)
        outer.append({"held_out": g, "params": best["params"], "inner_score": best["score"],
                      "outer_score": score(best["params"], tr, te)})
    final = search(np.arange(len(groups)))
    s = [o["outer_score"] for o in outer]
    return {"selected": final["params"], "selected_cv_score": final["score"],
            "nested_mean": float(np.nanmean(s)), "nested_std": float(np.nanstd(s)), "outer": outer}
