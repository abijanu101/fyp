# The FYP-1 Codebase: A Contributor's Guide

This guide explains how the retention-graph pipeline is built, from first principles up to the
details of every stage. It is written to be read in order: each chapter assumes the previous
ones. If you only want to change one thing, Chapter 9 has step-by-step recipes, but they will
make much more sense after Chapters 3 and 4.

The design document (*FYP-1 Baseline: Technical Design v3*) says **what** the system should do and
**why**. This guide says **how the code does it**. Section references like "§5c" point at the
design document.

## Reading order

| # | Chapter | You will learn |
|---|---|---|
| 1 | [The big picture](01_big_picture.md) | What goes in, what comes out, and the journey in between |
| 2 | [The Python toolkit](02_python_toolkit.md) | The handful of language features the codebase leans on |
| 3 | [The registry pattern, from scratch](03_registry.md) | How a name in a JSON file becomes a running object |
| 4 | [Contracts and configuration](04_contracts_and_config.md) | The dataclasses stages exchange, the config file, hashing |
| 5 | [The frame pass](05_frame_pass.md) | Ingestion, detection, tracking, pixel-level features |
| 6 | [The feature pass](06_feature_pass.md) | Tracklet filtering, window features, the vector Vec |
| 7 | [Models: gate, experts, labels](07_models.md) | Context prediction, engagement experts, the label pipeline |
| 8 | [Scripts, caches and tuning](08_scripts_caching_tuning.md) | The entry points, what is cached where, how tuning works |
| 9 | [Tests and recipes](09_testing_and_recipes.md) | How to test, and how to make the common changes |
| 10 | [Pitfalls and status](10_pitfalls_and_status.md) | Environment quirks, data quirks, what is open |

## Conventions used in this guide

- File references are relative to the repository root, e.g. `src/core/registry.py`.
- Code excerpts are copied from the codebase, sometimes trimmed; trimmed lines are marked `...`.
- "Session" always means one *session-camera*: one group, one experiment, one camera, e.g.
  `group_01/experiment_03/context_cam_02`.
- All commands run from the repository root inside the `fyp-local` conda environment.

## Keeping this guide true

The guide describes code that changes. When you change a stage's contract (its dataclass, its
config keys, or what it caches), update the matching chapter in the same commit.
