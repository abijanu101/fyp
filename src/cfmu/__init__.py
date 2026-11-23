"""CFMU (design §10): part of FYP-1, not active in Deliverable III. The slot exists so turning it
on only widens Vec. `spread` holds the §10.4 maths; `classrel` the class-relative features and
clusterings of guide 15 (used offline by scripts/clustering/; no extractor registered yet)."""

from .extractor import CLUSTER_EXTRACTORS, ClusterFeatureExtractor
from .spread import spread_decomposition

__all__ = ["CLUSTER_EXTRACTORS", "ClusterFeatureExtractor", "spread_decomposition"]
