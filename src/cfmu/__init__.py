"""CFMU (design §10): part of FYP-1, not active in Deliverable III. The slot exists so turning it
on only widens Vec. `spread` holds the §10.4 maths; clustering (§10.3) is not built yet."""

from .extractor import CLUSTER_EXTRACTORS, ClusterFeatureExtractor
from .spread import spread_decomposition

__all__ = ["CLUSTER_EXTRACTORS", "ClusterFeatureExtractor", "spread_decomposition"]
