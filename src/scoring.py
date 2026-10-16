"""Small wrapper so any Model-B family exposes the same `score` method (higher = more unusual)."""
from __future__ import annotations

import numpy as np


class AnomalyScorer:
    def __init__(self, model, kind: str, lo: float = 0.0, hi: float = 1.0):
        self.model, self.kind, self.lo, self.hi = model, kind, lo, hi

    def score(self, X) -> np.ndarray:
        if self.kind == "iforest":
            raw = -self.model.score_samples(X)
            return np.clip((raw - self.lo) / (self.hi - self.lo + 1e-12), 0.0, 1.0)
        return self.model.predict_proba(X)[:, 1]
