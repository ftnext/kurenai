from __future__ import annotations

from rouge_score.rouge_scorer import RougeScorer as OriginalRougeScorer
from rouge_score.scoring import BaseScorer


class RougeScorer(BaseScorer):
    def __init__(self, rouge_types: list[str]) -> None:
        self._scorer = OriginalRougeScorer(rouge_types)

    def score(self, target, prediction):
        raise NotImplementedError
