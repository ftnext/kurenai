from __future__ import annotations

from rouge_score.rouge_scorer import RougeScorer as OriginalRougeScorer
from rouge_score.scoring import BaseScorer

from kurenai.tokenizers import AllCharacterSupportTokenizer


class RougeScorer(BaseScorer):
    def __init__(self, rouge_types: list[str]) -> None:
        self._scorer = OriginalRougeScorer(
            rouge_types, tokenizer=AllCharacterSupportTokenizer()
        )

    def score(self, target, prediction):
        return self._scorer.score(target, prediction)
