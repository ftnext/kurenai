from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypedDict

from rouge_score.rouge_scorer import RougeScorer as OriginalRougeScorer
from rouge_score.scoring import BaseScorer, Score

from kurenai.tokenizers import AllCharacterSupportTokenizer

RougeType = Literal[
    "rouge1",
    "rouge2",
    "rouge3",
    "rouge4",
    "rouge5",
    "rouge6",
    "rouge7",
    "rouge8",
    "rouge9",
    "rougeL",
    # "rougeLsum",  # TODO
]


class RougeScoreDict(TypedDict, total=False):
    rouge1: Score
    rouge2: Score
    rouge3: Score
    rouge4: Score
    rouge5: Score
    rouge6: Score
    rouge7: Score
    rouge8: Score
    rouge9: Score
    rougeL: Score


class RougeScorer(BaseScorer):
    def __init__(self, rouge_types: Iterable[RougeType]) -> None:
        self._scorer = OriginalRougeScorer(
            list(rouge_types), tokenizer=AllCharacterSupportTokenizer()
        )

    def score(self, target: str, prediction: str) -> RougeScoreDict:
        return self._scorer.score(target, prediction)

    def score_multi(
        self, targets: Iterable[str], prediction: str
    ) -> RougeScoreDict:
        return self._scorer.score_multi(targets, prediction)
