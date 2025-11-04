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
    """Calculate rouges scores between two blobs of text.

    Sample usage:
        scorer = RougeScorer(["rouge1", "rougeL"])
        scores = scorer.score("The quick brown fox jumps over the lazy dog",
                              "The quick brown dog jumps on the log.")
    """

    def __init__(self, rouge_types: Iterable[RougeType]) -> None:
        """Initializes a new RougeScorer.

        Valid rouge types that can be computed are:
            rougeN (e.g. rouge1, rouge2, ..., rouge9): n-gram based scoring.
            rougeL: Longest common subsequence based scoring.

        Args:
            rouge_types: An iterable of rouge types to calculate.
        """
        self._scorer = OriginalRougeScorer(
            list(rouge_types), tokenizer=AllCharacterSupportTokenizer()
        )

    def score(self, target: str, prediction: str) -> RougeScoreDict:
        """Calculates rouge scores between the target and prediction.

        Args:
            target: Ground truth text.
            prediction: Predicted text.

        Returns:
            A dict mapping each rouge type to a Score object.
        """
        return self._scorer.score(target, prediction)

    def score_multi(
        self, targets: Iterable[str], prediction: str
    ) -> RougeScoreDict:
        """Calculates rouge scores between targets and prediction.

        The target with the maximum f-measure is used for the final score for
        each score type.

        Args:
            targets: An iterable of ground truth texts.
            prediction: Predicted text.

        Returns:
            A dict mapping each rouge type to a Score object.
        """
        return self._scorer.score_multi(targets, prediction)
