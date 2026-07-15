from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypedDict

from rouge_score.rouge_scorer import RougeScorer as OriginalRougeScorer
from rouge_score.scoring import BaseScorer, Score
from rouge_score.tokenizers import Tokenizer

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
    "rougeLsum",
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
    rougeLsum: Score


class RougeScorer(BaseScorer):
    """Calculate rouges scores between two blobs of text.

    Sample usage:
        scorer = RougeScorer(["rouge1", "rougeL"])
        scores = scorer.score("The quick brown fox jumps over the lazy dog",
                              "The quick brown dog jumps on the log.")
    """

    def __init__(
        self,
        rouge_types: Iterable[RougeType],
        use_stemmer: bool = False,
        split_summaries: bool = False,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        """Initializes a new RougeScorer.

        Valid rouge types that can be computed are:
            rougeN (e.g. rouge1, rouge2, ..., rouge9): n-gram based scoring.
            rougeL: Longest common subsequence based scoring.
            rougeLsum: Summary-level longest common subsequence based
                scoring. Sentences are assumed to be separated by newlines;
                each sentence is tokenized independently before scoring.

        Args:
            rouge_types: An iterable of rouge types to calculate.
            use_stemmer: Bool indicating whether the Porter stemmer should
                be used to strip word suffixes to improve matching. This is
                honored by kurenai's own default tokenizer
                (AllCharacterSupportTokenizer), which only stems ASCII
                alphanumeric tokens longer than 3 characters and leaves
                non-ASCII tokens (e.g. Japanese) untouched. Because kurenai
                never deletes or rewrites characters, tokens that still
                carry punctuation (e.g. "dogs.") are not stemmed; unlike
                rouge-score, no non-alphanumeric characters are stripped
                before stemming, so pass in pre-tokenized, space-separated
                text for the best results. Unlike rouge-score's original
                RougeScorer, this value is *not* forwarded to rouge-score's
                DefaultTokenizer, because that tokenizer drops non-ASCII
                characters. If a custom ``tokenizer`` is given, this arg
                has no effect on it -- the same as how rouge-score's
                use_stemmer only affects its DefaultTokenizer.
            split_summaries: Whether to add newlines between sentences for
                rougeLsum. This is passed through to rouge-score as-is.
                rouge-score splits sentences with nltk.sent_tokenize, which
                assumes English text, so this option is only reliable for
                English summaries.
            tokenizer: Tokenizer object which has a tokenize() method. When
                omitted, kurenai's AllCharacterSupportTokenizer(use_stemmer=
                use_stemmer) is used so that non-ASCII text keeps working by
                default.
        """
        if tokenizer is None:
            tokenizer = AllCharacterSupportTokenizer(use_stemmer=use_stemmer)
        self._scorer = OriginalRougeScorer(
            list(rouge_types),
            split_summaries=split_summaries,
            tokenizer=tokenizer,
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
