from rouge_score.rouge_scorer import RougeScorer as OriginalRougeScorer
from rouge_score.scoring import BaseScorer, Score

from kurenai.rouge_scorer import RougeScorer


class TestRougeScorer:
    def test_can_create(self) -> None:
        sut = RougeScorer(["rouge1", "rougeL"])

        assert isinstance(sut, BaseScorer)
        assert isinstance(sut._scorer, OriginalRougeScorer)
        assert sut._scorer.rouge_types == ["rouge1", "rougeL"]

    def test_rouge1_ascii(self) -> None:
        # ref: https://github.com/google-research/google-research/blob/c34656f25265e717cc7f051a99185594892fd041/rouge/rouge_scorer_test.py#L58-L63  # NOQA* E501
        scorer = RougeScorer(["rouge1"])
        actual = scorer.score("testing one two", "testing")

        precision = 1 / 1
        recall = 1 / 3
        fscore = 2 * precision * recall / (precision + recall)
        expected = {"rouge1": Score(precision, recall, fscore)}
        assert actual == expected

    def test_rouge1_non_ascii(self) -> None:
        scorer = RougeScorer(["rouge1"])
        actual = scorer.score("テスト いち に", "テスト に")

        precision = 1 / 1
        recall = 2 / 3
        fscore = 2 * precision * recall / (precision + recall)
        expected = {"rouge1": Score(precision, recall, fscore)}
        assert actual == expected
