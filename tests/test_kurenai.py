from rouge_score.rouge_scorer import RougeScorer as OriginalRougeScorer
from rouge_score.scoring import BaseScorer

from kurenai.rouge_scorer import RougeScorer


class TestRougeScorer:
    def test_can_create(self) -> None:
        sut = RougeScorer(["rouge1", "rougeL"])

        assert isinstance(sut, BaseScorer)
        assert isinstance(sut._scorer, OriginalRougeScorer)
        assert sut._scorer.rouge_types == ["rouge1", "rougeL"]
