from rouge_score import scoring as original_scoring

from kurenai import scoring
from kurenai.rouge_scorer import RougeScorer


class TestReExport:
    def test_score(self) -> None:
        assert scoring.Score is original_scoring.Score

    def test_aggregate_score(self) -> None:
        assert scoring.AggregateScore is original_scoring.AggregateScore

    def test_bootstrap_aggregator(self) -> None:
        assert (
            scoring.BootstrapAggregator is original_scoring.BootstrapAggregator
        )

    def test_base_scorer(self) -> None:
        assert scoring.BaseScorer is original_scoring.BaseScorer

    def test_fmeasure(self) -> None:
        assert scoring.fmeasure is original_scoring.fmeasure


class TestBootstrapAggregatorIntegration:
    def test_aggregate_with_kurenai_rouge_scorer(self) -> None:
        scorer = RougeScorer(["rouge1"])
        aggregator = scoring.BootstrapAggregator()

        aggregator.add_scores(scorer.score("テスト いち に", "テスト に"))
        aggregator.add_scores(scorer.score("テスト いち", "テスト いち に"))

        result = aggregator.aggregate()

        assert "rouge1" in result
        for aggregate_score in result.values():
            assert isinstance(aggregate_score, scoring.AggregateScore)
            assert isinstance(aggregate_score.low, scoring.Score)
            assert isinstance(aggregate_score.mid, scoring.Score)
            assert isinstance(aggregate_score.high, scoring.Score)
