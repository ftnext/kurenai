from rouge_score.rouge_scorer import RougeScorer as OriginalRougeScorer
from rouge_score.scoring import BaseScorer, Score

from kurenai.rouge_scorer import RougeScorer


def fscore_helper(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class TestRougeScorer:
    def test_can_create(self) -> None:
        sut = RougeScorer(["rouge1", "rougeL"])

        assert isinstance(sut, BaseScorer)
        assert isinstance(sut._scorer, OriginalRougeScorer)
        assert sut._scorer.rouge_types == ["rouge1", "rougeL"]

    def test_rouge1_ascii(self) -> None:
        # ref: https://github.com/google-research/google-research/blob/c34656f25265e717cc7f051a99185594892fd041/rouge/rouge_scorer_test.py#L58-L63  # NOQA: E501
        scorer = RougeScorer(["rouge1"])
        actual = scorer.score("testing one two", "testing")

        precision = 1 / 1
        recall = 1 / 3
        fscore = fscore_helper(precision, recall)
        expected = {"rouge1": Score(precision, recall, fscore)}
        assert actual == expected

    class TestNonAscii:
        class TestScore:
            def test_rouge1(self) -> None:
                scorer = RougeScorer(["rouge1"])
                actual = scorer.score("テスト いち に", "テスト に")

                precision = 2 / 2
                recall = 2 / 3
                fscore = fscore_helper(precision, recall)
                expected = {"rouge1": Score(precision, recall, fscore)}
                assert actual == expected

            def test_rouge2(self) -> None:
                # ref: https://github.com/google-research/google-research/blob/c34656f25265e717cc7f051a99185594892fd041/rouge/rouge_scorer_test.py#L87-L92  # NOQA: E501
                scorer = RougeScorer(["rouge2"])
                actual = scorer.score("テスト いち に", "テスト いち")

                precision = 1 / 1
                recall = 1 / 2  # 「テスト いち」「いち に」
                fscore = fscore_helper(precision, recall)
                expected = {"rouge2": Score(precision, recall, fscore)}
                assert actual == expected

            def test_rougeL(self) -> None:
                # ref: https://github.com/google-research/google-research/blob/4e9dcd23ab81f6bf3d0f09ba5557e991cd56658d/rouge/rouge_scorer_test.py#L94-L99  # NOQA: E501
                scorer = RougeScorer(["rougeL"])
                actual = scorer.score("テスト いち に", "テスト いち")

                precision = 2 / 2
                recall = 2 / 3
                fscore = fscore_helper(precision, recall)
                expected = {"rougeL": Score(precision, recall, fscore)}
                assert actual == expected

            def test_multiple_rouge_types(self) -> None:
                scorer = RougeScorer(["rouge1", "rougeL"])
                actual = scorer.score("テスト いち に", "テスト いち")

                precision_1 = 2 / 2
                recall_1 = 2 / 3
                fscore_1 = fscore_helper(precision_1, recall_1)

                precision_l = 2 / 2
                recall_l = 2 / 3
                fscore_l = fscore_helper(precision_l, recall_l)

                expected = {
                    "rouge1": Score(precision_1, recall_1, fscore_1),
                    "rougeL": Score(precision_l, recall_l, fscore_l),
                }
                assert actual == expected

        class TestScoreMulti:
            def test_rouge1(self) -> None:
                scorer = RougeScorer(["rouge1"])
                actual = scorer.score_multi(["テスト いち に"], "テスト に")

                precision = 2 / 2
                recall = 2 / 3
                fscore = fscore_helper(precision, recall)
                expected = {"rouge1": Score(precision, recall, fscore)}
                assert actual == expected

            def test_multiple_rouge_types(self) -> None:
                scorer = RougeScorer(["rouge1", "rouge2", "rougeL"])
                actual = scorer.score_multi(
                    ["最初 テキスト", "最初 何か"], "テキスト 最初"
                )

                precision_1 = 2 / 2
                recall_1 = 2 / 2  # 0番目のテキストによるrecallが最大なので使用
                fscore_1 = fscore_helper(precision_1, recall_1)

                precision_2 = 0 / 1
                recall_2 = 0 / 2
                fscore_2 = fscore_helper(precision_2, recall_2)

                # LCS（最長共通部分列）は「テキスト」または「最初」の1語
                precision_L = 1 / 2
                recall_L = 1 / 2
                fscore_L = fscore_helper(precision_L, recall_L)

                expected = {
                    "rouge1": Score(precision_1, recall_1, fscore_1),
                    "rouge2": Score(precision_2, recall_2, fscore_2),
                    "rougeL": Score(precision_L, recall_L, fscore_L),
                }
                assert actual == expected
