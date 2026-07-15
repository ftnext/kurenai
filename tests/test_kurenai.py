from rouge_score.rouge_scorer import RougeScorer as OriginalRougeScorer
from rouge_score.scoring import BaseScorer, Score
from rouge_score.tokenizers import Tokenizer

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

            def test_rougeLsum_single_line_matches_rougeL(self) -> None:
                # 改行を含まない1文だけの入力では、rougeLsumはrougeLと一致する
                scorer = RougeScorer(["rougeL", "rougeLsum"])
                actual = scorer.score("テスト いち に", "テスト いち")

                precision = 2 / 2
                recall = 2 / 3
                fscore = fscore_helper(precision, recall)
                expected = {
                    "rougeL": Score(precision, recall, fscore),
                    "rougeLsum": Score(precision, recall, fscore),
                }
                assert actual == expected

            def test_rougeLsum(self) -> None:
                # ref: https://github.com/google-research/google-research/blob/c34656f25265e717cc7f051a99185594892fd041/rouge/rouge_scorer_test.py#L271-L276  # NOQA: E501
                # 改行区切りで複数文を渡すと、文ごとのLCSの和集合でsummary-levelに
                # スコアリングされる
                scorer = RougeScorer(["rougeLsum"])
                actual = scorer.score(
                    "あ い う え お", "あ い か き く\nあ う く け お"
                )

                # target: あ い う え お (5トークン)
                # prediction 1文目: あ い か き く / 2文目: あ う く け お
                #   (計10トークン)
                # 1文目とのLCS: 「あ い」(2)
                # 2文目とのLCS: 「あ う ... お」の「あ」「う」「お」(3)
                # 和集合（重複除く）: 「あ」「い」「う」「お」の4トークンがhit
                precision = 4 / 10
                recall = 4 / 5
                fscore = fscore_helper(precision, recall)
                expected = {"rougeLsum": Score(precision, recall, fscore)}
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

    class TestConstructorCompat:
        def test_use_stemmer_matches_original_rouge_score(self) -> None:
            # "dogs"/"dog", "running"/"runs" are different tokens unless
            # stemmed; use_stemmer=True should make kurenai and rouge-score
            # agree on purely ASCII text.
            target = "The dogs are running in the park"
            prediction = "The dog runs in the park"

            sut = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
            original = OriginalRougeScorer(
                ["rouge1", "rougeL"], use_stemmer=True
            )

            assert sut.score(target, prediction) == original.score(
                target, prediction
            )

        def test_use_stemmer_true_changes_ascii_score(self) -> None:
            # Without stemming, "dogs" != "dog" and "running" != "runs", so
            # the score should differ from the stemmed version above.
            target = "The dogs are running in the park"
            prediction = "The dog runs in the park"

            without_stemmer = RougeScorer(["rouge1"]).score(target, prediction)
            with_stemmer = RougeScorer(["rouge1"], use_stemmer=True).score(
                target, prediction
            )

            assert without_stemmer != with_stemmer

        def test_use_stemmer_true_keeps_non_ascii_score_unchanged(
            self,
        ) -> None:
            target = "テスト いち に"
            prediction = "テスト に"

            without_stemmer = RougeScorer(["rouge1"]).score(target, prediction)
            with_stemmer = RougeScorer(["rouge1"], use_stemmer=True).score(
                target, prediction
            )

            assert without_stemmer == with_stemmer

        def test_custom_tokenizer_is_used(self) -> None:
            class UpperCaseSplitTokenizer(Tokenizer):
                def tokenize(self, text: str) -> list[str]:
                    return text.split()

            tokenizer = UpperCaseSplitTokenizer()
            sut = RougeScorer(["rouge1"], tokenizer=tokenizer)

            assert sut._scorer._tokenizer is tokenizer
            # "Testing" (custom tokenizer keeps the original case) only
            # matches itself, not the lowercased default tokenization.
            actual = sut.score("Testing one two", "testing")
            expected = {"rouge1": Score(0.0, 0.0, 0.0)}
            assert actual == expected

        def test_split_summaries_is_passed_to_original_scorer(self) -> None:
            sut = RougeScorer(["rougeLsum"], split_summaries=True)

            assert sut._scorer._split_summaries is True

        def test_split_summaries_defaults_to_false(self) -> None:
            sut = RougeScorer(["rougeLsum"])

            assert sut._scorer._split_summaries is False
