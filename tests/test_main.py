from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from kurenai.__main__ import main


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestMainScoresCsv:
    """--use-aggregator を指定しない場合はレコードごとのスコアを出力する。"""

    def test_writes_per_record_scores_to_output_file(
        self, tmp_path: Path
    ) -> None:
        target_file = tmp_path / "targets.txt"
        prediction_file = tmp_path / "predictions.txt"
        output_file = tmp_path / "scores.csv"

        _write_lines(target_file, ["one two three four", "testing one two"])
        _write_lines(prediction_file, ["four three two one", "testing"])

        exit_code = main(
            [
                "--rouge-types",
                "rouge1,rougeL",
                "--target-file",
                str(target_file),
                "--prediction-file",
                str(prediction_file),
                "--output",
                str(output_file),
            ]
        )

        assert exit_code == 0
        # rouge1 (record0): 全単語一致だが順序無視なので precision=recall=1.0
        # rougeL (record0): 完全に逆順のため LCS は1語のみ -> 0.25
        # rouge1/rougeL (record1): "testing" のみ一致 -> precision=1.0,
        # recall=1/3, fscore=0.5 (LCSも1語のみで同値)
        expected = (
            "id,rouge1-P,rouge1-R,rouge1-F,rougeL-P,rougeL-R,rougeL-F\n"
            "0,1.000000,1.000000,1.000000,0.250000,0.250000,0.250000\n"
            "1,1.000000,0.333333,0.500000,1.000000,0.333333,0.500000\n"
        )
        assert output_file.read_text(encoding="utf-8") == expected

    def test_writes_per_record_scores_to_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target_file = tmp_path / "targets.txt"
        prediction_file = tmp_path / "predictions.txt"

        _write_lines(target_file, ["testing one two"])
        _write_lines(prediction_file, ["testing"])

        exit_code = main(
            [
                "--rouge-types",
                "rouge1",
                "--target-file",
                str(target_file),
                "--prediction-file",
                str(prediction_file),
            ]
        )

        assert exit_code == 0
        expected = (
            "id,rouge1-P,rouge1-R,rouge1-F\n0,1.000000,0.333333,0.500000\n"
        )
        assert capsys.readouterr().out == expected


class TestMainAggregatedScoresCsv:
    """--use-aggregator を指定した場合は AggregateScore を出力する。"""

    def test_writes_aggregate_scores_to_output_file(
        self, tmp_path: Path
    ) -> None:
        target_file = tmp_path / "targets.txt"
        prediction_file = tmp_path / "predictions.txt"
        output_file = tmp_path / "scores.csv"

        # 全レコードが同一の入力なので、ブートストラップ再抽出しても
        # low/mid/high は常に同じ値になり、決定的に検証できる。
        # rouge1: target="テスト いち に", prediction="テスト に"
        # -> precision=2/2=1.0, recall=2/3, fscore=0.8
        _write_lines(target_file, ["テスト いち に"] * 3)
        _write_lines(prediction_file, ["テスト に"] * 3)

        exit_code = main(
            [
                "--rouge-types",
                "rouge1",
                "--target-file",
                str(target_file),
                "--prediction-file",
                str(prediction_file),
                "--use-aggregator",
                "--output",
                str(output_file),
            ]
        )

        assert exit_code == 0
        expected = (
            "score_type,low,mid,high\n"
            "rouge1-R,0.666667,0.666667,0.666667\n"
            "rouge1-P,1.000000,1.000000,1.000000\n"
            "rouge1-F,0.800000,0.800000,0.800000\n"
        )
        assert output_file.read_text(encoding="utf-8") == expected


class TestMainErrors:
    def test_mismatched_line_count_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target_file = tmp_path / "targets.txt"
        prediction_file = tmp_path / "predictions.txt"

        _write_lines(target_file, ["one", "two"])
        _write_lines(prediction_file, ["one", "two", "three"])

        exit_code = main(
            [
                "--target-file",
                str(target_file),
                "--prediction-file",
                str(prediction_file),
            ]
        )

        assert exit_code != 0
        assert "same number of lines" in capsys.readouterr().err

    def test_invalid_rouge_type_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target_file = tmp_path / "targets.txt"
        prediction_file = tmp_path / "predictions.txt"

        _write_lines(target_file, ["one"])
        _write_lines(prediction_file, ["one"])

        exit_code = main(
            [
                "--rouge-types",
                "not-a-real-rouge-type",
                "--target-file",
                str(target_file),
                "--prediction-file",
                str(prediction_file),
            ]
        )

        assert exit_code != 0
        assert "invalid rouge type" in capsys.readouterr().err


class TestMainSubprocess:
    """python -m kurenai として起動できることを確認する。"""

    def test_can_be_invoked_as_a_module(self, tmp_path: Path) -> None:
        target_file = tmp_path / "targets.txt"
        prediction_file = tmp_path / "predictions.txt"

        _write_lines(target_file, ["testing one two"])
        _write_lines(prediction_file, ["testing"])

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kurenai",
                "--rouge-types",
                "rouge1",
                "--target-file",
                str(target_file),
                "--prediction-file",
                str(prediction_file),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        expected = (
            "id,rouge1-P,rouge1-R,rouge1-F\n0,1.000000,0.333333,0.500000\n"
        )
        assert result.stdout == expected
