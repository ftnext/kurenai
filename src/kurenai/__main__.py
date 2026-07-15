"""Command line interface for kurenai.

Sample usage:

    python -m kurenai --rouge-types rouge1,rouge2,rougeL \\
        --target-file targets.txt --prediction-file predictions.txt \\
        --use-aggregator --output scores.csv

Both ``--target-file`` and ``--prediction-file`` are expected to be
newline-delimited text files where one line corresponds to one record.

Note:
    Tokenizer selection (e.g. a Japanese morphological tokenizer) is not
    yet configurable from this CLI. ``kurenai.rouge_scorer.RougeScorer``
    always uses ``AllCharacterSupportTokenizer``. Support for pluggable
    tokenizers is planned for a later phase.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import IO, cast, get_args

from kurenai.rouge_scorer import RougeScoreDict, RougeScorer, RougeType
from kurenai.scoring import AggregateScore, BootstrapAggregator, Score

_VALID_ROUGE_TYPES: tuple[str, ...] = get_args(RougeType)
_DEFAULT_ROUGE_TYPES = "rouge1,rouge2,rougeL"


def _parse_rouge_types(value: str) -> list[str]:
    rouge_types = [item.strip() for item in value.split(",") if item.strip()]
    invalid_types = [t for t in rouge_types if t not in _VALID_ROUGE_TYPES]
    if invalid_types:
        raise argparse.ArgumentTypeError(
            "invalid rouge type(s): {}. Valid types are: {}".format(
                ", ".join(invalid_types), ", ".join(_VALID_ROUGE_TYPES)
            )
        )
    return rouge_types


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kurenai",
        description=(
            "Calculate ROUGE scores between a target file and a "
            "prediction file."
        ),
    )
    parser.add_argument(
        "--rouge-types",
        type=_parse_rouge_types,
        default=_parse_rouge_types(_DEFAULT_ROUGE_TYPES),
        help=(
            "Comma-separated list of ROUGE types to calculate. Valid "
            "values: {}. (default: {})".format(
                ", ".join(_VALID_ROUGE_TYPES), _DEFAULT_ROUGE_TYPES
            )
        ),
    )
    parser.add_argument(
        "--target-file",
        required=True,
        type=Path,
        help="Newline-delimited file containing target text.",
    )
    parser.add_argument(
        "--prediction-file",
        required=True,
        type=Path,
        help="Newline-delimited file containing prediction text.",
    )
    parser.add_argument(
        "--use-aggregator",
        action="store_true",
        help="Aggregate scores with BootstrapAggregator.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="File to write the CSV output to. Defaults to stdout.",
    )
    return parser


def _read_records(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _compute_scores(
    rouge_types: list[RougeType],
    targets: list[str],
    predictions: list[str],
) -> list[RougeScoreDict]:
    scorer = RougeScorer(rouge_types)
    return [
        scorer.score(target, prediction)
        for target, prediction in zip(targets, predictions)
    ]


def _write_scores_csv(scores: list[RougeScoreDict], output: IO[str]) -> None:
    """Writes per-example scores, following rouge_score's io.py format."""
    if not scores:
        return
    rouge_types = sorted(scores[0].keys())

    output.write("id")
    for rouge_type in rouge_types:
        output.write(",{t}-P,{t}-R,{t}-F".format(t=rouge_type))
    output.write("\n")
    for i, result in enumerate(scores):
        output.write(str(i))
        result_by_type = cast("dict[str, Score]", result)
        for rouge_type in rouge_types:
            score = result_by_type[rouge_type]
            output.write(
                ",{:.6f},{:.6f},{:.6f}".format(
                    score.precision, score.recall, score.fmeasure
                )
            )
        output.write("\n")


def _write_aggregates_csv(
    aggregates: dict[str, AggregateScore], output: IO[str]
) -> None:
    """Writes aggregate scores, following rouge_score's io.py format."""
    output.write("score_type,low,mid,high\n")
    for score_type, aggregate in sorted(aggregates.items()):
        output.write(
            "{t}-R,{low:.6f},{mid:.6f},{high:.6f}\n".format(
                t=score_type,
                low=aggregate.low.recall,
                mid=aggregate.mid.recall,
                high=aggregate.high.recall,
            )
        )
        output.write(
            "{t}-P,{low:.6f},{mid:.6f},{high:.6f}\n".format(
                t=score_type,
                low=aggregate.low.precision,
                mid=aggregate.mid.precision,
                high=aggregate.high.precision,
            )
        )
        output.write(
            "{t}-F,{low:.6f},{mid:.6f},{high:.6f}\n".format(
                t=score_type,
                low=aggregate.low.fmeasure,
                mid=aggregate.mid.fmeasure,
                high=aggregate.high.fmeasure,
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    targets = _read_records(args.target_file)
    predictions = _read_records(args.prediction_file)

    if len(targets) != len(predictions):
        print(
            "Error: --target-file and --prediction-file must have the "
            "same number of lines (target: {}, prediction: {}).".format(
                len(targets), len(predictions)
            ),
            file=sys.stderr,
        )
        return 1

    scores = _compute_scores(args.rouge_types, targets, predictions)

    if args.use_aggregator:
        aggregator = BootstrapAggregator()
        for score in scores:
            aggregator.add_scores(score)
        aggregates = aggregator.aggregate()
        if args.output is not None:
            with args.output.open("w", encoding="utf-8") as f:
                _write_aggregates_csv(aggregates, f)
        else:
            _write_aggregates_csv(aggregates, sys.stdout)
    else:
        if args.output is not None:
            with args.output.open("w", encoding="utf-8") as f:
                _write_scores_csv(scores, f)
        else:
            _write_scores_csv(scores, sys.stdout)

    return 0


if __name__ == "__main__":
    sys.exit(main())
