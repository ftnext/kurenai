"""Re-export of ``rouge_score.scoring``.

This module lets users write ``from kurenai.scoring import Score``
instead of mixing ``rouge_score`` and ``kurenai`` imports.
"""

from __future__ import annotations

from rouge_score.scoring import AggregateScore as AggregateScore
from rouge_score.scoring import BaseScorer as BaseScorer
from rouge_score.scoring import BootstrapAggregator as BootstrapAggregator
from rouge_score.scoring import Score as Score
from rouge_score.scoring import fmeasure as fmeasure

__all__ = [
    "Score",
    "AggregateScore",
    "BootstrapAggregator",
    "BaseScorer",
    "fmeasure",
]
