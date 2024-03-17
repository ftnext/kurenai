from __future__ import annotations

from rouge_score.tokenize import SPACES_RE
from rouge_score.tokenizers import Tokenizer


class AllCharacterSupportTokenizer(Tokenizer):
    """
    >>> AllCharacterSupportTokenizer().tokenize("いぬ ねこ")
    ['いぬ', 'ねこ']
    """

    def tokenize(self, text: str) -> list[str]:
        return SPACES_RE.split(text.lower())
