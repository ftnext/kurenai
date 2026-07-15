from __future__ import annotations

import re

from nltk.stem import porter
from rouge_score.tokenize import SPACES_RE
from rouge_score.tokenizers import Tokenizer

# ref: https://github.com/google-research/google-research/blob/master/rouge/tokenize.py  # NOQA: E501
# rouge-score only stems tokens that consist solely of ASCII
# alphanumeric characters and are longer than 3 characters; this mirrors
# that rule so non-ASCII (e.g. Japanese) tokens are never touched.
STEMMABLE_TOKEN_RE = re.compile(r"^[a-z0-9]+$")


class AllCharacterSupportTokenizer(Tokenizer):
    """
    >>> AllCharacterSupportTokenizer().tokenize("いぬ ねこ")
    ['いぬ', 'ねこ']
    >>> AllCharacterSupportTokenizer(use_stemmer=True).tokenize(
    ...     "The dogs are running"
    ... )
    ['the', 'dog', 'are', 'run']
    >>> AllCharacterSupportTokenizer(use_stemmer=True).tokenize(
    ...     "いぬ が はしる"
    ... )
    ['いぬ', 'が', 'はしる']
    """

    def __init__(self, use_stemmer: bool = False) -> None:
        self._use_stemmer = use_stemmer
        self._stemmer = porter.PorterStemmer() if use_stemmer else None

    def tokenize(self, text: str) -> list[str]:
        tokens = SPACES_RE.split(text.lower())
        if self._stemmer is None:
            return tokens
        return [self._maybe_stem(token) for token in tokens]

    def _maybe_stem(self, token: str) -> str:
        assert self._stemmer is not None
        if len(token) > 3 and STEMMABLE_TOKEN_RE.match(token):
            return str(self._stemmer.stem(token))
        return token
