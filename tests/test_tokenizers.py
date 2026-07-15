from kurenai.tokenizers import AllCharacterSupportTokenizer


class TestAllCharacterSupportTokenizer:
    def test_tokenize_without_stemmer(self) -> None:
        sut = AllCharacterSupportTokenizer()

        actual = sut.tokenize("いぬ ねこ")

        assert actual == ["いぬ", "ねこ"]

    class TestWithStemmer:
        def test_tokens_longer_than_3_chars_are_stemmed(self) -> None:
            sut = AllCharacterSupportTokenizer(use_stemmer=True)

            # "dogs" (4 chars) and "running" (7 chars) are stemmed;
            # rouge-score only stems tokens of length > 3.
            actual = sut.tokenize("The dogs are running")

            assert actual == ["the", "dog", "are", "run"]

        def test_tokens_of_3_chars_or_fewer_are_not_stemmed(self) -> None:
            sut = AllCharacterSupportTokenizer(use_stemmer=True)

            # "run" is 3 chars, at or below rouge-score's "len > 3"
            # threshold for stemming, so it is left untouched.
            actual = sut.tokenize("cat run")

            assert actual == ["cat", "run"]

        def test_non_ascii_tokens_are_not_stemmed(self) -> None:
            sut = AllCharacterSupportTokenizer(use_stemmer=True)

            actual = sut.tokenize("いぬ が はしる")

            assert actual == ["いぬ", "が", "はしる"]

        def test_alnum_token_with_digits_can_be_stemmed(self) -> None:
            sut = AllCharacterSupportTokenizer(use_stemmer=True)

            # "2dogs" (5 chars) is made of lowercase letters and digits
            # only, so it matches rouge-score's [a-z0-9]+ token pattern
            # and gets stemmed like a pure-alphabetic token would.
            actual = sut.tokenize("2dogs")

            assert actual == ["2dog"]

        def test_token_with_non_alnum_char_is_not_stemmed(self) -> None:
            sut = AllCharacterSupportTokenizer(use_stemmer=True)

            # "run_ning" contains an underscore, so it does not match
            # rouge-score's [a-z0-9]+ token pattern and is left untouched,
            # even though it is longer than 3 characters.
            actual = sut.tokenize("run_ning")

            assert actual == ["run_ning"]

        def test_punctuated_token_is_not_stemmed(self) -> None:
            sut = AllCharacterSupportTokenizer(use_stemmer=True)

            # "dogs." keeps its trailing period because this tokenizer only
            # splits on whitespace and never deletes or rewrites
            # characters. That period makes the token fail rouge-score's
            # [a-z0-9]+ pattern, so it is left unstemmed -- unlike
            # rouge-score, which strips non-alphanumeric characters before
            # stemming. This is expected: kurenai's tokenizer is designed
            # for pre-tokenized, space-separated input.
            actual = sut.tokenize("The dogs. are running.")

            assert actual == ["the", "dogs.", "are", "running."]
