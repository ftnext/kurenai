# 紅 (kurenai)

紅 (kurenai) is a thin wrapper of [rouge-score](https://pypi.org/project/rouge-score/).  
rouge-score remove non-ascii characters by default, so ROUGE of Japanese text becomes 0.

```python
>>> from rouge_score.rouge_scorer import RougeScorer
>>> scorer = RougeScorer(["rouge1"])
>>> scorer.score('いぬ ねこ', 'いぬ ねこ')
{'rouge1': Score(precision=0.0, recall=0.0, fmeasure=0.0)}
```

紅 (kurenai) resolves this, it **supports** ascii and **non-ascii**

Currently, It is at a developing status:

* Supports ROUGE-N (1, 2, ..., 9), ROUGE-L and ROUGE-Lsum
* Supports both `RougeScorer.score()` and `RougeScorer.score_multi()`
* Supports `RougeScorer`'s constructor arguments (`use_stemmer`,
  `split_summaries`, `tokenizer`), same as rouge-score's original
  `RougeScorer`
* Re-exports rouge-score's scoring utilities (`Score`, `AggregateScore`,
  `BootstrapAggregator`, `BaseScorer`, `fmeasure`) as `kurenai.scoring`
* Provides a `python -m kurenai` CLI that does not require `absl`
* TODO: Tokenizing raw Japanese text (e.g. a `CharacterTokenizer` or a
  morphological analyzer-based tokenizer) and Japanese-aware
  `split_summaries` are planned for a later phase. For now, text passed to
  `RougeScorer` is expected to already be space-separated tokens (as in
  the examples below).

## Usage

紅 (kurenai) has the same interface as [rouge-score](https://pypi.org/project/rouge-score/).

### Basic scoring

```python
>>> from kurenai.rouge_scorer import RougeScorer
>>> scorer = RougeScorer(["rouge1"])
>>> scorer.score('いぬ ねこ', 'いぬ ねこ')
{'rouge1': Score(precision=1.0, recall=1.0, fmeasure=1.0)}
>>> scorer.score('The quick brown fox jumps over the lazy dog', 'The quick brown dog jumps on the log.')
{'rouge1': Score(precision=0.75, recall=0.6666666666666666, fmeasure=0.7058823529411765)}
```

### ROUGE-Lsum

For `rougeLsum`, sentences are assumed to be separated by newlines; each
line is tokenized and scored as its own sentence, and the per-sentence LCS
matches are unioned at the summary level.

```python
>>> from kurenai.rouge_scorer import RougeScorer
>>> scorer = RougeScorer(["rougeLsum"])
>>> target = "今日 は 晴れ\n明日 は 雨"
>>> prediction = "今日 は 曇り\n明日 は 雨"
>>> scorer.score(target, prediction)
{'rougeLsum': Score(precision=0.8333333333333334, recall=0.8333333333333334, fmeasure=0.8333333333333334)}
```

### `use_stemmer`

`use_stemmer=True` enables Porter-stemmer-based matching, same as
rouge-score. Unlike rouge-score's original tokenizer (which drops
non-ascii characters), kurenai's default tokenizer
(`AllCharacterSupportTokenizer`) only stems ASCII alphanumeric tokens of
4+ characters, so Japanese text is left untouched even when
`use_stemmer=True`. Because kurenai never deletes or rewrites
characters, tokens that still carry punctuation (e.g. `"dogs."`) are
also left untouched and are not stemmed; split punctuation off into its
own token (as in the space-separated examples above) if you want it
stemmed.

```python
>>> from kurenai.rouge_scorer import RougeScorer
>>> scorer = RougeScorer(["rouge1"], use_stemmer=True)
>>> scorer.score("The dogs are running", "The dog runs")
{'rouge1': Score(precision=1.0, recall=0.75, fmeasure=0.8571428571428571)}
```

### Custom `tokenizer`

Like rouge-score, a custom tokenizer (implementing
`rouge_score.tokenizers.Tokenizer`) can be passed to replace kurenai's
default `AllCharacterSupportTokenizer`, e.g. to plug in a Japanese
morphological tokenizer:

```python
>>> from kurenai.rouge_scorer import RougeScorer
>>> from rouge_score.tokenizers import Tokenizer
>>> class WhitespaceKeepCaseTokenizer(Tokenizer):
...     def tokenize(self, text):
...         return text.split()
>>> scorer = RougeScorer(["rouge1"], tokenizer=WhitespaceKeepCaseTokenizer())
>>> scorer.score("Testing one two", "testing")
{'rouge1': Score(precision=0.0, recall=0.0, fmeasure=0.0)}
```

### Aggregating scores with `kurenai.scoring`

`kurenai.scoring` re-exports rouge-score's `scoring` module, so
`BootstrapAggregator` and friends can be imported from `kurenai` directly:

```python
>>> from kurenai.rouge_scorer import RougeScorer
>>> from kurenai.scoring import BootstrapAggregator
>>> scorer = RougeScorer(["rouge1"])
>>> aggregator = BootstrapAggregator()
>>> aggregator.add_scores(scorer.score("テスト いち に", "テスト に"))
>>> aggregator.add_scores(scorer.score("テスト いち", "テスト いち に"))
>>> result = aggregator.aggregate()
>>> result["rouge1"].mid.fmeasure
0.8
```

### CLI

`python -m kurenai` calculates ROUGE scores between a newline-delimited
target file and prediction file (one record per line), and writes a CSV
report to stdout or `--output`. Unlike rouge-score's own CLI, it is built
with `argparse` and does not depend on `absl`.

```console
$ python -m kurenai --rouge-types rouge1,rougeL \
    --target-file targets.txt --prediction-file predictions.txt
id,rouge1-P,rouge1-R,rouge1-F,rougeL-P,rougeL-R,rougeL-F
0,0.666667,0.666667,0.666667,0.666667,0.666667,0.666667
1,0.750000,0.666667,0.705882,0.625000,0.555556,0.588235
```

With `--use-aggregator`, per-record scores are aggregated with
`BootstrapAggregator` instead:

```console
$ python -m kurenai --rouge-types rouge1,rougeL \
    --target-file targets.txt --prediction-file predictions.txt \
    --use-aggregator
score_type,low,mid,high
rouge1-R,0.666667,0.666667,0.666667
rouge1-P,0.666667,0.708333,0.750000
rouge1-F,0.666667,0.686275,0.705882
rougeL-R,0.555556,0.611111,0.666667
rougeL-P,0.625000,0.645833,0.666667
rougeL-F,0.588235,0.627451,0.666667
```

Add `--output scores.csv` to write the CSV to a file instead of stdout.

## Compatibility with rouge-score

|                             | Status                                                  |
| --------------------------- | -------------------------------------------------------- |
| ROUGE-N (1-9), ROUGE-L, ROUGE-Lsum | Compatible                                        |
| `score()` / `score_multi()` | Compatible                                                |
| Constructor signature (`use_stemmer`, `split_summaries`, `tokenizer`) | Compatible |
| `scoring` utilities (`Score`, `AggregateScore`, `BootstrapAggregator`, `BaseScorer`, `fmeasure`) | Compatible, re-exported as `kurenai.scoring` |
| Default tokenizer | **Extended**: supports non-ascii text (rouge-score's `DefaultTokenizer` drops it) |
| `use_stemmer` | **Extended**: does not corrupt non-ascii tokens (only ASCII alphanumeric tokens of 4+ chars are stemmed) |
| CLI | **Extended**: `python -m kurenai` uses `argparse` and does not require `absl` |
| Tokenizing raw Japanese text (`CharacterTokenizer`, morphological analyzers, etc.) | Not yet supported (planned for a later phase) |
| Japanese-aware `split_summaries` | Not yet supported; rouge-score's `split_summaries` uses `nltk.sent_tokenize`, which assumes English text |
