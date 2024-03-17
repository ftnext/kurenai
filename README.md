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

* Supports ROUGE1 only

## Usage

紅 (kurenai) has the same interface as [rouge-score](https://pypi.org/project/rouge-score/).

```python
>>> from kurenai.rouge_scorer import RougeScorer
>>> scorer = RougeScorer(["rouge1"])
>>> scorer.score('いぬ ねこ', 'いぬ ねこ')
{'rouge1': Score(precision=1.0, recall=1.0, fmeasure=1.0)}
>> scorer.score('The quick brown fox jumps over the lazy dog', 'The quick brown dog jumps on the log.')
{'rouge1': Score(precision=0.75, recall=0.6666666666666666, fmeasure=0.7058823529411765)}
```
