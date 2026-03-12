# autoresearch-glm

`autoresearch-glm` is a minimal fork of [Andrej Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch).

The original repo studies autonomous code-editing loops on a tiny GPT training benchmark. This fork keeps that spirit, but pivots the benchmark to tabular binary classification with a logistic GLM and autonomous feature search.

The idea is simple:

- `prepare.py` defines a fixed tabular benchmark and validation split.
- `train.py` is the main editable research surface.
- the agent searches over variable selection, variable transformation, interaction screening, and simple regularization.
- the score is one scalar: validation AUC.

That makes the fork about one concrete problem:

**autonomous fixed-budget feature discovery for GLMs on tabular data**

## Why this is interesting

Classical tabular modeling still contains a hard combinatorial core. Even when the final estimator is just logistic regression, the choice of variables, transformations, caps, and interactions can dominate performance. In domains like credit scoring, fraud, AML, and loss forecasting, that search space has historically been important, labor-intensive, and difficult to systematize.

`autoresearch-glm` applies the autoresearch loop to exactly that surface. The agent edits a small piece of experiment code, runs a short benchmark, observes a single metric, and keeps only changes that actually help.

## v1 scope

Version 1 is intentionally narrow:

- binary classification only
- logistic regression / GLM only
- validation AUC only
- compact feature search inside `train.py`

No multiclass support, no regression support, no deep learning, no feature platform, and no large framework abstractions.

## Quickstart

```bash
python prepare.py
python train.py
```

By default, `prepare.py` creates a deterministic synthetic credit-style dataset. You can also point it at a CSV or Parquet file with a binary target:

```bash
python prepare.py --input data.csv --target target
```

If you want the cache inside the repo instead of `~/.cache/autoresearch-glm`, set:

```bash
export AUTORESEARCH_GLM_CACHE=.cache/autoresearch-glm
```

## Repo shape

- `prepare.py`: fixed tabular prep and metric logic
- `train.py`: compact agent-editable feature search and GLM fit
- `program.md`: policy for autonomous improvement

The goal is not to build AutoML. The goal is to keep the code small enough that an agent can rewrite the benchmark itself, while still operating on a research surface that matters.
