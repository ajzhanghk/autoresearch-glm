# autoresearch-glm

`autoresearch-glm` is a minimal fork of [Andrej Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch).

The original repo studies autonomous code-editing loops on a tiny GPT training benchmark. This fork keeps that spirit, but pivots the benchmark to tabular binary classification with a logistic GLM and autonomous feature search.

The canonical v1 benchmark is the Taiwan credit card default dataset from UCI. That keeps the repo grounded in a real credit-scoring problem while preserving the tiny, fixed-benchmark character of the upstream project.

Compared with the upstream GPT benchmark, this fork has a much smaller dependency footprint and no GPU requirement. It is a plain CPU-first Python benchmark with NumPy, pandas, `ucimlrepo`, and Matplotlib for analysis.

The idea is simple:

- `prepare.py` defines a fixed tabular benchmark and validation split.
- `train.py` is the main editable research surface.
- the agent searches over variable selection, variable transformation, interaction screening, and simple regularization.
- the score is one scalar: validation AUC.

That makes the fork about one concrete problem:

**autonomous fixed-budget feature discovery for GLMs on tabular data**

![Autoresearch-GLM progress](progress.png)

## Why this is interesting

Classical tabular modeling still contains a hard combinatorial core. Even when the final estimator is just logistic regression, the choice of variables, transformations, caps, and interactions can dominate performance. In domains like credit scoring, fraud, AML, and loss forecasting, that search space has historically been important, labor-intensive, and difficult to systematize.

`autoresearch-glm` applies the autoresearch loop to exactly that surface. The agent edits a small piece of experiment code, runs a short benchmark, observes a single metric, and keeps only changes that actually help.

Using TaiwanCredit as the default example makes the benchmark concrete. It is a compact, classical binary credit dataset with exactly the sort of variable handling questions GLM workflows care about: screening, monotone-friendly transforms, nonlinear corrections, and tightly controlled interaction search.

## v1 scope

Version 1 is intentionally narrow:

- binary classification only
- logistic regression / GLM only
- validation AUC only
- compact feature search inside `train.py`

No multiclass support, no regression support, no deep learning, no feature platform, and no large framework abstractions.

## Quickstart

Requirements:

- Python 3.11+
- internet access the first time `prepare.py` fetches TaiwanCredit from UCI

```bash
# 1. Install the minimal dependencies
python -m pip install numpy pandas matplotlib ucimlrepo

# 2. Prepare the fixed TaiwanCredit benchmark
python prepare.py

# 3. Run the compact GLM feature search benchmark
python train.py
```

`prepare.py` fetches the TaiwanCredit benchmark from UCI through `ucimlrepo`, builds the fixed train/validation split, and caches the prepared arrays.

There is no PyTorch stack, no tokenizer, and no GPU dependency. If you can run a normal scientific Python environment, you can usually try this fork.

If you want the cache inside the repo instead of `~/.cache/autoresearch-glm`, set:

```bash
export AUTORESEARCH_GLM_CACHE=.cache/autoresearch-glm
```

## Repo shape

- `prepare.py`: fixed tabular prep and metric logic
- `train.py`: compact agent-editable feature search and GLM fit
- `program.md`: policy for autonomous improvement
- `analysis.ipynb`: experiment analysis notebook, in the style of upstream `autoresearch`

The goal is not to build AutoML. The goal is to keep the code small enough that an agent can rewrite the benchmark itself, while still operating on a research surface that matters.
