# autoresearch-glm

`autoresearch-glm` is a minimal fork of [Andrej Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch).

The original repo studies autonomous code-editing loops on a tiny GPT training benchmark. This fork keeps that spirit, but pivots the benchmark to tabular binary classification with a logistic GLM and autonomous feature search.

The canonical v1 benchmark is the Taiwan credit card default dataset from UCI. That keeps the repo grounded in a real credit-scoring problem while preserving the tiny, fixed-benchmark character of the upstream project.

Compared with the upstream GPT benchmark, this fork has a much smaller dependency footprint and no GPU requirement. It is a plain CPU-first Python benchmark with NumPy, pandas, `ucimlrepo`, and Matplotlib for analysis.

The idea is simple: give an AI agent a small but real tabular modeling setup and let it experiment autonomously. It modifies the feature-policy code, runs the benchmark, checks whether validation AUC improved, keeps or discards the change, and repeats. You wake up later to a log of experiments and, ideally, a better GLM.

Instead of touching a large framework, you mainly program the `program.md` file that sets the operating rules for the agent, while the agent edits `train.py`.

That makes the fork about one concrete problem:

**autonomous fixed-budget feature discovery for GLMs on tabular data**

![Autoresearch-GLM progress](progress.png)

## How it works

The repo is deliberately kept small and only really has three files that matter:

- `prepare.py` — fixed benchmark setup, TaiwanCredit download/cache, validation split, and AUC metric. Not modified during normal experiments.
- `train.py` — the single file the agent edits. It contains the current feature-search policy, compact GLM fitting code, and evaluation logic.
- `program.md` — baseline instructions for one agent. Point your agent here and let it go. This file is edited and iterated on by the human.

By design, the benchmark is narrow and fixed:

- binary classification only
- logistic regression / GLM only
- validation AUC only

One run evaluates one current policy. The point is not to build AutoML. The point is to keep the code small enough that an agent can rewrite the benchmark itself while still working on a classical modeling surface that matters.

## Why this is interesting

Classical tabular modeling still contains a hard combinatorial core. Even when the final estimator is just logistic regression, the choice of variables, transformations, caps, and interactions can dominate performance. In domains like credit scoring, fraud, AML, and loss forecasting, that search space has historically been important, labor-intensive, and difficult to systematize.

`autoresearch-glm` applies the autoresearch loop to exactly that surface. The agent edits a small piece of experiment code, runs a short benchmark, observes a single metric, and keeps only changes that actually help.

Using TaiwanCredit as the default example makes the benchmark concrete. It is a compact, classical binary credit dataset with exactly the sort of variable handling questions GLM workflows care about: screening, monotone-friendly transforms, nonlinear corrections, and tightly controlled interaction search.

## Quick start

Requirements:

- Python 3.11+
- internet access the first time `prepare.py` fetches TaiwanCredit from UCI

```bash
# 1. Install the minimal dependencies
python -m pip install numpy pandas matplotlib ucimlrepo

# 2. Prepare the fixed TaiwanCredit benchmark
python prepare.py

# 3. Manually run a single experiment
python train.py
```

If the above commands work, your setup is ready and you can go into autonomous research mode.

`prepare.py` fetches TaiwanCredit through `ucimlrepo`, builds the fixed train/validation split, and caches the prepared arrays. There is no PyTorch stack, no tokenizer, and no GPU dependency. If you can run a normal scientific Python environment, you can usually run this fork.

If you want the cache inside the repo instead of `~/.cache/autoresearch-glm`, set:

```bash
export AUTORESEARCH_GLM_CACHE=.cache/autoresearch-glm
```

## Running the agent

Simply spin up your Claude, Codex, or whatever agent you want in this repo, then prompt something like:

```text
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight skill. It tells the agent what it can edit, what it should optimize, how to log experiments, and when to keep or revert a change.

The intended loop is:

1. `prepare.py` stays fixed.
2. The agent reads `program.md`.
3. The agent edits `train.py`.
4. The agent runs `python train.py`.
5. The agent keeps or discards the code change based on `val_auc`.

`train.py` should represent one current policy, not an internal sweep over many configs.

## Project structure

```text
prepare.py      fixed benchmark setup and runtime utilities
train.py        feature policy, GLM fit, evaluation (agent modifies this)
program.md      agent instructions
analysis.ipynb  experiment analysis notebook
pyproject.toml  dependencies
```

## Design choices

- Single file to modify. The agent should mainly touch `train.py`. This keeps the scope manageable and diffs reviewable.
- Fixed benchmark. The data split and metric stay fixed in `prepare.py`, so experiments remain directly comparable.
- Single scalar objective. Validation AUC is the only score that matters.
- Self-contained and lightweight. No GPU, no distributed stack, no giant config system, and no platform-specific training complexity.

This fork has a much lower barrier to entry than the upstream GPT benchmark. It is ready to try on normal laptops and CPU boxes, and it opens a different research direction: autonomous feature discovery for interpretable GLM workflows.

## v1 scope

Version 1 is intentionally narrow:

- binary classification only
- logistic regression / GLM only
- validation AUC only
- compact feature search inside `train.py`

No multiclass support, no regression support, no deep learning, no feature platform, and no large framework abstractions.
