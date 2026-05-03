# autoresearch-glm

`autoresearch-glm` is a minimal fork of [Andrej Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch).

The original repo studies autonomous code-editing loops on a tiny GPT training benchmark. This fork keeps that spirit, but pivots the benchmark to tabular binary classification with a logistic GLM and autonomous feature search.

The canonical benchmark is the Taiwan credit card default dataset from UCI. The March 2026 `v1` line established that fixed TaiwanCredit GLM benchmark, the April 2026 `v2` line expanded the feature-search machinery with XGBoost-seeded splines and rectangle interactions, and the May 2026 `v3` line replaces XGBoost with GAMI-Net-style ReLU subnetworks as the feature engine. The dataset and metric stay fixed across all three lines, keeping the repo grounded in a real credit-scoring problem while preserving the tiny, fixed-benchmark character of the upstream project.

Compared with the upstream GPT benchmark, this fork has a much smaller dependency footprint and no GPU requirement. It is a plain CPU-first Python benchmark with NumPy, pandas, `ucimlrepo`, and Matplotlib for analysis.

The idea is simple: give an AI agent a small but real tabular modeling setup and let it experiment autonomously. It modifies the feature-policy code, runs the benchmark, checks whether validation AUC improved, keeps or discards the change, and repeats. You wake up later to a log of experiments and, ideally, a better GLM.

Instead of touching a large framework, you mainly program the `program.md` file that sets the operating rules for the agent, while the agent edits `train.py`.

That makes the fork about one concrete problem:

**autonomous fixed-budget feature discovery for GLMs on tabular data**

![Autoresearch-GLM progress](progress.png)

## Current frontier

This README describes the current `v3` line of the project: the GAMI-Net-style ReLU neural network feature-engine version. The `v2` line (XGBoost-seeded splines and rectangle interactions) is preserved as a comparison frontier.

The best kept `v2` policy on the fixed TaiwanCredit validation split (Apr 8, 2026) is:

- commit `ea71156`
- validation AUC `0.781135`
- 36 final GLM terms after pruning

The first `v3` baseline (May 2026) lands at validation AUC `0.777844` with the same 36-feature budget, using:

- raw identity terms plus per-variable ReLU MLP shape functions (`nn_main`) as the main-effect path
- post-fit pruning to keep the final GLM compact
- residual-based two-way interaction screening over the top 12 screened raw variables
- a small bivariate ReLU MLP per surviving pair (`nn_pair`), one feature column per interaction

The agent's job in `v3` is to push the NN frontier above the `v2` frontier while keeping the GLM compact and the feature engine readable.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- `prepare.py` — fixed benchmark setup, TaiwanCredit download/cache, validation split, and AUC metric. Not modified during normal experiments.
- `train.py` — the single file the agent edits. It contains the current feature-search policy, compact GLM fitting code, and evaluation logic.
- `program.md` — baseline instructions for one agent. Point your agent here and let it go. This file is edited and iterated on by the human.

Inside that small surface, the current `train.py` policy already supports a useful set of compact tabular feature-search components:

- variable screening by marginal correlation
- optional tail clipping for raw variables
- `identity` raw terms
- `nn_main` per-variable ReLU MLP shape functions (v3 primary main-effect path)
- `nn_pair` bivariate ReLU MLP per surviving interaction pair (v3 interaction path)
- `xgb_bin` and `xgb_spline` legacy XGBoost-seeded paths (v2, kept for replay and ablations)
- residual-based FAST interaction screening
- L1/L2 regularization and a post-fit pruning pass

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

- Python 3.12+
- internet access the first time `prepare.py` fetches TaiwanCredit from UCI

```bash
# 1. Create the project virtualenv
python3.12 -m venv .venv

# 2. Install the package and dependencies
.venv/bin/pip install -e .

# 3. Prepare the fixed TaiwanCredit benchmark
.venv/bin/python prepare.py

# 4. Manually run a single experiment
.venv/bin/python train.py
```

If the above commands work, your setup is ready and you can go into autonomous research mode.

`prepare.py` fetches TaiwanCredit through `ucimlrepo`, builds the fixed train/validation split, and caches the prepared arrays. There is no PyTorch stack, no tokenizer, and no GPU dependency. If you can run a normal scientific Python environment, you can usually run this fork.

The current agent workflow assumes the repo-local `.venv` is the canonical environment and that subsequent commands use `.venv/bin/python`.

If you want the cache inside the repo instead of `~/.cache/autoresearch-glm`, set:

```bash
export AUTORESEARCH_GLM_CACHE=.cache/autoresearch-glm
```

## Running the agent

Simply spin up your Claude, Codex, or whatever agent you want in this repo, then prompt something like:

```text
Hi have a look at program.md and let's kick off a new experiment! 
```

The `program.md` file is essentially a super lightweight skill. It tells the agent what it can edit, what it should optimize, how to log experiments, and when to keep or revert a change.

The intended loop is:

1. `prepare.py` stays fixed.
2. The agent reads `program.md`.
3. The agent edits `train.py`.
4. The agent runs `.venv/bin/python train.py`.
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

## Current feature machinery

The benchmark intentionally keeps the final estimator fixed as a logistic GLM, but `train.py` now contains a richer feature-policy surface than the initial raw baseline.

### GAMI-Net-style ReLU subnetworks (v3)

The current code uses small ReLU MLPs as feature-discovery subnetworks, following the [GAMI-Net](https://arxiv.org/abs/2003.07132) architecture: each subnetwork is a shape function that maps either one raw variable (main effect) or one bivariate raw pair (interaction) to a single scalar feature column for the downstream GLM.

- `nn_main` fits a per-variable MLP from the standardized raw variable to `y` (binary, treated as 0/1 regression). The MLP's prediction becomes one continuous main-effect column.
- `nn_pair` fits a per-pair MLP from the standardized raw bivariate input to the residual left by the main-effect GLM. The MLP's prediction becomes one interaction column per surviving pair.
- `identity` keeps the raw clipped variable available alongside these nonlinear corrections so the GLM can still anchor a linear shape when it is best.

The default subnetwork is a `(16, 16)` ReLU MLP trained with Adam, L2 weight decay, and early stopping. The downstream GLM provides the logistic link, the L1/L2 sparsification, and the post-fit pruning.

### Legacy XGBoost-seeded main effects (v2)

The v2 main-effect engine remains available for replay and ablations:

- `xgb_bin` turns discovered depth-1 stump split locations into a compact piecewise-constant univariate feature.
- `xgb_spline` turns those same split locations into truncated linear bases, giving a continuous piecewise-linear main effect.

In v2, `xgb_spline` was the primary main-effect path; v3 replaces it with `nn_main`.

### Post-fit pruning

After the candidate main-effect and interaction terms are selected and standardized, the code fits the GLM once, ranks terms by absolute coefficient magnitude, prunes weak terms, and re-standardizes the reduced design. The current frontier uses `PRUNE_KEEP = 36`.

### Residual-based interaction screening

The interaction path is explicitly residual-based:

1. Fit the current main-effect GLM.
2. Compute the residual left by that fit.
3. Coarsely score raw variable pairs with a FAST-style interaction score on the residual.
4. Send only the top screened pairs into the active interaction engine.
5. Materialize each surviving pair as a GLM column.

The interaction engine itself is selected by `INTERACTION_ENGINE`:

- `"nn"` (v3 default) fits one bivariate ReLU MLP per surviving pair on the residual and uses its prediction as a single interaction column.
- `"xgb"` (v2) fits a constrained depth-2 XGBoost model and converts its leaves into explicit rectangle indicator terms.

## Recent empirical notes

The v2 search loop (XGBoost feature engine) established a few practical conclusions that still inform the v3 line:

- broader pre-prune candidate pools at 45 or 50 features were worse than `FEATURE_CAP = 40`.
- coefficient-threshold pruning tied or underperformed the `keep36` pruning rule.
- interaction screening mattered only after broadening the raw pair source pool to 12 raws; once widened, `INTERACTION_CAP = 4` produced the v2 best kept model.

In v3 the natural directions to explore are NN-specific:

- subnetwork width and depth (`NN_HIDDEN`) — narrower or deeper than the `(16, 16)` baseline.
- L2 weight decay (`NN_ALPHA`) and learning rate (`NN_LR`).
- whether identity terms still help once `nn_main` is the main-effect path.
- whether `nn_pair` benefits from a smaller architecture than `nn_main` since it sees only two inputs.

## Design choices

- Single file to modify. The agent should mainly touch `train.py`. This keeps the scope manageable and diffs reviewable.
- Fixed benchmark. The data split and metric stay fixed in `prepare.py`, so experiments remain directly comparable.
- Single scalar objective. Validation AUC is the only score that matters.
- Self-contained and lightweight. No GPU, no distributed stack, no giant config system, and no platform-specific training complexity.

This fork has a much lower barrier to entry than the upstream GPT benchmark. It is ready to try on normal laptops and CPU boxes, and it opens a different research direction: autonomous feature discovery for interpretable GLM workflows.

## v1 scope (March 2026)

Version 1 was the March 2026 line of the project. It established the fixed TaiwanCredit benchmark and kept the problem intentionally narrow:

- binary classification only
- logistic regression / GLM only
- validation AUC only
- compact feature search inside `train.py`

No multiclass support, no regression support, no deep learning, no feature platform, and no large framework abstractions.

## v2 scope (April 2026)

Version 2 keeps the same fixed benchmark, but upgrades the feature-search machinery beyond the original March raw-only baseline:

- binary classification only
- logistic regression / GLM only
- validation AUC only
- XGBoost-seeded `xgb_bin` and `xgb_spline` main effects
- residual-based XGBoost interaction screening with explicit rectangle terms
- compact feature search inside `train.py`

No multiclass support, no regression support, no deep learning, no feature platform, and no large framework abstractions.

## v3 scope (May 2026)

Version 3 keeps the same fixed benchmark and the same logistic GLM as the final estimator, but replaces the v2 XGBoost feature engine with a [GAMI-Net](https://arxiv.org/abs/2003.07132)-style ReLU neural network feature engine:

- binary classification only
- logistic regression / GLM only
- validation AUC only
- per-variable ReLU MLP subnetwork main effects (`nn_main`) replacing `xgb_spline`
- residual-based FAST interaction screening (unchanged from v2)
- per-pair bivariate ReLU MLP subnetwork interactions (`nn_pair`) replacing the v2 rectangle indicator terms
- compact feature search inside `train.py`

The neural networks act as feature-discovery subnetworks, not as the final model. Each subnetwork outputs a single shape function that becomes a GLM column. The L1/L2 regularization, post-fit pruning, and standardization machinery from v2 carry over unchanged.

The implementation uses `sklearn.neural_network.MLPRegressor` to keep the dependency footprint minimal — no PyTorch, no TensorFlow, no GPU. The legacy v2 XGBoost paths (`xgb_bin`, `xgb_spline`, and `INTERACTION_ENGINE="xgb"`) are preserved in `train.py` so older v2 commits can still be replayed via `build_model_forms.py`.
