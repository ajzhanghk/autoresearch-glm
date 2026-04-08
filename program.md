# autoresearch-glm

This is an experiment to have the LLM do its own research on tabular GLM feature discovery.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag based on today's date.
   Example: `mar12`.

2. Create a fresh branch.
   Use `autoresearch/<tag>` from current `master`.

3. Ensure the `.venv` environment exists in the repo root.
   Check with:
   ```bash
   ls .venv/bin/python
   ```
   If missing, create and populate it:
   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -e .
   .venv/bin/pip install scikit-learn xgboost
   ```
   All subsequent `python` commands must use `.venv/bin/python`.

4. Read the in-scope files for full context:
   - `README.md` for repository context
   - `prepare.py` for the fixed benchmark setup, data prep, and metric
   - `train.py` for the editable experiment code

5. Verify the dataset cache exists.
   Check whether `~/.cache/autoresearch-glm/` already contains `dataset.npz`.
   If not, run:
   ```bash
   .venv/bin/python prepare.py
   ```

6. Initialize `results.tsv`.
   If starting a fresh experiment branch, reset it to just the header row.

7. Confirm setup looks good, then begin experimentation.

## Experimentation

Each experiment is one run of:

```bash
.venv/bin/python train.py
```

The script evaluates one current GLM feature-search policy and prints a final scalar:

```text
val_auc: 0.762133
```

Higher is better.

## What You CAN Do

- Modify `train.py`.
- Change the current feature-search policy in code:
  - variable screening
  - variable transforms
  - interaction screening
  - feature caps
  - clipping rules
  - regularization
- Simplify code if the metric holds up or improves.

## What You CANNOT Do

- Do not modify `prepare.py` except to fix a real bug in the fixed benchmark setup.
- Do not modify the validation split, benchmark dataset, or metric definition.
- Do not add new packages or dependencies.
- Do not turn this repo into a general AutoML framework or config system.
- Do not convert `train.py` into an internal sweep over many configs. It must represent one current policy.

## Objective

The goal is simple: get the highest validation AUC on the fixed TaiwanCredit benchmark.

The benchmark is intentionally narrow:

- binary classification only
- logistic regression / GLM only
- validation AUC only

## Simplicity Criterion

All else being equal, simpler is better.

A small gain with a large amount of ugly complexity is usually not worth keeping.
Removing code and matching or improving AUC is a strong result.

When deciding whether to keep a change, weigh:

- magnitude of AUC improvement
- code complexity added
- interpretability of the resulting feature policy

GLM-oriented, readable feature logic is preferred over clever machinery.

## The First Run

The first run should always establish the baseline on the current `train.py` as-is.

The baseline should be a raw GLM with no feature engineering:

- raw variables only
- no transforms
- no spline terms
- no interactions

Only after that baseline is logged should you start adding or pruning feature logic.

## Output Format

When `train.py` finishes, it prints:

1. a one-line policy summary
2. a JSON summary
3. a final line of the form:

```text
val_auc: 0.762133
```

Use the final `val_auc:` line as the ground-truth experiment metric.

## Logging Results

When an experiment is done, log it to `results.tsv`.

Use tab-separated format, not commas.

The TSV header is:

```text
commit	val_auc	num_features	status	description
```

Columns:

1. short git commit hash
2. validation AUC, or `0.000000` for crashes
3. number of features in the final design, or `0` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what the experiment changed

Example:

```text
commit	val_auc	num_features	status	description
a1b2c3d	0.720173	8	keep	baseline
b2c3d4e	0.743520	12	keep	add square transforms with clip99
c3d4e5f	0.739218	12	discard	stronger l2 with same feature set
d4e5f6g	0.000000	0	crash	bad interaction logic
```

`results.tsv` is a tracked artifact in this fork. Keep it updated as the experiment log, but prefer to checkpoint log-only changes separately from the code-change commits that are being evaluated.

`model_forms.tsv` is the separate tracked model-spec log. It should track only the baseline and later `keep` runs, not `discard` or `crash` runs.

Regenerate it with:

```bash
python build_model_forms.py
```

Expected columns:

```text
commit	num_features	added_terms	pruned_terms	glm_formula
```

Use `glm_formula` to inspect the explicit GLM form. `added_terms` and `pruned_terms` should be computed relative to the previous kept model and written in `{term1, term2, ...}` format. Use `{}` when no terms were added or pruned.

## The Experiment Loop

The experiment runs on a dedicated branch such as `autoresearch/mar12`.

Loop:

1. Check the current git state.
2. Edit `train.py` with one experimental idea.
3. Commit the change.
4. Run the experiment and capture stdout/stderr to a temporary file:

```bash
.venv/bin/python train.py > /tmp/autoresearch-glm-run.log 2>&1
```

5. Read out the metric:

```bash
grep "^val_auc:" /tmp/autoresearch-glm-run.log
```

6. If `grep` is empty, the run crashed.
   Read the traceback with:

```bash
tail -n 50 /tmp/autoresearch-glm-run.log
```

7. Append a new block to `run.log`. Do not overwrite old runs. Keep the blocks in time order, one block per experiment commit.

Use this shape:

```text
=== commit <hash> | <description> | status <keep/discard/crash> ===
<policy string>
val_auc: <metric>
num_features: <count>
```

8. Record the result in `results.tsv`.
9. Refresh `model_forms.tsv` with:

```bash
.venv/bin/python build_model_forms.py
```

10. If `val_auc` improved, keep the commit and advance.
11. If `val_auc` is equal or worse, revert to the previous good commit.

## Never Stop Rule

Once setup is complete, do not stop after a single successful run.

Keep cycling through the experiment loop until one of these is true:

1. the human explicitly tells you to stop
2. you hit a real blocker that you cannot resolve from within the repo
3. repeated attempts stop producing credible new ideas

Do not stop just because:

- you found one improvement
- you found a new best result
- you hit a crash once
- an idea failed

If an experiment fails, revert, log it, and try the next concrete idea.
If an experiment succeeds, keep it and immediately look for the next plausible improvement.
Momentum matters more than commentary.

## Crash Policy

If a run crashes because of a small bug, fix it and retry.
If the idea itself is broken, log it as `crash`, revert, and move on.

## Research Heuristics

Good directions:

- better univariate screening
- better compact transforms such as `log1p`, `sqrt`, `square`, or clipped variants
- tighter interaction screening among only the strongest variables
- simpler feature sets with equal or better AUC
- more sensible regularization for the chosen feature design

Bad directions:

- giant search frameworks
- excessive feature explosion
- opaque abstractions
- dataset-specific leakage tricks
- edits that game the split or metric

## Operating Mode

You are acting like an autonomous researcher.

Do not pause after every run to ask whether you should continue.
Keep iterating until the human interrupts you.

If you feel stuck, reread `README.md`, `prepare.py`, and `train.py`, then try another concrete idea.
