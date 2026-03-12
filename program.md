# Program: autoresearch-glm

Goal: improve the validation AUC of the fixed binary tabular benchmark, with TaiwanCredit as the canonical v1 example.

The core benchmark is intentionally small. `prepare.py` owns the data split and the AUC definition. `train.py` is the main research surface and should stay compact, readable, and hackable.

## Workflow

1. Run `python prepare.py` once to materialize the benchmark dataset.
2. Run `python train.py` and optimize the final `val_auc:` line.
3. Make small, testable edits and keep only changes that improve the score without making the code needlessly larger or more obscure.

## Primary edit surface

- Edit `train.py` first.
- Avoid editing `prepare.py` unless there is a clear bug in data preparation or metric computation.
- Do not change the validation split, the task definition, or the reported objective just to make the score look better.
- Treat TaiwanCredit as the fixed benchmark.

## Good directions

- Better variable screening before fitting the GLM.
- Better compact transformations such as `identity`, `log1p`, `sqrt`, `square`, or clipped variants.
- Better screening of pairwise interactions among a small set of promising variables.
- Better simple regularization or feature caps that improve AUC while preserving interpretability.

## Constraints

- Optimize one scalar objective: validation AUC.
- Keep experiments short and fixed-budget.
- Control feature explosion aggressively.
- Prefer interpretable GLM-oriented improvements over large search frameworks.
- Do not add heavy dependencies or infrastructure.
- Roll back changes when they hurt either AUC or simplicity.

## Style

- Prefer clear code over clever code.
- Prefer a small number of strong candidate features over a huge design matrix.
- Keep the repo faithful to the original autoresearch spirit: compact code edits, empirical testing, and disciplined iteration.
