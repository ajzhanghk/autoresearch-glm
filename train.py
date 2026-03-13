import itertools
import json
from dataclasses import dataclass

import numpy as np

from prepare import auc_score, load_dataset

# The agent should primarily edit this policy block.
SCREEN_K = 10
FEATURE_CAP = 18
INTERACTION_CAP = 5
CLIP_Q = 0.99
L2 = 0.0
TRANSFORMS = ("identity", "log1p", "sqrt", "square")


@dataclass(frozen=True)
class Candidate:
    name: str
    train: np.ndarray
    val: np.ndarray
    score: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x @ x) * (y @ y))
    if denom <= 1e-12:
        return 0.0
    return float(abs((x @ y) / denom))


def maybe_clip(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if CLIP_Q is None:
        return train, val, ""
    lo, hi = np.quantile(train, [1.0 - CLIP_Q, CLIP_Q])
    return np.clip(train, lo, hi), np.clip(val, lo, hi), f"_clip{int(CLIP_Q * 100)}"


def transform_feature(name: str, train: np.ndarray, val: np.ndarray, transform: str) -> Candidate | None:
    train, val, clip_suffix = maybe_clip(train, val)

    if transform == "identity":
        feat_train = train
        feat_val = val
        feat_name = f"{name}{clip_suffix}"
    elif transform == "square":
        feat_train = train**2
        feat_val = val**2
        feat_name = f"{name}{clip_suffix}__square"
    elif transform == "log1p":
        if train.min() < 0.0 or val.min() < 0.0:
            return None
        feat_train = np.log1p(train)
        feat_val = np.log1p(val)
        feat_name = f"{name}{clip_suffix}__log1p"
    elif transform == "sqrt":
        if train.min() < 0.0 or val.min() < 0.0:
            return None
        feat_train = np.sqrt(train)
        feat_val = np.sqrt(val)
        feat_name = f"{name}{clip_suffix}__sqrt"
    else:
        raise ValueError(f"Unknown transform: {transform}")

    return Candidate(name=feat_name, train=feat_train, val=feat_val, score=0.0)


def screen_variables(x_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]) -> list[int]:
    scores = [safe_corr(x_train[:, idx], y_train) for idx in range(x_train.shape[1])]
    return sorted(range(len(feature_names)), key=lambda idx: scores[idx], reverse=True)


def standardize(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (val - mean) / std


def build_design(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ranked = screen_variables(x_train, y_train, feature_names)
    screened = ranked[:SCREEN_K]

    singles: list[Candidate] = []
    for idx in screened:
        for transform in TRANSFORMS:
            candidate = transform_feature(
                name=feature_names[idx],
                train=x_train[:, idx],
                val=x_val[:, idx],
                transform=transform,
            )
            if candidate is not None:
                singles.append(candidate)

    rescored = [
        Candidate(name=c.name, train=c.train, val=c.val, score=safe_corr(c.train, y_train))
        for c in singles
    ]
    rescored.sort(key=lambda item: item.score, reverse=True)

    chosen: list[Candidate] = []
    seen = set()
    for candidate in rescored:
        if candidate.name in seen:
            continue
        chosen.append(candidate)
        seen.add(candidate.name)
        if len(chosen) >= FEATURE_CAP:
            break

    if INTERACTION_CAP > 0:
        interaction_pool: list[Candidate] = []
        source = screened[: min(len(screened), max(2, INTERACTION_CAP + 1))]
        for left, right in itertools.combinations(source, 2):
            left_name = feature_names[left]
            right_name = feature_names[right]
            left_train, left_val, _ = maybe_clip(x_train[:, left], x_val[:, left])
            right_train, right_val, _ = maybe_clip(x_train[:, right], x_val[:, right])
            train_term = left_train * right_train
            val_term = left_val * right_val
            interaction_pool.append(
                Candidate(
                    name=f"{left_name}__x__{right_name}",
                    train=train_term,
                    val=val_term,
                    score=safe_corr(train_term, y_train),
                )
            )
        interaction_pool.sort(key=lambda item: item.score, reverse=True)
        chosen.extend(interaction_pool[:INTERACTION_CAP])

    train_matrix = np.column_stack([candidate.train for candidate in chosen])
    val_matrix = np.column_stack([candidate.val for candidate in chosen])
    train_matrix, val_matrix = standardize(train_matrix, val_matrix)
    return train_matrix, val_matrix, [candidate.name for candidate in chosen]


def fit_logistic_glm(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_train)), x_train])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.ones_like(beta)
    penalty[0] = 0.0

    for _ in range(50):
        logits = design @ beta
        probs = sigmoid(logits)
        weights = np.clip(probs * (1.0 - probs), 1e-6, None)
        grad = design.T @ (y_train - probs) - L2 * penalty * beta
        hess = design.T @ (weights[:, None] * design)
        hess.flat[:: hess.shape[0] + 1] += L2 * penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta += step
        if np.max(np.abs(step)) < 1e-6:
            break

    return beta


def predict_scores(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return design @ beta


def describe_policy() -> str:
    clip = "none" if CLIP_Q is None else f"clip{int(CLIP_Q * 100)}"
    transforms = "+".join(TRANSFORMS)
    return (
        f"screen_k={SCREEN_K} feature_cap={FEATURE_CAP} "
        f"interaction_cap={INTERACTION_CAP} clip={clip} "
        f"l2={L2:.3f} transforms={transforms}"
    )


def run_experiment() -> dict:
    dataset = load_dataset()
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    x_val = dataset["x_val"]
    y_val = dataset["y_val"]
    feature_names = dataset["feature_names"]

    train_matrix, val_matrix, names = build_design(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        feature_names=feature_names,
    )
    beta = fit_logistic_glm(train_matrix, y_train)
    val_scores = predict_scores(val_matrix, beta)
    val_auc = auc_score(y_val, val_scores)

    return {
        "val_auc": val_auc,
        "num_features": len(names),
        "feature_names": names,
        "policy": describe_policy(),
    }


def main() -> None:
    result = run_experiment()
    print(result["policy"])
    print(json.dumps(result, indent=2))
    print(f"val_auc: {result['val_auc']:.6f}")


if __name__ == "__main__":
    main()
