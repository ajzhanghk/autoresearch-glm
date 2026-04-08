import itertools
import json
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from prepare import load_dataset

# The agent should primarily edit this policy block.
SCREEN_K = None
FEATURE_CAP = 40
INTERACTION_CAP = 0
FAST_BINS = 8
XGB_INTERACTION_TREES = 40
XGB_INTERACTION_ETA = 0.05
CLIP_Q = 0.97
L1 = 0.01
L2 = 0.03
# Primary main-effect path: nonparametric XGBoost-seeded splines.
# Optional main-effect support: XGBoost joint bins (`xgb_bin`) and raw terms (`identity`).
TRANSFORMS = ("identity", "xgb_spline")
XGB_BIN_TREES = 100
XGB_BIN_ETA = 0.1
XGB_BIN_MAX_KNOTS = 4
# Budget of retained XGBoost-seeded knots per raw feature for continuous linear splines.
XGB_SPLINE_MAX_KNOTS = 6
ADASPLINE_LAMBDA = 1.0
ADASPLINE_STEPS = 15
ADASPLINE_EPS = 1e-4


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


def parse_stump(tree_json: str) -> tuple[str, float, float, float] | None:
    node = json.loads(tree_json)
    if "leaf" in node or len(node.get("children", [])) != 2:
        return None
    children = {child["nodeid"]: float(child["leaf"]) for child in node["children"] if "leaf" in child}
    if node["yes"] not in children or node["no"] not in children:
        return None
    return str(node["split"]), float(node["split_condition"]), children[node["yes"]], children[node["no"]]


def fit_xgb_depth1_stumps(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> dict[str, list[tuple[float, float, float]]]:
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
    booster = xgb.train(
        params={
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": XGB_BIN_ETA,
            "max_depth": 1,
            "lambda": 1.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "verbosity": 0,
        },
        dtrain=dtrain,
        num_boost_round=XGB_BIN_TREES,
    )

    stumps: dict[str, list[tuple[float, float, float]]] = {name: [] for name in feature_names}
    for tree_json in booster.get_dump(dump_format="json"):
        parsed = parse_stump(tree_json)
        if parsed is None:
            continue
        name, split, left_leaf, right_leaf = parsed
        stumps[name].append((split, left_leaf, right_leaf))
    return stumps


def fit_xgb_interaction_feature(
    x_train: np.ndarray,
    x_val: np.ndarray,
    residual: np.ndarray,
    feature_names: list[str],
    screened_pairs: list[tuple[int, int]],
) -> list[Candidate]:
    if not screened_pairs:
        return []

    constrained = sorted({idx for pair in screened_pairs for idx in pair})
    if len(constrained) < 2:
        return []

    local_names = [feature_names[idx] for idx in constrained]
    local_train = x_train[:, constrained]
    local_val = x_val[:, constrained]
    local_index = {raw_idx: local_idx for local_idx, raw_idx in enumerate(constrained)}
    constraints = [[local_index[left], local_index[right]] for left, right in screened_pairs]

    dtrain = xgb.DMatrix(local_train, label=residual, feature_names=local_names)
    dval = xgb.DMatrix(local_val, feature_names=local_names)
    booster = xgb.train(
        params={
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": XGB_INTERACTION_ETA,
            "max_depth": 2,
            "lambda": 1.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "interaction_constraints": json.dumps(constraints),
            "verbosity": 0,
        },
        dtrain=dtrain,
        num_boost_round=XGB_INTERACTION_TREES,
    )

    pair_regions: dict[tuple[str, str], dict[tuple[tuple[str, float, float], ...], float]] = {}

    def visit(node: dict, bounds: dict[str, tuple[float, float]]) -> None:
        if "leaf" in node:
            if len(bounds) == 2:
                pair = tuple(sorted(bounds))
                region = tuple((name, *bounds[name]) for name in pair)
                pair_regions.setdefault(pair, {})
                pair_regions[pair][region] = pair_regions[pair].get(region, 0.0) + abs(float(node["leaf"]))
            return
        split = str(node["split"])
        threshold = float(node["split_condition"])
        lo, hi = bounds.get(split, (-np.inf, np.inf))
        for child in node.get("children", []):
            child_bounds = dict(bounds)
            if child["nodeid"] == node["yes"]:
                child_bounds[split] = (lo, min(hi, threshold))
            else:
                child_bounds[split] = (max(lo, threshold), hi)
            visit(child, child_bounds)

    for tree_json in booster.get_dump(dump_format="json"):
        visit(json.loads(tree_json), {})

    candidates: list[Candidate] = []
    for left, right in screened_pairs:
        left_name = feature_names[left]
        right_name = feature_names[right]
        pair = tuple(sorted((left_name, right_name)))
        region_weights = pair_regions.get(pair, {})
        if not region_weights:
            continue

        left_train, left_val, left_suffix = maybe_clip(x_train[:, left], x_val[:, left])
        right_train, right_val, right_suffix = maybe_clip(x_train[:, right], x_val[:, right])
        ordered = sorted(region_weights.items(), key=lambda item: item[1], reverse=True)
        for region, weight in ordered[:INTERACTION_CAP]:
            region_map = {name: (lo, hi) for name, lo, hi in region}
            left_lo, left_hi = region_map[left_name]
            right_lo, right_hi = region_map[right_name]
            train_term = (
                (left_train >= left_lo)
                & (left_train < left_hi)
                & (right_train >= right_lo)
                & (right_train < right_hi)
            ).astype(np.float64)
            val_term = (
                (left_val >= left_lo)
                & (left_val < left_hi)
                & (right_val >= right_lo)
                & (right_val < right_hi)
            ).astype(np.float64)
            if train_term.sum() <= 1.0:
                continue
            candidates.append(
                Candidate(
                    name=(
                        f"{left_name}{left_suffix}__rect_[{left_lo:g},{left_hi:g})"
                        f"__x__{right_name}{right_suffix}__rect_[{right_lo:g},{right_hi:g})"
                    ),
                    train=train_term,
                    val=val_term,
                    score=safe_corr(train_term, residual) * weight,
                )
            )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:INTERACTION_CAP]


def xgb_step_values(values: np.ndarray, stumps: list[tuple[float, float, float]]) -> np.ndarray:
    step = np.zeros(len(values), dtype=np.float64)
    for split, left_leaf, right_leaf in stumps:
        step += np.where(values < split, left_leaf, right_leaf)
    return step


def fit_weighted_ridge(design: np.ndarray, y: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += penalty
    rhs = design.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def adaspline_knots(
    x: np.ndarray,
    step_target: np.ndarray,
    initial_knots: list[float],
    max_knots: int,
) -> list[float]:
    knots = sorted({float(k) for k in initial_knots if np.isfinite(k)})
    if not knots or max_knots <= 0:
        return []

    basis = np.column_stack([np.maximum(0.0, x - knot) for knot in knots])
    design = np.column_stack([np.ones(len(x)), x, basis])
    weights = np.ones(len(knots), dtype=np.float64)

    for _ in range(ADASPLINE_STEPS):
        penalty = np.concatenate(
            [np.zeros(2, dtype=np.float64), ADASPLINE_LAMBDA * weights]
        )
        beta = fit_weighted_ridge(design, step_target, penalty)
        gamma = beta[2:]
        weights = 1.0 / (gamma * gamma + ADASPLINE_EPS**2)

    order = np.argsort(np.abs(gamma))[::-1]
    chosen = [knots[idx] for idx in order if abs(gamma[idx]) > 1e-8]
    return sorted(chosen[:max_knots])


def piecewise_constant_from_knots(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_target: np.ndarray,
    knots: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    if not knots:
        mean = float(train_target.mean())
        return np.full(len(train_x), mean), np.full(len(val_x), mean)

    bounds = np.asarray(sorted(knots), dtype=np.float64)
    train_bins = np.searchsorted(bounds, train_x, side="right")
    val_bins = np.searchsorted(bounds, val_x, side="right")
    num_bins = len(bounds) + 1
    sums = np.bincount(train_bins, weights=train_target, minlength=num_bins).astype(np.float64)
    counts = np.bincount(train_bins, minlength=num_bins).astype(np.float64)
    global_mean = float(train_target.mean())
    means = np.where(counts > 0.0, sums / counts, global_mean)
    return means[train_bins], means[val_bins]


def xgb_joint_bin_candidates(
    x_train: np.ndarray,
    x_val: np.ndarray,
    stumps: dict[str, list[tuple[float, float, float]]],
    feature_names: list[str],
    screened: list[int],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for idx in screened:
        name = feature_names[idx]
        feature_stumps = stumps[name]
        if not feature_stumps:
            continue
        raw_train = x_train[:, idx]
        raw_val = x_val[:, idx]
        raw_train, raw_val, clip_suffix = maybe_clip(raw_train, raw_val)
        step_target = xgb_step_values(raw_train, feature_stumps)
        initial_knots = [split for split, _, _ in feature_stumps]
        knots = adaspline_knots(raw_train, step_target, initial_knots, XGB_BIN_MAX_KNOTS)
        feat_train, feat_val = piecewise_constant_from_knots(raw_train, raw_val, step_target, knots)
        candidates.append(
            Candidate(
                name=f"{name}{clip_suffix}__xgb_bin",
                train=feat_train,
                val=feat_val,
                score=0.0,
            )
        )
    return candidates


def xgb_joint_spline_candidates(
    x_train: np.ndarray,
    x_val: np.ndarray,
    stumps: dict[str, list[tuple[float, float, float]]],
    feature_names: list[str],
    screened: list[int],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for idx in screened:
        name = feature_names[idx]
        feature_stumps = stumps[name]
        if not feature_stumps:
            continue
        raw_train, raw_val, clip_suffix = maybe_clip(x_train[:, idx], x_val[:, idx])
        step_target = xgb_step_values(raw_train, feature_stumps)
        initial_knots = [split for split, _, _ in feature_stumps]
        for knot in adaspline_knots(raw_train, step_target, initial_knots, XGB_SPLINE_MAX_KNOTS):
            candidates.append(
                Candidate(
                    # Truncated linear basis yields a continuous piecewise-linear main effect.
                    name=f"{name}{clip_suffix}__xgb_spline_{knot:g}",
                    train=np.maximum(0.0, raw_train - knot),
                    val=np.maximum(0.0, raw_val - knot),
                    score=0.0,
                )
            )
    return candidates


def identity_candidates(
    x_train: np.ndarray,
    x_val: np.ndarray,
    feature_names: list[str],
    screened: list[int],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for idx in screened:
        raw_train, raw_val, clip_suffix = maybe_clip(x_train[:, idx], x_val[:, idx])
        candidates.append(
            Candidate(
                name=f"{feature_names[idx]}{clip_suffix}",
                train=raw_train,
                val=raw_val,
                score=0.0,
            )
        )
    return candidates


def screen_variables(x_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]) -> list[int]:
    scores = [safe_corr(x_train[:, idx], y_train) for idx in range(x_train.shape[1])]
    return sorted(range(len(feature_names)), key=lambda idx: scores[idx], reverse=True)


def standardize(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (val - mean) / std


def quantile_bin_edges(x: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 1:
        return np.empty(0, dtype=np.float64)
    probs = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    return np.unique(np.quantile(x, probs))


def fast_interaction_score(left: np.ndarray, right: np.ndarray, residual: np.ndarray) -> float:
    left_edges = quantile_bin_edges(left, FAST_BINS)
    right_edges = quantile_bin_edges(right, FAST_BINS)
    left_bins = np.searchsorted(left_edges, left, side="right")
    right_bins = np.searchsorted(right_edges, right, side="right")

    shape = (len(left_edges) + 1, len(right_edges) + 1)
    counts = np.zeros(shape, dtype=np.float64)
    sums = np.zeros(shape, dtype=np.float64)
    np.add.at(counts, (left_bins, right_bins), 1.0)
    np.add.at(sums, (left_bins, right_bins), residual)

    row_counts = counts.sum(axis=1)
    col_counts = counts.sum(axis=0)
    row_sums = sums.sum(axis=1)
    col_sums = sums.sum(axis=0)
    global_mean = float(residual.mean())

    with np.errstate(divide="ignore", invalid="ignore"):
        cell_mean = np.where(counts > 0.0, sums / counts, 0.0)
        row_mean = np.where(row_counts > 0.0, row_sums / row_counts, 0.0)
        col_mean = np.where(col_counts > 0.0, col_sums / col_counts, 0.0)

    interaction = cell_mean - row_mean[:, None] - col_mean[None, :] + global_mean
    return float(np.sqrt(np.sum(counts * interaction * interaction) / len(residual)))


def build_design(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ranked = screen_variables(x_train, y_train, feature_names)
    screened = ranked if SCREEN_K is None else ranked[:SCREEN_K]
    xgb_stumps = None

    singles: list[Candidate] = []
    unknown = set(TRANSFORMS) - {"identity", "xgb_bin", "xgb_spline"}
    if unknown:
        raise ValueError(f"Unknown transform(s): {sorted(unknown)}")

    if "identity" in TRANSFORMS:
        singles.extend(identity_candidates(x_train, x_val, feature_names, screened))
    if "xgb_bin" in TRANSFORMS:
        xgb_stumps = fit_xgb_depth1_stumps(x_train, y_train, feature_names)
        singles.extend(xgb_joint_bin_candidates(x_train, x_val, xgb_stumps, feature_names, screened))
    if "xgb_spline" in TRANSFORMS:
        xgb_stumps = fit_xgb_depth1_stumps(x_train, y_train, feature_names) if xgb_stumps is None else xgb_stumps
        singles.extend(xgb_joint_spline_candidates(x_train, x_val, xgb_stumps, feature_names, screened))

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

    main_train = np.column_stack([candidate.train for candidate in chosen])
    main_val = np.column_stack([candidate.val for candidate in chosen])
    main_train_std, main_val_std = standardize(main_train, main_val)

    if INTERACTION_CAP > 0:
        main_beta = fit_logistic_glm(main_train_std, y_train)
        residual = y_train - sigmoid(predict_scores(main_train_std, main_beta))
        pair_scores: list[tuple[float, int, int]] = []
        source = screened[: min(len(screened), max(2, INTERACTION_CAP + 1))]
        for left, right in itertools.combinations(source, 2):
            left_train, left_val, _ = maybe_clip(x_train[:, left], x_val[:, left])
            right_train, right_val, _ = maybe_clip(x_train[:, right], x_val[:, right])
            pair_scores.append((fast_interaction_score(left_train, right_train, residual), left, right))
        pair_scores.sort(reverse=True)
        top_pairs = [(left, right) for _, left, right in pair_scores[:INTERACTION_CAP]]
        interaction_features = fit_xgb_interaction_feature(
            x_train=x_train,
            x_val=x_val,
            residual=residual,
            feature_names=feature_names,
            screened_pairs=top_pairs,
        )
        chosen.extend(interaction_features)

    train_matrix = np.column_stack([candidate.train for candidate in chosen])
    val_matrix = np.column_stack([candidate.val for candidate in chosen])
    train_matrix, val_matrix = standardize(train_matrix, val_matrix)
    return train_matrix, val_matrix, [candidate.name for candidate in chosen]


def fit_logistic_glm(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    total_penalty = L1 + L2
    kwargs: dict[str, object] = {
        "fit_intercept": True,
        "max_iter": 2000,
        "tol": 1e-4,
        "random_state": 0,
    }
    if total_penalty <= 1e-12:
        kwargs["penalty"] = None
        kwargs["solver"] = "lbfgs"
    elif L1 <= 1e-12:
        kwargs["penalty"] = "l2"
        kwargs["solver"] = "lbfgs"
        kwargs["C"] = 1.0 / L2
    elif L2 <= 1e-12:
        kwargs["penalty"] = "l1"
        kwargs["solver"] = "saga"
        kwargs["C"] = 1.0 / L1
    else:
        kwargs["penalty"] = "elasticnet"
        kwargs["solver"] = "saga"
        kwargs["C"] = 1.0 / total_penalty
        kwargs["l1_ratio"] = L1 / total_penalty

    model = LogisticRegression(**kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*'penalty' was deprecated.*",
            category=FutureWarning,
        )
        model.fit(x_train, y_train)
    return np.concatenate([model.intercept_, model.coef_.ravel()])


def predict_scores(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return design @ beta


def describe_policy() -> str:
    clip = "none" if CLIP_Q is None else f"clip{int(CLIP_Q * 100)}"
    screen = "all" if SCREEN_K is None else str(SCREEN_K)
    transforms = "+".join(TRANSFORMS)
    xgb_bits = []
    if "xgb_bin" in TRANSFORMS or "xgb_spline" in TRANSFORMS:
        xgb_bits.append(f"xgb_trees={XGB_BIN_TREES}")
    if "xgb_bin" in TRANSFORMS:
        xgb_bits.append(f"xgb_bin_knots={XGB_BIN_MAX_KNOTS}")
    if "xgb_spline" in TRANSFORMS:
        xgb_bits.append(f"xgb_spline_knots={XGB_SPLINE_MAX_KNOTS}")
    xgb_suffix = "" if not xgb_bits else " " + " ".join(xgb_bits)
    return (
        f"screen_k={screen} feature_cap={FEATURE_CAP} "
        f"interaction_cap={INTERACTION_CAP} clip={clip} "
        f"l1={L1:.3f} l2={L2:.3f} transforms={transforms}{xgb_suffix}"
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
    val_auc = float(roc_auc_score(y_val, val_scores))

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
