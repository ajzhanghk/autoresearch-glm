from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 1337
VAL_FRACTION = 0.2
TIME_BUDGET = 60.0
DEFAULT_DATASET = "taiwancredit"
TAIWAN_CREDIT_ID = 350
TAIWAN_CREDIT_TARGET = "default.payment.next.month"


def default_cache_dir() -> Path:
    root = os.environ.get("AUTORESEARCH_GLM_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "autoresearch-glm"


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires both positive and negative examples.")

    order = np.argsort(y_score, kind="mergesort")
    scores = y_score[order]
    labels = y_true[order]

    concordant = 0.0
    negatives_seen = 0
    i = 0
    while i < len(scores):
        j = i
        pos = 0
        neg = 0
        while j < len(scores) and scores[j] == scores[i]:
            if labels[j] == 1:
                pos += 1
            else:
                neg += 1
            j += 1
        concordant += pos * (negatives_seen + 0.5 * neg)
        negatives_seen += neg
        i = j

    return concordant / (n_pos * n_neg)


def make_synthetic_credit_data(n_rows: int = 8000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    income_k = rng.lognormal(mean=4.0, sigma=0.35, size=n_rows)
    utilization = rng.beta(2.0, 4.0, size=n_rows)
    debt_to_income = rng.beta(2.5, 5.5, size=n_rows) * 1.4
    delinq_12m = rng.poisson(0.5, size=n_rows)
    inquiries_6m = rng.poisson(1.2, size=n_rows)
    credit_history_years = rng.gamma(shape=4.0, scale=2.0, size=n_rows)
    avg_txn_k = rng.lognormal(mean=2.2, sigma=0.45, size=n_rows)
    balance_k = rng.lognormal(mean=3.0, sigma=0.55, size=n_rows)
    savings_k = rng.lognormal(mean=2.5, sigma=0.7, size=n_rows)
    cashflow_vol = rng.lognormal(mean=1.4, sigma=0.6, size=n_rows)
    bureau_score = rng.normal(loc=690.0, scale=55.0, size=n_rows).clip(450.0, 850.0)
    recent_growth = rng.normal(loc=0.0, scale=1.0, size=n_rows)

    logit = (
        -3.2
        + 3.4 * utilization
        + 1.8 * debt_to_income
        + 0.55 * delinq_12m
        + 0.18 * inquiries_6m
        - 0.12 * np.log1p(income_k)
        - 0.14 * np.log1p(savings_k)
        + 0.08 * np.log1p(balance_k)
        + 0.22 * np.sqrt(cashflow_vol)
        - 0.03 * credit_history_years
        - 0.006 * (bureau_score - 650.0)
        + 1.1 * utilization * debt_to_income
        + 0.15 * delinq_12m * utilization
        + 0.22 * recent_growth**2
    )
    logit += rng.normal(scale=0.45, size=n_rows)
    target = rng.binomial(1, sigmoid(logit))

    return pd.DataFrame(
        {
            "income_k": income_k,
            "utilization": utilization,
            "debt_to_income": debt_to_income,
            "delinq_12m": delinq_12m.astype(float),
            "inquiries_6m": inquiries_6m.astype(float),
            "credit_history_years": credit_history_years,
            "avg_txn_k": avg_txn_k,
            "balance_k": balance_k,
            "savings_k": savings_k,
            "cashflow_vol": cashflow_vol,
            "bureau_score": bureau_score,
            "recent_growth": recent_growth,
            "target": target.astype(np.int64),
        }
    )


def load_taiwan_credit_from_uci() -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "TaiwanCredit download requires `ucimlrepo`. Install project dependencies first."
        ) from exc

    dataset = fetch_ucirepo(id=TAIWAN_CREDIT_ID)
    frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    return normalize_taiwan_credit_frame(frame)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def normalize_taiwan_credit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(how="all").dropna(axis=1, how="all").copy()
    frame.columns = [str(col).strip() for col in frame.columns]

    if TAIWAN_CREDIT_TARGET not in frame.columns and not frame.empty:
        header_row = [str(value).strip() for value in frame.iloc[0].tolist()]
        if TAIWAN_CREDIT_TARGET in header_row:
            frame.columns = header_row
            frame = frame.iloc[1:].reset_index(drop=True)
            frame.columns = [str(col).strip() for col in frame.columns]

    if "ID" in frame.columns:
        frame = frame.drop(columns=["ID"])

    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() >= max(1, int(0.9 * len(frame))):
            frame[column] = converted

    return frame


def read_frame(input_path: str | None, dataset: str, n_rows: int, seed: int) -> pd.DataFrame:
    if input_path is None:
        if dataset == "synthetic_credit":
            return make_synthetic_credit_data(n_rows=n_rows, seed=seed)
        if dataset == "taiwancredit":
            return load_taiwan_credit_from_uci()
        raise ValueError(f"Unknown dataset: {dataset}")

    path = Path(input_path)
    frame = read_table(path)
    if dataset == "taiwancredit":
        return normalize_taiwan_credit_frame(frame)
    return frame


def infer_target(frame: pd.DataFrame, target: str | None, dataset: str) -> str:
    if target is not None:
        return target

    if dataset == "taiwancredit":
        aliases = [
            TAIWAN_CREDIT_TARGET,
            "default payment next month",
            "Y",
            "target",
        ]
        for alias in aliases:
            if alias in frame.columns:
                return alias

    if "target" in frame.columns:
        return "target"
    return str(frame.columns[-1])


def coerce_binary_target(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(np.int64).to_numpy()

    unique = pd.Series(series.dropna().unique()).tolist()
    if len(unique) != 2:
        raise ValueError("Expected a binary target with exactly two distinct values.")

    if pd.api.types.is_numeric_dtype(series):
        values = set(float(v) for v in unique)
        if values <= {0.0, 1.0}:
            return series.astype(np.int64).to_numpy()

    mapping = {value: idx for idx, value in enumerate(sorted(unique))}
    return series.map(mapping).astype(np.int64).to_numpy()


def numeric_feature_frame(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} not found.")

    y = coerce_binary_target(frame[target])
    x_frame = frame.drop(columns=[target]).select_dtypes(include=["number"]).copy()
    if x_frame.empty:
        raise ValueError("No numeric predictor columns found.")

    x_frame = x_frame.replace([np.inf, -np.inf], np.nan)
    x_frame = x_frame.fillna(x_frame.median(numeric_only=True))
    return x_frame.to_numpy(dtype=np.float64), y, list(x_frame.columns)


def stratified_split(y: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []

    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        if len(idx) < 2:
            raise ValueError("Each class needs at least two rows for a train/validation split.")
        idx = rng.permutation(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        n_val = min(n_val, len(idx) - 1)
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])

    train = np.concatenate(train_idx)
    val = np.concatenate(val_idx)
    return rng.permutation(train), rng.permutation(val)


def prepare_dataset(
    input_path: str | None = None,
    target: str | None = None,
    cache_dir: Path | None = None,
    n_rows: int = 8000,
    seed: int = SEED,
    dataset: str = DEFAULT_DATASET,
) -> dict:
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    frame = read_frame(input_path=input_path, dataset=dataset, n_rows=n_rows, seed=seed)
    target = infer_target(frame=frame, target=target, dataset=dataset)
    x, y, feature_names = numeric_feature_frame(frame, target=target)
    train_idx, val_idx = stratified_split(y, val_fraction=VAL_FRACTION, seed=seed)

    np.savez_compressed(
        cache_dir / "dataset.npz",
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_val=x[val_idx],
        y_val=y[val_idx],
        feature_names=np.asarray(feature_names, dtype=object),
    )

    meta = {
        "task": "binary_classification",
        "model": "logistic_glm",
        "metric": "val_auc",
        "rows": int(len(frame)),
        "features": int(len(feature_names)),
        "target": target,
        "cache_dir": str(cache_dir),
        "source": str(input_path) if input_path else dataset,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_dataset(cache_dir: Path | None = None) -> dict:
    cache_dir = cache_dir or default_cache_dir()
    dataset_path = cache_dir / "dataset.npz"
    if not dataset_path.exists():
        prepare_dataset(cache_dir=cache_dir)

    dataset = np.load(dataset_path, allow_pickle=True)
    return {
        "x_train": dataset["x_train"].astype(np.float64),
        "y_train": dataset["y_train"].astype(np.int64),
        "x_val": dataset["x_val"].astype(np.float64),
        "y_val": dataset["y_val"].astype(np.int64),
        "feature_names": dataset["feature_names"].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the autoresearch-glm dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=["taiwancredit", "synthetic_credit"],
        help="Fixed benchmark dataset. TaiwanCredit is the canonical v1 benchmark.",
    )
    parser.add_argument("--input", type=str, default=None, help="Optional CSV or Parquet dataset.")
    parser.add_argument("--target", type=str, default=None, help="Binary target column name.")
    parser.add_argument("--rows", type=int, default=8000, help="Rows for the synthetic benchmark.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    meta = prepare_dataset(
        input_path=args.input,
        target=args.target,
        cache_dir=args.cache_dir,
        n_rows=args.rows,
        seed=args.seed,
        dataset=args.dataset,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
