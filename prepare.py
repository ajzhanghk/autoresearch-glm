import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

SEED = 1337
VAL_FRACTION = 0.2
TIME_BUDGET = 60.0
TAIWAN_CREDIT_ID = 350
TAIWAN_CREDIT_TARGET = "default.payment.next.month"
TAIWAN_CREDIT_ALIASES = (
    TAIWAN_CREDIT_TARGET,
    "default payment next month",
    "Y",
)


def default_cache_dir() -> Path:
    root = os.environ.get("AUTORESEARCH_GLM_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "autoresearch-glm"


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

    for alias in TAIWAN_CREDIT_ALIASES:
        if alias in frame.columns:
            frame = frame.rename(columns={alias: TAIWAN_CREDIT_TARGET})
            break

    return frame


def numeric_feature_frame(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} not found.")

    y_series = pd.to_numeric(frame[target], errors="raise")
    target_values = {float(value) for value in y_series.dropna().unique()}
    if target_values != {0.0, 1.0}:
        raise ValueError("TaiwanCredit target must contain both binary values 0 and 1.")

    y = y_series.astype(np.int64).to_numpy()
    x_frame = frame.drop(columns=[target]).select_dtypes(include=["number"]).copy()
    if x_frame.empty:
        raise ValueError("No numeric predictor columns found.")

    x_frame = x_frame.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(x_frame).astype(np.float64)
    return x, y, list(x_frame.columns)


def prepare_dataset(
    cache_dir: Path | None = None,
    seed: int = SEED,
) -> dict:
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    frame = load_taiwan_credit_from_uci()
    target = TAIWAN_CREDIT_TARGET
    x, y, feature_names = numeric_feature_frame(frame, target=target)
    if np.min(np.bincount(y)) < 2:
        raise ValueError("Each class needs at least two rows for a train/validation split.")
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=VAL_FRACTION,
        random_state=seed,
        stratify=y,
    )

    np.savez_compressed(
        cache_dir / "dataset.npz",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
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
        "source": "taiwancredit",
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    meta = prepare_dataset(
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
