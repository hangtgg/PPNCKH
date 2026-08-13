"""
evaluation.py

Evaluate final test predictions for all model-feature-seed runs.

Expected experimental design:
    Models       : RF, XGB, LGBM, LSTM
    Feature sets : F0, F1, F2, F3, F4
    Seeds        : 42, 84, 168
    Total runs   : 4 x 5 x 3 = 60

Main outputs:
    outputs/metrics/all_runs_metrics.csv
    outputs/metrics/summary_metrics.csv
    outputs/metrics/ranking.csv
    outputs/metrics/completeness_report.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


LOGGER = logging.getLogger("evaluation")


DEFAULT_MODELS = ["RF", "XGB", "LGBM", "LSTM"]
DEFAULT_FEATURE_SETS = ["F0", "F1", "F2", "F3", "F4"]
DEFAULT_SEEDS = [42, 84, 168]


TRUE_COLUMN_CANDIDATES = [
    "y_true",
    "actual",
    "actual_load",
    "true",
    "target",
    "observed",
    "load_actual",
]

PREDICTION_COLUMN_CANDIDATES = [
    "y_pred",
    "predicted",
    "prediction",
    "forecast",
    "predicted_load",
    "load_predicted",
]

TIME_COLUMN_CANDIDATES = {
    "training_time": [
        "training_time",
        "training_seconds",
        "training_time_seconds",
        "train_time",
        "train_time_seconds",
    ],
    "inference_time": [
        "inference_time",
        "inference_seconds",
        "inference_time_seconds",
        "prediction_time",
        "prediction_time_seconds",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate test predictions and produce per-run and "
            "model-feature summary tables."
        )
    )

    parser.add_argument(
        "--prediction-dir",
        "--predictions-dir",
        dest="prediction_dir",
        type=Path,
        required=True,
        help="Directory containing prediction CSV or Parquet files.",
    )

    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing run metadata or metrics JSON files. "
            "If omitted, sibling outputs/metadata and outputs/metrics folders "
            "are searched automatically."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where evaluation tables will be saved.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Expected model names.",
    )

    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=DEFAULT_FEATURE_SETS,
        help="Expected feature-set names.",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Expected random seeds.",
    )

    parser.add_argument(
        "--mape-epsilon",
        type=float,
        default=1e-8,
        help="Targets with absolute value <= epsilon are excluded from MAPE.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if runs are missing, duplicated, or invalid.",
    )

    parser.add_argument(
        "--save-excel",
        action="store_true",
        help="Also create evaluation_results.xlsx.",
    )

    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_model_name(value: Any) -> str:
    text = str(value).strip().upper()

    aliases = {
        "RANDOMFOREST": "RF",
        "RANDOM_FOREST": "RF",
        "RANDOM FOREST": "RF",
        "XGBOOST": "XGB",
        "LIGHTGBM": "LGBM",
    }

    return aliases.get(text, text)


def normalize_feature_set(value: Any) -> str:
    text = str(value).strip().upper().replace("-", "").replace("_", "")

    match = re.search(r"F([0-4])", text)
    if match:
        return f"F{match.group(1)}"

    return str(value).strip().upper()


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    normalized_columns = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in normalized_columns:
            return normalized_columns[candidate.lower()]

    return None


def read_prediction_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported prediction format: {path}")


def extract_run_identity_from_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[str | None, str | None, int | None]:
    model: str | None = None
    feature_set: str | None = None
    seed: int | None = None

    if "model" in dataframe.columns and not dataframe["model"].dropna().empty:
        model = normalize_model_name(dataframe["model"].dropna().iloc[0])

    if (
        "feature_set" in dataframe.columns
        and not dataframe["feature_set"].dropna().empty
    ):
        feature_set = normalize_feature_set(
            dataframe["feature_set"].dropna().iloc[0]
        )

    if "seed" in dataframe.columns and not dataframe["seed"].dropna().empty:
        seed = int(dataframe["seed"].dropna().iloc[0])

    return model, feature_set, seed


def extract_run_identity_from_filename(
    path: Path,
    expected_models: list[str],
    expected_feature_sets: list[str],
    expected_seeds: list[int],
) -> tuple[str | None, str | None, int | None]:
    searchable_text = " ".join(
        [
            path.stem,
            path.parent.name,
            path.parent.parent.name if path.parent.parent else "",
        ]
    ).upper()

    normalized_text = re.sub(r"[^A-Z0-9]+", "_", searchable_text)

    model: str | None = None
    feature_set: str | None = None
    seed: int | None = None

    model_aliases = {
        "RF": ["RF", "RANDOM_FOREST", "RANDOMFOREST"],
        "XGB": ["XGB", "XGBOOST"],
        "LGBM": ["LGBM", "LIGHTGBM"],
        "LSTM": ["LSTM"],
    }

    for expected_model in expected_models:
        normalized_model = normalize_model_name(expected_model)

        aliases = model_aliases.get(
            normalized_model,
            [normalized_model],
        )

        if any(
            re.search(rf"(^|_){re.escape(alias)}(_|$)", normalized_text)
            for alias in aliases
        ):
            model = normalized_model
            break

    for expected_feature_set in expected_feature_sets:
        normalized_feature = normalize_feature_set(expected_feature_set)

        if re.search(
            rf"(^|_){re.escape(normalized_feature)}(_|$)",
            normalized_text,
        ):
            feature_set = normalized_feature
            break

    seed_patterns = [
        r"(?:^|_)SEED[_-]?(\d+)(?:_|$)",
        r"(?:^|_)S[_-]?(\d+)(?:_|$)",
    ]

    for pattern in seed_patterns:
        match = re.search(pattern, normalized_text)

        if match:
            candidate_seed = int(match.group(1))

            if candidate_seed in expected_seeds:
                seed = candidate_seed
                break

    if seed is None:
        number_tokens = re.findall(r"(?:^|_)(\d+)(?:_|$)", normalized_text)

        for token in number_tokens:
            candidate_seed = int(token)

            if candidate_seed in expected_seeds:
                seed = candidate_seed
                break

    return model, feature_set, seed


def determine_run_identity(
    dataframe: pd.DataFrame,
    path: Path,
    expected_models: list[str],
    expected_feature_sets: list[str],
    expected_seeds: list[int],
) -> tuple[str, str, int]:
    df_model, df_feature_set, df_seed = (
        extract_run_identity_from_dataframe(dataframe)
    )

    file_model, file_feature_set, file_seed = (
        extract_run_identity_from_filename(
            path=path,
            expected_models=expected_models,
            expected_feature_sets=expected_feature_sets,
            expected_seeds=expected_seeds,
        )
    )

    model = df_model or file_model
    feature_set = df_feature_set or file_feature_set
    seed = df_seed if df_seed is not None else file_seed

    missing_fields: list[str] = []

    if model is None:
        missing_fields.append("model")

    if feature_set is None:
        missing_fields.append("feature_set")

    if seed is None:
        missing_fields.append("seed")

    if missing_fields:
        raise ValueError(
            f"Could not identify {', '.join(missing_fields)} "
            f"from file: {path}"
        )

    return (
        normalize_model_name(model),
        normalize_feature_set(feature_set),
        int(seed),
    )


def extract_constant_numeric_value(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> float:
    column = find_column(dataframe, candidates)

    if column is None:
        return float("nan")

    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()

    if values.empty:
        return float("nan")

    return float(values.iloc[0])


def safe_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float,
) -> tuple[float, int]:
    valid_mask = np.abs(y_true) > epsilon
    valid_count = int(valid_mask.sum())

    if valid_count == 0:
        return float("nan"), 0

    percentage_errors = np.abs(
        (y_true[valid_mask] - y_pred[valid_mask])
        / y_true[valid_mask]
    )

    return float(np.mean(percentage_errors) * 100.0), valid_count


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mape_epsilon: float,
) -> dict[str, float | int]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape, mape_valid_count = safe_mape(
        y_true=y_true,
        y_pred=y_pred,
        epsilon=mape_epsilon,
    )
    r2 = float(r2_score(y_true, y_pred))

    residuals = y_true - y_pred

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "mean_error": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)),
        "n_test_samples": int(len(y_true)),
        "mape_valid_samples": mape_valid_count,
    }


def build_metadata_candidates(
    prediction_file: Path,
    metadata_dirs: list[Path],
) -> list[Path]:
    stem = prediction_file.stem

    replacements = [
        stem,
        stem.replace("_predictions", ""),
        stem.replace("predictions_", ""),
        stem.replace("_prediction", ""),
    ]

    candidates: list[Path] = []

    for metadata_dir in metadata_dirs:
        if not metadata_dir.exists():
            continue

        for replacement in replacements:
            candidates.extend(
                [
                    metadata_dir / f"{replacement}.json",
                    metadata_dir / f"{replacement}_metadata.json",
                    metadata_dir / f"{replacement}_metrics.json",
                ]
            )

    return candidates


def search_matching_json(
    model: str,
    feature_set: str,
    seed: int,
    metadata_dirs: list[Path],
) -> Path | None:
    model_tokens = {
        "RF": ["rf", "random_forest", "randomforest"],
        "XGB": ["xgb", "xgboost"],
        "LGBM": ["lgbm", "lightgbm"],
        "LSTM": ["lstm"],
    }.get(model, [model.lower()])

    for directory in metadata_dirs:
        if not directory.exists():
            continue

        for path in directory.rglob("*.json"):
            text = path.stem.lower()

            has_model = any(token in text for token in model_tokens)
            has_feature = feature_set.lower() in text
            has_seed = bool(
                re.search(
                    rf"(?:seed[_-]?|s[_-]?)?{seed}(?:\D|$)",
                    text,
                )
            )

            if has_model and has_feature and has_seed:
                return path

    return None


def recursively_find_numeric(
    data: Any,
    candidate_keys: list[str],
) -> float:
    normalized_candidates = {
        candidate.lower().replace("-", "_")
        for candidate in candidate_keys
    }

    if isinstance(data, dict):
        for key, value in data.items():
            normalized_key = str(key).lower().replace("-", "_")

            if normalized_key in normalized_candidates:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

        for value in data.values():
            result = recursively_find_numeric(value, candidate_keys)

            if np.isfinite(result):
                return result

    if isinstance(data, list):
        for item in data:
            result = recursively_find_numeric(item, candidate_keys)

            if np.isfinite(result):
                return result

    return float("nan")


def read_time_metadata(
    prediction_file: Path,
    model: str,
    feature_set: str,
    seed: int,
    metadata_dirs: list[Path],
) -> tuple[float, float, str | None]:
    candidates = build_metadata_candidates(
        prediction_file=prediction_file,
        metadata_dirs=metadata_dirs,
    )

    metadata_file = next(
        (path for path in candidates if path.exists()),
        None,
    )

    if metadata_file is None:
        metadata_file = search_matching_json(
            model=model,
            feature_set=feature_set,
            seed=seed,
            metadata_dirs=metadata_dirs,
        )

    if metadata_file is None:
        return float("nan"), float("nan"), None

    try:
        with metadata_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        training_time = recursively_find_numeric(
            data,
            TIME_COLUMN_CANDIDATES["training_time"],
        )

        inference_time = recursively_find_numeric(
            data,
            TIME_COLUMN_CANDIDATES["inference_time"],
        )

        return training_time, inference_time, str(metadata_file)

    except (OSError, json.JSONDecodeError, TypeError) as error:
        LOGGER.warning(
            "Could not read metadata %s: %s",
            metadata_file,
            error,
        )

        return float("nan"), float("nan"), str(metadata_file)


def evaluate_prediction_file(
    path: Path,
    expected_models: list[str],
    expected_feature_sets: list[str],
    expected_seeds: list[int],
    mape_epsilon: float,
    metadata_dirs: list[Path],
) -> dict[str, Any]:
    dataframe = read_prediction_file(path)

    if dataframe.empty:
        raise ValueError(f"Prediction file is empty: {path}")

    true_column = find_column(dataframe, TRUE_COLUMN_CANDIDATES)
    prediction_column = find_column(
        dataframe,
        PREDICTION_COLUMN_CANDIDATES,
    )

    if true_column is None:
        raise ValueError(
            f"No actual-value column found in {path}. "
            f"Expected one of: {TRUE_COLUMN_CANDIDATES}"
        )

    if prediction_column is None:
        raise ValueError(
            f"No prediction column found in {path}. "
            f"Expected one of: {PREDICTION_COLUMN_CANDIDATES}"
        )

    model, feature_set, seed = determine_run_identity(
        dataframe=dataframe,
        path=path,
        expected_models=expected_models,
        expected_feature_sets=expected_feature_sets,
        expected_seeds=expected_seeds,
    )

    valid_data = pd.DataFrame(
        {
            "y_true": pd.to_numeric(
                dataframe[true_column],
                errors="coerce",
            ),
            "y_pred": pd.to_numeric(
                dataframe[prediction_column],
                errors="coerce",
            ),
        }
    )

    original_rows = len(valid_data)
    valid_data = valid_data.replace([np.inf, -np.inf], np.nan)
    valid_data = valid_data.dropna()

    dropped_rows = original_rows - len(valid_data)

    if valid_data.empty:
        raise ValueError(
            f"No valid prediction pairs found in {path}"
        )

    y_true = valid_data["y_true"].to_numpy(dtype=np.float64)
    y_pred = valid_data["y_pred"].to_numpy(dtype=np.float64)

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        mape_epsilon=mape_epsilon,
    )

    training_time = extract_constant_numeric_value(
        dataframe,
        TIME_COLUMN_CANDIDATES["training_time"],
    )

    inference_time = extract_constant_numeric_value(
        dataframe,
        TIME_COLUMN_CANDIDATES["inference_time"],
    )

    metadata_file: str | None = None

    if not np.isfinite(training_time) or not np.isfinite(inference_time):
        (
            metadata_training_time,
            metadata_inference_time,
            metadata_file,
        ) = read_time_metadata(
            prediction_file=path,
            model=model,
            feature_set=feature_set,
            seed=seed,
            metadata_dirs=metadata_dirs,
        )

        if not np.isfinite(training_time):
            training_time = metadata_training_time

        if not np.isfinite(inference_time):
            inference_time = metadata_inference_time

    return {
        "model": model,
        "feature_set": feature_set,
        "seed": seed,
        **metrics,
        "training_time": training_time,
        "inference_time": inference_time,
        "dropped_rows": dropped_rows,
        "prediction_file": str(path),
        "metadata_file": metadata_file,
    }


def discover_prediction_files(prediction_dir: Path) -> list[Path]:
    supported_extensions = {".csv", ".parquet", ".pq"}

    files = [
        path
        for path in prediction_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    ]

    excluded_names = {
        "all_runs_metrics.csv",
        "summary_metrics.csv",
        "ranking.csv",
        "completeness_report.csv",
    }

    files = [
        path
        for path in files
        if path.name.lower() not in excluded_names
    ]

    return sorted(files)


def create_expected_runs(
    models: list[str],
    feature_sets: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    records = []

    for model in models:
        for feature_set in feature_sets:
            for seed in seeds:
                records.append(
                    {
                        "model": normalize_model_name(model),
                        "feature_set": normalize_feature_set(feature_set),
                        "seed": int(seed),
                    }
                )

    return pd.DataFrame(records)


def validate_completeness(
    all_runs: pd.DataFrame,
    models: list[str],
    feature_sets: list[str],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected = create_expected_runs(
        models=models,
        feature_sets=feature_sets,
        seeds=seeds,
    )

    run_counts = (
        all_runs.groupby(
            ["model", "feature_set", "seed"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "file_count"})
    )

    report = expected.merge(
        run_counts,
        on=["model", "feature_set", "seed"],
        how="left",
    )

    report["file_count"] = report["file_count"].fillna(0).astype(int)

    report["status"] = np.select(
        [
            report["file_count"].eq(0),
            report["file_count"].eq(1),
            report["file_count"].gt(1),
        ],
        [
            "missing",
            "complete",
            "duplicate",
        ],
        default="invalid",
    )

    missing = report.loc[report["status"] == "missing"].copy()
    duplicates = report.loc[report["status"] == "duplicate"].copy()

    return report, missing, duplicates


def create_summary_metrics(all_runs: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "rmse",
        "mae",
        "mape",
        "r2",
        "training_time",
        "inference_time",
    ]

    summary = (
        all_runs.groupby(
            ["model", "feature_set"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            training_time_mean=("training_time", "mean"),
            training_time_std=("training_time", "std"),
            inference_time_mean=("inference_time", "mean"),
            inference_time_std=("inference_time", "std"),
        )
    )

    numeric_columns = [
        column
        for column in summary.columns
        if column not in {"model", "feature_set", "n_seeds"}
    ]

    summary[numeric_columns] = summary[numeric_columns].round(6)

    return summary.sort_values(
        ["rmse_mean", "mae_mean"],
        ascending=[True, True],
    ).reset_index(drop=True)


def create_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    ranking = summary.copy()

    ranking["rmse_rank"] = ranking["rmse_mean"].rank(
        method="min",
        ascending=True,
    )

    ranking["mae_rank"] = ranking["mae_mean"].rank(
        method="min",
        ascending=True,
    )

    ranking["mape_rank"] = ranking["mape_mean"].rank(
        method="min",
        ascending=True,
    )

    ranking["r2_rank"] = ranking["r2_mean"].rank(
        method="min",
        ascending=False,
    )

    ranking["average_rank"] = ranking[
        [
            "rmse_rank",
            "mae_rank",
            "mape_rank",
            "r2_rank",
        ]
    ].mean(axis=1)

    ranking = ranking.sort_values(
        ["average_rank", "rmse_mean"],
        ascending=[True, True],
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "overall_rank",
        np.arange(1, len(ranking) + 1),
    )

    return ranking


def save_excel_workbook(
    output_path: Path,
    all_runs: pd.DataFrame,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    completeness: pd.DataFrame,
    invalid_files: pd.DataFrame,
) -> None:
    try:
        with pd.ExcelWriter(
            output_path,
            engine="openpyxl",
        ) as writer:
            all_runs.to_excel(
                writer,
                sheet_name="All Runs",
                index=False,
            )

            summary.to_excel(
                writer,
                sheet_name="Summary",
                index=False,
            )

            ranking.to_excel(
                writer,
                sheet_name="Ranking",
                index=False,
            )

            completeness.to_excel(
                writer,
                sheet_name="Completeness",
                index=False,
            )

            invalid_files.to_excel(
                writer,
                sheet_name="Invalid Files",
                index=False,
            )

    except ImportError:
        LOGGER.warning(
            "openpyxl is not installed. Excel output was skipped."
        )


def main() -> int:
    args = parse_args()
    configure_logging()

    prediction_dir = args.prediction_dir.resolve()
    output_dir = args.output_dir.resolve()

    expected_models = [
        normalize_model_name(model)
        for model in args.models
    ]

    expected_feature_sets = [
        normalize_feature_set(feature_set)
        for feature_set in args.feature_sets
    ]

    expected_seeds = [int(seed) for seed in args.seeds]

    if not prediction_dir.exists():
        LOGGER.error(
            "Prediction directory does not exist: %s",
            prediction_dir,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_dirs: list[Path] = []

    if args.metadata_dir is not None:
        metadata_dirs.append(args.metadata_dir.resolve())

    parent_output_dir = prediction_dir.parent

    metadata_dirs.extend(
        [
            parent_output_dir / "metadata",
            parent_output_dir / "run_metrics",
            parent_output_dir / "metrics",
        ]
    )

    metadata_dirs = list(dict.fromkeys(metadata_dirs))

    prediction_files = discover_prediction_files(prediction_dir)

    if not prediction_files:
        LOGGER.error(
            "No CSV or Parquet prediction files found in %s",
            prediction_dir,
        )
        return 1

    LOGGER.info(
        "Found %d prediction files.",
        len(prediction_files),
    )

    run_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, str]] = []

    for index, path in enumerate(prediction_files, start=1):
        LOGGER.info(
            "[%d/%d] Evaluating %s",
            index,
            len(prediction_files),
            path.name,
        )

        try:
            record = evaluate_prediction_file(
                path=path,
                expected_models=expected_models,
                expected_feature_sets=expected_feature_sets,
                expected_seeds=expected_seeds,
                mape_epsilon=args.mape_epsilon,
                metadata_dirs=metadata_dirs,
            )

            run_records.append(record)

        except Exception as error:
            LOGGER.error(
                "Could not evaluate %s: %s",
                path,
                error,
            )

            invalid_records.append(
                {
                    "prediction_file": str(path),
                    "error": str(error),
                }
            )

    if not run_records:
        LOGGER.error("No valid prediction files were evaluated.")
        return 1

    all_runs = pd.DataFrame(run_records)

    model_order = {
        model: index
        for index, model in enumerate(expected_models)
    }

    feature_order = {
        feature_set: index
        for index, feature_set in enumerate(expected_feature_sets)
    }

    all_runs["_model_order"] = all_runs["model"].map(model_order)
    all_runs["_feature_order"] = all_runs["feature_set"].map(
        feature_order
    )

    all_runs = (
        all_runs.sort_values(
            ["_model_order", "_feature_order", "seed"],
        )
        .drop(columns=["_model_order", "_feature_order"])
        .reset_index(drop=True)
    )

    preferred_columns = [
        "model",
        "feature_set",
        "seed",
        "rmse",
        "mae",
        "mape",
        "r2",
        "training_time",
        "inference_time",
        "mean_error",
        "residual_std",
        "n_test_samples",
        "mape_valid_samples",
        "dropped_rows",
        "prediction_file",
        "metadata_file",
    ]

    all_runs = all_runs[
        [
            column
            for column in preferred_columns
            if column in all_runs.columns
        ]
    ]

    completeness, missing, duplicates = validate_completeness(
        all_runs=all_runs,
        models=expected_models,
        feature_sets=expected_feature_sets,
        seeds=expected_seeds,
    )

    summary = create_summary_metrics(all_runs)
    ranking = create_ranking(summary)

    invalid_files = pd.DataFrame(
        invalid_records,
        columns=["prediction_file", "error"],
    )

    all_runs_path = output_dir / "all_runs_metrics.csv"
    summary_path = output_dir / "summary_metrics.csv"
    ranking_path = output_dir / "ranking.csv"
    completeness_path = output_dir / "completeness_report.csv"
    invalid_path = output_dir / "invalid_prediction_files.csv"

    all_runs.to_csv(
        all_runs_path,
        index=False,
        float_format="%.6f",
    )

    summary.to_csv(
        summary_path,
        index=False,
        float_format="%.6f",
    )

    ranking.to_csv(
        ranking_path,
        index=False,
        float_format="%.6f",
    )

    completeness.to_csv(
        completeness_path,
        index=False,
    )

    invalid_files.to_csv(
        invalid_path,
        index=False,
    )

    if args.save_excel:
        save_excel_workbook(
            output_path=output_dir / "evaluation_results.xlsx",
            all_runs=all_runs,
            summary=summary,
            ranking=ranking,
            completeness=completeness,
            invalid_files=invalid_files,
        )

    expected_count = (
        len(expected_models)
        * len(expected_feature_sets)
        * len(expected_seeds)
    )

    LOGGER.info("=" * 72)
    LOGGER.info("Evaluation completed.")
    LOGGER.info("Expected runs       : %d", expected_count)
    LOGGER.info("Valid files         : %d", len(all_runs))
    LOGGER.info("Missing runs        : %d", len(missing))
    LOGGER.info("Duplicated runs     : %d", len(duplicates))
    LOGGER.info("Invalid files       : %d", len(invalid_files))
    LOGGER.info("All-run table       : %s", all_runs_path)
    LOGGER.info("Summary table       : %s", summary_path)
    LOGGER.info("Ranking table       : %s", ranking_path)
    LOGGER.info("=" * 72)

    if not ranking.empty:
        best = ranking.iloc[0]

        LOGGER.info(
            "Best configuration: %s-%s | "
            "RMSE = %.6f ± %.6f | "
            "MAE = %.6f ± %.6f",
            best["model"],
            best["feature_set"],
            best["rmse_mean"],
            best["rmse_std"],
            best["mae_mean"],
            best["mae_std"],
        )

    validation_failed = (
        len(missing) > 0
        or len(duplicates) > 0
        or len(invalid_files) > 0
    )

    if validation_failed:
        LOGGER.warning(
            "The experiment is not fully complete. "
            "Review completeness_report.csv and "
            "invalid_prediction_files.csv."
        )

        if args.strict:
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())