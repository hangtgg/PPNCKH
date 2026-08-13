"""
statistical_tests.py

Statistical analysis for:
H1: Feature-set improvements over F0 using paired forecast errors,
    Diebold-Mariano tests, Holm correction, and moving-block bootstrap.
H2: Model × feature-set interaction using blocked ANOVA with seed as a block,
    effect sizes, incremental improvements, and bootstrap confidence intervals.
H3: Accuracy-efficiency analysis using Pareto frontiers and ideal-point distance.

Expected metrics file:
    model,feature_set,seed,rmse,mae,mape,r2,training_time,inference_time

Expected prediction files:
    One file per model × feature_set × seed.
    Recommended filename:
        RF_F0_seed42_predictions.csv
        LSTM_F4_seed168_predictions.csv

Prediction columns:
    y_true / actual / observed / target
    y_pred / predicted / prediction / forecast

Examples:
    python statistical_tests.py --experiment H1 \
        --metrics-file outputs/metrics/all_runs_metrics.csv \
        --prediction-dir outputs/predictions \
        --output-dir outputs/statistics/H1 \
        --baseline F0 --alpha 0.05 --holm-correction --block-bootstrap

    python statistical_tests.py --experiment H2 \
        --metrics-file outputs/metrics/all_runs_metrics.csv \
        --output-dir outputs/statistics/H2 \
        --baseline F0 --final-feature-set F4

    python statistical_tests.py --experiment H3 \
        --metrics-file outputs/metrics/all_runs_metrics.csv \
        --output-dir outputs/statistics/H3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multitest import multipletests
except ImportError as exc:
    raise ImportError(
        "statsmodels is required. Install it with: pip install statsmodels"
    ) from exc


LOGGER = logging.getLogger("statistical_tests")

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

PRED_COLUMN_CANDIDATES = [
    "y_pred",
    "predicted",
    "prediction",
    "forecast",
    "predicted_load",
    "load_predicted",
]

TIME_COLUMN_CANDIDATES = [
    "timestamp",
    "datetime",
    "date_time",
    "time",
    "date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run H1, H2, or H3 statistical analyses."
    )

    parser.add_argument(
        "--experiment",
        choices=["H1", "H2", "H3", "all"],
        required=True,
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="Required for H1 Diebold-Mariano and bootstrap analyses.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline",
        default="F0",
    )
    parser.add_argument(
        "--final-feature-set",
        default="F4",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=DEFAULT_FEATURE_SETS,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--holm-correction",
        action="store_true",
    )
    parser.add_argument(
        "--block-bootstrap",
        action="store_true",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=24,
        help="Moving-block bootstrap block length in observations.",
    )
    parser.add_argument(
        "--dm-lag",
        type=int,
        default=24,
        help="Newey-West maximum lag for the DM statistic.",
    )
    parser.add_argument(
        "--loss",
        choices=["squared", "absolute"],
        default="squared",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--tradeoff-cost",
        choices=["inference_time", "training_time"],
        default="inference_time",
    )
    parser.add_argument(
        "--accuracy-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
    )

    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_model(value: Any) -> str:
    text = str(value).strip().upper().replace("-", "_")
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
    match = re.search(r"F(\d+)", text)
    return f"F{match.group(1)}" if match else str(value).strip().upper()


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "feature_set", "seed", "rmse", "mae"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Metrics file is missing required columns: {sorted(missing)}"
        )

    result = df.copy()
    result["model"] = result["model"].map(normalize_model)
    result["feature_set"] = result["feature_set"].map(normalize_feature_set)
    result["seed"] = pd.to_numeric(result["seed"], errors="raise").astype(int)

    numeric_columns = [
        "rmse",
        "mae",
        "mape",
        "r2",
        "training_time",
        "inference_time",
        "peak_memory",
        "model_size",
        "feature_generation_time",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    duplicated = result.duplicated(
        subset=["model", "feature_set", "seed"],
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = result.loc[
            duplicated,
            ["model", "feature_set", "seed"],
        ]
        raise ValueError(
            "Duplicate model-feature-seed rows found:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    return result


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    return normalize_metrics(pd.read_csv(path))


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    mapping = {
        str(column).strip().lower(): str(column)
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]

    return None


def discover_prediction_files(prediction_dir: Path) -> list[Path]:
    if not prediction_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: {prediction_dir}"
        )

    return sorted(
        path
        for path in prediction_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".pq"}
    )


def read_prediction(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    return pd.read_parquet(path)


def infer_run_identity(
    path: Path,
    df: pd.DataFrame,
    expected_models: list[str],
    expected_feature_sets: list[str],
    expected_seeds: list[int],
) -> tuple[str, str, int]:
    model = None
    feature_set = None
    seed = None

    if "model" in df.columns and not df["model"].dropna().empty:
        model = normalize_model(df["model"].dropna().iloc[0])

    if "feature_set" in df.columns and not df["feature_set"].dropna().empty:
        feature_set = normalize_feature_set(
            df["feature_set"].dropna().iloc[0]
        )

    if "seed" in df.columns and not df["seed"].dropna().empty:
        seed = int(df["seed"].dropna().iloc[0])

    searchable = re.sub(
        r"[^A-Z0-9]+",
        "_",
        f"{path.stem}_{path.parent.name}".upper(),
    )

    if model is None:
        aliases = {
            "RF": ["RF", "RANDOM_FOREST", "RANDOMFOREST"],
            "XGB": ["XGB", "XGBOOST"],
            "LGBM": ["LGBM", "LIGHTGBM"],
            "LSTM": ["LSTM"],
        }

        for candidate in expected_models:
            normalized = normalize_model(candidate)
            if any(
                re.search(rf"(^|_){re.escape(alias)}(_|$)", searchable)
                for alias in aliases.get(normalized, [normalized])
            ):
                model = normalized
                break

    if feature_set is None:
        for candidate in expected_feature_sets:
            normalized = normalize_feature_set(candidate)
            if re.search(
                rf"(^|_){re.escape(normalized)}(_|$)",
                searchable,
            ):
                feature_set = normalized
                break

    if seed is None:
        match = re.search(r"(?:^|_)SEED_?(\d+)(?:_|$)", searchable)
        if match:
            seed = int(match.group(1))
        else:
            for token in re.findall(r"(?:^|_)(\d+)(?:_|$)", searchable):
                value = int(token)
                if value in expected_seeds:
                    seed = value
                    break

    if model is None or feature_set is None or seed is None:
        raise ValueError(
            f"Could not infer model, feature_set, and seed from {path}"
        )

    return model, feature_set, seed


def build_prediction_index(
    prediction_dir: Path,
    models: list[str],
    feature_sets: list[str],
    seeds: list[int],
) -> dict[tuple[str, str, int], Path]:
    index: dict[tuple[str, str, int], Path] = {}

    for path in discover_prediction_files(prediction_dir):
        try:
            df = read_prediction(path)
            identity = infer_run_identity(
                path,
                df,
                models,
                feature_sets,
                seeds,
            )

            if identity in index:
                raise ValueError(
                    f"Duplicate prediction files for {identity}:\n"
                    f"{index[identity]}\n{path}"
                )

            index[identity] = path

        except Exception as error:
            LOGGER.warning("Skipped %s: %s", path, error)

    return index


def load_aligned_predictions(
    baseline_path: Path,
    comparison_path: Path,
) -> pd.DataFrame:
    baseline = read_prediction(baseline_path)
    comparison = read_prediction(comparison_path)

    base_true = find_column(baseline, TRUE_COLUMN_CANDIDATES)
    base_pred = find_column(baseline, PRED_COLUMN_CANDIDATES)
    comp_true = find_column(comparison, TRUE_COLUMN_CANDIDATES)
    comp_pred = find_column(comparison, PRED_COLUMN_CANDIDATES)

    if None in {base_true, base_pred, comp_true, comp_pred}:
        raise ValueError(
            "Prediction files must contain actual and predicted columns."
        )

    base_time = find_column(baseline, TIME_COLUMN_CANDIDATES)
    comp_time = find_column(comparison, TIME_COLUMN_CANDIDATES)

    if base_time is not None and comp_time is not None:
        left = baseline[[base_time, base_true, base_pred]].copy()
        right = comparison[[comp_time, comp_true, comp_pred]].copy()

        left.columns = ["timestamp", "y_true_base", "y_pred_base"]
        right.columns = ["timestamp", "y_true_comp", "y_pred_comp"]

        left["timestamp"] = pd.to_datetime(left["timestamp"], errors="coerce")
        right["timestamp"] = pd.to_datetime(right["timestamp"], errors="coerce")

        merged = left.merge(
            right,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )
    else:
        if len(baseline) != len(comparison):
            raise ValueError(
                "Prediction files have different lengths and no common "
                "timestamp column is available."
            )

        merged = pd.DataFrame(
            {
                "y_true_base": baseline[base_true].to_numpy(),
                "y_pred_base": baseline[base_pred].to_numpy(),
                "y_true_comp": comparison[comp_true].to_numpy(),
                "y_pred_comp": comparison[comp_pred].to_numpy(),
            }
        )

    numeric_columns = [
        "y_true_base",
        "y_pred_base",
        "y_true_comp",
        "y_pred_comp",
    ]

    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=numeric_columns
    )

    if merged.empty:
        raise ValueError("No aligned valid predictions remain.")

    true_difference = np.abs(
        merged["y_true_base"].to_numpy()
        - merged["y_true_comp"].to_numpy()
    )

    if np.nanmax(true_difference) > 1e-6:
        LOGGER.warning(
            "Actual values differ between paired prediction files. "
            "The baseline actual series will be used."
        )

    return merged.reset_index(drop=True)


def loss_differential(
    aligned: pd.DataFrame,
    loss: str,
) -> np.ndarray:
    error_base = (
        aligned["y_true_base"].to_numpy()
        - aligned["y_pred_base"].to_numpy()
    )
    error_comp = (
        aligned["y_true_base"].to_numpy()
        - aligned["y_pred_comp"].to_numpy()
    )

    if loss == "squared":
        return error_base ** 2 - error_comp ** 2

    return np.abs(error_base) - np.abs(error_comp)


def newey_west_long_run_variance(
    values: np.ndarray,
    max_lag: int,
) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)

    if n < 3:
        return float("nan")

    centered = values - np.mean(values)
    max_lag = min(max(0, int(max_lag)), n - 1)

    gamma0 = np.dot(centered, centered) / n
    long_run_variance = gamma0

    for lag in range(1, max_lag + 1):
        covariance = np.dot(
            centered[lag:],
            centered[:-lag],
        ) / n

        bartlett_weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * covariance

    return float(max(long_run_variance, 0.0))


def diebold_mariano_test(
    differential: np.ndarray,
    max_lag: int,
) -> dict[str, float | int]:
    differential = np.asarray(differential, dtype=float)
    differential = differential[np.isfinite(differential)]
    n = len(differential)

    if n < 3:
        return {
            "dm_statistic": float("nan"),
            "p_value": float("nan"),
            "mean_loss_difference": float("nan"),
            "n_observations": n,
        }

    mean_difference = float(np.mean(differential))
    long_run_variance = newey_west_long_run_variance(
        differential,
        max_lag=max_lag,
    )

    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        statistic = 0.0 if math.isclose(mean_difference, 0.0) else np.sign(
            mean_difference
        ) * np.inf
    else:
        standard_error = math.sqrt(long_run_variance / n)
        statistic = mean_difference / standard_error

    p_value = float(2.0 * stats.norm.sf(abs(statistic)))

    return {
        "dm_statistic": float(statistic),
        "p_value": p_value,
        "mean_loss_difference": mean_difference,
        "n_observations": n,
    }


def moving_block_bootstrap_means(
    values: np.ndarray,
    repetitions: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)

    if n == 0:
        return np.array([], dtype=float)

    block_length = min(max(1, int(block_length)), n)
    number_of_blocks = int(math.ceil(n / block_length))
    max_start = n - block_length

    means = np.empty(repetitions, dtype=float)

    for repetition in range(repetitions):
        sampled_parts = []

        for _ in range(number_of_blocks):
            start = int(rng.integers(0, max_start + 1))
            sampled_parts.append(values[start:start + block_length])

        sample = np.concatenate(sampled_parts)[:n]
        means[repetition] = np.mean(sample)

    return means


def bootstrap_paired_difference(
    differentials_by_seed: list[np.ndarray],
    repetitions: int,
    block_length: int,
    alpha: float,
    random_state: int,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    seed_bootstrap_means = []

    for values in differentials_by_seed:
        seed_bootstrap_means.append(
            moving_block_bootstrap_means(
                values=values,
                repetitions=repetitions,
                block_length=block_length,
                rng=rng,
            )
        )

    if not seed_bootstrap_means:
        return {
            "mean_difference": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "bootstrap_p_value": float("nan"),
        }

    stacked = np.vstack(seed_bootstrap_means)
    combined = np.mean(stacked, axis=0)

    observed_seed_means = [
        float(np.mean(values))
        for values in differentials_by_seed
        if len(values) > 0
    ]
    observed = float(np.mean(observed_seed_means))

    lower = float(np.quantile(combined, alpha / 2.0))
    upper = float(np.quantile(combined, 1.0 - alpha / 2.0))

    probability_nonpositive = float(np.mean(combined <= 0.0))
    probability_nonnegative = float(np.mean(combined >= 0.0))
    p_value = min(
        1.0,
        2.0 * min(probability_nonpositive, probability_nonnegative),
    )

    return {
        "mean_difference": observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_p_value": p_value,
    }


def combine_p_values_stouffer(
    p_values: list[float],
    effects: list[float],
    weights: list[float] | None = None,
) -> tuple[float, float]:
    valid = [
        (p, effect, 1.0 if weights is None else weights[index])
        for index, (p, effect) in enumerate(zip(p_values, effects))
        if np.isfinite(p) and 0 < p <= 1 and np.isfinite(effect)
    ]

    if not valid:
        return float("nan"), float("nan")

    z_values = []
    used_weights = []

    for p_value, effect, weight in valid:
        sign = 1.0 if effect >= 0 else -1.0
        z_value = sign * stats.norm.isf(p_value / 2.0)
        z_values.append(z_value)
        used_weights.append(weight)

    z_values_array = np.asarray(z_values)
    weights_array = np.asarray(used_weights)

    combined_z = float(
        np.sum(weights_array * z_values_array)
        / math.sqrt(np.sum(weights_array ** 2))
    )
    combined_p = float(2.0 * stats.norm.sf(abs(combined_z)))

    return combined_z, combined_p


def mean_metric(
    metrics: pd.DataFrame,
    model: str,
    feature_set: str,
    metric: str,
) -> float:
    values = metrics.loc[
        (metrics["model"] == model)
        & (metrics["feature_set"] == feature_set),
        metric,
    ]

    return float(values.mean()) if not values.empty else float("nan")


def run_h1(args: argparse.Namespace, metrics: pd.DataFrame) -> None:
    if args.prediction_dir is None:
        raise ValueError("--prediction-dir is required for H1.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [normalize_model(value) for value in args.models]
    feature_sets = [
        normalize_feature_set(value)
        for value in args.feature_sets
    ]
    seeds = [int(value) for value in args.seeds]
    baseline = normalize_feature_set(args.baseline)

    comparison_sets = [
        feature_set
        for feature_set in feature_sets
        if feature_set != baseline
    ]

    prediction_index = build_prediction_index(
        args.prediction_dir,
        models,
        feature_sets,
        seeds,
    )

    metrics_rows = []
    dm_rows = []
    bootstrap_rows = []

    for model in models:
        baseline_rmse = mean_metric(
            metrics,
            model,
            baseline,
            "rmse",
        )
        baseline_mae = mean_metric(
            metrics,
            model,
            baseline,
            "mae",
        )

        for feature_set in comparison_sets:
            comparison_rmse = mean_metric(
                metrics,
                model,
                feature_set,
                "rmse",
            )
            comparison_mae = mean_metric(
                metrics,
                model,
                feature_set,
                "mae",
            )
            comparison_mape = (
                mean_metric(metrics, model, feature_set, "mape")
                if "mape" in metrics.columns
                else float("nan")
            )
            comparison_r2 = (
                mean_metric(metrics, model, feature_set, "r2")
                if "r2" in metrics.columns
                else float("nan")
            )

            delta_rmse = baseline_rmse - comparison_rmse
            delta_mae = baseline_mae - comparison_mae

            improvement_rmse = (
                100.0 * delta_rmse / baseline_rmse
                if np.isfinite(baseline_rmse)
                and not math.isclose(baseline_rmse, 0.0)
                else float("nan")
            )
            improvement_mae = (
                100.0 * delta_mae / baseline_mae
                if np.isfinite(baseline_mae)
                and not math.isclose(baseline_mae, 0.0)
                else float("nan")
            )

            seed_p_values = []
            seed_effects = []
            seed_weights = []
            differentials_by_seed = []

            for seed in seeds:
                baseline_key = (model, baseline, seed)
                comparison_key = (model, feature_set, seed)

                if (
                    baseline_key not in prediction_index
                    or comparison_key not in prediction_index
                ):
                    LOGGER.warning(
                        "Missing prediction pair for %s %s vs %s seed=%s",
                        model,
                        baseline,
                        feature_set,
                        seed,
                    )
                    continue

                aligned = load_aligned_predictions(
                    prediction_index[baseline_key],
                    prediction_index[comparison_key],
                )
                differential = loss_differential(
                    aligned,
                    args.loss,
                )
                dm_result = diebold_mariano_test(
                    differential,
                    max_lag=args.dm_lag,
                )

                seed_p_values.append(float(dm_result["p_value"]))
                seed_effects.append(
                    float(dm_result["mean_loss_difference"])
                )
                seed_weights.append(
                    math.sqrt(int(dm_result["n_observations"]))
                )
                differentials_by_seed.append(differential)

                dm_rows.append(
                    {
                        "model": model,
                        "baseline": baseline,
                        "feature_set": feature_set,
                        "seed": seed,
                        "loss": args.loss,
                        **dm_result,
                    }
                )

            combined_z, combined_p = combine_p_values_stouffer(
                seed_p_values,
                seed_effects,
                seed_weights,
            )

            if args.block_bootstrap and differentials_by_seed:
                bootstrap_result = bootstrap_paired_difference(
                    differentials_by_seed=differentials_by_seed,
                    repetitions=args.bootstrap_repetitions,
                    block_length=args.block_length,
                    alpha=args.alpha,
                    random_state=(
                        args.random_state
                        + hash((model, feature_set)) % 1_000_000
                    ),
                )
            else:
                bootstrap_result = {
                    "mean_difference": float("nan"),
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                    "bootstrap_p_value": float("nan"),
                }

            bootstrap_rows.append(
                {
                    "model": model,
                    "baseline": baseline,
                    "feature_set": feature_set,
                    "loss": args.loss,
                    "block_length": args.block_length,
                    "bootstrap_repetitions": (
                        args.bootstrap_repetitions
                        if args.block_bootstrap
                        else 0
                    ),
                    **bootstrap_result,
                }
            )

            metrics_rows.append(
                {
                    "model": model,
                    "baseline": baseline,
                    "feature_set": feature_set,
                    "rmse_baseline": baseline_rmse,
                    "rmse": comparison_rmse,
                    "mae_baseline": baseline_mae,
                    "mae": comparison_mae,
                    "mape": comparison_mape,
                    "r2": comparison_r2,
                    "delta_rmse": delta_rmse,
                    "rmse_improvement_percent": improvement_rmse,
                    "delta_mae": delta_mae,
                    "mae_improvement_percent": improvement_mae,
                    "combined_dm_z": combined_z,
                    "raw_p_value": combined_p,
                    "n_seeds_tested": len(seed_p_values),
                    "bootstrap_ci_lower": bootstrap_result["ci_lower"],
                    "bootstrap_ci_upper": bootstrap_result["ci_upper"],
                    "bootstrap_p_value": bootstrap_result[
                        "bootstrap_p_value"
                    ],
                }
            )

    h1_table = pd.DataFrame(metrics_rows)
    dm_table = pd.DataFrame(dm_rows)
    bootstrap_table = pd.DataFrame(bootstrap_rows)

    if not h1_table.empty:
        valid_mask = h1_table["raw_p_value"].notna()
        h1_table["adjusted_p_value"] = np.nan

        if valid_mask.any():
            if args.holm_correction:
                adjusted = multipletests(
                    h1_table.loc[valid_mask, "raw_p_value"].to_numpy(),
                    alpha=args.alpha,
                    method="holm",
                )[1]
            else:
                adjusted = h1_table.loc[
                    valid_mask,
                    "raw_p_value",
                ].to_numpy()

            h1_table.loc[
                valid_mask,
                "adjusted_p_value",
            ] = adjusted

        h1_table["better_than_baseline"] = (
            h1_table["delta_rmse"] > 0
        )
        h1_table["significant_after_correction"] = (
            h1_table["adjusted_p_value"] < args.alpha
        )
        h1_table["bootstrap_ci_excludes_zero"] = (
            (h1_table["bootstrap_ci_lower"] > 0)
            | (h1_table["bootstrap_ci_upper"] < 0)
        )

        if args.block_bootstrap:
            h1_table["significant_improvement"] = (
                h1_table["better_than_baseline"]
                & h1_table["significant_after_correction"]
                & (h1_table["bootstrap_ci_lower"] > 0)
            )
        else:
            h1_table["significant_improvement"] = (
                h1_table["better_than_baseline"]
                & h1_table["significant_after_correction"]
            )

    h1_table.to_csv(
        output_dir / "H1_metrics_table.csv",
        index=False,
        float_format="%.8f",
    )
    dm_table.to_csv(
        output_dir / "H1_dm_tests.csv",
        index=False,
        float_format="%.8f",
    )
    bootstrap_table.to_csv(
        output_dir / "H1_bootstrap_ci.csv",
        index=False,
        float_format="%.8f",
    )

    summary_lines = [
        "H1: Feature-set improvement over baseline",
        f"Baseline: {baseline}",
        f"Alpha: {args.alpha}",
        f"Loss for DM test: {args.loss}",
        f"Holm correction: {args.holm_correction}",
        f"Block bootstrap: {args.block_bootstrap}",
        "",
    ]

    if h1_table.empty:
        summary_lines.append("No valid H1 comparisons were produced.")
    else:
        for _, row in h1_table.sort_values(
            ["model", "feature_set"]
        ).iterrows():
            conclusion = (
                "statistically significant improvement"
                if bool(row["significant_improvement"])
                else "no statistically significant improvement"
            )

            summary_lines.append(
                f"{row['model']} {baseline}→{row['feature_set']}: "
                f"ΔRMSE={row['delta_rmse']:.6f}, "
                f"improvement={row['rmse_improvement_percent']:.3f}%, "
                f"Holm-adjusted p={row['adjusted_p_value']:.6g}; "
                f"{conclusion}."
            )

    (output_dir / "H1_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    LOGGER.info("H1 outputs saved to %s", output_dir)


def bootstrap_mean_ci(
    values: np.ndarray,
    repetitions: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan"), float("nan")

    samples = rng.choice(
        values,
        size=(repetitions, len(values)),
        replace=True,
    )
    means = samples.mean(axis=1)

    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def run_h2(args: argparse.Namespace, metrics: pd.DataFrame) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [normalize_model(value) for value in args.models]
    feature_sets = [
        normalize_feature_set(value)
        for value in args.feature_sets
    ]
    baseline = normalize_feature_set(args.baseline)
    final_feature_set = normalize_feature_set(args.final_feature_set)

    filtered = metrics.loc[
        metrics["model"].isin(models)
        & metrics["feature_set"].isin(feature_sets)
    ].copy()

    improvement_rows = []

    for model in models:
        f0 = mean_metric(filtered, model, baseline, "rmse")
        ff = mean_metric(
            filtered,
            model,
            final_feature_set,
            "rmse",
        )
        delta = f0 - ff
        percent = (
            100.0 * delta / f0
            if np.isfinite(f0) and not math.isclose(f0, 0.0)
            else float("nan")
        )

        improvement_rows.append(
            {
                "model": model,
                "baseline": baseline,
                "final_feature_set": final_feature_set,
                "rmse_baseline": f0,
                "rmse_final": ff,
                "delta_rmse": delta,
                "improvement_percent": percent,
            }
        )

    model_improvement = pd.DataFrame(improvement_rows)

    incremental_rows = []

    for model in models:
        for first, second in zip(feature_sets[:-1], feature_sets[1:]):
            first_rmse = mean_metric(
                filtered,
                model,
                first,
                "rmse",
            )
            second_rmse = mean_metric(
                filtered,
                model,
                second,
                "rmse",
            )
            delta = first_rmse - second_rmse
            percent = (
                100.0 * delta / first_rmse
                if np.isfinite(first_rmse)
                and not math.isclose(first_rmse, 0.0)
                else float("nan")
            )

            incremental_rows.append(
                {
                    "model": model,
                    "from_feature_set": first,
                    "to_feature_set": second,
                    "rmse_from": first_rmse,
                    "rmse_to": second_rmse,
                    "delta_rmse": delta,
                    "improvement_percent": percent,
                }
            )

    incremental = pd.DataFrame(incremental_rows)

    fitted = ols(
        "rmse ~ C(seed) + C(model) * C(feature_set)",
        data=filtered,
    ).fit()

    anova = sm.stats.anova_lm(fitted, typ=2).reset_index()
    anova = anova.rename(columns={"index": "effect"})

    residual_row = anova.loc[anova["effect"] == "Residual"]
    residual_ss = (
        float(residual_row["sum_sq"].iloc[0])
        if not residual_row.empty
        else float("nan")
    )

    effect_sizes = anova.loc[
        anova["effect"] != "Residual",
        ["effect", "sum_sq", "df", "F", "PR(>F)"],
    ].copy()

    effect_sizes["partial_eta_squared"] = (
        effect_sizes["sum_sq"]
        / (effect_sizes["sum_sq"] + residual_ss)
    )

    rng = np.random.default_rng(args.random_state)
    bootstrap_interaction_rows = []

    for model in models:
        pivot = filtered.loc[
            filtered["model"] == model,
            ["seed", "feature_set", "rmse"],
        ].pivot(
            index="seed",
            columns="feature_set",
            values="rmse",
        )

        if baseline not in pivot.columns:
            continue

        for feature_set in feature_sets:
            if feature_set == baseline or feature_set not in pivot.columns:
                continue

            paired = pivot[[baseline, feature_set]].dropna()
            differences = (
                paired[baseline] - paired[feature_set]
            ).to_numpy(dtype=float)

            lower, upper = bootstrap_mean_ci(
                differences,
                repetitions=args.bootstrap_repetitions,
                alpha=args.alpha,
                rng=rng,
            )

            bootstrap_interaction_rows.append(
                {
                    "model": model,
                    "baseline": baseline,
                    "feature_set": feature_set,
                    "mean_delta_rmse": float(
                        np.mean(differences)
                    ) if len(differences) else float("nan"),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_seeds": len(differences),
                }
            )

    bootstrap_interactions = pd.DataFrame(
        bootstrap_interaction_rows
    )

    model_improvement.to_csv(
        output_dir / "H2_model_improvement.csv",
        index=False,
        float_format="%.8f",
    )
    incremental.to_csv(
        output_dir / "H2_incremental_improvement.csv",
        index=False,
        float_format="%.8f",
    )
    anova.to_csv(
        output_dir / "H2_anova.csv",
        index=False,
        float_format="%.8f",
    )
    effect_sizes.to_csv(
        output_dir / "H2_effect_sizes.csv",
        index=False,
        float_format="%.8f",
    )
    bootstrap_interactions.to_csv(
        output_dir / "H2_bootstrap_interactions.csv",
        index=False,
        float_format="%.8f",
    )

    interaction_row = anova.loc[
        anova["effect"] == "C(model):C(feature_set)"
    ]

    if interaction_row.empty:
        interaction_p = float("nan")
    else:
        interaction_p = float(interaction_row["PR(>F)"].iloc[0])

    summary_lines = [
        "H2: Model × feature-set interaction",
        "Blocked ANOVA model:",
        "RMSE ~ C(seed) + C(model) * C(feature_set)",
        f"Alpha: {args.alpha}",
        "",
        f"Interaction p-value: {interaction_p:.8g}",
    ]

    if np.isfinite(interaction_p) and interaction_p < args.alpha:
        summary_lines.append(
            "Conclusion: the influence of feature set depends "
            "significantly on the forecasting model."
        )
    else:
        summary_lines.append(
            "Conclusion: there is insufficient evidence that the "
            "feature-set effect differs across models."
        )

    summary_lines.extend(
        [
            "",
            "Interpret cautiously because only three random seeds are "
            "available and all models share the same test period.",
            "",
            "F0→F4 improvements:",
        ]
    )

    for _, row in model_improvement.iterrows():
        summary_lines.append(
            f"{row['model']}: ΔRMSE={row['delta_rmse']:.6f}, "
            f"improvement={row['improvement_percent']:.3f}%."
        )

    (output_dir / "H2_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    LOGGER.info("H2 outputs saved to %s", output_dir)


def aggregate_accuracy_efficiency(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_seeds": ("seed", "nunique"),
        "rmse_mean": ("rmse", "mean"),
        "rmse_std": ("rmse", "std"),
        "mae_mean": ("mae", "mean"),
        "mae_std": ("mae", "std"),
    }

    optional_columns = [
        "mape",
        "r2",
        "training_time",
        "inference_time",
        "peak_memory",
        "model_size",
        "feature_generation_time",
    ]

    for column in optional_columns:
        if column in metrics.columns:
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_std"] = (column, "std")

    return (
        metrics.groupby(
            ["model", "feature_set"],
            as_index=False,
        )
        .agg(**aggregations)
        .reset_index(drop=True)
    )


def identify_pareto_frontier(
    df: pd.DataFrame,
    minimize_columns: list[str],
) -> pd.DataFrame:
    result = df.copy()
    valid = result[minimize_columns].notna().all(axis=1)
    efficient = np.zeros(len(result), dtype=bool)

    valid_indices = np.flatnonzero(valid.to_numpy())
    values = result.loc[valid, minimize_columns].to_numpy(dtype=float)

    for local_i, global_i in enumerate(valid_indices):
        current = values[local_i]
        dominated = False

        for local_j, _ in enumerate(valid_indices):
            if local_i == local_j:
                continue

            candidate = values[local_j]
            no_worse = np.all(candidate <= current)
            strictly_better = np.any(candidate < current)

            if no_worse and strictly_better:
                dominated = True
                break

        efficient[global_i] = not dominated

    result["pareto_efficient"] = efficient
    return result


def min_max_normalize(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    value_range = maximum - minimum

    if not np.isfinite(value_range) or math.isclose(value_range, 0.0):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
            dtype=float,
        )

    return (series - minimum) / value_range


def calculate_tradeoff_score(
    df: pd.DataFrame,
    accuracy_col: str,
    cost_col: str,
    accuracy_weight: float,
    cost_weight: float,
) -> pd.DataFrame:
    if accuracy_weight < 0 or cost_weight < 0:
        raise ValueError("Trade-off weights must be non-negative.")

    weight_sum = accuracy_weight + cost_weight
    if math.isclose(weight_sum, 0.0):
        raise ValueError("At least one trade-off weight must be positive.")

    accuracy_weight /= weight_sum
    cost_weight /= weight_sum

    result = df.copy()
    result["rmse_normalized"] = min_max_normalize(
        result[accuracy_col]
    )
    result["cost_normalized"] = min_max_normalize(
        result[cost_col]
    )

    result["tradeoff_distance"] = np.sqrt(
        accuracy_weight * result["rmse_normalized"] ** 2
        + cost_weight * result["cost_normalized"] ** 2
    )

    return result


def run_h3(args: argparse.Namespace, metrics: pd.DataFrame) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = aggregate_accuracy_efficiency(metrics)

    cost_column = f"{args.tradeoff_cost}_mean"

    if cost_column not in aggregate.columns:
        raise ValueError(
            f"{cost_column} is unavailable. Ensure train_models.py "
            f"stores {args.tradeoff_cost}."
        )

    required = ["rmse_mean", cost_column]
    missing_values = aggregate[required].isna().any(axis=1)

    if missing_values.any():
        LOGGER.warning(
            "%d configurations have missing accuracy or cost values.",
            int(missing_values.sum()),
        )

    two_objective = identify_pareto_frontier(
        aggregate,
        minimize_columns=["rmse_mean", cost_column],
    )

    available_multiobjective = [
        column
        for column in [
            "rmse_mean",
            "training_time_mean",
            "inference_time_mean",
            "peak_memory_mean",
            "model_size_mean",
        ]
        if column in aggregate.columns
        and aggregate[column].notna().all()
    ]

    if len(available_multiobjective) >= 2:
        multiobjective = identify_pareto_frontier(
            aggregate,
            minimize_columns=available_multiobjective,
        ).rename(
            columns={
                "pareto_efficient": "multiobjective_pareto_efficient"
            }
        )
        two_objective = two_objective.merge(
            multiobjective[
                [
                    "model",
                    "feature_set",
                    "multiobjective_pareto_efficient",
                ]
            ],
            on=["model", "feature_set"],
            how="left",
        )

    scored = calculate_tradeoff_score(
        two_objective,
        accuracy_col="rmse_mean",
        cost_col=cost_column,
        accuracy_weight=args.accuracy_weight,
        cost_weight=args.cost_weight,
    )

    pareto_only = scored.loc[
        scored["pareto_efficient"]
    ].copy()
    pareto_only = pareto_only.sort_values(
        "tradeoff_distance"
    ).reset_index(drop=True)

    best_rows = []

    best_accuracy = scored.loc[
        scored["rmse_mean"].idxmin()
    ]
    best_rows.append(
        {
            "category": "Best accuracy",
            "model": best_accuracy["model"],
            "feature_set": best_accuracy["feature_set"],
            "rmse_mean": best_accuracy["rmse_mean"],
            "cost_metric": cost_column,
            "cost_mean": best_accuracy[cost_column],
            "tradeoff_distance": best_accuracy[
                "tradeoff_distance"
            ],
        }
    )

    fastest = scored.loc[scored[cost_column].idxmin()]
    best_rows.append(
        {
            "category": f"Fastest {args.tradeoff_cost}",
            "model": fastest["model"],
            "feature_set": fastest["feature_set"],
            "rmse_mean": fastest["rmse_mean"],
            "cost_metric": cost_column,
            "cost_mean": fastest[cost_column],
            "tradeoff_distance": fastest["tradeoff_distance"],
        }
    )

    if not pareto_only.empty:
        best_tradeoff = pareto_only.iloc[0]
        best_rows.append(
            {
                "category": "Best accuracy-efficiency trade-off",
                "model": best_tradeoff["model"],
                "feature_set": best_tradeoff["feature_set"],
                "rmse_mean": best_tradeoff["rmse_mean"],
                "cost_metric": cost_column,
                "cost_mean": best_tradeoff[cost_column],
                "tradeoff_distance": best_tradeoff[
                    "tradeoff_distance"
                ],
            }
        )

    best_configurations = pd.DataFrame(best_rows)

    scored.to_csv(
        output_dir / "H3_accuracy_efficiency.csv",
        index=False,
        float_format="%.8f",
    )
    pareto_only.to_csv(
        output_dir / "H3_pareto_frontier.csv",
        index=False,
        float_format="%.8f",
    )
    best_configurations.to_csv(
        output_dir / "H3_best_configurations.csv",
        index=False,
        float_format="%.8f",
    )

    summary_lines = [
        "H3: Accuracy-efficiency analysis",
        f"Accuracy metric: rmse_mean",
        f"Cost metric: {cost_column}",
        f"Accuracy weight: {args.accuracy_weight}",
        f"Cost weight: {args.cost_weight}",
        "",
        "The ideal-point distance is a decision-support criterion, "
        "not a forecasting metric.",
        "",
    ]

    for _, row in best_configurations.iterrows():
        summary_lines.append(
            f"{row['category']}: {row['model']}-{row['feature_set']}; "
            f"RMSE={row['rmse_mean']:.6f}; "
            f"{row['cost_metric']}={row['cost_mean']:.6f}; "
            f"distance={row['tradeoff_distance']:.6f}."
        )

    summary_lines.append("")
    summary_lines.append(
        f"Number of two-objective Pareto-efficient configurations: "
        f"{int(scored['pareto_efficient'].sum())}."
    )

    (output_dir / "H3_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    LOGGER.info("H3 outputs saved to %s", output_dir)


def write_run_metadata(
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    metadata = {
        "experiment": args.experiment,
        "metrics_file": str(args.metrics_file),
        "prediction_dir": (
            str(args.prediction_dir)
            if args.prediction_dir is not None
            else None
        ),
        "baseline": args.baseline,
        "final_feature_set": args.final_feature_set,
        "alpha": args.alpha,
        "holm_correction": args.holm_correction,
        "block_bootstrap": args.block_bootstrap,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "block_length": args.block_length,
        "dm_lag": args.dm_lag,
        "loss": args.loss,
        "random_state": args.random_state,
        "tradeoff_cost": args.tradeoff_cost,
        "accuracy_weight": args.accuracy_weight,
        "cost_weight": args.cost_weight,
    }

    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    configure_logging()

    metrics = load_metrics(args.metrics_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.experiment == "H1":
            run_h1(args, metrics)
            write_run_metadata(args, args.output_dir)

        elif args.experiment == "H2":
            run_h2(args, metrics)
            write_run_metadata(args, args.output_dir)

        elif args.experiment == "H3":
            run_h3(args, metrics)
            write_run_metadata(args, args.output_dir)

        else:
            root = args.output_dir

            h1_args = argparse.Namespace(**vars(args))
            h1_args.output_dir = root / "H1"
            run_h1(h1_args, metrics)
            write_run_metadata(h1_args, h1_args.output_dir)

            h2_args = argparse.Namespace(**vars(args))
            h2_args.output_dir = root / "H2"
            run_h2(h2_args, metrics)
            write_run_metadata(h2_args, h2_args.output_dir)

            h3_args = argparse.Namespace(**vars(args))
            h3_args.output_dir = root / "H3"
            run_h3(h3_args, metrics)
            write_run_metadata(h3_args, h3_args.output_dir)

    except Exception:
        LOGGER.exception("Statistical analysis failed.")
        return 1

    LOGGER.info("Statistical analysis completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())