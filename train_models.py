"""
train_models.py

Huấn luyện các mô hình cuối cùng sau khi hoàn thành Optuna tuning.

Thiết kế:
- Đọc best hyperparameters từ outputs/tuning.
- Gộp train và validation.
- Huấn luyện mô hình cuối trên train + validation.
- Dự báo trên test.
- Không sử dụng test để chọn siêu tham số.
- Chạy nhiều seed để đánh giá độ ổn định.
- Lưu model, prediction, metrics và metadata.

Ví dụ:
python train_models.py --split-dir outputs/splits --params-dir outputs/tuning_final --output-dir outputs --models rf xgb lgbm lstm --feature-sets F0 F1 F2 F3 F4 --seeds 42 84 168
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Constants
# ============================================================

TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target"

SUPPORTED_MODELS = ("rf", "xgb", "lgbm", "lstm")
SUPPORTED_FEATURE_SETS = ("F0", "F1", "F2", "F3", "F4")

SEQUENCE_LENGTH = 24

DEFAULT_MODELS = ["rf", "xgb", "lgbm", "lstm"]
DEFAULT_FEATURE_SETS = ["F0", "F1", "F2", "F3", "F4"]
DEFAULT_SEEDS = [42, 84, 168]

DEFAULT_LSTM_MAX_EPOCHS = 100
DEFAULT_LSTM_PATIENCE = 10
DEFAULT_LSTM_MIN_DELTA = 1e-5


# ============================================================
# 2. Data containers
# ============================================================

@dataclass
class FinalDataset:
    """Dữ liệu train+validation và test của một feature set."""

    feature_set: str
    feature_names: list[str]

    x_train_final: np.ndarray
    y_train_final: np.ndarray

    x_test: np.ndarray
    y_test: np.ndarray

    test_timestamps: pd.Series

    train_rows: int
    validation_rows: int
    final_train_rows: int
    test_rows: int


@dataclass
class LSTMFinalDataset:
    """Dữ liệu hai nhánh cho mô hình LSTM."""

    feature_set: str

    sequence_feature_names: list[str]
    static_feature_names: list[str]

    x_train_sequence: np.ndarray
    x_train_static: np.ndarray

    x_test_sequence: np.ndarray
    x_test_static: np.ndarray

    y_train_scaled: np.ndarray
    y_test_original: np.ndarray

    test_timestamps: pd.Series

    sequence_scaler: StandardScaler
    static_scaler: StandardScaler | None
    target_scaler: StandardScaler

    train_rows: int
    validation_rows: int
    final_train_rows: int
    test_rows: int

    sequence_length: int
    static_feature_count: int


# ============================================================
# 3. CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Huấn luyện các mô hình cuối cùng bằng best "
            "hyperparameters và đánh giá trên test."
        )
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("outputs/splits"),
    )

    parser.add_argument(
        "--params-dir",
        type=Path,
        default=Path("outputs/tuning_final"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=SUPPORTED_MODELS,
    )

    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=DEFAULT_FEATURE_SETS,
        choices=SUPPORTED_FEATURE_SETS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--n-jobs-model",
        type=int,
        default=-1,
        help="Số CPU thread cho mô hình cây.",
    )

    parser.add_argument(
        "--lstm-max-epochs",
        type=int,
        default=DEFAULT_LSTM_MAX_EPOCHS,
    )

    parser.add_argument(
        "--lstm-patience",
        type=int,
        default=DEFAULT_LSTM_PATIENCE,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Huấn luyện lại dù output đã tồn tại.",
    )

    return parser.parse_args()


# ============================================================
# 4. General utilities
# ============================================================

def set_global_seed(seed: int) -> None:
    """Đặt seed cho Python và NumPy."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)


def require_finite(
    array: np.ndarray,
    name: str,
) -> None:
    """Xác nhận array không chứa NaN hoặc infinity."""

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} chứa NaN hoặc giá trị vô hạn."
        )


def make_output_directories(
    output_dir: Path,
) -> dict[str, Path]:
    directories = {
        "models": output_dir / "models",
        "predictions": output_dir / "predictions",
        "metrics": output_dir / "metrics",
        "metadata": output_dir / "metadata",
        "summaries": output_dir / "summaries",
    }

    for directory in directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return directories


def write_json(
    path: Path,
    content: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            content,
            file,
            ensure_ascii=False,
            indent=4,
            default=convert_json_value,
        )


def convert_json_value(value: Any) -> Any:
    """Chuyển các kiểu NumPy sang kiểu JSON."""

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Không thể serialize kiểu {type(value)}."
    )


# ============================================================
# 5. Load and validate split data
# ============================================================

def load_split_file(
    split_dir: Path,
    feature_set: str,
    split_name: str,
) -> pd.DataFrame:
    path = split_dir / f"{feature_set}_{split_name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy split file: {path}"
        )

    frame = pd.read_csv(path)

    if TIMESTAMP_COLUMN not in frame.columns:
        raise KeyError(
            f"{path} thiếu cột '{TIMESTAMP_COLUMN}'."
        )

    if TARGET_COLUMN not in frame.columns:
        raise KeyError(
            f"{path} thiếu cột '{TARGET_COLUMN}'."
        )

    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN],
        errors="raise",
    )

    if frame[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError(
            f"{path} chứa timestamp trùng lặp."
        )

    if not frame[TIMESTAMP_COLUMN].is_monotonic_increasing:
        raise ValueError(
            f"{path} chưa được sắp xếp theo thời gian."
        )

    return frame


def verify_split_order(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_set: str,
) -> None:
    train_end = train_frame[TIMESTAMP_COLUMN].iloc[-1]
    validation_start = validation_frame[
        TIMESTAMP_COLUMN
    ].iloc[0]
    validation_end = validation_frame[
        TIMESTAMP_COLUMN
    ].iloc[-1]
    test_start = test_frame[
        TIMESTAMP_COLUMN
    ].iloc[0]

    if train_end >= validation_start:
        raise ValueError(
            f"{feature_set}: train và validation "
            "không theo thứ tự thời gian."
        )

    if validation_end >= test_start:
        raise ValueError(
            f"{feature_set}: validation và test "
            "không theo thứ tự thời gian."
        )


def load_final_dataset(
    split_dir: Path,
    feature_set: str,
) -> FinalDataset:
    train_frame = load_split_file(
        split_dir,
        feature_set,
        "train",
    )

    validation_frame = load_split_file(
        split_dir,
        feature_set,
        "validation",
    )

    test_frame = load_split_file(
        split_dir,
        feature_set,
        "test",
    )

    verify_split_order(
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        feature_set=feature_set,
    )

    excluded_columns = {
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
    }

    feature_names = [
        column
        for column in train_frame.columns
        if column not in excluded_columns
    ]

    if not feature_names:
        raise ValueError(
            f"{feature_set} không có feature nào."
        )

    expected_columns = (
        [TIMESTAMP_COLUMN]
        + feature_names
        + [TARGET_COLUMN]
    )

    for split_name, frame in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ):
        missing_columns = (
            set(expected_columns)
            - set(frame.columns)
        )

        if missing_columns:
            raise KeyError(
                f"{feature_set}-{split_name} thiếu cột: "
                f"{sorted(missing_columns)}"
            )

        if frame[feature_names].isna().any().any():
            raise ValueError(
                f"{feature_set}-{split_name} "
                "có NaN trong feature."
            )

        if frame[TARGET_COLUMN].isna().any():
            raise ValueError(
                f"{feature_set}-{split_name} "
                "có NaN trong target."
            )

    final_train_frame = pd.concat(
        [train_frame, validation_frame],
        axis=0,
        ignore_index=True,
    )

    x_train_final = final_train_frame[
        feature_names
    ].to_numpy(dtype=np.float32)

    y_train_final = final_train_frame[
        TARGET_COLUMN
    ].to_numpy(dtype=np.float32)

    x_test = test_frame[
        feature_names
    ].to_numpy(dtype=np.float32)

    y_test = test_frame[
        TARGET_COLUMN
    ].to_numpy(dtype=np.float32)

    require_finite(
        x_train_final,
        f"{feature_set} x_train_final",
    )
    require_finite(
        y_train_final,
        f"{feature_set} y_train_final",
    )
    require_finite(
        x_test,
        f"{feature_set} x_test",
    )
    require_finite(
        y_test,
        f"{feature_set} y_test",
    )

    return FinalDataset(
        feature_set=feature_set,
        feature_names=feature_names,
        x_train_final=x_train_final,
        y_train_final=y_train_final,
        x_test=x_test,
        y_test=y_test,
        test_timestamps=test_frame[
            TIMESTAMP_COLUMN
        ].copy(),
        train_rows=len(train_frame),
        validation_rows=len(validation_frame),
        final_train_rows=len(final_train_frame),
        test_rows=len(test_frame),
    )


# ============================================================
# 6. Load best hyperparameters
# ============================================================

def find_best_params_file(
    params_dir: Path,
    model_name: str,
    feature_set: str,
) -> Path:
    candidate_names = [
        f"{model_name}_{feature_set}_best_params.json",
        f"{model_name.lower()}_{feature_set}_best_params.json",
        f"{model_name.upper()}_{feature_set}_best_params.json",
    ]

    for candidate_name in candidate_names:
        candidate_path = params_dir / candidate_name

        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        "Không tìm thấy best parameters cho "
        f"{model_name}-{feature_set} trong {params_dir}. "
        f"Đã tìm: {candidate_names}"
    )


def load_best_params(
    params_dir: Path,
    model_name: str,
    feature_set: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    params_path = find_best_params_file(
        params_dir=params_dir,
        model_name=model_name,
        feature_set=feature_set,
    )

    with params_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if "best_params" in payload:
        best_params = payload["best_params"]
    elif "params" in payload:
        best_params = payload["params"]
    else:
        # Hỗ trợ trường hợp file chỉ chứa dictionary params.
        best_params = payload

    if not isinstance(best_params, dict):
        raise TypeError(
            f"best_params trong {params_path} "
            "không phải dictionary."
        )

    return best_params, params_path, payload


# ============================================================
# 7. Metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true và y_pred khác kích thước: "
            f"{y_true.shape} và {y_pred.shape}."
        )

    require_finite(y_true, "y_true")
    require_finite(y_pred, "y_pred")

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = math.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred,
    )

    nonzero_mask = y_true != 0

    if nonzero_mask.any():
        mape = np.mean(
            np.abs(
                (
                    y_true[nonzero_mask]
                    - y_pred[nonzero_mask]
                )
                / y_true[nonzero_mask]
            )
        ) * 100.0
    else:
        mape = float("nan")

    smape_denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
    )

    smape_mask = smape_denominator != 0

    if smape_mask.any():
        smape = np.mean(
            200.0
            * np.abs(
                y_true[smape_mask]
                - y_pred[smape_mask]
            )
            / smape_denominator[smape_mask]
        )
    else:
        smape = float("nan")

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape_percent": float(mape),
        "smape_percent": float(smape),
        "r2": float(r2),
    }


# ============================================================
# 8. Model parameter cleaning
# ============================================================

def normalize_tree_params(
    model_name: str,
    params: dict[str, Any],
    seed: int,
    n_jobs_model: int,
) -> dict[str, Any]:
    """
    Chuẩn hóa best params trước khi tạo model cuối.
    """

    normalized = dict(params)

    # Không để params từ tuning ghi đè random seed.
    for key in (
        "random_state",
        "seed",
        "random_seed",
    ):
        normalized.pop(key, None)

    if model_name == "rf":
        normalized["random_state"] = seed
        normalized["n_jobs"] = n_jobs_model

    elif model_name == "xgb":
        normalized["random_state"] = seed
        normalized["n_jobs"] = n_jobs_model
        normalized.setdefault(
            "objective",
            "reg:squarederror",
        )
        normalized.setdefault(
            "eval_metric",
            "rmse",
        )
        normalized.setdefault(
            "verbosity",
            0,
        )

    elif model_name == "lgbm":
        normalized["random_state"] = seed
        normalized["n_jobs"] = n_jobs_model
        normalized.setdefault(
            "verbosity",
            -1,
        )

    return normalized


# ============================================================
# 9. Tree models
# ============================================================

def create_tree_model(
    model_name: str,
    params: dict[str, Any],
    seed: int,
    n_jobs_model: int,
) -> Any:
    normalized_params = normalize_tree_params(
        model_name=model_name,
        params=params,
        seed=seed,
        n_jobs_model=n_jobs_model,
    )

    if model_name == "rf":
        return RandomForestRegressor(
            **normalized_params
        )

    if model_name == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError(
                "Thiếu xgboost. Chạy: pip install xgboost"
            ) from error

        return XGBRegressor(
            **normalized_params
        )

    if model_name == "lgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise ImportError(
                "Thiếu lightgbm. "
                "Chạy: pip install lightgbm"
            ) from error

        return LGBMRegressor(
            **normalized_params
        )

    raise ValueError(
        f"Không hỗ trợ tree model: {model_name}"
    )


def train_tree_model(
    model_name: str,
    dataset: FinalDataset,
    best_params: dict[str, Any],
    seed: int,
    n_jobs_model: int,
) -> tuple[Any, np.ndarray, float, float]:
    model = create_tree_model(
        model_name=model_name,
        params=best_params,
        seed=seed,
        n_jobs_model=n_jobs_model,
    )

    train_start = time.perf_counter()

    model.fit(
        dataset.x_train_final,
        dataset.y_train_final,
    )

    training_seconds = (
        time.perf_counter()
        - train_start
    )

    inference_start = time.perf_counter()

    predictions = model.predict(
        dataset.x_test
    )

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    ).reshape(-1)

    require_finite(
        predictions,
        f"{model_name}-{dataset.feature_set} predictions",
    )

    return (
        model,
        predictions,
        training_seconds,
        inference_seconds,
    )


# ============================================================
# 10. LSTM dataset preparation
# ============================================================

def get_sequence_columns(
    feature_names: list[str],
) -> list[str]:
    """
    Chuỗi thời gian được sắp:
    load_lag_23, ..., load_lag_0.

    Nghĩa là timestep đầu tiên là xa nhất,
    timestep cuối là gần thời điểm dự báo nhất.
    """

    sequence_columns = [
        f"load_lag_{lag}"
        for lag in range(
            SEQUENCE_LENGTH - 1,
            -1,
            -1,
        )
    ]

    missing_columns = [
        column
        for column in sequence_columns
        if column not in feature_names
    ]

    if missing_columns:
        raise KeyError(
            "Thiếu các lag bắt buộc cho LSTM: "
            f"{missing_columns}"
        )

    return sequence_columns


def prepare_lstm_final_dataset(
    split_dir: Path,
    feature_set: str,
) -> LSTMFinalDataset:
    train_frame = load_split_file(
        split_dir,
        feature_set,
        "train",
    )

    validation_frame = load_split_file(
        split_dir,
        feature_set,
        "validation",
    )

    test_frame = load_split_file(
        split_dir,
        feature_set,
        "test",
    )

    verify_split_order(
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        feature_set=feature_set,
    )

    feature_names = [
        column
        for column in train_frame.columns
        if column not in {
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
        }
    ]

    sequence_feature_names = get_sequence_columns(
        feature_names
    )

    static_feature_names = [
        column
        for column in feature_names
        if column not in sequence_feature_names
    ]

    final_train_frame = pd.concat(
        [train_frame, validation_frame],
        axis=0,
        ignore_index=True,
    )

    x_train_sequence_2d = final_train_frame[
        sequence_feature_names
    ].to_numpy(dtype=np.float32)

    x_test_sequence_2d = test_frame[
        sequence_feature_names
    ].to_numpy(dtype=np.float32)

    y_train_original = final_train_frame[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1)

    y_test_original = test_frame[
        TARGET_COLUMN
    ].to_numpy(dtype=np.float32)

    sequence_scaler = StandardScaler()

    x_train_sequence_scaled = (
        sequence_scaler.fit_transform(
            x_train_sequence_2d
        )
    )

    x_test_sequence_scaled = (
        sequence_scaler.transform(
            x_test_sequence_2d
        )
    )

    x_train_sequence = (
        x_train_sequence_scaled
        .reshape(
            -1,
            SEQUENCE_LENGTH,
            1,
        )
        .astype(np.float32)
    )

    x_test_sequence = (
        x_test_sequence_scaled
        .reshape(
            -1,
            SEQUENCE_LENGTH,
            1,
        )
        .astype(np.float32)
    )

    static_scaler: StandardScaler | None

    if static_feature_names:
        x_train_static_original = final_train_frame[
            static_feature_names
        ].to_numpy(dtype=np.float32)

        x_test_static_original = test_frame[
            static_feature_names
        ].to_numpy(dtype=np.float32)

        static_scaler = StandardScaler()

        x_train_static = (
            static_scaler.fit_transform(
                x_train_static_original
            )
            .astype(np.float32)
        )

        x_test_static = (
            static_scaler.transform(
                x_test_static_original
            )
            .astype(np.float32)
        )
    else:
        static_scaler = None

        x_train_static = np.empty(
            (
                len(final_train_frame),
                0,
            ),
            dtype=np.float32,
        )

        x_test_static = np.empty(
            (
                len(test_frame),
                0,
            ),
            dtype=np.float32,
        )

    target_scaler = StandardScaler()

    y_train_scaled = (
        target_scaler.fit_transform(
            y_train_original
        )
        .astype(np.float32)
    )

    require_finite(
        x_train_sequence,
        "LSTM x_train_sequence",
    )
    require_finite(
        x_test_sequence,
        "LSTM x_test_sequence",
    )
    require_finite(
        x_train_static,
        "LSTM x_train_static",
    )
    require_finite(
        x_test_static,
        "LSTM x_test_static",
    )
    require_finite(
        y_train_scaled,
        "LSTM y_train_scaled",
    )
    require_finite(
        y_test_original,
        "LSTM y_test_original",
    )

    return LSTMFinalDataset(
        feature_set=feature_set,
        sequence_feature_names=sequence_feature_names,
        static_feature_names=static_feature_names,
        x_train_sequence=x_train_sequence,
        x_train_static=x_train_static,
        x_test_sequence=x_test_sequence,
        x_test_static=x_test_static,
        y_train_scaled=y_train_scaled,
        y_test_original=y_test_original,
        test_timestamps=test_frame[
            TIMESTAMP_COLUMN
        ].copy(),
        sequence_scaler=sequence_scaler,
        static_scaler=static_scaler,
        target_scaler=target_scaler,
        train_rows=len(train_frame),
        validation_rows=len(validation_frame),
        final_train_rows=len(final_train_frame),
        test_rows=len(test_frame),
        sequence_length=SEQUENCE_LENGTH,
        static_feature_count=len(
            static_feature_names
        ),
    )


def make_lstm_inputs(
    sequence_data: np.ndarray,
    static_data: np.ndarray,
) -> np.ndarray | dict[str, np.ndarray]:
    if static_data.shape[1] > 0:
        return {
            "sequence_input": sequence_data,
            "static_input": static_data,
        }

    return sequence_data


# ============================================================
# 11. Build final LSTM
# ============================================================

def get_param(
    params: dict[str, Any],
    possible_names: list[str],
    default: Any,
) -> Any:
    for name in possible_names:
        if name in params:
            return params[name]

    return default


def build_final_lstm(
    best_params: dict[str, Any],
    sequence_input_shape: tuple[int, int],
    static_feature_count: int,
    seed: int,
) -> Any:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ImportError(
            "Thiếu TensorFlow. "
            "Chạy: pip install tensorflow"
        ) from error

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    lstm_layers = int(
        get_param(
            best_params,
            [
                "lstm_layers",
                "num_lstm_layers",
                "n_lstm_layers",
            ],
            1,
        )
    )

    units_1 = int(
        get_param(
            best_params,
            [
                "units_layer_1",
                "lstm_units_1",
                "units_1",
                "lstm_units",
            ],
            64,
        )
    )

    units_2 = int(
        get_param(
            best_params,
            [
                "units_layer_2",
                "lstm_units_2",
                "units_2",
            ],
            32,
        )
    )

    dense_units = int(
        get_param(
            best_params,
            [
                "dense_units",
                "fusion_dense_units",
            ],
            32,
        )
    )

    static_dense_units = int(
        get_param(
            best_params,
            [
                "static_dense_units",
                "static_units",
            ],
            16,
        )
    )

    dropout = float(
        get_param(
            best_params,
            [
                "dropout",
                "lstm_dropout",
            ],
            0.1,
        )
    )

    dense_dropout = float(
        get_param(
            best_params,
            [
                "dense_dropout",
                "fusion_dropout",
            ],
            dropout,
        )
    )

    learning_rate = float(
        get_param(
            best_params,
            [
                "learning_rate",
                "lr",
            ],
            1e-3,
        )
    )

    sequence_input = tf.keras.Input(
        shape=sequence_input_shape,
        name="sequence_input",
    )

    if lstm_layers >= 2:
        sequence_branch = tf.keras.layers.LSTM(
            units=units_1,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.0,
            name="lstm_1",
        )(sequence_input)

        sequence_branch = tf.keras.layers.LSTM(
            units=units_2,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=0.0,
            name="lstm_2",
        )(sequence_branch)
    else:
        sequence_branch = tf.keras.layers.LSTM(
            units=units_1,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=0.0,
            name="lstm_1",
        )(sequence_input)

    if static_feature_count > 0:
        static_input = tf.keras.Input(
            shape=(static_feature_count,),
            name="static_input",
        )

        static_branch = tf.keras.layers.Dense(
            static_dense_units,
            activation="relu",
            name="static_dense",
        )(static_input)

        combined = tf.keras.layers.Concatenate(
            name="feature_concatenation"
        )(
            [
                sequence_branch,
                static_branch,
            ]
        )

        model_inputs: Any = {
            "sequence_input": sequence_input,
            "static_input": static_input,
        }
    else:
        combined = sequence_branch
        model_inputs = sequence_input

    hidden = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        name="fusion_dense",
    )(combined)

    if dense_dropout > 0:
        hidden = tf.keras.layers.Dropout(
            dense_dropout,
            name="fusion_dropout",
        )(hidden)

    output = tf.keras.layers.Dense(
        1,
        activation="linear",
        name="load_forecast",
    )(hidden)

    model = tf.keras.Model(
        inputs=model_inputs,
        outputs=output,
        name=f"lstm_{seed}",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(
                name="rmse"
            )
        ],
    )

    return model


# ============================================================
# 12. Train final LSTM
# ============================================================

def train_lstm_model(
    dataset: LSTMFinalDataset,
    best_params: dict[str, Any],
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[Any, np.ndarray, float, float, dict[str, Any]]:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ImportError(
            "Thiếu TensorFlow."
        ) from error

    batch_size = int(
        get_param(
            best_params,
            ["batch_size"],
            128,
        )
    )

    model = build_final_lstm(
        best_params=best_params,
        sequence_input_shape=(
            dataset.sequence_length,
            1,
        ),
        static_feature_count=(
            dataset.static_feature_count
        ),
        seed=seed,
    )

    train_inputs = make_lstm_inputs(
        sequence_data=(
            dataset.x_train_sequence
        ),
        static_data=(
            dataset.x_train_static
        ),
    )

    test_inputs = make_lstm_inputs(
        sequence_data=(
            dataset.x_test_sequence
        ),
        static_data=(
            dataset.x_test_static
        ),
    )

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            mode="min",
            patience=patience,
            min_delta=DEFAULT_LSTM_MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        )
    )

    terminate_on_nan = (
        tf.keras.callbacks.TerminateOnNaN()
    )

    train_start = time.perf_counter()

    history = model.fit(
        train_inputs,
        dataset.y_train_scaled,
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[
            early_stopping,
            terminate_on_nan,
        ],
        verbose=2,
    )

    training_seconds = (
        time.perf_counter()
        - train_start
    )

    inference_start = time.perf_counter()

    scaled_predictions = model.predict(
        test_inputs,
        batch_size=batch_size,
        verbose=0,
    )

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    predictions = (
        dataset.target_scaler
        .inverse_transform(
            np.asarray(
                scaled_predictions,
                dtype=np.float32,
            ).reshape(-1, 1)
        )
        .reshape(-1)
    )

    require_finite(
        predictions,
        f"LSTM-{dataset.feature_set} predictions",
    )

    loss_history = history.history.get(
        "loss",
        [],
    )

    training_info = {
        "epochs_trained": len(loss_history),
        "final_training_loss": (
            float(loss_history[-1])
            if loss_history
            else None
        ),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "trainable_parameters": int(
            model.count_params()
        ),
    }

    return (
        model,
        predictions,
        training_seconds,
        inference_seconds,
        training_info,
    )


# ============================================================
# 13. Save artifacts
# ============================================================

def build_run_name(
    model_name: str,
    feature_set: str,
    seed: int,
) -> str:
    return (
        f"{model_name.upper()}_"
        f"{feature_set}_seed{seed}"
    )


def save_predictions(
    path: Path,
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    prediction_frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.to_datetime(
                timestamps
            ),
            "actual": np.asarray(
                y_true
            ).reshape(-1),
            "prediction": np.asarray(
                y_pred
            ).reshape(-1),
        }
    )

    prediction_frame["error"] = (
        prediction_frame["actual"]
        - prediction_frame["prediction"]
    )

    prediction_frame["absolute_error"] = (
        prediction_frame["error"].abs()
    )

    prediction_frame["squared_error"] = (
        prediction_frame["error"] ** 2
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_frame.to_csv(
        path,
        index=False,
    )


def save_tree_model(
    model: Any,
    model_path: Path,
) -> None:
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
        compress=3,
    )


def save_lstm_artifacts(
    model: Any,
    model_path: Path,
    scaler_path: Path,
    dataset: LSTMFinalDataset,
) -> None:
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(model_path)

    scaler_payload = {
        "sequence_scaler": (
            dataset.sequence_scaler
        ),
        "static_scaler": (
            dataset.static_scaler
        ),
        "target_scaler": (
            dataset.target_scaler
        ),
        "sequence_feature_names": (
            dataset.sequence_feature_names
        ),
        "static_feature_names": (
            dataset.static_feature_names
        ),
    }

    joblib.dump(
        scaler_payload,
        scaler_path,
        compress=3,
    )


# ============================================================
# 14. Single final run
# ============================================================

def train_single_configuration(
    model_name: str,
    feature_set: str,
    seed: int,
    args: argparse.Namespace,
    directories: dict[str, Path],
) -> dict[str, Any]:
    run_name = build_run_name(
        model_name=model_name,
        feature_set=feature_set,
        seed=seed,
    )

    prediction_path = (
        directories["predictions"]
        / f"{run_name}.csv"
    )

    metrics_path = (
        directories["metrics"]
        / f"{run_name}.json"
    )

    metadata_path = (
        directories["metadata"]
        / f"{run_name}.json"
    )

    if model_name == "lstm":
        model_path = (
            directories["models"]
            / f"{run_name}.keras"
        )
        scaler_path = (
            directories["models"]
            / f"{run_name}_scalers.joblib"
        )
    else:
        model_path = (
            directories["models"]
            / f"{run_name}.joblib"
        )
        scaler_path = None

    required_outputs = [
        prediction_path,
        metrics_path,
        metadata_path,
        model_path,
    ]

    if (
        not args.overwrite
        and all(
            path.exists()
            for path in required_outputs
        )
    ):
        print(
            f"[SKIP] {run_name}: output đã tồn tại."
        )

        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    print()
    print("=" * 72)
    print(f"RUN: {run_name}")
    print("=" * 72)

    set_global_seed(seed)

    best_params, params_path, params_payload = (
        load_best_params(
            params_dir=args.params_dir,
            model_name=model_name,
            feature_set=feature_set,
        )
    )

    print(f"Best params: {params_path}")

    if model_name == "lstm":
        dataset = prepare_lstm_final_dataset(
            split_dir=args.split_dir,
            feature_set=feature_set,
        )

        print(
            "LSTM sequence shape:",
            dataset.x_train_sequence.shape,
        )
        print(
            "LSTM static features:",
            dataset.static_feature_count,
        )

        (
            model,
            predictions,
            training_seconds,
            inference_seconds,
            training_info,
        ) = train_lstm_model(
            dataset=dataset,
            best_params=best_params,
            seed=seed,
            max_epochs=int(
    params_payload.get(
        "best_trial_user_attrs",
        {},
    ).get(
        "best_epoch",
        args.lstm_max_epochs,
    )
),
            patience=args.lstm_patience,
        )

        y_test = dataset.y_test_original
        test_timestamps = dataset.test_timestamps

        feature_names = (
            dataset.sequence_feature_names
            + dataset.static_feature_names
        )

        save_lstm_artifacts(
            model=model,
            model_path=model_path,
            scaler_path=scaler_path,
            dataset=dataset,
        )

        data_info = {
            "train_rows": dataset.train_rows,
            "validation_rows": (
                dataset.validation_rows
            ),
            "final_train_rows": (
                dataset.final_train_rows
            ),
            "test_rows": dataset.test_rows,
            "sequence_length": (
                dataset.sequence_length
            ),
            "static_feature_count": (
                dataset.static_feature_count
            ),
            "sequence_feature_names": (
                dataset.sequence_feature_names
            ),
            "static_feature_names": (
                dataset.static_feature_names
            ),
        }

    else:
        dataset = load_final_dataset(
            split_dir=args.split_dir,
            feature_set=feature_set,
        )

        (
            model,
            predictions,
            training_seconds,
            inference_seconds,
        ) = train_tree_model(
            model_name=model_name,
            dataset=dataset,
            best_params=best_params,
            seed=seed,
            n_jobs_model=args.n_jobs_model,
        )

        y_test = dataset.y_test
        test_timestamps = dataset.test_timestamps
        feature_names = dataset.feature_names

        save_tree_model(
            model=model,
            model_path=model_path,
        )

        training_info = {
            "trainable_parameters": None,
            "epochs_trained": None,
        }

        data_info = {
            "train_rows": dataset.train_rows,
            "validation_rows": (
                dataset.validation_rows
            ),
            "final_train_rows": (
                dataset.final_train_rows
            ),
            "test_rows": dataset.test_rows,
            "feature_count": len(
                dataset.feature_names
            ),
            "feature_names": (
                dataset.feature_names
            ),
        }

    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=predictions,
    )

    inference_ms_per_sample = (
        inference_seconds
        / len(predictions)
        * 1000.0
    )

    metrics_payload = {
        "run_name": run_name,
        "model": model_name.upper(),
        "feature_set": feature_set,
        "seed": seed,
        "evaluation_split": "test",
        **metrics,
        "training_seconds": float(
            training_seconds
        ),
        "inference_seconds": float(
            inference_seconds
        ),
        "inference_ms_per_sample": float(
            inference_ms_per_sample
        ),
        "test_rows": int(len(y_test)),
    }

    metadata_payload = {
        "run_name": run_name,
        "model": model_name.upper(),
        "feature_set": feature_set,
        "seed": seed,
        "best_params": best_params,
        "best_params_file": str(params_path),
        "best_tuning_payload": params_payload,
        "training_protocol": {
            "hyperparameter_tuning_data": [
                "train",
                "validation",
            ],
            "final_model_training_data": (
                "train_plus_validation"
            ),
            "final_evaluation_data": "test",
            "test_used_for_tuning": False,
            "test_used_for_early_stopping": False,
            "shuffle": False,
        },
        "timing": {
            "training_seconds": float(
                training_seconds
            ),
            "inference_seconds": float(
                inference_seconds
            ),
            "inference_ms_per_sample": float(
                inference_ms_per_sample
            ),
        },
        "data": data_info,
        "training_info": training_info,
        "feature_names": feature_names,
        "artifacts": {
            "model": str(model_path),
            "scalers": (
                str(scaler_path)
                if scaler_path is not None
                else None
            ),
            "predictions": str(
                prediction_path
            ),
            "metrics": str(metrics_path),
            "metadata": str(metadata_path),
        },
    }

    save_predictions(
        path=prediction_path,
        timestamps=test_timestamps,
        y_true=y_test,
        y_pred=predictions,
    )

    write_json(
        metrics_path,
        metrics_payload,
    )

    write_json(
        metadata_path,
        metadata_payload,
    )

    print(
        f"RMSE={metrics['rmse']:.6f} | "
        f"MAE={metrics['mae']:.6f} | "
        f"R²={metrics['r2']:.6f}"
    )

    print(
        f"Training time: "
        f"{training_seconds:.2f} s"
    )

    print(
        f"Inference time: "
        f"{inference_seconds:.4f} s"
    )

    print(
        f"Saved model: {model_path}"
    )

    del model
    gc.collect()

    if model_name == "lstm":
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
        except ImportError:
            pass

    return metrics_payload


# ============================================================
# 15. Summary output
# ============================================================

def save_training_summary(
    records: list[dict[str, Any]],
    summary_dir: Path,
) -> None:
    if not records:
        raise RuntimeError(
            "Không có kết quả huấn luyện."
        )

    summary_frame = pd.DataFrame(records)

    preferred_columns = [
        "run_name",
        "model",
        "feature_set",
        "seed",
        "rmse",
        "mae",
        "mape_percent",
        "smape_percent",
        "r2",
        "training_seconds",
        "inference_seconds",
        "inference_ms_per_sample",
        "test_rows",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in summary_frame.columns
    ]

    summary_frame = summary_frame[
        available_columns
    ].sort_values(
        by=[
            "model",
            "feature_set",
            "seed",
        ]
    )

    runs_path = (
        summary_dir
        / "final_model_runs.csv"
    )

    summary_frame.to_csv(
        runs_path,
        index=False,
    )

    numeric_metrics = [
        "rmse",
        "mae",
        "mape_percent",
        "smape_percent",
        "r2",
        "training_seconds",
        "inference_seconds",
        "inference_ms_per_sample",
    ]

    aggregation_map: dict[
        str,
        list[str],
    ] = {
        metric: ["mean", "std"]
        for metric in numeric_metrics
        if metric in summary_frame.columns
    }

    aggregate_frame = (
        summary_frame
        .groupby(
            ["model", "feature_set"],
            as_index=False,
        )
        .agg(aggregation_map)
    )

    aggregate_frame.columns = [
        "_".join(
            [
                str(part)
                for part in column
                if str(part)
            ]
        )
        if isinstance(column, tuple)
        else column
        for column in aggregate_frame.columns
    ]

    aggregate_path = (
        summary_dir
        / "final_model_summary_by_seed.csv"
    )

    aggregate_frame.to_csv(
        aggregate_path,
        index=False,
    )

    try:
        excel_path = (
            summary_dir
            / "final_model_results.xlsx"
        )

        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl",
        ) as writer:
            summary_frame.to_excel(
                writer,
                sheet_name="all_runs",
                index=False,
            )

            aggregate_frame.to_excel(
                writer,
                sheet_name="seed_summary",
                index=False,
            )

    except ImportError:
        warnings.warn(
            "Không tạo được Excel vì thiếu openpyxl.",
            RuntimeWarning,
        )

    print()
    print(f"Saved run summary: {runs_path}")
    print(
        "Saved seed summary:",
        aggregate_path,
    )


# ============================================================
# 16. Main loop: 4 × 5 × 3 = 60 models
# ============================================================

def main() -> None:
    args = parse_arguments()

    directories = make_output_directories(
        args.output_dir
    )

    total_runs = (
        len(args.models)
        * len(args.feature_sets)
        * len(args.seeds)
    )

    print("=" * 72)
    print("FINAL MODEL TRAINING")
    print("=" * 72)
    print(f"Models       : {args.models}")
    print(
        f"Feature sets : {args.feature_sets}"
    )
    print(f"Seeds        : {args.seeds}")
    print(f"Total runs   : {total_runs}")
    print(
        "Training data: train + validation"
    )
    print("Evaluation   : test only")
    print("=" * 72)

    records: list[dict[str, Any]] = []
    run_index = 0

    for model_name in args.models:
        for feature_set in args.feature_sets:
            for seed in args.seeds:
                run_index += 1

                print()
                print(
                    f"[{run_index}/{total_runs}] "
                    f"{model_name.upper()}-"
                    f"{feature_set}-seed{seed}"
                )

                try:
                    result = train_single_configuration(
                        model_name=model_name,
                        feature_set=feature_set,
                        seed=seed,
                        args=args,
                        directories=directories,
                    )

                    records.append(result)

                except Exception as error:
                    print(
                        f"[FAILED] "
                        f"{model_name.upper()}-"
                        f"{feature_set}-seed{seed}: "
                        f"{error}"
                    )

                    raise

    save_training_summary(
        records=records,
        summary_dir=(
            directories["summaries"]
        ),
    )

    print()
    print("=" * 72)
    print(
        f"HOÀN THÀNH {len(records)}/{total_runs} RUN"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()