"""
hyperparameter_tuning.py

PHẦN 1:
    - Parser CLI
    - Kiểm tra thư viện
    - Đọc train và validation
    - Kiểm tra dữ liệu
    - Hàm RMSE
    - Objective Random Forest
    - Objective XGBoost
    - Objective LightGBM

PHẦN 2 sẽ bổ sung:
    - Objective LSTM
    - Early stopping cho LSTM
    - Tạo Optuna Study
    - Chạy toàn bộ model × feature set
    - Lưu best_params.json
    - Lưu lịch sử các trial
    - Hàm main()

Nguyên tắc:
    - Model chỉ fit trên train.
    - Validation chỉ dùng để chọn siêu tham số.
    - Không đọc hoặc sử dụng tập test.
    - Objective là validation RMSE.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. Hằng số chung
# ============================================================

SUPPORTED_MODELS = (
    "rf",
    "xgb",
    "lgbm",
    "lstm",
)

SUPPORTED_FEATURE_SETS = (
    "F0",
    "F1",
    "F2",
    "F3",
    "F4",
)

REQUIRED_COLUMNS = (
    "timestamp",
    "target",
)

DEFAULT_SPLIT_DIR = "outputs/splits"
DEFAULT_OUTPUT_DIR = "outputs/tuning"

DEFAULT_MODELS = [
    "rf",
    "xgb",
    "lgbm",
    "lstm",
]

DEFAULT_FEATURE_SETS = [
    "F0",
    "F1",
    "F2",
    "F3",
    "F4",
]


# ============================================================
# 2. Kiểu dữ liệu dùng để lưu train và validation
# ============================================================

@dataclass
class TuningDataset:
    """
    Dữ liệu của một feature set phục vụ tuning.

    Thuộc tính
    ----------
    feature_set:
        Tên feature set, ví dụ F0 hoặc F4.

    feature_names:
        Danh sách các biến đầu vào.

    train_timestamps:
        Timestamp của tập train.

    validation_timestamps:
        Timestamp của tập validation.

    x_train:
        Ma trận đặc trưng train.

    y_train:
        Vector target train.

    x_validation:
        Ma trận đặc trưng validation.

    y_validation:
        Vector target validation.
    """

    feature_set: str
    feature_names: list[str]

    train_timestamps: pd.Series
    validation_timestamps: pd.Series

    x_train: pd.DataFrame
    y_train: np.ndarray

    x_validation: pd.DataFrame
    y_validation: np.ndarray


# ============================================================
# 3. Parser CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Đọc tham số dòng lệnh.

    Ví dụ PowerShell:

    python hyperparameter_tuning.py `
        --split-dir outputs/splits `
        --output-dir outputs/tuning `
        --models rf xgb lgbm lstm `
        --feature-sets F0 F1 F2 F3 F4 `
        --n-trials 10 `
        --seed 42
    """

    parser = argparse.ArgumentParser(
        description=(
            "Tối ưu siêu tham số cho RF, XGBoost, LightGBM và "
            "LSTM bằng Optuna trên tập train và validation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--split-dir",
        type=str,
        default=DEFAULT_SPLIT_DIR,
        help=(
            "Thư mục chứa các file đã chia, ví dụ "
            "F0_train.csv và F0_validation.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Thư mục lưu kết quả tối ưu siêu tham số.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=DEFAULT_MODELS,
        help=(
            "Danh sách mô hình cần tuning. "
            "Các giá trị hỗ trợ: rf xgb lgbm lstm."
        ),
    )

    parser.add_argument(
        "--feature-sets",
        nargs="+",
        type=str,
        default=DEFAULT_FEATURE_SETS,
        help=(
            "Danh sách feature set cần tuning. "
            "Các giá trị hỗ trợ: F0 F1 F2 F3 F4."
        ),
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Số Optuna trial cho mỗi cấu hình model–feature set.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed dùng trong Optuna và các mô hình.",
    )

    parser.add_argument(
        "--n-jobs-model",
        type=int,
        default=-1,
        help=(
            "Số CPU thread bên trong mô hình RF, XGB và LGBM. "
            "-1 nghĩa là sử dụng tất cả CPU khả dụng."
        ),
    )

    parser.add_argument(
        "--study-jobs",
        type=int,
        default=1,
        help=(
            "Số Optuna trial chạy song song. "
            "Khuyến nghị đặt 1 để bảo đảm khả năng tái lập."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            "Thời gian tối đa tính bằng giây cho mỗi Optuna study. "
            "Mặc định không giới hạn."
        ),
    )

    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Optuna storage URI. Ví dụ: "
            "sqlite:///outputs/tuning/optuna_studies.db. "
            "Nếu bỏ trống, study chỉ lưu trong bộ nhớ trong khi chạy."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Cho phép nạp lại Optuna study đã tồn tại khi dùng storage."
        ),
    )

    parser.add_argument(
        "--show-progress-bar",
        action="store_true",
        help="Hiển thị thanh tiến trình của Optuna.",
    )

    parser.add_argument(
        "--strict-hourly",
        action="store_true",
        help=(
            "Kiểm tra timestamp trong từng split phải liên tục từng giờ."
        ),
    )

    parser.add_argument(
        "--float-dtype",
        type=str,
        choices=("float32", "float64"),
        default="float32",
        help=(
            "Kiểu dữ liệu số cho ma trận đặc trưng. "
            "float32 giúp giảm bộ nhớ."
        ),
    )

    parser.add_argument(
        "--rf-min-estimators",
        type=int,
        default=100,
        help="Giới hạn dưới của n_estimators cho Random Forest.",
    )

    parser.add_argument(
        "--rf-max-estimators",
        type=int,
        default=700,
        help="Giới hạn trên của n_estimators cho Random Forest.",
    )

    parser.add_argument(
        "--xgb-min-estimators",
        type=int,
        default=100,
        help="Giới hạn dưới của n_estimators cho XGBoost.",
    )

    parser.add_argument(
        "--xgb-max-estimators",
        type=int,
        default=1000,
        help="Giới hạn trên của n_estimators cho XGBoost.",
    )

    parser.add_argument(
        "--lgbm-min-estimators",
        type=int,
        default=100,
        help="Giới hạn dưới của n_estimators cho LightGBM.",
    )

    parser.add_argument(
        "--lgbm-max-estimators",
        type=int,
        default=1000,
        help="Giới hạn trên của n_estimators cho LightGBM.",
    )

    args = parser.parse_args()

    args.models = normalize_model_names(args.models)

    args.feature_sets = normalize_feature_set_names(
        args.feature_sets
    )

    validate_cli_arguments(args)

    return args


def normalize_model_names(
    model_names: list[str],
) -> list[str]:
    """
    Chuẩn hóa tên mô hình về chữ thường và loại trùng lặp.
    """

    normalized: list[str] = []

    aliases = {
        "random_forest": "rf",
        "randomforest": "rf",
        "random-forest": "rf",
        "xgboost": "xgb",
        "lightgbm": "lgbm",
        "light_gbm": "lgbm",
    }

    for model_name in model_names:
        name = model_name.strip().lower()
        name = aliases.get(name, name)

        if name not in normalized:
            normalized.append(name)

    return normalized


def normalize_feature_set_names(
    feature_sets: list[str],
) -> list[str]:
    """
    Chuẩn hóa tên feature set về F0–F4 và loại trùng lặp.
    """

    normalized: list[str] = []

    for feature_set in feature_sets:
        name = feature_set.strip().upper()

        if name not in normalized:
            normalized.append(name)

    return normalized


def validate_cli_arguments(
    args: argparse.Namespace,
) -> None:
    """
    Kiểm tra tính hợp lệ của tham số dòng lệnh.
    """

    invalid_models = [
        model_name
        for model_name in args.models
        if model_name not in SUPPORTED_MODELS
    ]

    if invalid_models:
        raise ValueError(
            "Mô hình không được hỗ trợ: "
            f"{invalid_models}. "
            f"Các mô hình hợp lệ: {list(SUPPORTED_MODELS)}"
        )

    invalid_feature_sets = [
        feature_set
        for feature_set in args.feature_sets
        if feature_set not in SUPPORTED_FEATURE_SETS
    ]

    if invalid_feature_sets:
        raise ValueError(
            "Feature set không được hỗ trợ: "
            f"{invalid_feature_sets}. "
            "Các feature set hợp lệ: "
            f"{list(SUPPORTED_FEATURE_SETS)}"
        )

    if args.n_trials <= 0:
        raise ValueError(
            "--n-trials phải là số nguyên lớn hơn 0."
        )

    if args.study_jobs == 0 or args.study_jobs < -1:
        raise ValueError(
            "--study-jobs phải bằng -1 hoặc là số nguyên lớn hơn 0."
        )

    if args.timeout is not None and args.timeout <= 0:
        raise ValueError(
            "--timeout phải lớn hơn 0 nếu được chỉ định."
        )

    estimator_limits = [
        (
            "RF",
            args.rf_min_estimators,
            args.rf_max_estimators,
        ),
        (
            "XGB",
            args.xgb_min_estimators,
            args.xgb_max_estimators,
        ),
        (
            "LGBM",
            args.lgbm_min_estimators,
            args.lgbm_max_estimators,
        ),
    ]

    for (
        model_name,
        minimum,
        maximum,
    ) in estimator_limits:
        if minimum <= 0:
            raise ValueError(
                f"{model_name}: số cây tối thiểu phải lớn hơn 0."
            )

        if maximum < minimum:
            raise ValueError(
                f"{model_name}: giới hạn trên của số cây "
                "phải lớn hơn hoặc bằng giới hạn dưới."
            )

    split_dir = Path(args.split_dir)

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục split: "
            f"{split_dir.resolve()}"
        )


# ============================================================
# 4. Kiểm tra các thư viện phụ thuộc
# ============================================================

def require_module(
    module_name: str,
    install_name: str | None = None,
) -> Any:
    """
    Import một thư viện và cung cấp thông báo lỗi rõ ràng nếu thiếu.
    """

    package_name = install_name or module_name

    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"Chưa cài thư viện '{package_name}'. "
            f"Hãy chạy: pip install {package_name}"
        ) from error


def check_required_libraries(
    selected_models: list[str],
) -> None:
    """
    Kiểm tra các thư viện cần thiết theo các mô hình được chọn.
    """

    require_module("optuna")

    if "xgb" in selected_models:
        require_module(
            module_name="xgboost",
            install_name="xgboost",
        )

    if "lgbm" in selected_models:
        require_module(
            module_name="lightgbm",
            install_name="lightgbm",
        )

    if "lstm" in selected_models:
        require_module(
            module_name="tensorflow",
            install_name="tensorflow",
        )


# ============================================================
# 5. Thiết lập random seed
# ============================================================

def set_global_seed(seed: int) -> None:
    """
    Thiết lập seed cho Python và NumPy.

    TensorFlow seed sẽ được thiết lập thêm trong phần objective LSTM.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# 6. Đọc một file split
# ============================================================

def load_split_file(
    file_path: Path,
    split_name: str,
    feature_set: str,
    float_dtype: str = "float32",
    strict_hourly: bool = False,
) -> pd.DataFrame:
    """
    Đọc và kiểm tra một file train hoặc validation.

    Không sử dụng file test trong quá trình tuning.
    """

    if split_name not in {"train", "validation"}:
        raise ValueError(
            "Tuning chỉ được đọc split 'train' hoặc 'validation'."
        )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file {feature_set}_{split_name}: "
            f"{file_path.resolve()}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            f"{feature_set}_{split_name}.csv không có dữ liệu."
        )

    missing_required_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_required_columns:
        raise KeyError(
            f"{feature_set}_{split_name} thiếu cột bắt buộc: "
            f"{missing_required_columns}"
        )

    if len(df.columns) <= len(REQUIRED_COLUMNS):
        raise ValueError(
            f"{feature_set}_{split_name} không có biến đầu vào."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamp_count = int(
        df["timestamp"].isna().sum()
    )

    if invalid_timestamp_count > 0:
        raise ValueError(
            f"{feature_set}_{split_name} có "
            f"{invalid_timestamp_count} timestamp không hợp lệ."
        )

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            f"{feature_set}_{split_name} không được sắp xếp "
            "theo thời gian tăng dần."
        )

    duplicate_timestamp_count = int(
        df["timestamp"].duplicated().sum()
    )

    if duplicate_timestamp_count > 0:
        raise ValueError(
            f"{feature_set}_{split_name} có "
            f"{duplicate_timestamp_count} timestamp trùng lặp."
        )

    if strict_hourly:
        timestamp_differences = (
            df["timestamp"]
            .diff()
            .dropna()
        )

        abnormal_intervals = timestamp_differences[
            timestamp_differences
            != pd.Timedelta(hours=1)
        ]

        if not abnormal_intervals.empty:
            raise ValueError(
                f"{feature_set}_{split_name} không liên tục "
                "theo từng giờ. "
                f"Số khoảng bất thường: {len(abnormal_intervals)}."
            )

    numeric_columns = [
        column
        for column in df.columns
        if column != "timestamp"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    missing_values = (
        df[numeric_columns]
        .isna()
        .sum()
    )

    missing_values = missing_values[
        missing_values > 0
    ]

    if not missing_values.empty:
        raise ValueError(
            f"{feature_set}_{split_name} còn giá trị thiếu "
            "hoặc giá trị không chuyển được sang số: "
            f"{missing_values.to_dict()}"
        )

    numeric_array = df[numeric_columns].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    finite_mask = np.isfinite(numeric_array)

    if not finite_mask.all():
        invalid_count = int(
            np.size(finite_mask)
            - finite_mask.sum()
        )

        raise ValueError(
            f"{feature_set}_{split_name} có "
            f"{invalid_count} giá trị vô hạn hoặc không hữu hạn."
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in REQUIRED_COLUMNS
    ]

    df[feature_columns] = df[feature_columns].astype(
        float_dtype
    )

    df["target"] = df["target"].astype(float_dtype)

    return df


# ============================================================
# 7. Đọc train và validation của một feature set
# ============================================================

def load_tuning_dataset(
    split_dir: str | Path,
    feature_set: str,
    float_dtype: str = "float32",
    strict_hourly: bool = False,
) -> TuningDataset:
    """
    Đọc dữ liệu train và validation của một feature set.

    Tập test không được đọc.
    """

    split_directory = Path(split_dir)

    train_path = (
        split_directory
        / f"{feature_set}_train.csv"
    )

    validation_path = (
        split_directory
        / f"{feature_set}_validation.csv"
    )

    train_df = load_split_file(
        file_path=train_path,
        split_name="train",
        feature_set=feature_set,
        float_dtype=float_dtype,
        strict_hourly=strict_hourly,
    )

    validation_df = load_split_file(
        file_path=validation_path,
        split_name="validation",
        feature_set=feature_set,
        float_dtype=float_dtype,
        strict_hourly=strict_hourly,
    )

    validate_train_validation_pair(
        train_df=train_df,
        validation_df=validation_df,
        feature_set=feature_set,
    )

    feature_names = [
        column
        for column in train_df.columns
        if column not in REQUIRED_COLUMNS
    ]

    x_train = (
        train_df[feature_names]
        .copy()
    )

    y_train = (
        train_df["target"]
        .to_numpy(
            dtype=float_dtype,
            copy=True,
        )
    )

    x_validation = (
        validation_df[feature_names]
        .copy()
    )

    y_validation = (
        validation_df["target"]
        .to_numpy(
            dtype=float_dtype,
            copy=True,
        )
    )

    return TuningDataset(
        feature_set=feature_set,
        feature_names=feature_names,
        train_timestamps=(
            train_df["timestamp"].copy()
        ),
        validation_timestamps=(
            validation_df["timestamp"].copy()
        ),
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )


def validate_train_validation_pair(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_set: str,
) -> None:
    """
    Kiểm tra train và validation của cùng feature set.
    """

    train_columns = list(train_df.columns)
    validation_columns = list(validation_df.columns)

    if train_columns != validation_columns:
        train_only_columns = sorted(
            set(train_columns)
            - set(validation_columns)
        )

        validation_only_columns = sorted(
            set(validation_columns)
            - set(train_columns)
        )

        raise ValueError(
            f"{feature_set}: train và validation không có "
            "cùng cấu trúc cột. "
            f"Chỉ có ở train: {train_only_columns}. "
            f"Chỉ có ở validation: {validation_only_columns}."
        )

    train_last_timestamp = (
        train_df["timestamp"].iloc[-1]
    )

    validation_first_timestamp = (
        validation_df["timestamp"].iloc[0]
    )

    if train_last_timestamp >= validation_first_timestamp:
        raise ValueError(
            f"{feature_set}: train và validation bị chồng lấn "
            "hoặc không đúng thứ tự thời gian. "
            f"Train kết thúc tại {train_last_timestamp}, "
            f"validation bắt đầu tại "
            f"{validation_first_timestamp}."
        )

    if len(train_df) < 2:
        raise ValueError(
            f"{feature_set}: tập train quá nhỏ."
        )

    if len(validation_df) < 1:
        raise ValueError(
            f"{feature_set}: tập validation rỗng."
        )


# ============================================================
# 8. Hiển thị thông tin dữ liệu
# ============================================================

def print_dataset_summary(
    dataset: TuningDataset,
) -> None:
    """
    In thông tin train và validation.
    """

    print("-" * 70)

    print(
        f"Feature set       : {dataset.feature_set}"
    )

    print(
        f"Số biến đầu vào   : "
        f"{len(dataset.feature_names):,}"
    )

    print(
        f"Số dòng train     : "
        f"{len(dataset.x_train):,}"
    )

    print(
        f"Số dòng validation: "
        f"{len(dataset.x_validation):,}"
    )

    print(
        "Train period      : "
        f"{dataset.train_timestamps.iloc[0]} "
        "đến "
        f"{dataset.train_timestamps.iloc[-1]}"
    )

    print(
        "Validation period : "
        f"{dataset.validation_timestamps.iloc[0]} "
        "đến "
        f"{dataset.validation_timestamps.iloc[-1]}"
    )

    print(
        f"Target train mean : "
        f"{float(np.mean(dataset.y_train)):.6f}"
    )

    print(
        f"Target val mean   : "
        f"{float(np.mean(dataset.y_validation)):.6f}"
    )

    print("-" * 70)


# ============================================================
# 9. Hàm RMSE
# ============================================================

def calculate_rmse(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """
    Tính Root Mean Squared Error.

    RMSE = sqrt(mean((y_true - y_pred)^2))
    """

    true_values = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    predicted_values = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "y_true và y_pred phải có cùng kích thước. "
            f"Nhận được {true_values.shape} và "
            f"{predicted_values.shape}."
        )

    if true_values.size == 0:
        raise ValueError(
            "Không thể tính RMSE trên mảng rỗng."
        )

    if not np.isfinite(true_values).all():
        raise ValueError(
            "y_true chứa NaN hoặc giá trị vô hạn."
        )

    if not np.isfinite(predicted_values).all():
        raise ValueError(
            "y_pred chứa NaN hoặc giá trị vô hạn."
        )

    squared_errors = np.square(
        true_values - predicted_values
    )

    mean_squared_error = np.mean(
        squared_errors,
        dtype=np.float64,
    )

    rmse = math.sqrt(
        float(mean_squared_error)
    )

    return float(rmse)


# ============================================================
# 10. Kiểm tra giá trị objective
# ============================================================

def validate_objective_value(
    rmse: float,
    model_name: str,
    feature_set: str,
) -> float:
    """
    Kiểm tra RMSE trước khi trả về cho Optuna.
    """

    if not np.isfinite(rmse):
        raise ValueError(
            f"{model_name}-{feature_set} tạo RMSE không hữu hạn."
        )

    if rmse < 0:
        raise ValueError(
            f"{model_name}-{feature_set} tạo RMSE âm."
        )

    return float(rmse)


# ============================================================
# 11. Objective Random Forest
# ============================================================

def create_rf_objective(
    dataset: TuningDataset,
    seed: int,
    n_jobs_model: int,
    min_estimators: int = 100,
    max_estimators: int = 700,
) -> Callable[[Any], float]:
    """
    Tạo Optuna objective cho Random Forest.

    Model chỉ fit trên x_train và y_train.
    Validation chỉ dùng để tính RMSE.
    """

    def objective(trial: Any) -> float:
        trial_seed = seed

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                min_estimators,
                max_estimators,
                step=50,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                5,
                35,
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                20,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                10,
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                [
                    1.0,
                    "sqrt",
                    "log2",
                    0.5,
                    0.75,
                ],
            ),
            "bootstrap": trial.suggest_categorical(
                "bootstrap",
                [
                    True,
                    False,
                ],
            ),
            "criterion": trial.suggest_categorical(
                "criterion",
                [
                    "squared_error",
                    "friedman_mse",
                ],
            ),
        }

        if params["bootstrap"]:
            params["max_samples"] = trial.suggest_float(
                "max_samples",
                0.60,
                1.00,
                step=0.05,
            )
        else:
            params["max_samples"] = None

        model = RandomForestRegressor(
            **params,
            random_state=trial_seed,
            n_jobs=n_jobs_model,
        )

        start_time = time.perf_counter()

        try:
            model.fit(
                dataset.x_train,
                dataset.y_train,
            )

            validation_predictions = model.predict(
                dataset.x_validation
            )

            validation_rmse = calculate_rmse(
                y_true=dataset.y_validation,
                y_pred=validation_predictions,
            )

            elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            trial.set_user_attr(
                "model",
                "rf",
            )

            trial.set_user_attr(
                "feature_set",
                dataset.feature_set,
            )

            trial.set_user_attr(
                "validation_rmse",
                float(validation_rmse),
            )

            trial.set_user_attr(
                "fit_and_predict_seconds",
                float(elapsed_seconds),
            )

            trial.set_user_attr(
                "train_rows",
                int(len(dataset.x_train)),
            )

            trial.set_user_attr(
                "validation_rows",
                int(len(dataset.x_validation)),
            )

            trial.set_user_attr(
                "feature_count",
                int(len(dataset.feature_names)),
            )

            return validate_objective_value(
                rmse=validation_rmse,
                model_name="rf",
                feature_set=dataset.feature_set,
            )

        finally:
            del model
            gc.collect()

    return objective


# ============================================================
# 12. Objective XGBoost
# ============================================================

def create_xgb_objective(
    dataset: TuningDataset,
    seed: int,
    n_jobs_model: int,
    min_estimators: int = 100,
    max_estimators: int = 1000,
) -> Callable[[Any], float]:
    """
    Tạo Optuna objective cho XGBoost.

    Phiên bản tuning này không dùng early stopping cho XGBoost.
    Số boosting round được tối ưu thông qua n_estimators.

    Model chỉ fit trên train.
    Validation chỉ dùng để tính objective RMSE.
    """

    xgboost = require_module(
        module_name="xgboost",
        install_name="xgboost",
    )

    XGBRegressor = xgboost.XGBRegressor

    def objective(trial: Any) -> float:
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                min_estimators,
                max_estimators,
                step=50,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.30,
                log=True,
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                1.0,
                20.0,
                log=True,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.60,
                1.00,
                step=0.05,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.60,
                1.00,
                step=0.05,
            ),
            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                5.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                20.0,
                log=True,
            ),
            "max_bin": trial.suggest_categorical(
                "max_bin",
                [
                    128,
                    256,
                    512,
                ],
            ),
        }

        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=seed,
            seed=seed,
            n_jobs=n_jobs_model,
            verbosity=0,
        )

        start_time = time.perf_counter()

        try:
            model.fit(
                dataset.x_train,
                dataset.y_train,
                verbose=False,
            )

            validation_predictions = model.predict(
                dataset.x_validation
            )

            validation_rmse = calculate_rmse(
                y_true=dataset.y_validation,
                y_pred=validation_predictions,
            )

            elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            trial.set_user_attr(
                "model",
                "xgb",
            )

            trial.set_user_attr(
                "feature_set",
                dataset.feature_set,
            )

            trial.set_user_attr(
                "validation_rmse",
                float(validation_rmse),
            )

            trial.set_user_attr(
                "fit_and_predict_seconds",
                float(elapsed_seconds),
            )

            trial.set_user_attr(
                "train_rows",
                int(len(dataset.x_train)),
            )

            trial.set_user_attr(
                "validation_rows",
                int(len(dataset.x_validation)),
            )

            trial.set_user_attr(
                "feature_count",
                int(len(dataset.feature_names)),
            )

            return validate_objective_value(
                rmse=validation_rmse,
                model_name="xgb",
                feature_set=dataset.feature_set,
            )

        finally:
            del model
            gc.collect()

    return objective


# ============================================================
# 13. Objective LightGBM
# ============================================================

def create_lgbm_objective(
    dataset: TuningDataset,
    seed: int,
    n_jobs_model: int,
    min_estimators: int = 100,
    max_estimators: int = 1000,
) -> Callable[[Any], float]:
    """
    Tạo Optuna objective cho LightGBM.

    Model chỉ fit trên train.
    Validation chỉ dùng để tính objective RMSE.
    """

    lightgbm = require_module(
        module_name="lightgbm",
        install_name="lightgbm",
    )

    LGBMRegressor = lightgbm.LGBMRegressor

    def objective(trial: Any) -> float:
        max_depth = trial.suggest_int(
            "max_depth",
            3,
            16,
        )

        max_num_leaves = min(
            255,
            (2 ** max_depth) - 1,
        )

        min_num_leaves = min(
            8,
            max_num_leaves,
        )

        num_leaves = trial.suggest_int(
            "num_leaves",
            min_num_leaves,
            max_num_leaves,
        )

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                min_estimators,
                max_estimators,
                step=50,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.30,
                log=True,
            ),
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                5,
                100,
                step=5,
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                1e-4,
                10.0,
                log=True,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.60,
                1.00,
                step=0.05,
            ),
            "subsample_freq": trial.suggest_categorical(
                "subsample_freq",
                [
                    0,
                    1,
                    5,
                ],
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.60,
                1.00,
                step=0.05,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                20.0,
                log=True,
            ),
            "max_bin": trial.suggest_categorical(
                "max_bin",
                [
                    127,
                    255,
                    511,
                ],
            ),
        }

        model = LGBMRegressor(
            **params,
            objective="regression",
            random_state=seed,
            n_jobs=n_jobs_model,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )

        start_time = time.perf_counter()

        try:
            model.fit(
                dataset.x_train,
                dataset.y_train,
            )

            validation_predictions = model.predict(
                dataset.x_validation
            )

            validation_rmse = calculate_rmse(
                y_true=dataset.y_validation,
                y_pred=validation_predictions,
            )

            elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            trial.set_user_attr(
                "model",
                "lgbm",
            )

            trial.set_user_attr(
                "feature_set",
                dataset.feature_set,
            )

            trial.set_user_attr(
                "validation_rmse",
                float(validation_rmse),
            )

            trial.set_user_attr(
                "fit_and_predict_seconds",
                float(elapsed_seconds),
            )

            trial.set_user_attr(
                "train_rows",
                int(len(dataset.x_train)),
            )

            trial.set_user_attr(
                "validation_rows",
                int(len(dataset.x_validation)),
            )

            trial.set_user_attr(
                "feature_count",
                int(len(dataset.feature_names)),
            )

            return validate_objective_value(
                rmse=validation_rmse,
                model_name="lgbm",
                feature_set=dataset.feature_set,
            )

        finally:
            del model
            gc.collect()

    return objective


# ============================================================
# 14. Factory chọn objective
# ============================================================

def create_tree_model_objective(
    model_name: str,
    dataset: TuningDataset,
    args: argparse.Namespace,
) -> Callable[[Any], float]:
    """
    Chọn objective tương ứng với RF, XGB hoặc LGBM.

    LSTM sẽ được bổ sung trong Phần 2.
    """

    normalized_model_name = (
        model_name.strip().lower()
    )

    if normalized_model_name == "rf":
        return create_rf_objective(
            dataset=dataset,
            seed=args.seed,
            n_jobs_model=args.n_jobs_model,
            min_estimators=(
                args.rf_min_estimators
            ),
            max_estimators=(
                args.rf_max_estimators
            ),
        )

    if normalized_model_name == "xgb":
        return create_xgb_objective(
            dataset=dataset,
            seed=args.seed,
            n_jobs_model=args.n_jobs_model,
            min_estimators=(
                args.xgb_min_estimators
            ),
            max_estimators=(
                args.xgb_max_estimators
            ),
        )

    if normalized_model_name == "lgbm":
        return create_lgbm_objective(
            dataset=dataset,
            seed=args.seed,
            n_jobs_model=args.n_jobs_model,
            min_estimators=(
                args.lgbm_min_estimators
            ),
            max_estimators=(
                args.lgbm_max_estimators
            ),
        )

    if normalized_model_name == "lstm":
        raise NotImplementedError(
            "Objective LSTM sẽ được thêm trong Phần 2."
        )

    raise ValueError(
        f"Không hỗ trợ mô hình: {model_name}"
    )


# ============================================================
# 15. Cấu hình mặc định cho LSTM
# ============================================================

LSTM_MAX_EPOCHS = 100
LSTM_EARLY_STOPPING_PATIENCE = 10
LSTM_MIN_DELTA = 1e-5

LSTM_MIN_UNITS = 32
LSTM_MAX_UNITS = 256

LSTM_BATCH_SIZE_CHOICES = [
    32,
    64,
    128,
    256,
]

LSTM_OPTIMIZER_CHOICES = [
    "adam",
    "rmsprop",
]


# ============================================================
# 16. Chuẩn bị dữ liệu cho LSTM
# ============================================================

@dataclass
class LSTMTuningDataset:
    """
    Dữ liệu LSTM gồm hai nhánh:

    1. Sequence input:
       load_lag_23, ..., load_lag_0
       shape = (samples, 24, 1)

    2. Static input:
       các feature còn lại của F1–F4
       shape = (samples, static_feature_count)
    """

    feature_set: str

    sequence_feature_names: list[str]
    static_feature_names: list[str]

    x_train_sequence: np.ndarray
    x_validation_sequence: np.ndarray

    x_train_static: np.ndarray | None
    x_validation_static: np.ndarray | None

    y_train: np.ndarray
    y_validation: np.ndarray

    y_scaler: Any

    train_rows: int
    validation_rows: int
    sequence_length: int
    static_feature_count: int


def prepare_lstm_dataset(
    dataset: TuningDataset,
) -> LSTMTuningDataset:
    """
    Chuẩn bị dữ liệu LSTM thực sự theo 24 timestep.

    Thứ tự chuỗi:
        load_lag_23: thời điểm xa nhất
        ...
        load_lag_1
        load_lag_0 : thời điểm gần target nhất

    Tất cả scaler chỉ fit trên train.
    """

    from sklearn.preprocessing import StandardScaler

    # Phải sắp xếp từ quá khứ xa đến hiện tại.
    sequence_feature_names = [
        f"load_lag_{lag}"
        for lag in range(23, -1, -1)
    ]

    missing_sequence_columns = [
        column
        for column in sequence_feature_names
        if column not in dataset.feature_names
    ]

    if missing_sequence_columns:
        raise KeyError(
            "Không thể tạo chuỗi LSTM vì thiếu các cột: "
            f"{missing_sequence_columns}"
        )

    static_feature_names = [
        column
        for column in dataset.feature_names
        if column not in sequence_feature_names
    ]

    # ========================================================
    # Nhánh chuỗi 24 giờ
    # ========================================================

    x_train_sequence_raw = (
        dataset.x_train[
            sequence_feature_names
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    x_validation_sequence_raw = (
        dataset.x_validation[
            sequence_feature_names
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    # Fit một scaler chung cho tất cả giá trị tải lịch sử.
    # Dữ liệu được flatten thành một cột rồi reshape lại.
    sequence_scaler = StandardScaler()

    train_sequence_flat = (
        x_train_sequence_raw.reshape(-1, 1)
    )

    validation_sequence_flat = (
        x_validation_sequence_raw.reshape(-1, 1)
    )

    train_sequence_scaled = (
        sequence_scaler
        .fit_transform(train_sequence_flat)
        .reshape(
            len(x_train_sequence_raw),
            24,
            1,
        )
        .astype(np.float32)
    )

    validation_sequence_scaled = (
        sequence_scaler
        .transform(validation_sequence_flat)
        .reshape(
            len(x_validation_sequence_raw),
            24,
            1,
        )
        .astype(np.float32)
    )

    # ========================================================
    # Nhánh feature bổ sung
    # ========================================================

    if static_feature_names:
        static_scaler = StandardScaler()

        x_train_static_raw = (
            dataset.x_train[
                static_feature_names
            ]
            .to_numpy(
                dtype=np.float32,
                copy=True,
            )
        )

        x_validation_static_raw = (
            dataset.x_validation[
                static_feature_names
            ]
            .to_numpy(
                dtype=np.float32,
                copy=True,
            )
        )

        x_train_static = (
            static_scaler
            .fit_transform(x_train_static_raw)
            .astype(np.float32)
        )

        x_validation_static = (
            static_scaler
            .transform(
                x_validation_static_raw
            )
            .astype(np.float32)
        )

    else:
        # F0 chỉ có chuỗi 24 lag, không có nhánh static.
        x_train_static = None
        x_validation_static = None

    # ========================================================
    # Target
    # ========================================================

    y_scaler = StandardScaler()

    y_train_raw = np.asarray(
        dataset.y_train,
        dtype=np.float32,
    ).reshape(-1, 1)

    y_validation_raw = np.asarray(
        dataset.y_validation,
        dtype=np.float32,
    ).reshape(-1, 1)

    y_train_scaled = (
        y_scaler
        .fit_transform(y_train_raw)
        .reshape(-1)
        .astype(np.float32)
    )

    y_validation_scaled = (
        y_scaler
        .transform(y_validation_raw)
        .reshape(-1)
        .astype(np.float32)
    )

    # ========================================================
    # Validation
    # ========================================================

    if train_sequence_scaled.shape[1:] != (24, 1):
        raise RuntimeError(
            "Sequence train phải có shape "
            "(samples, 24, 1), nhưng nhận được "
            f"{train_sequence_scaled.shape}."
        )

    if validation_sequence_scaled.shape[1:] != (24, 1):
        raise RuntimeError(
            "Sequence validation phải có shape "
            "(samples, 24, 1), nhưng nhận được "
            f"{validation_sequence_scaled.shape}."
        )

    arrays_to_check = [
        train_sequence_scaled,
        validation_sequence_scaled,
        y_train_scaled,
        y_validation_scaled,
    ]

    if x_train_static is not None:
        arrays_to_check.extend(
            [
                x_train_static,
                x_validation_static,
            ]
        )

    for array in arrays_to_check:
        if not np.isfinite(array).all():
            raise ValueError(
                "Dữ liệu LSTM chứa NaN hoặc giá trị vô hạn "
                "sau khi chuẩn hóa."
            )

    print(
        f"LSTM-{dataset.feature_set}: "
        f"sequence shape={train_sequence_scaled.shape}, "
        f"static features={len(static_feature_names)}"
    )

    return LSTMTuningDataset(
        feature_set=dataset.feature_set,
        sequence_feature_names=(
            sequence_feature_names
        ),
        static_feature_names=(
            static_feature_names
        ),
        x_train_sequence=(
            train_sequence_scaled
        ),
        x_validation_sequence=(
            validation_sequence_scaled
        ),
        x_train_static=x_train_static,
        x_validation_static=(
            x_validation_static
        ),
        y_train=y_train_scaled,
        y_validation=y_validation_scaled,
        y_scaler=y_scaler,
        train_rows=len(
            train_sequence_scaled
        ),
        validation_rows=len(
            validation_sequence_scaled
        ),
        sequence_length=24,
        static_feature_count=len(
            static_feature_names
        ),
    )

def make_lstm_inputs(
    sequence_data: np.ndarray,
    static_data: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """
    Tạo dictionary đầu vào phù hợp với Keras Functional API.
    """

    inputs = {
        "sequence_input": sequence_data,
    }

    if static_data is not None:
        inputs["static_input"] = static_data

    return inputs


# ============================================================
# 17. Callback Optuna pruning cho Keras
# ============================================================

def create_optuna_pruning_callback(
    trial: Any,
) -> Any:
    """
    Tạo callback báo cáo val_loss sau mỗi epoch cho Optuna.

    Nếu pruner quyết định trial không còn triển vọng,
    trial sẽ được dừng sớm.
    """

    tensorflow = require_module(
        module_name="tensorflow",
        install_name="tensorflow",
    )

    optuna = require_module("optuna")

    class OptunaPruningCallback(
        tensorflow.keras.callbacks.Callback
    ):
        def __init__(
            self,
            optuna_trial: Any,
        ) -> None:
            super().__init__()
            self.optuna_trial = optuna_trial

        def on_epoch_end(
            self,
            epoch: int,
            logs: dict[str, Any] | None = None,
        ) -> None:
            logs = logs or {}

            current_value = logs.get(
                "val_loss"
            )

            if current_value is None:
                return

            current_value = float(
                current_value
            )

            self.optuna_trial.report(
                current_value,
                step=epoch,
            )

            if self.optuna_trial.should_prune():
                raise optuna.TrialPruned(
                    "Trial LSTM bị prune tại "
                    f"epoch {epoch + 1}; "
                    f"val_loss={current_value:.8f}."
                )

    return OptunaPruningCallback(trial)


# ============================================================
# 18. Xây dựng mô hình LSTM
# ============================================================

def build_lstm_model(
    trial: Any,
    sequence_input_shape: tuple[int, int],
    static_feature_count: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """
    Mô hình hai nhánh:

        Chuỗi 24 giờ -> LSTM
        Feature bổ sung -> Dense
        Concatenate -> Dense -> forecast
    """

    tensorflow = require_module(
        module_name="tensorflow",
        install_name="tensorflow",
    )

    keras = tensorflow.keras

    keras.utils.set_random_seed(seed)

    lstm_layers = trial.suggest_int(
        "lstm_layers",
        1,
        2,
    )

    units_layer_1 = trial.suggest_int(
        "units_layer_1",
        32,
        256,
        step=32,
    )

    lstm_dropout = trial.suggest_float(
        "lstm_dropout",
        0.0,
        0.4,
        step=0.1,
    )

    dense_units = trial.suggest_int(
        "dense_units",
        16,
        128,
        step=16,
    )

    dense_dropout = trial.suggest_float(
        "dense_dropout",
        0.0,
        0.4,
        step=0.1,
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        5e-3,
        log=True,
    )

    l2_regularization = trial.suggest_float(
        "l2_regularization",
        1e-8,
        1e-3,
        log=True,
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["adam", "rmsprop"],
    )

    clipnorm = trial.suggest_float(
        "clipnorm",
        0.5,
        5.0,
        log=True,
    )

    regularizer = keras.regularizers.l2(
        l2_regularization
    )

    # ========================================================
    # Nhánh sequence
    # ========================================================

    sequence_input = keras.Input(
        shape=sequence_input_shape,
        name="sequence_input",
    )

    if lstm_layers == 1:
        sequence_branch = keras.layers.LSTM(
            units=units_layer_1,
            return_sequences=False,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizer,
            name="lstm_1",
        )(sequence_input)

        units_layer_2 = None

    else:
        units_layer_2 = trial.suggest_int(
            "units_layer_2",
            32,
            192,
            step=32,
        )

        sequence_branch = keras.layers.LSTM(
            units=units_layer_1,
            return_sequences=True,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizer,
            name="lstm_1",
        )(sequence_input)

        sequence_branch = keras.layers.LSTM(
            units=units_layer_2,
            return_sequences=False,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizer,
            name="lstm_2",
        )(sequence_branch)

    if lstm_dropout > 0:
        sequence_branch = keras.layers.Dropout(
            lstm_dropout,
            name="lstm_output_dropout",
        )(sequence_branch)

    model_inputs: list[Any] = [
        sequence_input
    ]

    combined = sequence_branch

    # ========================================================
    # Nhánh static của F1–F4
    # ========================================================

    static_dense_units = None

    if static_feature_count > 0:
        static_dense_units = (
            trial.suggest_int(
                "static_dense_units",
                8,
                64,
                step=8,
            )
        )

        static_input = keras.Input(
            shape=(static_feature_count,),
            name="static_input",
        )

        static_branch = keras.layers.Dense(
            static_dense_units,
            activation="relu",
            kernel_regularizer=regularizer,
            name="static_dense",
        )(static_input)

        combined = keras.layers.Concatenate(
            name="merge_sequence_static"
        )(
            [
                sequence_branch,
                static_branch,
            ]
        )

        model_inputs.append(static_input)

    # ========================================================
    # Dự báo
    # ========================================================

    combined = keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizer,
        name="combined_dense",
    )(combined)

    if dense_dropout > 0:
        combined = keras.layers.Dropout(
            dense_dropout,
            name="combined_dropout",
        )(combined)

    output = keras.layers.Dense(
        1,
        activation="linear",
        name="load_forecast",
    )(combined)

    model = keras.Model(
        inputs=model_inputs,
        outputs=output,
        name="lstm_load_forecaster",
    )

    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=clipnorm,
        )
    else:
        optimizer = keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            clipnorm=clipnorm,
        )

    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=[
            keras.metrics.RootMeanSquaredError(
                name="rmse"
            )
        ],
    )

    architecture_info = {
        "lstm_layers": lstm_layers,
        "units_layer_1": units_layer_1,
        "units_layer_2": units_layer_2,
        "static_dense_units": (
            static_dense_units
        ),
        "sequence_length": (
            sequence_input_shape[0]
        ),
        "sequence_features_per_step": (
            sequence_input_shape[1]
        ),
        "static_feature_count": (
            static_feature_count
        ),
        "trainable_parameters": int(
            model.count_params()
        ),
    }

    return model, architecture_info

# ============================================================
# 19. Objective LSTM
# ============================================================

def create_lstm_objective(
    dataset: TuningDataset,
    seed: int,
    max_epochs: int = LSTM_MAX_EPOCHS,
    patience: int = LSTM_EARLY_STOPPING_PATIENCE,
) -> Callable[[Any], float]:
    """
    Tạo Optuna objective cho mô hình LSTM.

    Thiết kế đầu vào
    ----------------
    Nhánh sequence:
        load_lag_23, ..., load_lag_0
        shape = (samples, 24, 1)

    Nhánh static:
        các feature còn lại của F1–F4
        shape = (samples, static_feature_count)

    Quy trình mỗi trial
    -------------------
    1. Chuẩn hóa dữ liệu bằng scaler chỉ fit trên train.
    2. Xây dựng kiến trúc LSTM theo siêu tham số Optuna.
    3. Model chỉ học trên tập train.
    4. Validation được dùng cho:
       - val_loss;
       - early stopping;
       - pruning;
       - tính objective RMSE.
    5. Dự đoán validation được inverse transform.
    6. Objective là validation RMSE trên thang đo phụ tải gốc.
    7. Tập test tuyệt đối không được sử dụng.
    """

    # Hàm này chuẩn bị dữ liệu một lần cho toàn bộ study.
    # Các scaler chỉ được fit trên train.
    lstm_dataset = prepare_lstm_dataset(
        dataset=dataset,
    )

    def objective(trial: Any) -> float:
        """
        Một Optuna trial cho LSTM.
        """

        tensorflow = require_module(
            module_name="tensorflow",
            install_name="tensorflow",
        )

        keras = tensorflow.keras

        # Xóa graph/model của trial trước để giảm tích lũy RAM.
        keras.backend.clear_session()
        gc.collect()

        # Mỗi trial có seed khác nhau nhưng vẫn tái lập được.
        trial_seed = seed + int(trial.number)

        keras.utils.set_random_seed(
            trial_seed
        )

        # Batch size cũng là một siêu tham số.
        batch_size = trial.suggest_categorical(
            "batch_size",
            LSTM_BATCH_SIZE_CHOICES,
        )

        sequence_input_shape = (
            int(
                lstm_dataset
                .x_train_sequence
                .shape[1]
            ),
            int(
                lstm_dataset
                .x_train_sequence
                .shape[2]
            ),
        )

        if sequence_input_shape != (24, 1):
            raise RuntimeError(
                "Đầu vào sequence của LSTM phải có shape "
                "(24, 1), nhưng nhận được "
                f"{sequence_input_shape}."
            )

        # Tạo dictionary input đúng tên layer của Keras.
        train_inputs = make_lstm_inputs(
            sequence_data=(
                lstm_dataset
                .x_train_sequence
            ),
            static_data=(
                lstm_dataset
                .x_train_static
            ),
        )

        validation_inputs = make_lstm_inputs(
            sequence_data=(
                lstm_dataset
                .x_validation_sequence
            ),
            static_data=(
                lstm_dataset
                .x_validation_static
            ),
        )

        model = None
        history = None

        start_time = time.perf_counter()

        try:
            # Tạo model hai nhánh:
            # sequence -> LSTM
            # static -> Dense
            # concatenate -> Dense -> output
            model, architecture_info = build_lstm_model(
                trial=trial,
                sequence_input_shape=(
                    sequence_input_shape
                ),
                static_feature_count=(
                    lstm_dataset
                    .static_feature_count
                ),
                seed=trial_seed,
            )

            # Dừng khi validation loss không còn cải thiện.
            early_stopping = (
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=patience,
                    min_delta=LSTM_MIN_DELTA,
                    restore_best_weights=True,
                    verbose=1,
                )
            )

            # Dừng trial nếu loss trở thành NaN.
            terminate_on_nan = (
                keras.callbacks.TerminateOnNaN()
            )

            # Cho Optuna quan sát val_loss từng epoch
            # và prune trial kém triển vọng.
            pruning_callback = (
                create_optuna_pruning_callback(
                    trial=trial,
                )
            )

            history = model.fit(
                train_inputs,
                lstm_dataset.y_train,
                validation_data=(
                    validation_inputs,
                    lstm_dataset.y_validation,
                ),
                epochs=max_epochs,
                batch_size=batch_size,

                # Với dữ liệu chuỗi thời gian,
                # không shuffle thứ tự các mẫu.
                shuffle=False,

                callbacks=[
                    early_stopping,
                    terminate_on_nan,
                    pruning_callback,
                ],

                # Mỗi epoch hiển thị một dòng.
                verbose=2,
            )

            # Dự đoán trên validation đã chuẩn hóa.
            scaled_predictions = model.predict(
                validation_inputs,
                batch_size=batch_size,
                verbose=0,
            )

            scaled_predictions = np.asarray(
                scaled_predictions,
                dtype=np.float32,
            ).reshape(-1, 1)

            if (
                len(scaled_predictions)
                != lstm_dataset.validation_rows
            ):
                raise RuntimeError(
                    "Số dự đoán validation không khớp "
                    "với số dòng validation. "
                    f"Nhận được {len(scaled_predictions)}, "
                    f"mong đợi "
                    f"{lstm_dataset.validation_rows}."
                )

            if not np.isfinite(
                scaled_predictions
            ).all():
                raise ValueError(
                    "Dự đoán LSTM chứa NaN hoặc "
                    "giá trị vô hạn."
                )

            # Chuyển dự đoán về đơn vị phụ tải ban đầu.
            validation_predictions = (
                lstm_dataset
                .y_scaler
                .inverse_transform(
                    scaled_predictions
                )
                .reshape(-1)
            )

            # Target validation gốc, chưa scale.
            validation_targets = np.asarray(
                dataset.y_validation,
                dtype=np.float64,
            ).reshape(-1)

            if (
                validation_predictions.shape
                != validation_targets.shape
            ):
                raise RuntimeError(
                    "Kích thước prediction và target "
                    "validation không khớp. "
                    f"Prediction: "
                    f"{validation_predictions.shape}; "
                    f"target: "
                    f"{validation_targets.shape}."
                )

            validation_rmse = calculate_rmse(
                y_true=validation_targets,
                y_pred=validation_predictions,
            )

            elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            # Lấy lịch sử train/validation loss.
            training_loss_history = (
                history.history.get(
                    "loss",
                    [],
                )
            )

            validation_loss_history = (
                history.history.get(
                    "val_loss",
                    [],
                )
            )

            epochs_trained = len(
                training_loss_history
            )

            if epochs_trained <= 0:
                raise RuntimeError(
                    "LSTM không hoàn tất epoch nào."
                )

            if validation_loss_history:
                best_epoch_index = int(
                    np.argmin(
                        validation_loss_history
                    )
                )

                # Epoch hiển thị theo số đếm bắt đầu từ 1.
                best_epoch = (
                    best_epoch_index + 1
                )

                best_scaled_val_loss = float(
                    validation_loss_history[
                        best_epoch_index
                    ]
                )

                best_scaled_val_rmse = float(
                    math.sqrt(
                        max(
                            best_scaled_val_loss,
                            0.0,
                        )
                    )
                )
            else:
                best_epoch = epochs_trained
                best_scaled_val_loss = None
                best_scaled_val_rmse = None

            final_training_loss = float(
                training_loss_history[-1]
            )

            if validation_loss_history:
                final_validation_loss = float(
                    validation_loss_history[-1]
                )
            else:
                final_validation_loss = None

            # Lưu thông tin của trial để xuất ra CSV/JSON.
            trial.set_user_attr(
                "model",
                "lstm",
            )

            trial.set_user_attr(
                "feature_set",
                dataset.feature_set,
            )

            trial.set_user_attr(
                "validation_rmse",
                float(validation_rmse),
            )

            trial.set_user_attr(
                "epochs_trained",
                int(epochs_trained),
            )

            trial.set_user_attr(
                "best_epoch",
                int(best_epoch),
            )

            trial.set_user_attr(
                "best_scaled_val_loss",
                best_scaled_val_loss,
            )

            trial.set_user_attr(
                "best_scaled_val_rmse",
                best_scaled_val_rmse,
            )

            trial.set_user_attr(
                "final_training_loss",
                final_training_loss,
            )

            trial.set_user_attr(
                "final_validation_loss",
                final_validation_loss,
            )

            trial.set_user_attr(
                "fit_and_predict_seconds",
                float(elapsed_seconds),
            )

            trial.set_user_attr(
                "train_rows",
                int(
                    lstm_dataset.train_rows
                ),
            )

            trial.set_user_attr(
                "validation_rows",
                int(
                    lstm_dataset.validation_rows
                ),
            )

            trial.set_user_attr(
                "timesteps",
                int(
                    lstm_dataset
                    .sequence_length
                ),
            )

            trial.set_user_attr(
                "sequence_feature_count",
                int(
                    len(
                        lstm_dataset
                        .sequence_feature_names
                    )
                ),
            )

            trial.set_user_attr(
                "static_feature_count",
                int(
                    lstm_dataset
                    .static_feature_count
                ),
            )

            trial.set_user_attr(
                "sequence_feature_names",
                list(
                    lstm_dataset
                    .sequence_feature_names
                ),
            )

            trial.set_user_attr(
                "static_feature_names",
                list(
                    lstm_dataset
                    .static_feature_names
                ),
            )

            trial.set_user_attr(
                "trainable_parameters",
                int(
                    architecture_info[
                        "trainable_parameters"
                    ]
                ),
            )

            trial.set_user_attr(
                "sequence_input_shape",
                list(
                    sequence_input_shape
                ),
            )

            trial.set_user_attr(
                "uses_static_branch",
                bool(
                    lstm_dataset
                    .static_feature_count
                    > 0
                ),
            )

            trial.set_user_attr(
                "batch_size",
                int(batch_size),
            )

            trial.set_user_attr(
                "max_epochs",
                int(max_epochs),
            )

            trial.set_user_attr(
                "early_stopping_patience",
                int(patience),
            )

            trial.set_user_attr(
                "early_stopping_min_delta",
                float(
                    LSTM_MIN_DELTA
                ),
            )

            trial.set_user_attr(
                "restore_best_weights",
                True,
            )

            trial.set_user_attr(
                "shuffle",
                False,
            )

            trial.set_user_attr(
                "x_scaler_fit_on",
                "train_only",
            )

            trial.set_user_attr(
                "y_scaler_fit_on",
                "train_only",
            )

            trial.set_user_attr(
                "test_data_used",
                False,
            )

            return validate_objective_value(
                rmse=validation_rmse,
                model_name="lstm",
                feature_set=(
                    dataset.feature_set
                ),
            )

        finally:
            # Giải phóng model và graph sau mỗi trial.
            if history is not None:
                del history

            if model is not None:
                del model

            keras.backend.clear_session()
            gc.collect()

    return objective


# ============================================================
# 20. Factory objective hoàn chỉnh
# ============================================================

def create_model_objective(
    model_name: str,
    dataset: TuningDataset,
    args: argparse.Namespace,
) -> Callable[[Any], float]:
    """
    Trả về objective tương ứng cho RF, XGB, LGBM hoặc LSTM.
    """

    normalized_model_name = (
        model_name.strip().lower()
    )

    if normalized_model_name in {
        "rf",
        "xgb",
        "lgbm",
    }:
        return create_tree_model_objective(
            model_name=(
                normalized_model_name
            ),
            dataset=dataset,
            args=args,
        )

    if normalized_model_name == "lstm":
        return create_lstm_objective(
            dataset=dataset,
            seed=args.seed,
            max_epochs=(
                LSTM_MAX_EPOCHS
            ),
            patience=(
                LSTM_EARLY_STOPPING_PATIENCE
            ),
        )

    raise ValueError(
        f"Không hỗ trợ mô hình: "
        f"{model_name}"
    )


# ============================================================
# 21. Chuyển dữ liệu sang kiểu có thể ghi JSON
# ============================================================

def make_json_serializable(
    value: Any,
) -> Any:
    """
    Chuyển NumPy, Path và các kiểu đặc biệt sang kiểu JSON.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        (
            np.integer,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):
        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return str(value)


def save_json_file(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Ghi JSON theo cách an toàn.

    Ghi vào file tạm trước rồi thay thế file đích để hạn chế
    file kết quả bị hỏng nếu chương trình dừng giữa lúc ghi.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    serializable_data = (
        make_json_serializable(data)
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    temporary_path.replace(output_path)


# ============================================================
# 22. Tạo tên study
# ============================================================

def build_study_name(
    model_name: str,
    feature_set: str,
) -> str:
    """
    Tạo tên study ổn định cho Optuna storage.
    """

    return (
        f"load_forecasting_"
        f"{model_name.lower()}_"
        f"{feature_set.upper()}"
    )


# ============================================================
# 23. Tạo Optuna study
# ============================================================

def create_optuna_study(
    model_name: str,
    feature_set: str,
    args: argparse.Namespace,
) -> Any:
    """
    Tạo Optuna study với:
        - direction = minimize
        - TPESampler
        - MedianPruner

    TPESampler được đặt seed để tăng khả năng tái lập.
    """

    optuna = require_module("optuna")

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(
            2,
            min(5, args.n_trials // 3),
        ),
        n_warmup_steps=5,
        interval_steps=1,
    )

    study_name = build_study_name(
        model_name=model_name,
        feature_set=feature_set,
    )

    if (
        args.resume
        and args.storage is None
    ):
        warnings.warn(
            "--resume chỉ có tác dụng khi "
            "--storage được cung cấp.",
            RuntimeWarning,
        )

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=bool(
            args.resume
            and args.storage is not None
        ),
    )

    study.set_user_attr(
        "model",
        model_name,
    )

    study.set_user_attr(
        "feature_set",
        feature_set,
    )

    study.set_user_attr(
        "seed",
        int(args.seed),
    )

    study.set_user_attr(
        "objective_metric",
        "validation_rmse",
    )

    study.set_user_attr(
        "test_data_used",
        False,
    )

    return study


# ============================================================
# 24. Lưu lịch sử trial
# ============================================================

def save_trials_dataframe(
    study: Any,
    output_path: Path,
) -> None:
    """
    Lưu toàn bộ lịch sử trial thành CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trials_df = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )
    )

    trials_df.to_csv(
        output_path,
        index=False,
    )


# ============================================================
# 25. Tạo báo cáo best params
# ============================================================

def build_best_params_report(
    study: Any,
    model_name: str,
    feature_set: str,
    dataset: TuningDataset,
    args: argparse.Namespace,
    study_elapsed_seconds: float,
) -> dict[str, Any]:
    """
    Tạo nội dung file:
        <model>_<feature_set>_best_params.json
    """

    optuna = require_module("optuna")

    completed_trials = [
        trial
        for trial in study.trials
        if (
            trial.state
            == optuna.trial.TrialState.COMPLETE
        )
    ]

    pruned_trials = [
        trial
        for trial in study.trials
        if (
            trial.state
            == optuna.trial.TrialState.PRUNED
        )
    ]

    failed_trials = [
        trial
        for trial in study.trials
        if (
            trial.state
            == optuna.trial.TrialState.FAIL
        )
    ]

    if not completed_trials:
        raise RuntimeError(
            f"{model_name.upper()}-{feature_set} "
            "không có trial hoàn tất thành công."
        )

    best_trial = study.best_trial

    report = {
        "model": model_name,
        "feature_set": feature_set,
        "study_name": study.study_name,
        "direction": "minimize",
        "objective_metric": (
            "validation_rmse"
        ),
        "best_validation_rmse": float(
            study.best_value
        ),
        "best_trial_number": int(
            best_trial.number
        ),
        "best_params": dict(
            study.best_params
        ),
        "best_trial_user_attributes": (
            dict(best_trial.user_attrs)
        ),
        "data": {
            "train_rows": int(
                len(dataset.x_train)
            ),
            "validation_rows": int(
                len(
                    dataset.x_validation
                )
            ),
            "feature_count": int(
                len(dataset.feature_names)
            ),
            "feature_names": list(
                dataset.feature_names
            ),
            "train_start": str(
                dataset
                .train_timestamps.iloc[0]
            ),
            "train_end": str(
                dataset
                .train_timestamps.iloc[-1]
            ),
            "validation_start": str(
                dataset
                .validation_timestamps
                .iloc[0]
            ),
            "validation_end": str(
                dataset
                .validation_timestamps
                .iloc[-1]
            ),
        },
        "tuning": {
            "requested_trials": int(
                args.n_trials
            ),
            "total_trials_in_study": int(
                len(study.trials)
            ),
            "completed_trials": int(
                len(completed_trials)
            ),
            "pruned_trials": int(
                len(pruned_trials)
            ),
            "failed_trials": int(
                len(failed_trials)
            ),
            "seed": int(args.seed),
            "study_jobs_requested": int(
                args.study_jobs
            ),
            "model_jobs": int(
                args.n_jobs_model
            ),
            "timeout_seconds": (
                args.timeout
            ),
            "study_elapsed_seconds": float(
                study_elapsed_seconds
            ),
            "storage": args.storage,
            "resume": bool(args.resume),
        },
        "data_usage": {
            "model_fit_data": "train",
            "hyperparameter_selection_data": (
                "validation"
            ),
            "test_data_used": False,
        },
    }

    if model_name == "lstm":
        report["lstm_training"] = {
            "input_representation": (
                "one timestep containing "
                "all engineered features"
            ),
            "timesteps": 1,
            "max_epochs": int(
                LSTM_MAX_EPOCHS
            ),
            "early_stopping_monitor": (
                "val_loss"
            ),
            "early_stopping_patience": int(
                LSTM_EARLY_STOPPING_PATIENCE
            ),
            "early_stopping_min_delta": float(
                LSTM_MIN_DELTA
            ),
            "restore_best_weights": True,
            "shuffle": False,
            "x_scaler_fit_on": "train_only",
            "y_scaler_fit_on": "train_only",
        }

    return report


# ============================================================
# 26. Chạy một configuration
# ============================================================

def run_single_configuration(
    model_name: str,
    feature_set: str,
    dataset: TuningDataset,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Chạy tuning cho một cặp model–feature set.

    Ví dụ:
        RF-F0
        XGB-F3
        LSTM-F4
    """

    optuna = require_module("optuna")

    configuration_name = (
        f"{model_name.upper()}-"
        f"{feature_set}"
    )

    print()
    print("=" * 72)
    print(
        f"BẮT ĐẦU TUNING: "
        f"{configuration_name}"
    )
    print("=" * 72)

    print_dataset_summary(dataset)

    objective = create_model_objective(
        model_name=model_name,
        dataset=dataset,
        args=args,
    )

    study = create_optuna_study(
        model_name=model_name,
        feature_set=feature_set,
        args=args,
    )

    # Không chạy nhiều trial LSTM cùng lúc vì TensorFlow có thể
    # cạnh tranh GPU/RAM và gây thiếu bộ nhớ.
    if model_name == "lstm":
        effective_study_jobs = 1

        if args.study_jobs != 1:
            print(
                "LSTM sử dụng study_jobs=1 "
                "để tránh xung đột bộ nhớ."
            )
    else:
        effective_study_jobs = (
            args.study_jobs
        )

    initial_trial_count = len(
        study.trials
    )

    start_time = time.perf_counter()

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=effective_study_jobs,
        show_progress_bar=(
            args.show_progress_bar
        ),
        gc_after_trial=True,
        catch=(
            ValueError,
            RuntimeError,
            FloatingPointError,
        ),
    )

    study_elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    new_trial_count = (
        len(study.trials)
        - initial_trial_count
    )

    report = build_best_params_report(
        study=study,
        model_name=model_name,
        feature_set=feature_set,
        dataset=dataset,
        args=args,
        study_elapsed_seconds=(
            study_elapsed_seconds
        ),
    )

    report["tuning"][
        "trials_added_this_run"
    ] = int(new_trial_count)

    best_params_path = (
        output_dir
        / (
            f"{model_name}_"
            f"{feature_set}_"
            f"best_params.json"
        )
    )

    trials_path = (
        output_dir
        / (
            f"{model_name}_"
            f"{feature_set}_"
            f"trials.csv"
        )
    )

    save_json_file(
        data=report,
        output_path=best_params_path,
    )

    save_trials_dataframe(
        study=study,
        output_path=trials_path,
    )

    print()
    print(
        f"Hoàn tất {configuration_name}"
    )

    print(
        "Best validation RMSE: "
        f"{study.best_value:,.6f}"
    )

    print(
        f"Best trial          : "
        f"{study.best_trial.number}"
    )

    print(
        f"Best params         : "
        f"{study.best_params}"
    )

    print(
        f"Đã lưu JSON         : "
        f"{best_params_path}"
    )

    print(
        f"Đã lưu trials       : "
        f"{trials_path}"
    )

    return report


# ============================================================
# 27. Lưu báo cáo tổng hợp
# ============================================================

def save_tuning_summary(
    configuration_reports: list[
        dict[str, Any]
    ],
    output_dir: Path,
    args: argparse.Namespace,
    total_elapsed_seconds: float,
) -> None:
    """
    Lưu báo cáo tổng hợp của tất cả configuration.
    """

    summary_rows = []

    for report in configuration_reports:
        summary_rows.append(
            {
                "model": report["model"],
                "feature_set": (
                    report["feature_set"]
                ),
                "best_validation_rmse": (
                    report[
                        "best_validation_rmse"
                    ]
                ),
                "best_trial_number": (
                    report[
                        "best_trial_number"
                    ]
                ),
                "completed_trials": (
                    report["tuning"][
                        "completed_trials"
                    ]
                ),
                "pruned_trials": (
                    report["tuning"][
                        "pruned_trials"
                    ]
                ),
                "failed_trials": (
                    report["tuning"][
                        "failed_trials"
                    ]
                ),
                "elapsed_seconds": (
                    report["tuning"][
                        "study_elapsed_seconds"
                    ]
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=[
                "model",
                "feature_set",
            ],
            kind="stable",
        ).reset_index(drop=True)

    summary_csv_path = (
        output_dir
        / "tuning_summary.csv"
    )

    summary_df.to_csv(
        summary_csv_path,
        index=False,
    )

    summary_json = {
        "models": args.models,
        "feature_sets": (
            args.feature_sets
        ),
        "configuration_count": int(
            len(configuration_reports)
        ),
        "requested_trials_per_configuration": (
            int(args.n_trials)
        ),
        "total_requested_tuning_runs": int(
            len(args.models)
            * len(args.feature_sets)
            * args.n_trials
        ),
        "total_elapsed_seconds": float(
            total_elapsed_seconds
        ),
        "test_data_used": False,
        "results": configuration_reports,
    }

    summary_json_path = (
        output_dir
        / "tuning_summary.json"
    )

    save_json_file(
        data=summary_json,
        output_path=summary_json_path,
    )

    print()
    print(
        f"Báo cáo CSV tổng hợp : "
        f"{summary_csv_path}"
    )

    print(
        f"Báo cáo JSON tổng hợp: "
        f"{summary_json_path}"
    )


# ============================================================
# 28. Kiểm tra timestamp giữa feature set
# ============================================================

def validate_tuning_period_alignment(
    datasets: dict[str, TuningDataset],
) -> None:
    """
    Kiểm tra F0–F4 dùng cùng giai đoạn train và validation.
    """

    if not datasets:
        raise ValueError(
            "Không có dataset để kiểm tra."
        )

    reference_feature_set = next(
        iter(datasets)
    )

    reference_dataset = datasets[
        reference_feature_set
    ]

    reference_train = (
        reference_dataset
        .train_timestamps
        .reset_index(drop=True)
    )

    reference_validation = (
        reference_dataset
        .validation_timestamps
        .reset_index(drop=True)
    )

    reference_y_train = np.asarray(
        reference_dataset.y_train
    )

    reference_y_validation = np.asarray(
        reference_dataset.y_validation
    )

    for (
        feature_set,
        current_dataset,
    ) in datasets.items():
        current_train = (
            current_dataset
            .train_timestamps
            .reset_index(drop=True)
        )

        current_validation = (
            current_dataset
            .validation_timestamps
            .reset_index(drop=True)
        )

        if not current_train.equals(
            reference_train
        ):
            raise ValueError(
                f"Timestamp train của "
                f"{feature_set} không trùng với "
                f"{reference_feature_set}."
            )

        if not current_validation.equals(
            reference_validation
        ):
            raise ValueError(
                f"Timestamp validation của "
                f"{feature_set} không trùng với "
                f"{reference_feature_set}."
            )

        if not np.array_equal(
            np.asarray(
                current_dataset.y_train
            ),
            reference_y_train,
        ):
            raise ValueError(
                f"Target train của {feature_set} "
                f"không trùng với "
                f"{reference_feature_set}."
            )

        if not np.array_equal(
            np.asarray(
                current_dataset.y_validation
            ),
            reference_y_validation,
        ):
            raise ValueError(
                f"Target validation của "
                f"{feature_set} không trùng với "
                f"{reference_feature_set}."
            )

    print(
        "Đã xác nhận các feature set có cùng "
        "timestamp và target trong train/validation."
    )


# ============================================================
# 29. Đọc trước các feature set
# ============================================================

def load_selected_datasets(
    args: argparse.Namespace,
) -> dict[str, TuningDataset]:
    """
    Đọc mỗi feature set một lần và tái sử dụng cho các mô hình.
    """

    datasets: dict[
        str,
        TuningDataset,
    ] = {}

    print()
    print("ĐANG ĐỌC DỮ LIỆU TRAIN/VALIDATION")
    print("-" * 72)

    for feature_set in args.feature_sets:
        dataset = load_tuning_dataset(
            split_dir=args.split_dir,
            feature_set=feature_set,
            float_dtype=args.float_dtype,
            strict_hourly=(
                args.strict_hourly
            ),
        )

        datasets[feature_set] = dataset

        print(
            f"Đã đọc {feature_set}: "
            f"{len(dataset.x_train):,} train, "
            f"{len(dataset.x_validation):,} validation, "
            f"{len(dataset.feature_names):,} features."
        )

    validate_tuning_period_alignment(
        datasets
    )

    return datasets


# ============================================================
# 30. Main hoàn chỉnh
# ============================================================

def main() -> None:
    """
    Chạy toàn bộ model × feature set.

    Với cấu hình mặc định:
        4 models × 5 feature sets
        = 20 configurations.

    Nếu n_trials=10:
        20 × 10 = 200 tuning runs.
    """

    args = parse_arguments()

    set_global_seed(args.seed)

    check_required_libraries(
        selected_models=args.models,
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    number_of_configurations = (
        len(args.models)
        * len(args.feature_sets)
    )

    total_requested_runs = (
        number_of_configurations
        * args.n_trials
    )

    print("=" * 72)
    print("TỐI ƯU SIÊU THAM SỐ BẰNG OPTUNA")
    print("=" * 72)

    print(
        f"Split directory       : "
        f"{args.split_dir}"
    )

    print(
        f"Output directory      : "
        f"{args.output_dir}"
    )

    print(
        f"Models                : "
        f"{args.models}"
    )

    print(
        f"Feature sets          : "
        f"{args.feature_sets}"
    )

    print(
        f"Trials/configuration  : "
        f"{args.n_trials}"
    )

    print(
        f"Configurations        : "
        f"{number_of_configurations}"
    )

    print(
        f"Requested tuning runs : "
        f"{total_requested_runs}"
    )

    print(
        f"Seed                  : "
        f"{args.seed}"
    )

    print(
        "Objective             : "
        "validation RMSE"
    )

    print(
        "Model fit data        : train only"
    )

    print(
        "Hyperparameter choice : validation only"
    )

    print(
        "Test data used        : False"
    )

    datasets = load_selected_datasets(
        args
    )

    configuration_reports: list[
        dict[str, Any]
    ] = []

    failed_configurations: list[
        dict[str, str]
    ] = []

    total_start_time = (
        time.perf_counter()
    )

    configuration_index = 0

    for model_name in args.models:
        for feature_set in args.feature_sets:
            configuration_index += 1

            print()
            print(
                f"[{configuration_index}/"
                f"{number_of_configurations}] "
                f"{model_name.upper()}-"
                f"{feature_set}"
            )

            try:
                report = (
                    run_single_configuration(
                        model_name=model_name,
                        feature_set=feature_set,
                        dataset=datasets[
                            feature_set
                        ],
                        output_dir=output_dir,
                        args=args,
                    )
                )

                configuration_reports.append(
                    report
                )

            except KeyboardInterrupt:
                print()
                print(
                    "Người dùng đã dừng chương trình."
                )
                raise

            except Exception as error:
                error_message = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                failed_configurations.append(
                    {
                        "model": model_name,
                        "feature_set": (
                            feature_set
                        ),
                        "error": error_message,
                    }
                )

                print()
                print(
                    f"LỖI tại "
                    f"{model_name.upper()}-"
                    f"{feature_set}: "
                    f"{error_message}"
                )

                warnings.warn(
                    "Bỏ qua configuration lỗi và "
                    "tiếp tục configuration tiếp theo.",
                    RuntimeWarning,
                )

            finally:
                gc.collect()

    total_elapsed_seconds = (
        time.perf_counter()
        - total_start_time
    )

    if configuration_reports:
        save_tuning_summary(
            configuration_reports=(
                configuration_reports
            ),
            output_dir=output_dir,
            args=args,
            total_elapsed_seconds=(
                total_elapsed_seconds
            ),
        )

    if failed_configurations:
        failure_path = (
            output_dir
            / "failed_configurations.json"
        )

        save_json_file(
            data={
                "failure_count": len(
                    failed_configurations
                ),
                "failures": (
                    failed_configurations
                ),
            },
            output_path=failure_path,
        )

        print()
        print(
            f"Có {len(failed_configurations)} "
            "configuration bị lỗi."
        )

        print(
            f"Chi tiết lỗi: {failure_path}"
        )

    print()
    print("=" * 72)
    print("HOÀN TẤT TỐI ƯU SIÊU THAM SỐ")
    print("=" * 72)

    print(
        f"Configuration thành công: "
        f"{len(configuration_reports)}/"
        f"{number_of_configurations}"
    )

    print(
        f"Configuration thất bại : "
        f"{len(failed_configurations)}"
    )

    print(
        f"Tổng thời gian          : "
        f"{total_elapsed_seconds:,.2f} giây"
    )

    print(
        f"Kết quả được lưu tại    : "
        f"{output_dir.resolve()}"
    )

    if not configuration_reports:
        raise RuntimeError(
            "Không có configuration nào "
            "tuning thành công."
        )


if __name__ == "__main__":
    main()