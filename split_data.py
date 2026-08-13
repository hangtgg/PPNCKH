"""
split_data.py

Mục đích:
    Chia các feature set F0–F4 thành:
        - 70% train
        - 15% validation
        - 15% test

Nguyên tắc:
    - Chia theo thứ tự thời gian.
    - Không shuffle.
    - Tất cả feature set phải có cùng số dòng.
    - Tất cả feature set phải có cùng timestamp.
    - Các giai đoạn train, validation và test phải giống nhau.

Đầu vào:
    outputs/features/F0.csv
    outputs/features/F1.csv
    outputs/features/F2.csv
    outputs/features/F3.csv
    outputs/features/F4.csv

Đầu ra:
    outputs/splits/F0_train.csv
    outputs/splits/F0_validation.csv
    outputs/splits/F0_test.csv
    ...
    outputs/splits/F4_test.csv

    outputs/splits/split_report.json

Cách chạy PowerShell:
    python split_data.py `
        --feature-dir outputs/features `
        --output-dir outputs/splits `
        --train-ratio 0.70 `
        --validation-ratio 0.15 `
        --test-ratio 0.15

Hoặc một dòng:
    python split_data.py --feature-dir outputs/features --output-dir outputs/splits --train-ratio 0.70 --validation-ratio 0.15 --test-ratio 0.15
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


FEATURE_SET_NAMES = ["F0", "F1", "F2", "F3", "F4"]


def validate_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    """
    Kiểm tra các tỷ lệ chia dữ liệu.
    """

    ratios = {
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
    }

    for ratio_name, ratio_value in ratios.items():
        if ratio_value <= 0 or ratio_value >= 1:
            raise ValueError(
                f"{ratio_name} phải lớn hơn 0 và nhỏ hơn 1. "
                f"Giá trị nhận được: {ratio_value}"
            )

    total_ratio = train_ratio + validation_ratio + test_ratio

    if not math.isclose(
        total_ratio,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Tổng train_ratio, validation_ratio và test_ratio "
            f"phải bằng 1. Hiện tại tổng bằng {total_ratio:.12f}."
        )


def load_feature_set(
    feature_path: Path,
    feature_set_name: str,
) -> pd.DataFrame:
    """
    Đọc và kiểm tra một feature set.
    """

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file {feature_set_name}: "
            f"{feature_path.resolve()}"
        )

    df = pd.read_csv(feature_path)

    if df.empty:
        raise ValueError(
            f"{feature_set_name} không có dữ liệu."
        )

    required_columns = ["timestamp", "target"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{feature_set_name} thiếu các cột bắt buộc: "
            f"{missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["target"] = pd.to_numeric(
        df["target"],
        errors="coerce",
    )

    invalid_timestamp_count = int(
        df["timestamp"].isna().sum()
    )

    invalid_target_count = int(
        df["target"].isna().sum()
    )

    if invalid_timestamp_count > 0:
        raise ValueError(
            f"{feature_set_name} có "
            f"{invalid_timestamp_count} timestamp không hợp lệ."
        )

    if invalid_target_count > 0:
        raise ValueError(
            f"{feature_set_name} có "
            f"{invalid_target_count} target không hợp lệ."
        )

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            f"{feature_set_name} chưa được sắp xếp "
            "theo thời gian tăng dần."
        )

    duplicate_count = int(
        df["timestamp"].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{feature_set_name} có "
            f"{duplicate_count} timestamp trùng lặp."
        )

    missing_value_count = int(
        df.isna().sum().sum()
    )

    if missing_value_count > 0:
        missing_by_column = (
            df.isna()
            .sum()
            .loc[lambda values: values > 0]
            .to_dict()
        )

        raise ValueError(
            f"{feature_set_name} còn "
            f"{missing_value_count} giá trị thiếu. "
            f"Chi tiết: {missing_by_column}"
        )

    return df


def load_all_feature_sets(
    feature_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Đọc toàn bộ F0–F4.
    """

    feature_sets: dict[str, pd.DataFrame] = {}

    for feature_set_name in FEATURE_SET_NAMES:
        feature_path = (
            feature_dir
            / f"{feature_set_name}.csv"
        )

        feature_sets[feature_set_name] = load_feature_set(
            feature_path=feature_path,
            feature_set_name=feature_set_name,
        )

        print(
            f"Đã đọc {feature_set_name}: "
            f"{len(feature_sets[feature_set_name]):,} dòng, "
            f"{len(feature_sets[feature_set_name].columns):,} cột."
        )

    return feature_sets


def validate_feature_set_alignment(
    feature_sets: dict[str, pd.DataFrame],
) -> pd.Series:
    """
    Xác nhận F0–F4 có cùng số dòng và cùng timestamp.
    """

    reference_name = FEATURE_SET_NAMES[0]
    reference_df = feature_sets[reference_name]

    reference_timestamps = (
        reference_df["timestamp"]
        .reset_index(drop=True)
    )

    reference_row_count = len(reference_df)

    for feature_set_name in FEATURE_SET_NAMES[1:]:
        current_df = feature_sets[feature_set_name]

        if len(current_df) != reference_row_count:
            raise ValueError(
                f"{feature_set_name} có {len(current_df):,} dòng, "
                f"nhưng {reference_name} có "
                f"{reference_row_count:,} dòng. "
                "F0–F4 phải được tạo từ cùng master feature table."
            )

        current_timestamps = (
            current_df["timestamp"]
            .reset_index(drop=True)
        )

        if not current_timestamps.equals(
            reference_timestamps
        ):
            mismatch_mask = (
                current_timestamps
                != reference_timestamps
            )

            mismatch_indices = (
                mismatch_mask[mismatch_mask]
                .index[:10]
                .tolist()
            )

            mismatch_examples = []

            for index in mismatch_indices:
                mismatch_examples.append(
                    {
                        "index": int(index),
                        reference_name: str(
                            reference_timestamps.iloc[index]
                        ),
                        feature_set_name: str(
                            current_timestamps.iloc[index]
                        ),
                    }
                )

            raise ValueError(
                f"Timestamp của {feature_set_name} "
                f"không trùng với {reference_name}. "
                f"Ví dụ sai khác: {mismatch_examples}"
            )

        reference_target = (
            reference_df["target"]
            .reset_index(drop=True)
        )

        current_target = (
            current_df["target"]
            .reset_index(drop=True)
        )

        if not current_target.equals(reference_target):
            raise ValueError(
                f"Target của {feature_set_name} "
                f"không trùng với {reference_name}."
            )

    print(
        "Xác nhận F0–F4 có cùng số dòng, "
        "timestamp và target."
    )

    return reference_timestamps


def calculate_split_indices(
    total_rows: int,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[int, int]:
    """
    Tính chỉ số kết thúc train và validation.

    Train:
        [0, train_end)

    Validation:
        [train_end, validation_end)

    Test:
        [validation_end, total_rows)
    """

    train_end = int(
        math.floor(total_rows * train_ratio)
    )

    validation_size = int(
        math.floor(total_rows * validation_ratio)
    )

    validation_end = train_end + validation_size

    train_size = train_end
    test_size = total_rows - validation_end

    if train_size <= 0:
        raise ValueError(
            "Tập train không có dữ liệu."
        )

    if validation_size <= 0:
        raise ValueError(
            "Tập validation không có dữ liệu."
        )

    if test_size <= 0:
        raise ValueError(
            "Tập test không có dữ liệu."
        )

    return train_end, validation_end


def split_dataframe(
    df: pd.DataFrame,
    train_end: int,
    validation_end: int,
) -> dict[str, pd.DataFrame]:
    """
    Chia một DataFrame theo thứ tự thời gian.
    """

    train_df = (
        df.iloc[:train_end]
        .copy()
        .reset_index(drop=True)
    )

    validation_df = (
        df.iloc[train_end:validation_end]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        df.iloc[validation_end:]
        .copy()
        .reset_index(drop=True)
    )

    return {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }


def validate_split_boundaries(
    splits: dict[str, pd.DataFrame],
    feature_set_name: str,
) -> None:
    """
    Kiểm tra các tập không bị chồng lấn và đúng thứ tự thời gian.
    """

    train_df = splits["train"]
    validation_df = splits["validation"]
    test_df = splits["test"]

    if train_df.empty:
        raise ValueError(
            f"{feature_set_name}: train rỗng."
        )

    if validation_df.empty:
        raise ValueError(
            f"{feature_set_name}: validation rỗng."
        )

    if test_df.empty:
        raise ValueError(
            f"{feature_set_name}: test rỗng."
        )

    train_last = train_df["timestamp"].iloc[-1]
    validation_first = validation_df["timestamp"].iloc[0]
    validation_last = validation_df["timestamp"].iloc[-1]
    test_first = test_df["timestamp"].iloc[0]

    if train_last >= validation_first:
        raise ValueError(
            f"{feature_set_name}: train và validation "
            "bị chồng lấn theo thời gian."
        )

    if validation_last >= test_first:
        raise ValueError(
            f"{feature_set_name}: validation và test "
            "bị chồng lấn theo thời gian."
        )


def save_split_files(
    feature_sets: dict[str, pd.DataFrame],
    output_dir: Path,
    train_end: int,
    validation_end: int,
) -> dict[str, dict]:
    """
    Chia và lưu train, validation, test của F0–F4.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report: dict[str, dict] = {}

    reference_split_timestamps: dict[
        str,
        pd.Series,
    ] = {}

    for feature_set_name in FEATURE_SET_NAMES:
        df = feature_sets[feature_set_name]

        splits = split_dataframe(
            df=df,
            train_end=train_end,
            validation_end=validation_end,
        )

        validate_split_boundaries(
            splits=splits,
            feature_set_name=feature_set_name,
        )

        report[feature_set_name] = {}

        for split_name, split_df in splits.items():
            output_path = (
                output_dir
                / f"{feature_set_name}_{split_name}.csv"
            )

            split_df.to_csv(
                output_path,
                index=False,
                date_format="%Y-%m-%d %H:%M:%S",
            )

            split_timestamps = (
                split_df["timestamp"]
                .reset_index(drop=True)
            )

            if feature_set_name == "F0":
                reference_split_timestamps[
                    split_name
                ] = split_timestamps
            else:
                if not split_timestamps.equals(
                    reference_split_timestamps[
                        split_name
                    ]
                ):
                    raise RuntimeError(
                        f"Timestamp {split_name} của "
                        f"{feature_set_name} không trùng "
                        "với F0."
                    )

            report[feature_set_name][split_name] = {
                "file": str(output_path),
                "rows": int(len(split_df)),
                "columns": int(
                    len(split_df.columns)
                ),
                "input_feature_count": int(
                    len(split_df.columns) - 2
                ),
                "first_timestamp": str(
                    split_df["timestamp"].iloc[0]
                ),
                "last_timestamp": str(
                    split_df["timestamp"].iloc[-1]
                ),
                "target_min": float(
                    split_df["target"].min()
                ),
                "target_max": float(
                    split_df["target"].max()
                ),
                "target_mean": float(
                    split_df["target"].mean()
                ),
            }

            print(
                f"Đã lưu {feature_set_name}_{split_name}.csv: "
                f"{len(split_df):,} dòng."
            )

    return report


def build_split_report(
    feature_sets: dict[str, pd.DataFrame],
    split_report: dict[str, dict],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    train_end: int,
    validation_end: int,
) -> dict:
    """
    Tạo báo cáo tổng hợp.
    """

    total_rows = len(feature_sets["F0"])

    train_rows = train_end
    validation_rows = (
        validation_end - train_end
    )
    test_rows = (
        total_rows - validation_end
    )

    actual_train_ratio = (
        train_rows / total_rows
    )

    actual_validation_ratio = (
        validation_rows / total_rows
    )

    actual_test_ratio = (
        test_rows / total_rows
    )

    return {
        "split_method": (
            "chronological_no_shuffle"
        ),
        "feature_sets": FEATURE_SET_NAMES,
        "feature_sets_aligned": True,
        "total_rows_per_feature_set": int(
            total_rows
        ),
        "requested_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "actual_ratios": {
            "train": actual_train_ratio,
            "validation": actual_validation_ratio,
            "test": actual_test_ratio,
        },
        "split_sizes": {
            "train": int(train_rows),
            "validation": int(
                validation_rows
            ),
            "test": int(test_rows),
        },
        "split_indices": {
            "train_start": 0,
            "train_end_exclusive": int(
                train_end
            ),
            "validation_start": int(
                train_end
            ),
            "validation_end_exclusive": int(
                validation_end
            ),
            "test_start": int(
                validation_end
            ),
            "test_end_exclusive": int(
                total_rows
            ),
        },
        "common_periods": {
            "train": {
                "start": split_report[
                    "F0"
                ]["train"]["first_timestamp"],
                "end": split_report[
                    "F0"
                ]["train"]["last_timestamp"],
            },
            "validation": {
                "start": split_report[
                    "F0"
                ]["validation"]["first_timestamp"],
                "end": split_report[
                    "F0"
                ]["validation"]["last_timestamp"],
            },
            "test": {
                "start": split_report[
                    "F0"
                ]["test"]["first_timestamp"],
                "end": split_report[
                    "F0"
                ]["test"]["last_timestamp"],
            },
        },
        "files": split_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Chia F0–F4 theo thời gian thành "
            "train, validation và test."
        )
    )

    parser.add_argument(
        "--feature-dir",
        type=str,
        default="outputs/features",
        help=(
            "Thư mục chứa F0.csv đến F4.csv. "
            "Mặc định: outputs/features"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/splits",
        help=(
            "Thư mục lưu các tập dữ liệu. "
            "Mặc định: outputs/splits"
        ),
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help=(
            "Tỷ lệ train. Mặc định: 0.70"
        ),
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help=(
            "Tỷ lệ validation. Mặc định: 0.15"
        ),
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help=(
            "Tỷ lệ test. Mặc định: 0.15"
        ),
    )

    args = parser.parse_args()

    validate_ratios(
        train_ratio=args.train_ratio,
        validation_ratio=(
            args.validation_ratio
        ),
        test_ratio=args.test_ratio,
    )

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)

    print("=" * 68)
    print(
        "BẮT ĐẦU CHIA TRAIN, VALIDATION VÀ TEST"
    )
    print("=" * 68)
    print(
        f"Thư mục feature: {feature_dir}"
    )
    print(
        f"Thư mục output : {output_dir}"
    )
    print(
        f"Tỷ lệ train    : "
        f"{args.train_ratio:.2%}"
    )
    print(
        f"Tỷ lệ validation: "
        f"{args.validation_ratio:.2%}"
    )
    print(
        f"Tỷ lệ test     : "
        f"{args.test_ratio:.2%}"
    )
    print("Shuffle         : False")
    print()

    feature_sets = load_all_feature_sets(
        feature_dir=feature_dir,
    )

    validate_feature_set_alignment(
        feature_sets=feature_sets,
    )

    total_rows = len(feature_sets["F0"])

    train_end, validation_end = (
        calculate_split_indices(
            total_rows=total_rows,
            train_ratio=args.train_ratio,
            validation_ratio=(
                args.validation_ratio
            ),
        )
    )

    train_rows = train_end
    validation_rows = (
        validation_end - train_end
    )
    test_rows = total_rows - validation_end

    print()
    print(
        f"Tổng số dòng: {total_rows:,}"
    )
    print(
        f"Train         : {train_rows:,} dòng"
    )
    print(
        f"Validation    : "
        f"{validation_rows:,} dòng"
    )
    print(
        f"Test          : {test_rows:,} dòng"
    )
    print()

    split_report = save_split_files(
        feature_sets=feature_sets,
        output_dir=output_dir,
        train_end=train_end,
        validation_end=validation_end,
    )

    complete_report = build_split_report(
        feature_sets=feature_sets,
        split_report=split_report,
        train_ratio=args.train_ratio,
        validation_ratio=(
            args.validation_ratio
        ),
        test_ratio=args.test_ratio,
        train_end=train_end,
        validation_end=validation_end,
    )

    report_path = (
        output_dir
        / "split_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            complete_report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 68)
    print("HOÀN TẤT CHIA DỮ LIỆU")
    print("=" * 68)

    common_periods = complete_report[
        "common_periods"
    ]

    print(
        "Train      : "
        f"{common_periods['train']['start']} "
        "đến "
        f"{common_periods['train']['end']}"
    )

    print(
        "Validation : "
        f"{common_periods['validation']['start']} "
        "đến "
        f"{common_periods['validation']['end']}"
    )

    print(
        "Test       : "
        f"{common_periods['test']['start']} "
        "đến "
        f"{common_periods['test']['end']}"
    )

    print(f"Báo cáo: {report_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()