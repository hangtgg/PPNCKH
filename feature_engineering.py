"""
feature_engineering.py

Mục đích:
    Tạo 5 bộ đặc trưng thời gian F0–F4 cho bài toán dự báo phụ tải
    trước một giờ.

Đầu vào mặc định:
    outputs/features/clean_data.csv

Đầu ra:
    outputs/features/F0.csv
    outputs/features/F1.csv
    outputs/features/F2.csv
    outputs/features/F3.csv
    outputs/features/F4.csv
    outputs/features/feature_engineering_report.json

Cách chạy:
    python feature_engineering.py ^
        --input outputs/features/clean_data.csv ^
        --output-dir outputs/features

PowerShell:
    python feature_engineering.py `
        --input outputs/features/clean_data.csv `
        --output-dir outputs/features
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Đọc và chuẩn hóa tên cột
# ============================================================

def load_clean_data(input_path: str) -> pd.DataFrame:
    """
    Đọc dữ liệu sạch từ preprocessing.py.

    Hàm hỗ trợ một số tên cột phổ biến:
        - Datetime / Load
        - timestamp / load
        - datetime / PJM_Load_MW
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {input_path}"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("File dữ liệu đầu vào không có dòng dữ liệu nào.")

    # Tìm cột thời gian.
    datetime_candidates = [
        "timestamp",
        "datetime",
        "date_time",
        "date",
        "Datetime",
        "Timestamp",
    ]

    datetime_column = next(
        (column for column in datetime_candidates if column in df.columns),
        None,
    )

    # Nếu không tìm thấy theo tên phổ biến, sử dụng cột đầu tiên.
    if datetime_column is None:
        datetime_column = df.columns[0]
        warnings.warn(
            f"Không tìm thấy cột thời gian theo tên chuẩn. "
            f"Sử dụng cột đầu tiên: {datetime_column}"
        )

    # Tìm cột phụ tải.
    load_candidates = [
        "load",
        "Load",
        "PJM_Load_MW",
        "load_mw",
        "energy",
        "value",
    ]

    load_column = next(
        (column for column in load_candidates if column in df.columns),
        None,
    )

    # Nếu không tìm thấy, chọn cột số đầu tiên khác cột thời gian.
    if load_column is None:
        remaining_columns = [
            column
            for column in df.columns
            if column != datetime_column
        ]

        numeric_candidates = []

        for column in remaining_columns:
            converted = pd.to_numeric(df[column], errors="coerce")

            if converted.notna().sum() > 0:
                numeric_candidates.append(column)

        if not numeric_candidates:
            raise ValueError(
                "Không tìm thấy cột phụ tải trong dữ liệu."
            )

        load_column = numeric_candidates[0]

        warnings.warn(
            f"Không tìm thấy cột phụ tải theo tên chuẩn. "
            f"Sử dụng cột: {load_column}"
        )

    # Chuẩn hóa tên cột.
    df = df.rename(
        columns={
            datetime_column: "timestamp",
            load_column: "load",
        }
    )

    df = df[["timestamp", "load"]].copy()

    # Chuyển đổi kiểu dữ liệu.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["load"] = pd.to_numeric(
        df["load"],
        errors="coerce",
    )

    # Loại dòng không chuyển đổi được.
    invalid_datetime = int(df["timestamp"].isna().sum())
    invalid_load = int(df["load"].isna().sum())

    if invalid_datetime > 0:
        warnings.warn(
            f"Loại {invalid_datetime} dòng có timestamp không hợp lệ."
        )

    if invalid_load > 0:
        warnings.warn(
            f"Loại {invalid_load} dòng có load không hợp lệ."
        )

    df = df.dropna(subset=["timestamp", "load"])

    # Sắp xếp theo thời gian.
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Không cho phép timestamp trùng lặp.
    duplicate_count = int(df["timestamp"].duplicated().sum())

    if duplicate_count > 0:
        raise ValueError(
            f"Phát hiện {duplicate_count} timestamp trùng lặp. "
            "Hãy xử lý ở preprocessing.py trước khi tạo đặc trưng."
        )

    # Không cho phép phụ tải âm.
    negative_count = int((df["load"] < 0).sum())

    if negative_count > 0:
        raise ValueError(
            f"Phát hiện {negative_count} giá trị phụ tải âm."
        )

    return df


# ============================================================
# 2. Kiểm tra tính liên tục theo giờ
# ============================================================

def validate_hourly_frequency(df: pd.DataFrame) -> None:
    """
    Kiểm tra khoảng cách giữa hai timestamp liên tiếp có đúng một giờ
    hay không.

    Nếu dữ liệu không liên tục, việc dùng shift() sẽ không còn tương
    ứng chính xác với lag theo giờ.
    """

    time_difference = df["timestamp"].diff().dropna()

    invalid_intervals = time_difference[
        time_difference != pd.Timedelta(hours=1)
    ]

    if not invalid_intervals.empty:
        example_indices = invalid_intervals.index[:5].tolist()

        raise ValueError(
            "Dữ liệu không liên tục theo tần suất một giờ. "
            f"Phát hiện {len(invalid_intervals)} khoảng thời gian bất thường. "
            f"Ví dụ tại các chỉ số: {example_indices}. "
            "Hãy xử lý missing timestamp trong preprocessing.py."
        )


# ============================================================
# 3. Tạo biến ngày nghỉ lễ
# ============================================================

def create_holiday_feature(
    timestamps: pd.Series,
    holiday_country: str | None,
) -> pd.Series:
    """
    Tạo biến is_holiday.

    Với dữ liệu PJM của Hoa Kỳ, sử dụng:
        --holiday-country US

    Nếu không muốn sử dụng ngày nghỉ:
        --holiday-country none
    """

    if (
        holiday_country is None
        or holiday_country.strip().lower() in {"none", "no", "false"}
    ):
        return pd.Series(
            np.zeros(len(timestamps), dtype=np.int8),
            index=timestamps.index,
            name="is_holiday",
        )

    try:
        import holidays
    except ImportError as error:
        raise ImportError(
            "Bạn đã yêu cầu tạo biến ngày nghỉ nhưng chưa cài thư viện "
            "'holidays'. Hãy chạy:\n"
            "pip install holidays\n"
            "Hoặc sử dụng --holiday-country none để không dùng ngày nghỉ."
        ) from error

    country_code = holiday_country.upper()

    try:
        years = sorted(timestamps.dt.year.unique().tolist())

        country_holidays = holidays.country_holidays(
            country_code,
            years=years,
        )
    except Exception as error:
        raise ValueError(
            f"Không thể tạo lịch ngày nghỉ cho mã quốc gia "
            f"'{country_code}'."
        ) from error

    holiday_values = timestamps.dt.date.map(
        lambda value: int(value in country_holidays)
    )

    return holiday_values.astype(np.int8).rename("is_holiday")


# ============================================================
# 4. Tạo bảng đặc trưng tổng
# ============================================================

def build_master_feature_table(
    df: pd.DataFrame,
    holiday_country: str | None = "US",
) -> pd.DataFrame:
    """
    Tạo toàn bộ đặc trưng cần thiết cho F0–F4.

    Target:
        target tại thời điểm t là load tại thời điểm t+1.

    F0:
        load_lag_0, ..., load_lag_23

    Trong đó:
        load_lag_0 = y_t
        load_lag_1 = y_(t-1)
        ...
        load_lag_23 = y_(t-23)

    Rolling feature chỉ sử dụng load tại thời điểm hiện tại và quá khứ,
    không sử dụng target hoặc dữ liệu tương lai.
    """

    features = df.copy()

    # --------------------------------------------------------
    # Target: dự báo phụ tải trước một giờ
    # --------------------------------------------------------
    features["target"] = features["load"].shift(-1)

    # --------------------------------------------------------
    # F0: lịch sử phụ tải 24 giờ
    # --------------------------------------------------------
    for lag in range(24):
        features[f"load_lag_{lag}"] = features["load"].shift(lag)

    # --------------------------------------------------------
    # F1: đặc trưng lịch
    # --------------------------------------------------------
    features["hour"] = features["timestamp"].dt.hour.astype(np.int8)

    features["day_of_week"] = (
        features["timestamp"].dt.dayofweek.astype(np.int8)
    )

    features["month"] = (
        features["timestamp"].dt.month.astype(np.int8)
    )

    # Thứ Hai = 0, ..., Chủ nhật = 6.
    features["is_weekend"] = (
        features["day_of_week"] >= 5
    ).astype(np.int8)

    features["is_holiday"] = create_holiday_feature(
        timestamps=features["timestamp"],
        holiday_country=holiday_country,
    )

    # --------------------------------------------------------
    # F2: mã hóa chu kỳ
    # --------------------------------------------------------

    # Chu kỳ giờ: 24 giờ.
    features["hour_sin"] = np.sin(
        2.0 * math.pi * features["hour"] / 24.0
    )

    features["hour_cos"] = np.cos(
        2.0 * math.pi * features["hour"] / 24.0
    )

    # Chu kỳ thứ trong tuần: 7 ngày.
    features["weekday_sin"] = np.sin(
        2.0 * math.pi * features["day_of_week"] / 7.0
    )

    features["weekday_cos"] = np.cos(
        2.0 * math.pi * features["day_of_week"] / 7.0
    )

    # Month có giá trị từ 1 đến 12, nên trừ 1 trước khi mã hóa.
    month_zero_based = features["month"] - 1

    features["month_sin"] = np.sin(
        2.0 * math.pi * month_zero_based / 12.0
    )

    features["month_cos"] = np.cos(
        2.0 * math.pi * month_zero_based / 12.0
    )

    # --------------------------------------------------------
    # F3: lag dài hạn
    # --------------------------------------------------------
    features["load_lag_48"] = features["load"].shift(48)
    features["load_lag_168"] = features["load"].shift(168)

    # Không tạo lag24 riêng vì load_lag_23...load_lag_0 đã chứa
    # cửa sổ 24 giờ. Theo thiết kế nghiên cứu, lag24 được bỏ có chủ ý.

    # --------------------------------------------------------
    # F4: rolling statistics
    # --------------------------------------------------------
    # Các phép rolling bên dưới sử dụng chính cột load hiện tại và quá
    # khứ. Vì target là load ở t+1 nên không gây rò rỉ dữ liệu tương lai.

    rolling_24 = features["load"].rolling(
        window=24,
        min_periods=24,
    )

    features["rolling_mean_24"] = rolling_24.mean()
    features["rolling_std_24"] = rolling_24.std()
    features["rolling_min_24"] = rolling_24.min()
    features["rolling_max_24"] = rolling_24.max()

    rolling_168 = features["load"].rolling(
        window=168,
        min_periods=168,
    )

    features["rolling_mean_168"] = rolling_168.mean()
    features["rolling_std_168"] = rolling_168.std()

    return features


# ============================================================
# 5. Khai báo cột của từng feature set
# ============================================================

def get_feature_set_columns() -> dict[str, list[str]]:
    """
    Trả về danh sách biến đầu vào của F0–F4.

    F2 thay thế hour, day_of_week và month bằng mã hóa sin/cos,
    không giữ lại ba biến số nguyên này.
    """

    history_columns = [
        f"load_lag_{lag}"
        for lag in range(24)
    ]

    f0_columns = history_columns

    f1_columns = (
        history_columns
        + [
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "is_holiday",
        ]
    )

    f2_columns = (
        history_columns
        + [
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "is_weekend",
            "is_holiday",
        ]
    )

    f3_columns = (
        f2_columns
        + [
            "load_lag_48",
            "load_lag_168",
        ]
    )

    f4_columns = (
        f3_columns
        + [
            "rolling_mean_24",
            "rolling_std_24",
            "rolling_min_24",
            "rolling_max_24",
            "rolling_mean_168",
            "rolling_std_168",
        ]
    )

    return {
        "F0": f0_columns,
        "F1": f1_columns,
        "F2": f2_columns,
        "F3": f3_columns,
        "F4": f4_columns,
    }


# ============================================================
# 6. Loại dòng thiếu một lần cho toàn bộ F0–F4
# ============================================================

def align_feature_sets(
    master_table: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Đồng bộ toàn bộ feature set trên cùng tập timestamp.

    Nguyên tắc:
        - Giữ tất cả các cột cần thiết của F0, F1, F2, F3 và F4.
        - Loại dòng thiếu dựa trên các biến của F4 và target.
        - Sau khi loại, tất cả F0–F4 có cùng số dòng và timestamp.
    """

    # Tạo danh sách hợp của tất cả các biến trong F0–F4.
    all_feature_columns = []

    for columns in feature_sets.values():
        for column in columns:
            if column not in all_feature_columns:
                all_feature_columns.append(column)

    # Những cột dùng để xác định dòng hợp lệ.
    # F4 yêu cầu lịch sử dài nhất nên dùng F4 để drop NaN.
    validity_columns = [
        "timestamp",
        "target",
        *feature_sets["F4"],
    ]

    rows_before = len(master_table)

    # Xác định các dòng đủ lịch sử và có target.
    valid_mask = master_table[validity_columns].notna().all(axis=1)

    # Giữ toàn bộ các cột cần cho cả F0–F4.
    required_columns = [
        "timestamp",
        "target",
        *all_feature_columns,
    ]

    aligned_table = (
        master_table.loc[valid_mask, required_columns]
        .reset_index(drop=True)
    )

    rows_after = len(aligned_table)

    if rows_after == 0:
        raise ValueError(
            "Không còn dữ liệu sau khi loại các dòng thiếu. "
            "Dataset phải có nhiều hơn 169 quan sát theo giờ."
        )

    # Kiểm tra các cột cần thiết có tồn tại.
    missing_columns = [
        column
        for column in required_columns
        if column not in aligned_table.columns
    ]

    if missing_columns:
        raise KeyError(
            "Thiếu các cột sau trong bảng đặc trưng tổng: "
            f"{missing_columns}"
        )

    print(
        f"Đã loại {rows_before - rows_after:,} dòng "
        "chưa đủ lịch sử hoặc chưa có target."
    )

    return aligned_table

# ============================================================
# 7. Xuất F0–F4
# ============================================================

def save_feature_sets(
    aligned_table: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    output_dir: str,
) -> dict[str, dict]:
    """
    Xuất mỗi feature set thành một file CSV riêng.
    """

    os.makedirs(output_dir, exist_ok=True)

    report: dict[str, dict] = {}

    expected_timestamps = aligned_table["timestamp"]

    for feature_set_name, feature_columns in feature_sets.items():
        output_columns = (
            ["timestamp", "target"]
            + feature_columns
        )

        output_df = aligned_table[output_columns].copy()

        # Kiểm tra không còn NaN.
        missing_values = int(output_df.isna().sum().sum())

        if missing_values > 0:
            raise ValueError(
                f"{feature_set_name} vẫn còn "
                f"{missing_values} giá trị thiếu."
            )

        # Kiểm tra timestamp không bị thay đổi.
        if not output_df["timestamp"].equals(expected_timestamps):
            raise RuntimeError(
                f"Timestamp của {feature_set_name} không đồng nhất."
            )

        output_path = os.path.join(
            output_dir,
            f"{feature_set_name}.csv",
        )

        output_df.to_csv(
            output_path,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
        )

        report[feature_set_name] = {
            "file": output_path,
            "rows": int(len(output_df)),
            "input_feature_count": int(len(feature_columns)),
            "total_column_count": int(len(output_df.columns)),
            "first_timestamp": str(
                output_df["timestamp"].iloc[0]
            ),
            "last_timestamp": str(
                output_df["timestamp"].iloc[-1]
            ),
            "features": feature_columns,
        }

        print(
            f"Đã tạo {feature_set_name}: "
            f"{len(output_df):,} dòng, "
            f"{len(feature_columns)} biến đầu vào."
        )

    return report


# ============================================================
# 8. Hàm chạy chính
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tạo các bộ đặc trưng thời gian F0–F4 "
            "cho dự báo phụ tải trước một giờ."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default="outputs/features/clean_data.csv",
        help=(
            "Đường dẫn đến dữ liệu sạch. "
            "Mặc định: outputs/features/clean_data.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/features",
        help=(
            "Thư mục lưu F0–F4. "
            "Mặc định: outputs/features"
        ),
    )

    parser.add_argument(
        "--holiday-country",
        type=str,
        default="US",
        help=(
            "Mã quốc gia dùng để tạo biến ngày nghỉ. "
            "Dữ liệu PJM sử dụng US. "
            "Nhập 'none' nếu không dùng ngày nghỉ."
        ),
    )

    args = parser.parse_args()

    start_time = time.perf_counter()

    print("=" * 65)
    print("BẮT ĐẦU TẠO ĐẶC TRƯNG F0–F4")
    print("=" * 65)
    print(f"File đầu vào : {args.input}")
    print(f"Thư mục output: {args.output_dir}")
    print(f"Lịch ngày nghỉ: {args.holiday_country}")
    print()

    # Đọc dữ liệu.
    df = load_clean_data(args.input)

    print(f"Số dòng dữ liệu sạch: {len(df):,}")
    print(
        f"Khoảng thời gian: "
        f"{df['timestamp'].min()} đến {df['timestamp'].max()}"
    )

    # Kiểm tra tần suất một giờ.
    validate_hourly_frequency(df)

    print("Tần suất dữ liệu: liên tục theo giờ.")

    # Tạo bảng đặc trưng tổng.
    master_table = build_master_feature_table(
        df=df,
        holiday_country=args.holiday_country,
    )

    feature_sets = get_feature_set_columns()

    # Đồng bộ timestamp giữa F0–F4.
    aligned_table = align_feature_sets(
        master_table=master_table,
        feature_sets=feature_sets,
    )

    # Lưu năm file CSV.
    feature_report = save_feature_sets(
        aligned_table=aligned_table,
        feature_sets=feature_sets,
        output_dir=args.output_dir,
    )

    elapsed_time = time.perf_counter() - start_time

    complete_report = {
        "input_file": args.input,
        "output_directory": args.output_dir,
        "holiday_country": args.holiday_country,
        "forecast_horizon_hours": 1,
        "input_window_hours": 24,
        "original_rows": int(len(df)),
        "aligned_rows": int(len(aligned_table)),
        "removed_rows": int(len(df) - len(aligned_table)),
        "first_aligned_timestamp": str(
            aligned_table["timestamp"].iloc[0]
        ),
        "last_aligned_timestamp": str(
            aligned_table["timestamp"].iloc[-1]
        ),
        "feature_generation_time_seconds": round(
            elapsed_time,
            6,
        ),
        "feature_sets": feature_report,
    }

    report_path = os.path.join(
        args.output_dir,
        "feature_engineering_report.json",
    )

    with open(
        report_path,
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
    print("=" * 65)
    print("HOÀN TẤT TẠO ĐẶC TRƯNG")
    print("=" * 65)
    print(f"Số dòng dùng chung: {len(aligned_table):,}")
    print(f"Thời gian thực hiện: {elapsed_time:.3f} giây")
    print(f"Báo cáo: {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()