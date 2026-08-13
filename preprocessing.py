"""
preprocessing.py

Mục đích:
    - Đọc dữ liệu PJM_Load_hourly.csv
    - Chuẩn hóa tên cột thành Datetime và Load
    - Kiểm tra timestamp trùng lặp
    - Kiểm tra timestamp bị thiếu
    - Tạo lại chuỗi thời gian liên tục theo giờ
    - Nội suy giá trị phụ tải bị thiếu
    - Kiểm tra giá trị âm
    - Lưu dữ liệu sạch
    - Lưu báo cáo kiểm tra dữ liệu

Cách chạy trên PowerShell:

python preprocessing.py `
    --input data/PJM_Load_hourly.csv `
    --output outputs/features/clean_data.csv

Hoặc chạy trên một dòng:

python preprocessing.py --input data/PJM_Load_hourly.csv --output outputs/features/clean_data.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Tự động xác định cột thời gian và cột phụ tải.

    Ưu tiên các tên cột phổ biến. Nếu không tìm thấy,
    sử dụng cột đầu tiên làm thời gian và cột thứ hai làm phụ tải.
    """

    if df.shape[1] < 2:
        raise ValueError(
            "File CSV phải có ít nhất hai cột: "
            "một cột thời gian và một cột phụ tải."
        )

    datetime_candidates = [
        "Datetime",
        "datetime",
        "Timestamp",
        "timestamp",
        "DateTime",
        "date_time",
        "date",
    ]

    load_candidates = [
        "Load",
        "load",
        "PJM_Load_MW",
        "PJM_Load",
        "load_mw",
        "energy",
        "value",
    ]

    datetime_column = next(
        (
            column
            for column in datetime_candidates
            if column in df.columns
        ),
        None,
    )

    load_column = next(
        (
            column
            for column in load_candidates
            if column in df.columns
        ),
        None,
    )

    if datetime_column is None:
        datetime_column = df.columns[0]

    if load_column is None:
        remaining_columns = [
            column
            for column in df.columns
            if column != datetime_column
        ]

        numeric_columns = []

        for column in remaining_columns:
            converted = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if converted.notna().sum() > 0:
                numeric_columns.append(column)

        if not numeric_columns:
            raise ValueError(
                "Không tìm thấy cột phụ tải dạng số trong file CSV."
            )

        load_column = numeric_columns[0]

    if datetime_column == load_column:
        raise ValueError(
            "Cột thời gian và cột phụ tải không được trùng nhau."
        )

    return datetime_column, load_column


def validate_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Kiểm tra, làm sạch và chuẩn hóa dữ liệu về tần suất một giờ.

    Các bước chính:
        1. Xác định cột thời gian và phụ tải.
        2. Chuyển đổi kiểu dữ liệu.
        3. Loại timestamp không hợp lệ.
        4. Sắp xếp theo thời gian.
        5. Xử lý timestamp trùng bằng giá trị trung bình.
        6. Tạo lại dải thời gian liên tục theo giờ.
        7. Nội suy các giá trị phụ tải bị thiếu.
        8. Kiểm tra dữ liệu âm và tần suất cuối cùng.
    """

    report: dict = {}

    report["original_rows"] = int(len(df))
    report["original_columns"] = list(df.columns)

    # =========================================================
    # 1. Xác định cột thời gian và cột phụ tải
    # =========================================================
    datetime_column, load_column = detect_columns(df)

    report["detected_datetime_column"] = datetime_column
    report["detected_load_column"] = load_column

    df = df.rename(
        columns={
            datetime_column: "Datetime",
            load_column: "Load",
        }
    )

    df = df[["Datetime", "Load"]].copy()

    # =========================================================
    # 2. Chuyển đổi kiểu dữ liệu
    # =========================================================
    df["Datetime"] = pd.to_datetime(
        df["Datetime"],
        errors="coerce",
    )

    df["Load"] = pd.to_numeric(
        df["Load"],
        errors="coerce",
    )

    invalid_datetime_count = int(
        df["Datetime"].isna().sum()
    )

    invalid_load_count = int(
        df["Load"].isna().sum()
    )

    report["invalid_datetime_values"] = (
        invalid_datetime_count
    )

    report["invalid_or_missing_load_values_before_cleaning"] = (
        invalid_load_count
    )

    # Không thể phục hồi dòng không có timestamp,
    # nên loại các dòng này.
    df = df.dropna(subset=["Datetime"]).copy()

    # =========================================================
    # 3. Sắp xếp theo thời gian
    # =========================================================
    df = (
        df.sort_values("Datetime")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "Không còn dữ liệu hợp lệ sau khi xử lý timestamp."
        )

    # =========================================================
    # 4. Xử lý timestamp trùng lặp
    # =========================================================
    duplicate_count = int(
        df["Datetime"].duplicated(
            keep=False
        ).sum()
    )

    duplicate_timestamp_count = int(
        df["Datetime"].duplicated().sum()
    )

    report["rows_in_duplicate_groups"] = duplicate_count
    report["duplicate_timestamp_count"] = (
        duplicate_timestamp_count
    )

    if duplicate_timestamp_count > 0:
        # Nếu cùng timestamp xuất hiện nhiều lần,
        # lấy trung bình phụ tải.
        df = (
            df.groupby(
                "Datetime",
                as_index=False,
            )["Load"]
            .mean()
            .sort_values("Datetime")
            .reset_index(drop=True)
        )

    report["rows_after_duplicate_handling"] = int(
        len(df)
    )

    # =========================================================
    # 5. Kiểm tra giá trị phụ tải âm
    # =========================================================
    negative_count_before = int(
        (df["Load"] < 0).sum()
    )

    report["negative_load_values"] = (
        negative_count_before
    )

    if negative_count_before > 0:
        negative_examples = (
            df.loc[
                df["Load"] < 0,
                ["Datetime", "Load"],
            ]
            .head(10)
            .astype(str)
            .to_dict(orient="records")
        )

        report["negative_load_examples"] = (
            negative_examples
        )

        raise ValueError(
            f"Phát hiện {negative_count_before} "
            "giá trị phụ tải âm. "
            "Không tự động sửa vì có thể làm sai dữ liệu."
        )

    # =========================================================
    # 6. Tạo chuỗi timestamp đầy đủ theo giờ
    # =========================================================
    start_time = df["Datetime"].min()
    end_time = df["Datetime"].max()

    full_hourly_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq="h",
    )

    original_timestamp_index = pd.DatetimeIndex(
        df["Datetime"]
    )

    missing_timestamps = (
        full_hourly_index.difference(
            original_timestamp_index
        )
    )

    report["start_timestamp"] = str(start_time)
    report["end_timestamp"] = str(end_time)

    report["expected_hourly_rows"] = int(
        len(full_hourly_index)
    )

    report["missing_timestamps_before_reindex"] = int(
        len(missing_timestamps)
    )

    report["missing_timestamp_examples"] = [
        str(timestamp)
        for timestamp in missing_timestamps[:20]
    ]

    # Đưa Datetime thành index để bổ sung các giờ còn thiếu.
    df = df.set_index("Datetime")

    df = df.reindex(full_hourly_index)

    df.index.name = "Datetime"

    # =========================================================
    # 7. Nội suy giá trị phụ tải bị thiếu
    # =========================================================
    missing_load_before_interpolation = int(
        df["Load"].isna().sum()
    )

    report[
        "missing_load_before_interpolation"
    ] = missing_load_before_interpolation

    interpolated_mask = df["Load"].isna()

    # Nội suy tuyến tính dựa trên trục thời gian.
    df["Load"] = df["Load"].interpolate(
        method="time",
        limit_direction="both",
    )

    missing_load_after_interpolation = int(
        df["Load"].isna().sum()
    )

    report[
        "missing_load_after_interpolation"
    ] = missing_load_after_interpolation

    report["interpolated_value_count"] = int(
        interpolated_mask.sum()
    )

    if missing_load_after_interpolation > 0:
        raise ValueError(
            "Vẫn còn giá trị phụ tải thiếu "
            "sau khi nội suy."
        )

    # =========================================================
    # 8. Khôi phục Datetime thành cột
    # =========================================================
    df = df.reset_index()

    # =========================================================
    # 9. Kiểm tra lại giá trị âm sau nội suy
    # =========================================================
    negative_count_after = int(
        (df["Load"] < 0).sum()
    )

    report[
        "negative_load_values_after_interpolation"
    ] = negative_count_after

    if negative_count_after > 0:
        raise ValueError(
            "Nội suy tạo ra giá trị phụ tải âm. "
            "Hãy kiểm tra lại dữ liệu."
        )

    # =========================================================
    # 10. Xác nhận tần suất một giờ
    # =========================================================
    intervals = (
        df["Datetime"]
        .diff()
        .dropna()
    )

    hourly_frequency = bool(
        (
            intervals
            == pd.Timedelta(hours=1)
        ).all()
    )

    report[
        "hourly_frequency_after_cleaning"
    ] = hourly_frequency

    if not hourly_frequency:
        abnormal_intervals = intervals[
            intervals != pd.Timedelta(hours=1)
        ]

        raise ValueError(
            "Dữ liệu vẫn chưa liên tục theo giờ "
            "sau khi xử lý. "
            f"Số khoảng bất thường: "
            f"{len(abnormal_intervals)}."
        )

    # =========================================================
    # 11. Kiểm tra timestamp trùng sau xử lý
    # =========================================================
    remaining_duplicates = int(
        df["Datetime"].duplicated().sum()
    )

    report[
        "duplicate_timestamps_after_cleaning"
    ] = remaining_duplicates

    if remaining_duplicates > 0:
        raise ValueError(
            "Vẫn còn timestamp trùng lặp "
            "sau khi làm sạch."
        )

    # =========================================================
    # 12. Báo cáo cuối
    # =========================================================
    report["clean_rows"] = int(len(df))

    report["inserted_timestamp_rows"] = int(
        len(full_hourly_index)
        - len(original_timestamp_index)
    )

    report["remaining_missing_values"] = int(
        df[["Datetime", "Load"]]
        .isna()
        .sum()
        .sum()
    )

    report["load_min"] = float(df["Load"].min())
    report["load_max"] = float(df["Load"].max())
    report["load_mean"] = float(df["Load"].mean())
    report["load_std"] = float(df["Load"].std())

    # Bảo đảm đúng thứ tự cột.
    cleaned_df = df[["Datetime", "Load"]].copy()

    return cleaned_df, report


def save_outputs(
    cleaned_df: pd.DataFrame,
    report: dict,
    output_path: str,
) -> tuple[Path, Path]:
    """
    Lưu dữ liệu sạch và báo cáo JSON.
    """

    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_df.to_csv(
        output_file,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    report_path = (
        output_file.parent
        / "data_validation_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_file, report_path


def print_report(
    report: dict,
    output_file: Path,
    report_file: Path,
) -> None:
    """
    Hiển thị kết quả chính trên terminal.
    """

    print()
    print("=" * 65)
    print("HOÀN TẤT TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 65)

    print(
        f"Số dòng ban đầu: "
        f"{report['original_rows']:,}"
    )

    print(
        f"Timestamp trùng lặp: "
        f"{report['duplicate_timestamp_count']:,}"
    )

    print(
        f"Timestamp bị thiếu: "
        f"{report['missing_timestamps_before_reindex']:,}"
    )

    print(
        f"Giá trị được nội suy: "
        f"{report['interpolated_value_count']:,}"
    )

    print(
        f"Giá trị phụ tải âm: "
        f"{report['negative_load_values']:,}"
    )

    print(
        f"Dữ liệu liên tục theo giờ: "
        f"{report['hourly_frequency_after_cleaning']}"
    )

    print(
        f"Số dòng sau làm sạch: "
        f"{report['clean_rows']:,}"
    )

    print(
        f"Khoảng thời gian: "
        f"{report['start_timestamp']} "
        f"đến {report['end_timestamp']}"
    )

    print("-" * 65)
    print(f"Dữ liệu sạch: {output_file}")
    print(f"Báo cáo kiểm tra: {report_file}")
    print("=" * 65)


def main() -> None:
    """
    Hàm chạy chính của chương trình.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Làm sạch và chuẩn hóa dữ liệu "
            "phụ tải PJM theo tần suất một giờ."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/PJM_Load_hourly.csv",
        help=(
            "Đường dẫn file CSV đầu vào. "
            "Mặc định: data/PJM_Load_hourly.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "outputs/features/clean_data.csv"
        ),
        help=(
            "Đường dẫn file dữ liệu sạch. "
            "Mặc định: "
            "outputs/features/clean_data.csv"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: "
            f"{input_path.resolve()}"
        )

    print("=" * 65)
    print("BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 65)
    print(f"File đầu vào: {input_path}")
    print(f"File đầu ra : {args.output}")

    df = pd.read_csv(input_path)

    cleaned_df, report = validate_dataset(df)

    output_file, report_file = save_outputs(
        cleaned_df=cleaned_df,
        report=report,
        output_path=args.output,
    )

    print_report(
        report=report,
        output_file=output_file,
        report_file=report_file,
    )


if __name__ == "__main__":
    main()