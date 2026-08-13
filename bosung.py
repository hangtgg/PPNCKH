from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first matching column name."""
    normalized = {col.strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    raise KeyError(
        f"Không tìm thấy cột phù hợp. Các cột hiện có: {list(df.columns)}"
    )


def main() -> None:
    input_file = Path("outputs/metrics/summary_metrics.csv")
    output_dir = Path("outputs/figures/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_file}")

    df = pd.read_csv(input_file)

    model_col = find_column(df, ["model"])
    feature_col = find_column(
        df,
        ["feature_set", "feature set", "featureset", "feature"],
    )
    rmse_col = find_column(
        df,
        ["rmse_mean", "mean_rmse", "rmse"],
    )

    plot_df = df[[model_col, feature_col, rmse_col]].copy()
    plot_df.columns = ["model", "feature_set", "rmse"]

    model_order = ["RF", "XGB", "LGBM", "LSTM"]
    feature_order = ["F0", "F1", "F2", "F3", "F4"]

    plot_df["model"] = pd.Categorical(
        plot_df["model"],
        categories=model_order,
        ordered=True,
    )
    plot_df["feature_set"] = pd.Categorical(
        plot_df["feature_set"],
        categories=feature_order,
        ordered=True,
    )

    plot_df = plot_df.sort_values(["model", "feature_set"])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    for model in model_order:
        model_data = plot_df[plot_df["model"] == model]

        if model_data.empty:
            print(f"Cảnh báo: không tìm thấy dữ liệu cho mô hình {model}")
            continue

        ax.plot(
            model_data["feature_set"].astype(str),
            model_data["rmse"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=model,
        )

    ax.set_xlabel("Feature set")
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Model × feature-set interaction")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Model")
    ax.margins(x=0.03)

    fig.tight_layout()

    png_file = output_dir / "model_feature_interaction.png"
    pdf_file = output_dir / "model_feature_interaction.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã lưu: {png_file}")
    print(f"Đã lưu: {pdf_file}")


if __name__ == "__main__":
    main()