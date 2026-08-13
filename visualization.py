"""Publication-ready visualization for the load-forecasting experiments."""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("visualization")
MODELS = ["RF", "XGB", "LGBM", "LSTM"]
FEATURES = ["F0", "F1", "F2", "F3", "F4"]
TRUE_CANDIDATES = ["actual", "y_true", "true", "target", "observed"]
PRED_CANDIDATES = ["prediction", "y_pred", "predicted", "forecast"]
TIME_CANDIDATES = ["timestamp", "datetime", "date", "time"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create figures and report tables.")
    p.add_argument("--metrics-file", type=Path, required=True)
    p.add_argument("--summary-file", type=Path)
    p.add_argument("--prediction-dir", type=Path)
    p.add_argument("--h1-dir", type=Path)
    p.add_argument("--h2-dir", type=Path)
    p.add_argument("--h3-dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--feature-sets", nargs="+", default=FEATURES)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--forecast-points", type=int, default=336)
    p.add_argument("--skip-pdf", action="store_true")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def setup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    plt.rcParams.update({
        "figure.figsize": (9.2, 5.8), "font.size": 10,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linestyle": "--",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def norm_model(v: Any) -> str:
    s = str(v).strip().upper()
    return {"RANDOMFOREST": "RF", "RANDOM_FOREST": "RF", "XGBOOST": "XGB", "LIGHTGBM": "LGBM"}.get(s, s)


def norm_feature(v: Any) -> str:
    s = str(v).strip().upper().replace("-", "").replace("_", "")
    m = re.search(r"F(\d+)", s)
    return f"F{m.group(1)}" if m else s


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    mapping = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in mapping:
            return mapping[c.lower()]
    return None


def read_metrics(path: Path, models: list[str], features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", "feature_set", "seed", "rmse", "mae", "r2"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    df["model"] = df["model"].map(norm_model)
    df["feature_set"] = df["feature_set"].map(norm_feature)
    for c in ["seed", "rmse", "mae", "mape", "r2", "training_time", "inference_time"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df.model.isin(models) & df.feature_set.isin(features)].copy()
    df = df.dropna(subset=["rmse", "mae", "r2"])
    if df.empty:
        raise RuntimeError("No valid metric rows.")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg: dict[str, tuple[str, str]] = {
        "n_seeds": ("seed", "nunique"), "rmse_mean": ("rmse", "mean"),
        "rmse_std": ("rmse", "std"), "mae_mean": ("mae", "mean"),
        "mae_std": ("mae", "std"), "r2_mean": ("r2", "mean"),
        "r2_std": ("r2", "std"),
    }
    for c in ["mape", "training_time", "inference_time"]:
        if c in df.columns:
            agg[f"{c}_mean"] = (c, "mean")
            agg[f"{c}_std"] = (c, "std")
    return df.groupby(["model", "feature_set"], as_index=False).agg(**agg)


def save_fig(fig: plt.Figure, stem: str, out: Path, dpi: int, pdf: bool,
             manifest: list[dict[str, str]], desc: str, show: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{stem}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    manifest.append({"artifact": stem, "type": "png", "path": str(png), "description": desc})
    if pdf:
        p = out / f"{stem}.pdf"
        fig.savefig(p, bbox_inches="tight")
        manifest.append({"artifact": stem, "type": "pdf", "path": str(p), "description": desc})
    if show:
        plt.show()
    plt.close(fig)


def save_table(df: pd.DataFrame, name: str, out: Path,
               manifest: list[dict[str, str]], desc: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    df.to_csv(path, index=False, float_format="%.6f")
    manifest.append({"artifact": name, "type": "csv", "path": str(path), "description": desc})


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, models: list[str],
                features: list[str], out: Path, args: argparse.Namespace,
                manifest: list[dict[str, str]], number: str) -> None:
    mean, std = f"{metric}_mean", f"{metric}_std"
    if mean not in summary.columns:
        return
    fig, ax = plt.subplots()
    x = np.arange(len(features))
    for model in models:
        s = summary[summary.model == model].set_index("feature_set").reindex(features)
        ax.errorbar(x, s[mean], yerr=s[std].fillna(0), marker="o", capsize=3, linewidth=1.8, label=model)
    ax.set_xticks(x, features); ax.set_xlabel("Feature set"); ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} across temporal feature sets")
    ax.legend(title="Model", frameon=False, ncol=2); ax.grid(axis="x", visible=False)
    save_fig(fig, f"{number}_{metric}_by_feature_set", out, args.dpi, not args.skip_pdf,
             manifest, f"Mean {ylabel} and standard deviation across seeds.", args.show)


def plot_per_model_rmse(summary: pd.DataFrame, models: list[str], features: list[str],
                        out: Path, args: argparse.Namespace, manifest: list[dict[str, str]]) -> None:
    for i, model in enumerate(models, 1):
        s = summary[summary.model == model].set_index("feature_set").reindex(features)
        if s.empty: continue
        values = s.rmse_mean.to_numpy(float); std = s.rmse_std.fillna(0).to_numpy(float)
        fig, ax = plt.subplots(figsize=(7.6, 5.1)); x = np.arange(len(features))
        ax.bar(x, values, yerr=std, capsize=4); ax.set_xticks(x, features)
        ax.set_xlabel("Feature set"); ax.set_ylabel("RMSE"); ax.set_ylim(bottom=0)
        ax.set_title(f"{model}: RMSE by feature set"); ax.grid(axis="x", visible=False)
        span = np.nanmax(values) - np.nanmin(values); offset = span * .025 if span else np.nanmax(values) * .02
        for xx, value in zip(x, values):
            if np.isfinite(value): ax.text(xx, value + offset, f"{value:.2f}", ha="center", fontsize=8)
        save_fig(fig, f"05_{i:02d}_{model.lower()}_rmse", out, args.dpi, not args.skip_pdf,
                 manifest, f"{model} RMSE across feature sets.", args.show)


def plot_heatmap(summary: pd.DataFrame, models: list[str], features: list[str], out: Path,
                 args: argparse.Namespace, manifest: list[dict[str, str]]) -> None:
    mat = summary.pivot(index="model", columns="feature_set", values="rmse_mean").reindex(index=models, columns=features)
    values = mat.to_numpy(float); fig, ax = plt.subplots(figsize=(8.2, 4.8)); im = ax.imshow(values, aspect="auto")
    ax.set_xticks(range(len(features)), features); ax.set_yticks(range(len(models)), models)
    ax.set_xlabel("Feature set"); ax.set_ylabel("Model"); ax.set_title("Mean RMSE heatmap"); ax.grid(False)
    midpoint = (np.nanmin(values) + np.nanmax(values)) / 2
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            if np.isfinite(values[r, c]):
                ax.text(c, r, f"{values[r,c]:.2f}", ha="center", va="center",
                        color="white" if values[r,c] > midpoint else "black", fontsize=9)
    cb = fig.colorbar(im, ax=ax); cb.set_label("RMSE")
    save_fig(fig, "06_rmse_heatmap", out, args.dpi, not args.skip_pdf,
             manifest, "RMSE heatmap for all configurations.", args.show)


def improvement_table(summary: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        s = summary[summary.model == model].set_index("feature_set")
        if "F0" not in s.index: continue
        base = float(s.loc["F0", "rmse_mean"])
        for f in ["F1", "F2", "F3", "F4"]:
            if f in s.index:
                value = float(s.loc[f, "rmse_mean"])
                rows.append({"model": model, "feature_set": f, "rmse_f0": base,
                             "rmse_feature": value, "rmse_reduction": base-value,
                             "improvement_percent": (base-value)/base*100 if base else np.nan})
    return pd.DataFrame(rows)


def plot_improvement(df: pd.DataFrame, models: list[str], out: Path,
                     args: argparse.Namespace, manifest: list[dict[str, str]]) -> None:
    if df.empty: return
    feats = ["F1", "F2", "F3", "F4"]; x = np.arange(len(feats)); width = .8/len(models)
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    for i, model in enumerate(models):
        s = df[df.model == model].set_index("feature_set").reindex(feats)
        ax.bar(x-.4+width/2+i*width, s.improvement_percent, width=width, label=model)
    ax.axhline(0, linewidth=1); ax.set_xticks(x, feats); ax.set_xlabel("Feature set compared with F0")
    ax.set_ylabel("RMSE improvement (%)"); ax.set_title("RMSE improvement relative to F0")
    ax.legend(title="Model", frameon=False, ncol=2); ax.grid(axis="x", visible=False)
    save_fig(fig, "07_rmse_improvement_vs_f0", out, args.dpi, not args.skip_pdf,
             manifest, "Percentage RMSE improvement relative to F0.", args.show)


def plot_efficiency(summary: pd.DataFrame, metric: str, label: str, models: list[str],
                    features: list[str], out: Path, args: argparse.Namespace,
                    manifest: list[dict[str, str]], number: str) -> None:
    mean, std = f"{metric}_mean", f"{metric}_std"
    if mean not in summary.columns or not np.isfinite(summary[mean]).any(): return
    x = np.arange(len(features)); width = .8/len(models); fig, ax = plt.subplots(figsize=(10.2, 5.8))
    for i, model in enumerate(models):
        s = summary[summary.model == model].set_index("feature_set").reindex(features)
        ax.bar(x-.4+width/2+i*width, s[mean], width=width, yerr=s[std].fillna(0), capsize=2, label=model)
    ax.set_xticks(x, features); ax.set_xlabel("Feature set"); ax.set_ylabel(label); ax.set_yscale("log")
    ax.set_title(f"{label} by model and feature set"); ax.legend(title="Model", frameon=False, ncol=2)
    ax.grid(axis="x", visible=False)
    save_fig(fig, f"{number}_{metric}", out, args.dpi, not args.skip_pdf,
             manifest, f"{label} comparison on a log scale.", args.show)


def pareto_mask(df: pd.DataFrame, x: str, y: str) -> np.ndarray:
    a = df[[x, y]].to_numpy(float); keep = np.ones(len(a), bool)
    for i, p in enumerate(a):
        keep[i] = not (((a[:,0] <= p[0]) & (a[:,1] <= p[1]) & ((a[:,0] < p[0]) | (a[:,1] < p[1]))).any())
    return keep


def plot_pareto(summary: pd.DataFrame, cost: str, label: str, out: Path,
                args: argparse.Namespace, manifest: list[dict[str, str]], number: str) -> pd.DataFrame | None:
    col = f"{cost}_mean"
    if col not in summary.columns: return None
    df = summary[["model", "feature_set", "rmse_mean", col]].dropna().copy()
    if df.empty: return None
    df["is_pareto"] = pareto_mask(df, "rmse_mean", col)
    fig, ax = plt.subplots(figsize=(9, 6.2)); markers = {m:k for m,k in zip(MODELS,["o","s","^","D"])}
    for model in df.model.unique():
        s = df[df.model == model]; ax.scatter(s[col], s.rmse_mean, marker=markers.get(model,"o"), s=58, label=model)
        for _, r in s.iterrows(): ax.annotate(r.feature_set, (r[col], r.rmse_mean), xytext=(4,4), textcoords="offset points", fontsize=7)
    front = df[df.is_pareto].sort_values(col); ax.plot(front[col], front.rmse_mean, "--", linewidth=1.4, label="Pareto frontier")
    ax.set_xscale("log"); ax.set_xlabel(label); ax.set_ylabel("RMSE"); ax.set_title(f"Accuracy–efficiency: RMSE vs {label}")
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, f"{number}_pareto_{cost}", out, args.dpi, not args.skip_pdf,
             manifest, f"Pareto frontier for RMSE versus {label}.", args.show)
    return df


def plot_stability(metrics: pd.DataFrame, models: list[str], features: list[str], out: Path,
                   args: argparse.Namespace, manifest: list[dict[str, str]]) -> None:
    groups, labels = [], []
    for m in models:
        for f in features:
            v = metrics[(metrics.model==m)&(metrics.feature_set==f)].rmse.dropna().to_numpy(float)
            if len(v): groups.append(v); labels.append(f"{m}-{f}")
    fig, ax = plt.subplots(figsize=(13.2,6.2)); ax.boxplot(groups, labels=labels, showmeans=True, meanline=True)
    ax.set_ylabel("RMSE"); ax.set_xlabel("Model–feature configuration"); ax.set_title("RMSE stability across seeds")
    ax.tick_params(axis="x", rotation=55); ax.grid(axis="x", visible=False)
    save_fig(fig, "11_seed_stability_rmse", out, args.dpi, not args.skip_pdf,
             manifest, "RMSE distribution across seeds.", args.show)


def ranking_table(summary: pd.DataFrame) -> pd.DataFrame:
    r = summary.copy(); ranks=[]
    for c, asc in [("rmse_mean",True),("mae_mean",True),("r2_mean",False)]:
        rc=c.replace("_mean","_rank"); r[rc]=r[c].rank(ascending=asc, method="min"); ranks.append(rc)
    if "mape_mean" in r.columns:
        r["mape_rank"] = r.mape_mean.rank(ascending=True, method="min"); ranks.append("mape_rank")
    r["average_accuracy_rank"] = r[ranks].mean(axis=1)
    r = r.sort_values(["average_accuracy_rank","rmse_mean"]).reset_index(drop=True)
    r.insert(0,"overall_rank",np.arange(1,len(r)+1)); return r


def locate_prediction(directory: Path, model: str, feature: str, seed: int) -> Path | None:
    for pattern in [f"{model}_{feature}_seed{seed}.csv", f"{model.lower()}_{feature}_seed{seed}.csv", f"*{model}*{feature}*seed{seed}*.csv"]:
        matches=sorted(directory.rglob(pattern))
        if matches: return matches[0]
    return None


def plot_best_prediction(metrics: pd.DataFrame, directory: Path | None, points: int, out: Path,
                         args: argparse.Namespace, manifest: list[dict[str, str]]) -> None:
    if directory is None or not directory.exists(): return
    best=metrics.sort_values(["rmse","mae"]).iloc[0]; path=locate_prediction(directory,str(best.model),str(best.feature_set),int(best.seed))
    if path is None: LOGGER.warning("Best-run prediction file not found."); return
    raw=pd.read_csv(path); tc=find_col(raw,TRUE_CANDIDATES); pc=find_col(raw,PRED_CANDIDATES); dc=find_col(raw,TIME_CANDIDATES)
    if tc is None or pc is None: return
    df=pd.DataFrame({"actual":pd.to_numeric(raw[tc],errors="coerce"),"prediction":pd.to_numeric(raw[pc],errors="coerce")}).dropna()
    df["timestamp"] = pd.to_datetime(raw.loc[df.index,dc],errors="coerce") if dc else np.arange(len(df))
    df["error"] = df.actual-df.prediction; shown=df.iloc[:min(points,len(df))]
    fig,ax=plt.subplots(figsize=(12,5.6)); ax.plot(shown.timestamp,shown.actual,label="Actual",linewidth=1.4); ax.plot(shown.timestamp,shown.prediction,label="Predicted",linewidth=1.2)
    ax.set_xlabel("Time"); ax.set_ylabel("Electricity load"); ax.set_title(f"Best run: {best.model}-{best.feature_set}, seed {int(best.seed)} (RMSE={best.rmse:.2f})"); ax.legend(frameon=False)
    save_fig(fig,"12_best_run_forecast",out,args.dpi,not args.skip_pdf,manifest,"Actual and predicted load for the best run.",args.show)
    fig,ax=plt.subplots(figsize=(8.5,5.5)); ax.hist(df.error,bins=50); ax.axvline(0,linewidth=1); ax.set_xlabel("Residual: actual − predicted"); ax.set_ylabel("Frequency"); ax.set_title("Best-run residual distribution")
    save_fig(fig,"13_best_run_residuals",out,args.dpi,not args.skip_pdf,manifest,"Residual distribution for the best run.",args.show)
    fig,ax=plt.subplots(figsize=(6.6,6.1)); ax.scatter(df.actual,df.prediction,s=9,alpha=.45); lo=min(df.actual.min(),df.prediction.min()); hi=max(df.actual.max(),df.prediction.max()); ax.plot([lo,hi],[lo,hi],"--")
    ax.set_xlabel("Actual load"); ax.set_ylabel("Predicted load"); ax.set_title("Best-run actual vs predicted"); ax.set_aspect("equal",adjustable="box")
    save_fig(fig,"14_best_run_scatter",out,args.dpi,not args.skip_pdf,manifest,"Actual-versus-predicted scatter for the best run.",args.show)


def copy_stat_tables(dirs: list[tuple[str, Path | None, list[str]]], out: Path,
                     manifest: list[dict[str,str]]) -> None:
    for prefix,directory,names in dirs:
        if directory is None: continue
        for name in names:
            src=directory/name
            if src.exists():
                df=pd.read_csv(src); save_table(df,f"{prefix}_{src.stem}",out,manifest,f"Copy of {src} for report use.")


def main() -> int:
    args=parse_args(); setup(); models=[norm_model(x) for x in args.models]; features=[norm_feature(x) for x in args.feature_sets]
    if not args.metrics_file.exists(): LOGGER.error("Metrics file not found: %s",args.metrics_file); return 1
    out=args.output_dir.resolve(); figs=out/"figures"; tables=out/"tables"; manifest=[]
    metrics=read_metrics(args.metrics_file,models,features)
    summary=pd.read_csv(args.summary_file) if args.summary_file and args.summary_file.exists() else summarize(metrics)
    summary["model"]=summary.model.map(norm_model); summary["feature_set"]=summary.feature_set.map(norm_feature)
    save_table(metrics,"report_all_runs_metrics",tables,manifest,"All validated runs.")
    save_table(summary,"report_summary_metrics",tables,manifest,"Mean and standard deviation across seeds.")
    ranking=ranking_table(summary); save_table(ranking,"report_accuracy_ranking",tables,manifest,"Overall accuracy ranking.")
    improvement=improvement_table(summary,models); save_table(improvement,"report_rmse_improvement_vs_f0",tables,manifest,"RMSE improvements relative to F0.")
    plot_metric(summary,"rmse","RMSE",models,features,figs,args,manifest,"01")
    plot_metric(summary,"mae","MAE",models,features,figs,args,manifest,"02")
    plot_metric(summary,"mape","MAPE (%)",models,features,figs,args,manifest,"03")
    plot_metric(summary,"r2","R²",models,features,figs,args,manifest,"04")
    plot_per_model_rmse(summary,models,features,figs,args,manifest); plot_heatmap(summary,models,features,figs,args,manifest)
    plot_improvement(improvement,models,figs,args,manifest)
    plot_efficiency(summary,"training_time","Training time (s)",models,features,figs,args,manifest,"08")
    plot_efficiency(summary,"inference_time","Inference time (s)",models,features,figs,args,manifest,"09")
    p_inf=plot_pareto(summary,"inference_time","Inference time (s)",figs,args,manifest,"10")
    p_train=plot_pareto(summary,"training_time","Training time (s)",figs,args,manifest,"10b")
    if p_inf is not None: save_table(p_inf,"report_pareto_inference",tables,manifest,"Computed inference-time Pareto frontier.")
    if p_train is not None: save_table(p_train,"report_pareto_training",tables,manifest,"Computed training-time Pareto frontier.")
    plot_stability(metrics,models,features,figs,args,manifest)
    plot_best_prediction(metrics,args.prediction_dir.resolve() if args.prediction_dir else None,args.forecast_points,figs,args,manifest)
    copy_stat_tables([
        ("H1",args.h1_dir,["H1_metrics_table.csv","H1_dm_tests.csv","H1_bootstrap_ci.csv"]),
        ("H2",args.h2_dir,["H2_model_improvement.csv","H2_incremental_improvement.csv","H2_anova.csv","H2_effect_sizes.csv","H2_bootstrap_interactions.csv"]),
        ("H3",args.h3_dir,["H3_accuracy_efficiency.csv","H3_pareto_frontier.csv","H3_best_configurations.csv"]),
    ],tables,manifest)
    best=ranking.iloc[0]; best_run=metrics.sort_values(["rmse","mae"]).iloc[0]
    lines=["VISUALIZATION SUMMARY","="*72,f"Runs: {len(metrics)}",f"Configurations: {len(summary)}","",
           "Best mean configuration:",f"{best.model}-{best.feature_set} | RMSE={best.rmse_mean:.6f} | MAE={best.mae_mean:.6f} | R²={best.r2_mean:.6f}","",
           "Best individual run:",f"{best_run.model}-{best_run.feature_set}-seed{int(best_run.seed)} | RMSE={best_run.rmse:.6f} | MAE={best_run.mae:.6f} | R²={best_run.r2:.6f}"]
    if not improvement.empty:
        b=improvement.sort_values("improvement_percent",ascending=False).iloc[0]; lines += ["","Largest improvement over F0:",f"{b.model}-{b.feature_set}: {b.improvement_percent:.4f}%"]
    (out/"visualization_summary.txt").write_text("\n".join(lines),encoding="utf-8")
    pd.DataFrame(manifest).to_csv(out/"visualization_manifest.csv",index=False)
    LOGGER.info("Completed. Figures: %s | Tables: %s",figs,tables); return 0


if __name__ == "__main__":
    sys.exit(main())