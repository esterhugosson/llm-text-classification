# src/evaluation/report_charts.py

import json
from pathlib import Path
from src.evaluation.metrics import Metrics
from src.data.loaders.results_loader import load_results

import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# CONFIG
# ==================================================

REPORT_PATH = Path(__file__).parent / "evaluation_report.json"
OUTPUT_DIR = Path(__file__).parent / "metrics_charts"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use("ggplot")


# ==================================================
# HELPERS
# ==================================================

def load_report():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_plot(fig, filename):
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def metric_table(report):
    rows = []

    for dataset_name, data in report.items():
        m = data["metrics"]

        rows.append({
            "dataset": dataset_name,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "cohen_kappa": m["cohen_kappa"]
        })

    return pd.DataFrame(rows)


def bar_chart(df, x, y, title, filename, rotate=False):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df[x], df[y])

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel(y)
    ax.set_xlabel("")

    if rotate:
        plt.xticks(rotation=45, ha="right")

    save_plot(fig, filename)


def grouped_bar(df, title, filename):
    fig, ax = plt.subplots(figsize=(12, 7))

    df.plot(kind="bar", ax=ax)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")

    save_plot(fig, filename)


def heatmap(df, title, filename):
    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(df.values, aspect="auto")

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)

    ax.set_title(title, fontsize=14, pad=15)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(
                j, i,
                round(df.iloc[i, j], 3),
                ha="center",
                va="center",
                fontsize=9
            )

    fig.colorbar(im)
    save_plot(fig, filename)


# ==================================================
# MAIN CHARTS
# ==================================================

def overall_metric_charts(report):
    df = metric_table(report)

    bar_chart(
        df.sort_values("accuracy", ascending=False),
        "dataset",
        "accuracy",
        "Accuracy by Dataset",
        "01_accuracy_by_dataset.png",
        rotate=True
    )

    metrics_only = df.set_index("dataset")[
        ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa"]
    ]

    grouped_bar(
        metrics_only,
        "All Metrics by Dataset",
        "02_all_metrics_grouped.png"
    )


def global_accuracy_charts(report):
    if "all_results" not in report:
        return

    acc = report["all_results"]["accuracy_analysis"]

    # accuracy by model
    model_df = pd.DataFrame(
        list(acc["accuracy_by_model"].items()),
        columns=["model", "accuracy"]
    )

    bar_chart(
        model_df.sort_values("accuracy", ascending=False),
        "model",
        "accuracy",
        "Overall Accuracy by Model",
        "03_accuracy_by_model.png"
    )

    # strategy
    strat_df = pd.DataFrame(
        list(acc["accuracy_by_strategy"].items()),
        columns=["strategy", "accuracy"]
    )

    bar_chart(
        strat_df.sort_values("accuracy", ascending=False),
        "strategy",
        "accuracy",
        "Overall Accuracy by Strategy",
        "04_accuracy_by_strategy.png"
    )

    # category
    cat_df = pd.DataFrame(
        list(acc["accuracy_by_category"].items()),
        columns=["category", "accuracy"]
    )

    bar_chart(
        cat_df.sort_values("accuracy", ascending=False),
        "category",
        "accuracy",
        "Overall Accuracy by Category",
        "05_accuracy_by_category.png",
        rotate=True
    )


def heatmap_charts(report):
    if "all_results" not in report:
        return

    acc = report["all_results"]["accuracy_analysis"]

    # model vs strategy
    pivot = pd.DataFrame(acc["pivot_model_strategy"]).fillna(0)

    heatmap(
        pivot,
        "Accuracy Heatmap: Model vs Strategy",
        "06_heatmap_model_strategy.png"
    )

    # model vs category
    pivot2 = pd.DataFrame(acc["pivot_model_category"]).fillna(0)

    heatmap(
        pivot2,
        "Accuracy Heatmap: Model vs Category",
        "07_heatmap_model_category.png"
    )


def dataset_focus_charts(report):
    """
    Nice charts for each single dataset file.
    Example:
    claude_prompt_type
    gpt4o_cps_behavior
    """

    rows = []

    for name, data in report.items():

        if name == "all_results":
            continue

        rows.append({
            "dataset": name,
            "accuracy": data["metrics"]["accuracy"]
        })

    df = pd.DataFrame(rows)

    bar_chart(
        df.sort_values("accuracy", ascending=False),
        "dataset",
        "accuracy",
        "Single Experiment Accuracy",
        "08_single_experiment_accuracy.png",
        rotate=True
    )


# ==================================================
# RUN
# ==================================================

def main():
    report = load_report()

    overall_metric_charts(report)
    global_accuracy_charts(report)
    heatmap_charts(report)
    dataset_focus_charts(report)

    print("\nAll charts created in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()