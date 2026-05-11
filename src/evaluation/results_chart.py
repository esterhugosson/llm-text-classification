from src.data.loaders.results_loader import load_results

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns


# ======================================
# LOAD RESULTS
# ======================================

with open(
    "src/evaluation/clean_experiment_results.json",
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)


# ======================================
# CONVERT JSON -> DATAFRAME
# ======================================

def build_dataframe(results):

    rows = []

    for item in results:

        rows.append({

            "model": item["model"],

            "task": item["task"],

            "context": (
                "With Context"
                if item["context"]
                else "Without Context"
            ),

            "strategy": (
                "Few-shot"
                if item["strategy"] == "few_shot"
                else "Zero-shot"
            ),

            "precision": item["metrics"]["precision"],

            "recall": item["metrics"]["recall"],

            "macro_f1": item["metrics"]["macro_f1"],

            "accuracy": item["metrics"]["accuracy"],

            "cohen_kappa": item["metrics"]["cohen_kappa"],

            "n_samples": item["n_samples"]
        })

    return pd.DataFrame(rows)


# ======================================
# CREATE OUTPUT FOLDER
# ======================================

OUTPUT_DIR = Path("src/evaluation/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================
# SAVE FIGURE HELPER
# ======================================

def save_plot(filename):

    path = OUTPUT_DIR / filename

    plt.tight_layout()

    plt.savefig(path, dpi=300)

    print(f"Saved: {path}")

    plt.close()


# ======================================
# 1. MODEL COMPARISON
# ======================================

def plot_model_consistency(df):

    plt.figure(figsize=(8, 6))

    df.boxplot(
        column="macro_f1",
        by="model"
    )

    plt.title("Model Consistency Across Tasks and Configurations")

    plt.suptitle("")

    plt.ylabel("Macro-F1")

    save_plot("10_model_consistency_boxplot.png")




# ======================================
# MODEL PERFORMANCE BY TASK
# ======================================

def plot_model_by_task(df):

    pivot = df.pivot_table(

        values="macro_f1",

        index="task",

        columns="model",

        aggfunc="mean"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_ylabel("Average Macro-F1")

    ax.set_title("Model Performance Across Classification Tasks")

    save_plot("11_model_by_task.png")

# ======================================
# CONTEXT IMPACT PER TASK
# ======================================

def plot_context_by_task(df):

    pivot = df.pivot_table(

        values="macro_f1",

        index="task",

        columns="context",

        aggfunc="mean"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_ylabel("Average Macro-F1")

    ax.set_title("Impact of Context Across Tasks")

    save_plot("12_context_by_task.png")


# ======================================
# STRATEGY EFFECT PER MODEL
# ======================================

def plot_strategy_by_model(df):

    pivot = df.pivot_table(

        values="macro_f1",

        index="strategy",

        columns="model",

        aggfunc="mean"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(8, 6)
    )

    ax.set_ylabel("Average Macro-F1")

    ax.set_title("Effect of Prompting Strategy per Model")

    save_plot("13_strategy_by_model.png")

# ======================================
# CONFIGURATION HEATMAP
# ======================================

def plot_configuration_heatmap(df):

    df["config"] = (
        df["model"]
        + "\n"
        + df["strategy"]
        + "\n"
        + df["context"]
    )

    pivot = df.pivot_table(

        values="macro_f1",

        index="task",

        columns="config",

        aggfunc="mean"
    )

    plt.figure(figsize=(12, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f"
    )

    plt.title("Performance Across All Experimental Configurations")

    save_plot("14_configuration_heatmap.png")

# ======================================
# MODEL STABILITY
# ======================================

def plot_model_stability(df):

    stability = (
        df.groupby("model")["macro_f1"]
        .std()
    )

    ax = stability.plot(
        kind="bar",
        figsize=(6, 5)
    )

    ax.set_ylabel("Standard Deviation")

    ax.set_title("Model Stability Across Experiments")

    save_plot("15_model_stability.png")

# ======================================
# PRINT SUMMARY TABLES
# ======================================

def print_summary_tables(df):

    print("\n")
    print("=" * 50)
    print("OVERALL MODEL PERFORMANCE")
    print("=" * 50)

    print(
        df.groupby("model")[
            ["macro_f1", "accuracy", "cohen_kappa"]
        ].mean()
    )

    print("\n")
    print("=" * 50)
    print("PROMPTING STRATEGY PERFORMANCE")
    print("=" * 50)

    print(
        df.groupby("strategy")[
            ["macro_f1", "accuracy", "cohen_kappa"]
        ].mean()
    )

    print("\n")
    print("=" * 50)
    print("CONTEXT PERFORMANCE")
    print("=" * 50)

    print(
        df.groupby("context")[
            ["macro_f1", "accuracy", "cohen_kappa"]
        ].mean()
    )

# ======================================
# MAIN
# ======================================

def run():

    df = build_dataframe(results)

    print(df.head())



    # ------------------------------
    # GENERATE CHARTS
    # ------------------------------

    plot_model_consistency(df)

    plot_model_by_task(df)

    plot_context_by_task(df)

    plot_strategy_by_model(df)

    plot_configuration_heatmap(df)

    plot_model_stability(df)

    print("\nAll charts generated successfully.")


# ======================================
# START
# ======================================


run()