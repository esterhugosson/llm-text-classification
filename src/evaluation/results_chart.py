from src.data.loaders.results_loader import load_results

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import json


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

def plot_model_comparison(df):

    grouped = (
        df.groupby("model")["macro_f1"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(6, 5))

    grouped.plot(kind="bar")

    plt.ylabel("Average Macro-F1")

    plt.title("Overall Model Performance")

    save_plot("01_model_comparison.png")


# ======================================
# 2. STRATEGY COMPARISON
# ======================================

def plot_strategy_comparison(df):

    grouped = (
        df.groupby("strategy")["macro_f1"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(6, 5))

    grouped.plot(kind="bar")

    plt.ylabel("Average Macro-F1")

    plt.title("Prompting Strategy Comparison")

    save_plot("02_strategy_comparison.png")


# ======================================
# 3. CONTEXT COMPARISON
# ======================================

def plot_context_comparison(df):

    grouped = (
        df.groupby("context")["macro_f1"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(6, 5))

    grouped.plot(kind="bar")

    plt.ylabel("Average Macro-F1")

    plt.title("Effect of Context")

    save_plot("03_context_comparison.png")


# ======================================
# 4. MODEL × STRATEGY INTERACTION
# ======================================

def plot_model_strategy(df):

    pivot = df.pivot_table(

        values="macro_f1",

        index="strategy",

        columns="model",

        aggfunc="mean"
    )

    plt.figure(figsize=(7, 5))

    pivot.plot(marker="o")

    plt.ylabel("Average Macro-F1")

    plt.title("Model vs Prompting Strategy")

    plt.grid(True)

    save_plot("04_model_strategy_interaction.png")


# ======================================
# 5. MODEL × CONTEXT INTERACTION
# ======================================

def plot_model_context(df):

    pivot = df.pivot_table(

        values="macro_f1",

        index="context",

        columns="model",

        aggfunc="mean"
    )

    plt.figure(figsize=(7, 5))

    pivot.plot(marker="o")

    plt.ylabel("Average Macro-F1")

    plt.title("Model vs Context")

    plt.grid(True)

    save_plot("05_model_context_interaction.png")


# ======================================
# 6. STRATEGY × CONTEXT
# ======================================

def plot_strategy_context(df):

    pivot = df.pivot_table(

        values="macro_f1",

        index="strategy",

        columns="context",

        aggfunc="mean"
    )

    plt.figure(figsize=(7, 5))

    pivot.plot(marker="o")

    plt.ylabel("Average Macro-F1")

    plt.title("Prompting Strategy vs Context")

    plt.grid(True)

    save_plot("06_strategy_context_interaction.png")


# ======================================
# 7. TASK COMPARISON
# ======================================

def plot_task_comparison(df):

    grouped = (
        df.groupby("task")["macro_f1"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))

    grouped.plot(kind="bar")

    plt.ylabel("Average Macro-F1")

    plt.title("Task Difficulty Comparison")

    save_plot("07_task_comparison.png")


# ======================================
# 8. FULL CONFIGURATION COMPARISON
# ======================================

def plot_full_configuration(df):

    df["configuration"] = (

        df["model"]
        + " | "
        + df["strategy"]
        + " | "
        + df["context"]
    )

    grouped = (
        df.groupby("configuration")["macro_f1"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))

    grouped.plot(kind="bar")

    plt.ylabel("Average Macro-F1")

    plt.title("All Experimental Configurations")

    save_plot("08_full_configuration_comparison.png")


# ======================================
# 9. KAPPA COMPARISON
# ======================================

def plot_kappa_comparison(df):

    grouped = (
        df.groupby("model")["cohen_kappa"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(6, 5))

    grouped.plot(kind="bar")

    plt.ylabel("Average Cohen's Kappa")

    plt.title("Model Agreement Comparison")

    save_plot("09_kappa_comparison.png")


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

    print_summary_tables(df)

    # ------------------------------
    # GENERATE CHARTS
    # ------------------------------

    plot_model_comparison(df)

    plot_strategy_comparison(df)

    plot_context_comparison(df)

    plot_model_strategy(df)

    plot_model_context(df)

    plot_strategy_context(df)

    plot_task_comparison(df)

    plot_full_configuration(df)

    plot_kappa_comparison(df)

    print("\nAll charts generated successfully.")


# ======================================
# START
# ======================================


run()