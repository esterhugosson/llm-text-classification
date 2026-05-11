import pandas as pd
import matplotlib.pyplot as plt
from src.data.loaders.results_loader import load_results
from src.evaluation.metrics import Metrics


# -----------------------------------
# LOAD FILES
# -----------------------------------

datasets = {
    "claude_cps": load_results("src/results/raw/claude_cps_behavior.json"),
    "claude_inter": load_results("src/results/raw/claude_interactional_move.json"),
    "claude_is": load_results("src/results/raw/claude_is_followup.json"),
    "claude_prompt": load_results("src/results/raw/claude_prompt_type.json"),

    "gpt_cps": load_results("src/results/raw/gpt4o_cps_behavior.json"),
    "gpt_inter": load_results("src/results/raw/gpt4o_interactional_move.json"),
    "gpt_is": load_results("src/results/raw/gpt4o_is_followup.json"),
    "gpt_prompt": load_results("src/results/raw/gpt4o_prompt_type.json"),
}


# -----------------------------------
# CATEGORY NAME HELPER
# -----------------------------------

def get_category(dataset_name):
    if "cps" in dataset_name:
        return "cps_behavior"
    elif "inter" in dataset_name:
        return "interactional_move"
    elif "is" in dataset_name:
        return "is_followup"
    elif "prompt" in dataset_name:
        return "prompt_type"

def kappaByCategoryChart():

    rows = []

    for name, df in datasets.items():

        metrics = Metrics(df)
        category = get_category(name)

        for strategy in ["basic", "few_shot"]:

            sub = metrics.filter(strategy=strategy)
            result = metrics.evaluate(sub)

            rows.append({
                "category": category,
                "strategy": strategy,
                "cohen_kappa": result["cohen_kappa"]
            })

    results_df = pd.DataFrame(rows)

    # Average kappa across model + strategy combinations
    summary = (
        results_df
        .groupby("category")["cohen_kappa"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    print(summary)

    # -----------------------------------
    # Plot
    # -----------------------------------

    plt.figure(figsize=(10, 6))

    plt.barh(
        summary["category"],
        summary["cohen_kappa"]
    )

    plt.xlabel("Average Cohen's Kappa")
    plt.title("Agreement with Human Coding by Category")
    plt.gca().invert_yaxis()   # highest on top
    plt.tight_layout()
    plt.show()

# -----------------------------------
# HARDEST / EASIEST CATEGORY
# -----------------------------------

def categoryDifficultyChart():

    rows = []

    for name, df in datasets.items():

        metrics = Metrics(df)
        category = get_category(name)

        for strategy in ["basic", "few_shot"]:

            sub = metrics.filter(strategy=strategy)
            result = metrics.evaluate(sub)

            rows.append({
                "category": category,
                "strategy": strategy,
                "macro_f1": result["macro_f1"]
            })

    results_df = pd.DataFrame(rows)

    # Average across model + strategy combinations
    summary = (
        results_df
        .groupby("category")["macro_f1"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    print(summary)

    # -----------------------------------
    # Plot ranking chart
    # -----------------------------------

    plt.figure(figsize=(10, 6))

    plt.barh(
        summary["category"],
        summary["macro_f1"]
    )

    plt.xlabel("Average Macro F1")
    plt.title("Category Difficulty Ranking (Higher = Easier)")
    plt.gca().invert_yaxis()   # best on top
    plt.tight_layout()
    plt.show()


# -----------------------------------
# RUN
# -----------------------------------

kappaByCategoryChart()


# -----------------------------------
# AVG MACRO F1 BY STRATEGY
# -----------------------------------

def avgF1ByStrategy():

    rows = []

    for name, df in datasets.items():

        metrics = Metrics(df)

        # detect model name from dataset key
        if "claude" in name:
            model = "Claude"
        else:
            model = "GPT4o"

        for strategy in ["basic", "few_shot"]:

            sub = metrics.filter(strategy=strategy)

            result = metrics.evaluate(sub)

            rows.append({
                "dataset": name,
                "model": model,
                "strategy": strategy,
                "macro_f1": result["macro_f1"]
            })

    results_df = pd.DataFrame(rows)

    # -----------------------------------
    # Average over categories
    # -----------------------------------

    summary = (
        results_df
        .groupby(["model", "strategy"])["macro_f1"]
        .mean()
        .reset_index()
    )

    print(summary)

    # -----------------------------------
    # Prepare chart data
    # -----------------------------------

    models = summary["model"].unique()

    basic_scores = []
    few_scores = []

    for model in models:

        basic_val = summary[
            (summary["model"] == model) &
            (summary["strategy"] == "basic")
        ]["macro_f1"].values[0]

        few_val = summary[
            (summary["model"] == model) &
            (summary["strategy"] == "few_shot")
        ]["macro_f1"].values[0]

        basic_scores.append(basic_val)
        few_scores.append(few_val)

    # -----------------------------------
    # Plot
    # -----------------------------------

    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - width/2 for i in x],
        basic_scores,
        width=width,
        label="Zero-shot (basic)"
    )

    plt.bar(
        [i + width/2 for i in x],
        few_scores,
        width=width,
        label="Few-shot"
    )

    plt.xticks(x, models)
    plt.ylabel("Average Macro F1")
    plt.title("Zero-shot vs Few-shot Across All Categories")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------
# RUN
# -----------------------------------

# -----------------------------------
# COLLECT RESULTS
# -----------------------------------

""" rows = []

for name, df in datasets.items():

    metrics = Metrics(df)

    for strategy in ["basic", "few_shot"]:

        filtered_df = metrics.filter(strategy=strategy)

        result = metrics.evaluate(filtered_df)

        rows.append({
            "dataset": name,
            "strategy": strategy,
            "name": f"{name}_{strategy}",
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "cohen_kappa": result["cohen_kappa"]
        })


results_df = pd.DataFrame(rows)


# -----------------------------------
# FIND BESTS
# -----------------------------------

best_accuracy = results_df.sort_values(
    "accuracy",
    ascending=False
)

best_f1 = results_df.sort_values(
    "macro_f1",
    ascending=False
)

best_kappa = results_df.sort_values(
    "cohen_kappa",
    ascending=False
)


# -----------------------------------
# CHART FUNCTION
# -----------------------------------

def plot_metric(df, metric, title):

    plt.figure(figsize=(12, 6))

    plt.bar(df["name"], df[metric])

    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# -----------------------------------
# SHOW CHARTS
# -----------------------------------

plot_metric(
    best_accuracy,
    "accuracy",
    "Best Model + Category + Strategy by Accuracy"
)

plot_metric(
    best_f1,
    "macro_f1",
    "Best Model + Category + Strategy by Macro F1"
)

plot_metric(
    best_kappa,
    "cohen_kappa",
    "Best Model + Category + Strategy by Cohen Kappa"
) """