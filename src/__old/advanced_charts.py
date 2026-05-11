# src/evaluation/advanced_charts.py

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)

# ==================================================
# CONFIG
# ==================================================

DATA_PATH = Path(__file__).parent.parent / "results" / "raw" / "experiment_1_22_april.json"
OUTPUT_DIR = Path(__file__).parent / "advanced_charts_output"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use("ggplot")


# ==================================================
# LOAD DATA
# ==================================================

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# normalize labels
for col in ["true_label", "predicted_label"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

df["match"] = df["true_label"] == df["predicted_label"]


# ==================================================
# HELPERS
# ==================================================

def save_plot(fig, filename):
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def metric_scores(sub):
    if len(sub) == 0:
        return None

    y_true = sub["true_label"]
    y_pred = sub["predicted_label"]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def grouped_bar(df_plot, title, filename):
    fig, ax = plt.subplots(figsize=(12, 7))
    df_plot.plot(kind="bar", ax=ax)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")

    save_plot(fig, filename)


def simple_bar(df_plot, x, y, title, filename, rotate=False):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df_plot[x], df_plot[y])

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel(y)
    ax.set_xlabel("")

    if rotate:
        plt.xticks(rotation=45, ha="right")

    save_plot(fig, filename)


def heatmap(matrix, labels, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    ax.set_title(title, fontsize=14, pad=15)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i,
                matrix[i][j],
                ha="center",
                va="center",
                fontsize=8
            )

    fig.colorbar(im)
    save_plot(fig, filename)


# ==================================================
# 1. CLAUDE VS GPT4O OVERALL
# ==================================================

def model_comparison():
    rows = []

    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        m = metric_scores(sub)

        rows.append({
            "model": model,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "kappa": m["kappa"],
        })

    result = pd.DataFrame(rows).set_index("model")

    grouped_bar(
        result,
        "Claude vs GPT4o Overall Performance",
        "01_model_comparison.png"
    )


# ==================================================
# 2. HUMAN VS MODEL LABEL USAGE
# ==================================================

def label_distribution_vs_human():

    for category in sorted(df["category"].unique()):

        sub = df[df["category"] == category]

        # human = deduplicated truth rows
        human = (
            sub.drop_duplicates(
                subset=["thread_id", "message_id", "category"]
            )["true_label"]
            .value_counts(normalize=True)
        )

        rows = []

        for label in sorted(set(sub["true_label"]).union(set(sub["predicted_label"]))):

            row = {
                "label": label,
                "human": human.get(label, 0),
            }

            for model in sorted(sub["model"].unique()):
                pred = (
                    sub[sub["model"] == model]["predicted_label"]
                    .value_counts(normalize=True)
                )

                row[model] = pred.get(label, 0)

            rows.append(row)

        plot_df = pd.DataFrame(rows).set_index("label")

        grouped_bar(
            plot_df,
            f"Label Usage vs Human - {category}",
            f"02_label_usage_{category}.png"
        )


# ==================================================
# 3. CATEGORY PERFORMANCE
# ==================================================

def category_performance():

    rows = []

    for category in sorted(df["category"].unique()):
        sub = df[df["category"] == category]
        m = metric_scores(sub)

        rows.append({
            "category": category,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "kappa": m["kappa"],
        })

    plot_df = pd.DataFrame(rows).set_index("category")

    grouped_bar(
        plot_df,
        "Performance Per Category",
        "03_category_performance.png"
    )


# ==================================================
# 4. ASSISTANT COMPARISON
# ==================================================

def assistant_comparison():

    rows = []

    for assistant in sorted(df["assistant_name"].unique()):
        sub = df[df["assistant_name"] == assistant]
        m = metric_scores(sub)

        rows.append({
            "assistant": assistant,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "kappa": m["kappa"],
        })

    plot_df = pd.DataFrame(rows).set_index("assistant")

    grouped_bar(
        plot_df,
        "Conversation Assistant Comparison",
        "04_assistant_comparison.png"
    )


# ==================================================
# 5. BASIC VS FEW SHOT PER MODEL/CATEGORY
# ==================================================

def strategy_by_model_category():

    rows = []

    for (model, strategy, category), sub in df.groupby(
        ["model", "strategy", "category"]
    ):

        rows.append({
            "combo": f"{model} | {strategy} | {category}",
            "accuracy": sub["match"].mean()
        })

    plot_df = pd.DataFrame(rows)

    simple_bar(
        plot_df.sort_values("accuracy", ascending=False),
        "combo",
        "accuracy",
        "Few-shot vs Basic per Model & Category",
        "05_strategy_model_category.png",
        rotate=True
    )


# ==================================================
# 6. CONFUSION MATRICES
# ==================================================

def confusion_by_model():

    for model in sorted(df["model"].unique()):

        for category in sorted(df["category"].unique()):

            sub = df[
                (df["model"] == model) &
                (df["category"] == category)
            ]

            labels = sorted(
                list(
                    set(sub["true_label"]).union(set(sub["predicted_label"]))
                )
            )

            if len(sub) == 0:
                continue

            cm = confusion_matrix(
                sub["true_label"],
                sub["predicted_label"],
                labels=labels
            )

            heatmap(
                cm,
                labels,
                f"Confusion Matrix - {model} - {category}",
                f"06_confusion_{model}_{category}.png"
            )


# ==================================================
# 7. ROLE COMPARISON (teacher/user vs chatbot)
# ==================================================

def role_comparison():

    rows = []

    for role in sorted(df["role"].unique()):

        sub = df[df["role"] == role]
        m = metric_scores(sub)

        rows.append({
            "role": f"role_{role}",
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "kappa": m["kappa"],
        })

    plot_df = pd.DataFrame(rows).set_index("role")

    grouped_bar(
        plot_df,
        "Role Comparison (0=user, 1=assistant)",
        "07_role_comparison.png"
    )


# ==================================================
# 8. CONTEXT HELPS? FIRST vs LATER MESSAGE
# ==================================================

def context_effect():

    temp = df.copy()

    # rank messages inside each thread
    temp["position_rank"] = (
        temp.groupby("thread_id")["message_id"]
        .rank(method="dense")
    )

    temp["position"] = temp["position_rank"].apply(
        lambda x: "first" if x == 1 else "later"
    )

    rows = []

    for pos in ["first", "later"]:
        sub = temp[temp["position"] == pos]
        m = metric_scores(sub)

        rows.append({
            "position": pos,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "kappa": m["kappa"],
        })

    plot_df = pd.DataFrame(rows).set_index("position")

    grouped_bar(
        plot_df,
        "Context Effect: First Message vs Later Messages",
        "08_context_effect.png"
    )


# ==================================================
# RUN ALL
# ==================================================

def main():
    model_comparison()
    label_distribution_vs_human()
    category_performance()
    assistant_comparison()
    strategy_by_model_category()
    confusion_by_model()
    role_comparison()
    context_effect()

    print("\nAll advanced charts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()