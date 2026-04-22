import json
from collections import Counter, defaultdict
from pathlib import Path
from src.evaluation.metrics import Metrics
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# CONFIG
# ==================================================

DATA_PATH = Path(__file__).parent.parent / "results" / "raw" / "experiment_1_22_april.json"
OUTPUT_DIR = Path(__file__).parent / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

COLUMNS_TO_COMPARE = ["model", "strategy", "assistant_name"]
TOP_N = 20


# ==================================================
# LOAD DATA
# ==================================================

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# normalize text labels
df["true_label"] = (
    df["true_label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["predicted_label"] = (
    df["predicted_label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["match"] = df["true_label"] == df["predicted_label"]

print("Loaded rows:", len(df))
print("Columns:", list(df.columns))


# ==================================================
# BASIC OVERVIEW
# ==================================================

print("\n========== DATA OVERVIEW ==========")
print(df[["model", "strategy", "category", "assistant_name"]].nunique())

print("\nRows per category:")
print(df["category"].value_counts())

print("\nRows per model:")
print(df["model"].value_counts())

print("\nRows per strategy:")
print(df["strategy"].value_counts())


# ==================================================
# OVERALL ACCURACY
# ==================================================

print("\n========== OVERALL ACCURACY ==========")
print(round(df["match"].mean(), 4))


# ==================================================
# ACCURACY BY GROUP
# ==================================================

for col in COLUMNS_TO_COMPARE:
    print(f"\n========== ACCURACY BY {col.upper()} ==========")
    print(df.groupby(col)["match"].mean().round(4))


print("\n========== ACCURACY BY MODEL + STRATEGY ==========")
pivot = df.pivot_table(
    values="match",
    index="model",
    columns="strategy",
    aggfunc="mean"
).round(4)

print(pivot)


print("\n========== ACCURACY BY CATEGORY + MODEL ==========")
pivot2 = df.pivot_table(
    values="match",
    index="category",
    columns="model",
    aggfunc="mean"
).round(4)

print(pivot2)


# ==================================================
# TRUE LABEL DISTRIBUTION
# ==================================================

print("\n========== TRUE LABEL DISTRIBUTION PER CATEGORY ==========")

ground_truth_df = df.drop_duplicates(
    subset=["thread_id", "message_id", "category"]
)

categories = ground_truth_df["category"].unique()

for cat in categories:
    sub = ground_truth_df[ground_truth_df["category"] == cat]

    counter = Counter(sub["true_label"])
    total = sum(counter.values())

    print(f"\n--- {cat} ({total} rows) ---")

    for label, count in counter.most_common(TOP_N):
        pct = count / total * 100
        print(f"{label}: {count} ({pct:.1f}%)")


# ==================================================
# PREDICTED LABEL DISTRIBUTION
# ==================================================

print("\n========== PREDICTED LABEL DISTRIBUTION PER CATEGORY ==========")

for cat in categories:
    sub = df[df["category"] == cat]

    counter = Counter(sub["predicted_label"])
    total = sum(counter.values())

    print(f"\n--- {cat} ({total} rows) ---")

    for label, count in counter.most_common(TOP_N):
        pct = count / total * 100
        print(f"{label}: {count} ({pct:.1f}%)")


# ==================================================
# MODEL COMPARISON PER CATEGORY
# ==================================================

print("\n========== MODEL PERFORMANCE PER CATEGORY ==========")

for cat in categories:
    sub = df[df["category"] == cat]

    print(f"\n--- {cat} ---")
    print(
        sub.groupby("model")["match"]
        .mean()
        .sort_values(ascending=False)
        .round(4)
    )


# ==================================================
# STRATEGY COMPARISON PER CATEGORY
# ==================================================

print("\n========== STRATEGY PERFORMANCE PER CATEGORY ==========")

for cat in categories:
    sub = df[df["category"] == cat]

    print(f"\n--- {cat} ---")
    print(
        sub.groupby("strategy")["match"]
        .mean()
        .sort_values(ascending=False)
        .round(4)
    )


# ==================================================
# CHARTS
# ==================================================

def save_bar(counter, title, filename):
    labels = list(counter.keys())
    values = list(counter.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values)

    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


# true label charts
for cat in categories:
    sub = df[df["category"] == cat]
    counter = Counter(sub["true_label"])

    save_bar(
        counter,
        f"True label distribution - {cat}",
        f"{cat}_true_distribution.png"
    )


# predicted label charts
for cat in categories:
    sub = df[df["category"] == cat]
    counter = Counter(sub["predicted_label"])

    save_bar(
        counter,
        f"Predicted label distribution - {cat}",
        f"{cat}_pred_distribution.png"
    )


# ==================================================
# MOST COMMON ERRORS
# ==================================================

print("\n========== MOST COMMON MISCLASSIFICATIONS ==========")

errors = df[df["match"] == False].copy()
errors["error_pair"] = (
    errors["true_label"] + " -> " + errors["predicted_label"]
)

for cat in categories:
    sub = errors[errors["category"] == cat]

    print(f"\n--- {cat} ---")

    counter = Counter(sub["error_pair"])

    for pair, count in counter.most_common(10):
        print(pair, ":", count)


# ==================================================
# SAVE SUMMARY CSV
# ==================================================

summary = df.groupby(
    ["category", "model", "strategy"]
)["match"].agg(
    rows="count",
    accuracy="mean"
).reset_index()

summary["accuracy"] = summary["accuracy"].round(4)

summary.to_csv(
    OUTPUT_DIR / "summary_accuracy.csv",
    index=False
)

print("\nSaved:", OUTPUT_DIR / "summary_accuracy.csv")


# ==================================================
# DONE
# ==================================================

print("\nAnalysis complete.")