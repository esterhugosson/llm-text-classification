import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score
)

def load_and_combine_results(paths: list) -> pd.DataFrame:
    """Load multiple result files and combine them"""
    dfs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dfs.append(pd.DataFrame(data))
    return pd.concat(dfs, ignore_index=True)

def normalize_labels(df):
    """Normalize label columns"""
    df = df.copy()
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
    return df

def calculate_metrics(df):
    """Calculate metrics for the results"""
    
    y_true = df["true_label"]
    y_pred = df["predicted_label"]
    
    # Handle cases where there might be NaN or missing values
    valid_mask = (y_true.notna()) & (y_pred.notna())
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        return None
    
    metrics = {
        "total_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "cohens_kappa": cohen_kappa_score(y_true, y_pred),
    }
    
    return metrics

print("\n" + "="*70)
print("IS_FOLLOWUP: WITH CONTEXT vs WITHOUT CONTEXT COMPARISON")
print("="*70)

# Load WITH CONTEXT
print("\nLoading WITH CONTEXT results...")
with_context_paths = [
    "src/results/raw/claude_is_followup_with_context.json",
    "src/results/raw/gpt4o_is_followup_with_context.json"
]
df_with_context = load_and_combine_results(with_context_paths)
df_with_context = normalize_labels(df_with_context)

# Load WITHOUT CONTEXT
print("Loading WITHOUT CONTEXT results...")
without_context_paths = [
    "src/results/raw/claude_is_followup.json",
    "src/results/raw/gpt4o_is_followup.json"
]
df_without_context = load_and_combine_results(without_context_paths)
df_without_context = normalize_labels(df_without_context)

# Calculate metrics
print("\nCalculating metrics...")
metrics_with = calculate_metrics(df_with_context)
metrics_without = calculate_metrics(df_without_context)


# Print results
print("\n" + "-"*70)
print(f"{'Metric':<20} {'WITH CONTEXT':>20} {'WITHOUT CONTEXT':>20} {'Difference':>15}")
print("-"*70)

if metrics_with and metrics_without:
    print(f"{'Total Samples':<20} {metrics_with['total_samples']:>20} {metrics_without['total_samples']:>20}")
    
    for metric in ["accuracy", "precision", "recall", "f1", "cohens_kappa"]:
        with_val = metrics_with[metric]
        without_val = metrics_without[metric]
        diff = with_val - without_val
        
        metric_display = "Cohen's Kappa" if metric == "cohens_kappa" else metric.title()
        print(f"{metric_display:<20} {with_val:>19.4f} {without_val:>19.4f} {diff:>14.4f}")

print("-"*70)

print("\n" + "="*70)
print("BREAKDOWN BY MODEL AND STRATEGY")
print("="*70)

for context_type, df in [("WITH CONTEXT", df_with_context), ("WITHOUT CONTEXT", df_without_context)]:
    print(f"\n{context_type}:")
    print("-"*70)
    
    for model in df["model"].unique():
        for strategy in df["strategy"].unique():
            subset = df[(df["model"] == model) & (df["strategy"] == strategy)]
            
            if len(subset) == 0:
                continue
            
            metrics = calculate_metrics(subset)
            if metrics:
                print(
                    f"  {model:<20} {strategy:<15} "
                    f"Acc: {metrics['accuracy']:.4f} | "
                    f"κ: {metrics['cohens_kappa']:.4f} | "
                    f"F1: {metrics['f1']:.4f} | "
                    f"N={metrics['total_samples']}"
                )

print("\n" + "="*70)
print("Done!")
print("="*70 + "\n")
