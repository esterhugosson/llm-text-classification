from src.data.loaders.results_loader import load_results
from src.evaluation.metrics import Metrics

import json
from pathlib import Path


# ======================================
# LOAD ALL FILES
# ======================================

datasets = {
    "all_results": load_results("src/results/raw/experiment_1_22_april.json"),

    "claude_cps_behavior": load_results("src/results/raw/claude_cps_behavior.json"),
    "claude_interactional_move": load_results("src/results/raw/claude_interactional_move.json"),
    "claude_is_followup": load_results("src/results/raw/claude_is_followup.json"),
    "claude_prompt_type": load_results("src/results/raw/claude_prompt_type.json"),

    "gpt4o_cps_behavior": load_results("src/results/raw/gpt4o_cps_behavior.json"),
    "gpt4o_interactional_move": load_results("src/results/raw/gpt4o_interactional_move.json"),
    "gpt4o_is_followup": load_results("src/results/raw/gpt4o_is_followup.json"),
    "gpt4o_prompt_type": load_results("src/results/raw/gpt4o_prompt_type.json"),
}


# ======================================
# PRINT HELPER
# ======================================

def print_eval(title, result):
    print(f"\n===== {title.upper()} =====")
    print("Accuracy:", round(result["accuracy"], 3))
    print("Macro F1:", round(result["macro_f1"], 3))
    print("Weighted F1:", round(result["weighted_f1"], 3))
    print("Kappa:", round(result["cohen_kappa"], 3))


# ======================================
# SAVE JSON
# ======================================

def save(results):
    output_path = Path(__file__).parent / "evaluation_report.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {output_path}")


# ======================================
# CONVERT PANDAS TO JSON SAFE
# ======================================

def safe(value):

    if hasattr(value, "to_dict"):
        data = value.to_dict()

        cleaned = {}

        for key, val in data.items():

            if isinstance(key, tuple):
                key = " | ".join(str(x) for x in key)

            cleaned[str(key)] = val

        return cleaned

    return value


# ======================================
# BUILD REPORT FOR ONE DATASET
# ======================================

def build_report(df):
    metrics = Metrics(df)

    report = {
        "metrics": metrics.evaluate(),

        "accuracy_analysis": {
            "overall_accuracy": metrics.overall_accuracy(),

            "accuracy_by_model": safe(metrics.accuracy_by("model")),
            "accuracy_by_strategy": safe(metrics.accuracy_by("strategy")),
            "accuracy_by_category": safe(metrics.accuracy_by("category")),

            "pivot_model_strategy": safe(
                metrics.accuracy_pivot("model", "strategy")
            ),

            "pivot_model_category": safe(
                metrics.accuracy_pivot("model", "category")
            ),

            "pivot_category_strategy": safe(
                metrics.accuracy_pivot("category", "strategy")
            ),

            "assistant_category": safe(
                metrics.accuracy_by_two(
                    "assistant_name",
                    "category"
                )
            ),

            "model_category_strategy": safe(
                metrics.accuracy_by_three(
                    "model",
                    "category",
                    "strategy"
                )
            )
        }
    }

    return report


# ======================================
# RUN ALL
# ======================================

def main():

    all_results = {}

    for name, df in datasets.items():

        report = build_report(df)

        print_eval(name, report["metrics"])

        all_results[name] = report

    save(all_results)


# ======================================
# START
# ======================================

if __name__ == "__main__":
    main()