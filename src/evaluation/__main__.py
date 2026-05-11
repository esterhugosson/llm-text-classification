from src.data.loaders.results_loader import load_results
from src.evaluation.metrics import Metrics

import json
from pathlib import Path


# ======================================
# EXPERIMENT DEFINITIONS
# ======================================

experiments = [

    # ==================================
    # CLAUDE — WITH CONTEXT
    # ==================================

    {
        "model": "claude_sonnet",
        "task": "cps_behavior",
        "context": True,
        "path": "src/results/raw/claude_cps_behavior_with_context.json"
    },

    {
        "model": "claude_sonnet",
        "task": "interactional_move",
        "context": True,
        "path": "src/results/raw/claude_interactional_move_with_context.json"
    },

    {
        "model": "claude_sonnet",
        "task": "is_followup",
        "context": True,
        "path": "src/results/raw/claude_is_followup_with_context.json"
    },

    {
        "model": "claude_sonnet",
        "task": "prompt_type",
        "context": True,
        "path": "src/results/raw/claude_prompt_type_with_context.json"
    },

    # ==================================
    # GPT-4o — WITH CONTEXT
    # ==================================

    {
        "model": "gpt4o",
        "task": "cps_behavior",
        "context": True,
        "path": "src/results/raw/gpt4o_cps_behavior_with_context.json"
    },

    {
        "model": "gpt4o",
        "task": "interactional_move",
        "context": True,
        "path": "src/results/raw/gpt4o_interactional_move_with_context.json"
    },

    {
        "model": "gpt4o",
        "task": "is_followup",
        "context": True,
        "path": "src/results/raw/gpt4o_is_followup_with_context.json"
    },

    {
        "model": "gpt4o",
        "task": "prompt_type",
        "context": True,
        "path": "src/results/raw/gpt4o_prompt_type_with_context.json"
    },

    # ==================================
    # CLAUDE — WITHOUT CONTEXT
    # ==================================

    {
        "model": "claude_sonnet",
        "task": "cps_behavior",
        "context": False,
        "path": "src/results/raw/claude_cps_behavior.json"
    },

    {
        "model": "claude_sonnet",
        "task": "interactional_move",
        "context": False,
        "path": "src/results/raw/claude_interactional_move.json"
    },

    {
        "model": "claude_sonnet",
        "task": "is_followup",
        "context": False,
        "path": "src/results/raw/claude_is_followup.json"
    },

    {
        "model": "claude_sonnet",
        "task": "prompt_type",
        "context": False,
        "path": "src/results/raw/claude_prompt_type.json"
    },

    # ==================================
    # GPT-4o — WITHOUT CONTEXT
    # ==================================

    {
        "model": "gpt4o",
        "task": "cps_behavior",
        "context": False,
        "path": "src/results/raw/gpt4o_cps_behavior.json"
    },

    {
        "model": "gpt4o",
        "task": "interactional_move",
        "context": False,
        "path": "src/results/raw/gpt4o_interactional_move.json"
    },

    {
        "model": "gpt4o",
        "task": "is_followup",
        "context": False,
        "path": "src/results/raw/gpt4o_is_followup.json"
    },

    {
        "model": "gpt4o",
        "task": "prompt_type",
        "context": False,
        "path": "src/results/raw/gpt4o_prompt_type.json"
    }
]


# ======================================
# BUILD CLEAN RESULT STRUCTURE
# ======================================

def build_results():

    all_results = []

    for exp in experiments:

        print(f"Processing: {exp['model']} | {exp['task']} | context={exp['context']}")

        # ------------------------------
        # LOAD DATASET
        # ------------------------------

        df = load_results(exp["path"])

        metrics = Metrics(df)

        # ------------------------------
        # EVALUATE BOTH STRATEGIES
        # ------------------------------

        for strategy in ["basic", "few_shot"]:

            filtered_df = metrics.filter(strategy=strategy)

            result = metrics.evaluate(filtered_df)

            # --------------------------
            # BUILD CLEAN RESULT OBJECT
            # --------------------------

            experiment_result = {

                "model": exp["model"],

                "task": exp["task"],

                "context": exp["context"],

                "strategy": strategy,

                "n_samples": len(filtered_df),

                "metrics": {

                    "precision": round(result["precision"], 3),

                    "recall": round(result["recall"], 3),

                    "macro_f1": round(result["macro_f1"], 3),

                    "accuracy": round(result["accuracy"], 3),

                    "cohen_kappa": round(result["cohen_kappa"], 3)
                }
            }

            all_results.append(experiment_result)

    return all_results


# ======================================
# SAVE JSON
# ======================================

def save_results(results):

    output_path = (
        Path(__file__).parent / "clean_experiment_results.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nSaved JSON to:\n{output_path}")


# ======================================
# MAIN
# ======================================

def main():

    results = build_results()

    save_results(results)


# ======================================
# START
# ======================================

if __name__ == "__main__":
    main()