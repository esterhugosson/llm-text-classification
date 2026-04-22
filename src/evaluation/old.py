from src.data.loaders.results_loader import load_results
from src.evaluation.metrics import Metrics
import json
from pathlib import Path


# Files to all data
df = load_results("src/results/raw/experiment_1_22_april.json")

claude_cps = load_results("src/results/raw/claude_cps_behavior.json")
claude_inter = load_results("src/results/raw/claude_interactional_move.json")
claude_is = load_results("src/results/raw/claude_is_followup.json")
claude_prompt = load_results("src/results/raw/claude_prompt_type.json")
gpt_cps = load_results("src/results/raw/gpt4o_cps_behavior.json")
gpt_inter = load_results("src/results/raw/gpt4o_interactional_move.json")
gpt_is = load_results("src/results/raw/gpt4o_is_followup.json")
gpt_prompt = load_results("src/results/raw/gpt4o_prompt_type.json")

true_labels = []
pred_labels = []

metrics = Metrics(df)


def main():
    print("------")

    print("-------ACCURACY-------------")

    print("------")
    print("Overall Accuracy:")
    print(metrics.overall_accuracy())

    print("------")
    print(metrics.accuracy_by("model"))
    print("------")
    print(metrics.accuracy_by("strategy"))
    print("------")
    print(metrics.accuracy_by("category"))

    print("------")

    print(metrics.accuracy_pivot("model", "strategy"))
    print("------")

    print(metrics.accuracy_pivot("model", "category"))
    print("------")

    print(metrics.accuracy_pivot("category", "strategy"))
    print("------")
    print(
    metrics.accuracy_by_two(
        "assistant_name",
        "category"
    )
    )
    print("------")

    print(metrics.accuracy_by_three( "model", "category", "strategy"))
    


def print_eval(title, result):
    print("\n", title)
    print("Accuracy:", round(result["accuracy"], 3))
    print("Macro F1:", round(result["macro_f1"], 3))
    print("Weighted F1:", round(result["weighted_f1"], 3))
    print("Kappa:", round(result["cohen_kappa"], 3))


def save(results):
    output_path = Path(__file__).parent / 'report.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {output_path}") 

def test():

    print_eval("claude cps behavio",  metrics.evaluate())
    
    


main()