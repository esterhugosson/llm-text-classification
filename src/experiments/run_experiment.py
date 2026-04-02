import json
from datetime import datetime

from  src.llm.prompt_loader import PromptLoader
from src.llm.gpt_4o import LLMClassifierGpt4o
from src.llm.claude_sonnet import LLMClassifierClaudeSonnet
from src.llm.llama_3 import LLMClassifierLlama3

from src.experiments.config import MODELS, CATEGORIES, STRATEGIES


def get_model(model_name):
    if model_name == "gpt4o":
        return LLMClassifierGpt4o()
    elif model_name == "claude":
        return LLMClassifierClaudeSonnet()
    elif model_name == "llama3":
        return LLMClassifierLlama3    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run():

    loader = PromptLoader()

    with open("data/dataset/interactions.json") as f:
        data = json.load(f)

    results = []

    for model_name in MODELS:
        print(f"\nRunning model: {model_name}")
        classifier = get_model(model_name)

        for category, role_filter in CATEGORIES.items():
            print(f"  Category: {category} (role_filter={role_filter})")

            for strategy in STRATEGIES:
                print(f"    Strategy: {strategy}")

                prompt = loader.load_prompt(category, strategy)

                for thread_id, messages in data.items():

                    for msg in messages:

                        # Apply role filter for this category
                        if role_filter is not None:
                            msg_role = msg.get("role")
                            if msg_role != role_filter:
                                continue

                        text = msg.get("text", "")
                        true_label = msg.get(category)

                        if true_label is None:
                            continue

                        prediction = classifier.classify(prompt, text)

                        predicted_label = prediction.get("label")

                        result_row = {
                            "thread_id": thread_id,
                            "message_id": msg["message_id"],
                            "text": text,

                            "category": category,
                            "true_label": true_label,
                            "predicted_label": predicted_label,

                            "model": model_name,
                            "strategy": strategy,
                        }

                        results.append(result_row)

                        print(
                            f"      msg {msg['message_id']} → {predicted_label}"
                        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = f"src/results/raw/results_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    run()