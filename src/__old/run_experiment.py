import json
import traceback
from datetime import datetime
from pathlib import Path

from src.llm.prompt_loader import PromptLoader
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

    # Load interactions data (for text)
    with open("data/dataset/test_interactions.json") as f:
        interactions_data = json.load(f)

    # Load ground truth data (for true_label)
    with open("data/process_data/processed_ground_truths.json") as f:
        ground_truths = json.load(f)

    # Build lookup: (thread_id, message_id) -> true_labels dict
    ground_truth_lookup = {}
    for thread_id, messages in ground_truths.items():
        for msg in messages:
            msg_id = msg.get("message_id")
            key = (thread_id, msg_id)
            ground_truth_lookup[key] = msg

    results = []
    stats = {
        "total_predictions": 0,
        "successful_predictions": 0,
        "failed_predictions": 0,
        "skipped_messages": 0,
    }

    print(f"\n{'='*70}")
    print(f"  🚀 STARTING LLM CLASSIFICATION EXPERIMENT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for model_idx, model_name in enumerate(MODELS, 1):
        print(f"\n{'─'*70}")
        print(f"  [{model_idx}/{len(MODELS)}] 🤖 Model: {model_name}")
        print(f"{'─'*70}")

        try:
            classifier = get_model(model_name)
        except ValueError as e:
            print(f"  ❌ {e}")
            continue

        for cat_idx, (category, role_filter) in enumerate(CATEGORIES.items(), 1):
            role_label = {0: "teacher", 1: "chatbot", None: "all"}.get(role_filter, role_filter)
            print(f"\n  [{cat_idx}/{len(CATEGORIES)}] 📁 Category: {category}")
            print(f"      Role filter: {role_label}")

            for strat_idx, strategy in enumerate(STRATEGIES, 1):
                print(f"    [{strat_idx}/{len(STRATEGIES)}] 📋 Strategy: {strategy}")

                try:
                    prompt = loader.load_prompt(category, strategy)
                except FileNotFoundError as e:
                    print(f"      ❌ Prompt not found: {e}")
                    continue

                category_strategy_count = 0

                for thread_id, messages in interactions_data.items():

                    for msg in messages:
                        msg_role = msg.get("role")
                        message_id = msg.get("message_id")

                        # Apply role filter for this category
                        if role_filter is not None:
                            if msg_role != role_filter:
                                stats["skipped_messages"] += 1
                                continue

                        text = msg.get("text", "").strip()

                        # Skip empty text
                        if not text or len(text) < 10:
                            stats["skipped_messages"] += 1
                            continue

                        # GET TRUE LABEL FROM GROUND TRUTH
                        gt_key = (thread_id, message_id)
                        if gt_key not in ground_truth_lookup:
                            stats["skipped_messages"] += 1
                            continue

                        gt_msg = ground_truth_lookup[gt_key]
                        true_label = gt_msg.get(category)

                        # Skip if no true label for this category
                        if true_label is None:
                            stats["skipped_messages"] += 1
                            continue

                        # Classify
                        try:
                            prediction = classifier.classify(prompt, text)
                            predicted_label = prediction.get("label")

                            # Check if label is valid
                            if predicted_label is None or predicted_label == "ERROR":
                                status = "✗"
                                stats["failed_predictions"] += 1
                            else:
                                status = "✓"
                                stats["successful_predictions"] += 1

                            stats["total_predictions"] += 1

                            result_row = {
                                "thread_id": thread_id,
                                "message_id": message_id,
                                "text": text[:200],  # Truncate long texts
                                "category": category,
                                "true_label": true_label,
                                "predicted_label": predicted_label,
                                "model": model_name,
                                "strategy": strategy,
                                "role": msg_role,
                            }

                            results.append(result_row)
                            category_strategy_count += 1

                            # Show match status
                            match = "✓" if predicted_label == true_label else "✗"
                            print(
                                f"      {status} msg {message_id}: {predicted_label} {match}"
                            )

                        except Exception as e:
                            print(f"      ✗ Error on msg {message_id}: {e}")
                            stats["failed_predictions"] += 1
                            stats["total_predictions"] += 1

                print(f"      → {category_strategy_count} predictions")

    # Save results
    print(f"\n{'='*70}")
    print(f"  💾 SAVING RESULTS")
    print(f"{'='*70}\n")

    output_dir = Path("src/results/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Results saved to: {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  📊 EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")
    print(f"  Total Predictions Attempted: {stats['total_predictions']}")
    print(f"  ✓ Successful:                {stats['successful_predictions']}")
    print(f"  ✗ Failed:                    {stats['failed_predictions']}")
    print(f"  ⊘ Skipped:                   {stats['skipped_messages']}")

    if stats["total_predictions"] > 0:
        success_rate = (
            stats["successful_predictions"] / stats["total_predictions"] * 100
        )
        print(f"  Success Rate:                {success_rate:.1f}%")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    run()