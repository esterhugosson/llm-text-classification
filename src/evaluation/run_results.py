from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.llm.gpt_4o import LLMClassifierGpt4o
from src.llm.prompt_loader import PromptLoader
from experiments.config import CATEGORIES, STRATEGIES

def run_evaluation():
    loader = PromptLoader()
    classifier = LLMClassifierGpt4o(model="gpt-4o")
    
    messages_per_category = 5
    
    # Load test interactions (text)
    with open("data/process_data/test_interactions.json", encoding="utf-8") as f:
        interactions_data = json.load(f)
    
    # Load ground truth (labels)
    with open("data/process_data/processed_ground_truths.json", encoding="utf-8") as f:
        ground_truths = json.load(f)
    
    # Build lookup: (thread_id, message_id) -> ground truth msg
    ground_truth_lookup = {}
    for thread_id, messages in ground_truths.items():
        for msg in messages:
            msg_id = msg.get("message_id")
            key = (thread_id, msg_id)
            ground_truth_lookup[key] = msg

    results = []
    metrics = {}

    print(f"\n{'='*70}")
    print(f"Test Experiment")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Messages per category: {messages_per_category}")
    print(f"{'='*70}\n")

    for category, role_filter in CATEGORIES.items():
        role_label = {0: "teacher", 1: "chatbot", None: "all"}.get(role_filter, role_filter)
        print(f"Category: {category}")
        print(f"Role filter: {role_label}")

        category_metrics = {}

        for strategy in STRATEGIES:
            print(f"\n   Strategy: {strategy}")
            try:
                prompt = loader.load_prompt(category, strategy)
            except FileNotFoundError as e:
                print(f"      Prompt not found: {e}")
                continue
            
            y_true = []
            y_pred = []
            classified_count = 0
            
            for thread_id, messages in interactions_data.items():
                for msg in messages:
                    msg_role = msg.get("role")
                    msg_id = msg.get("id")
                    
                    # APPLY ROLE FILTER FOR THIS CATEGORY
                    if role_filter is not None:
                        if msg_role != role_filter:
                            continue
                    
                    text = msg.get("text", "").strip()
                    
                    # Skip empty text
                    if not text or len(text) < 10:
                        continue
                    
                    # GET TRUE LABEL FROM GROUND TRUTH
                    # Note: test data uses "id" not "message_id", need to check thread matching
                    gt_key = (thread_id, msg_id)
                    if gt_key not in ground_truth_lookup:
                        # Try alternative: iterate ground truth to find matching message
                        if thread_id in ground_truths:
                            found = False
                            for gt_msg in ground_truths[thread_id]:
                                if gt_msg.get("message_id") == msg_id:
                                    gt_msg_data = gt_msg
                                    found = True
                                    break
                            if not found:
                                continue
                        else:
                            continue
                    else:
                        gt_msg_data = ground_truth_lookup[gt_key]
                    
                    true_label = gt_msg_data.get(category)
                    
                    # Skip if no true label for this category
                    if true_label is None:
                        continue

                    # Classify
                    try:
                        print(f"      Classifying msg {msg_id} (role={msg_role})...")
                        prediction = classifier.classify(prompt, text)
                        predicted_label = prediction.get("label", "ERROR")
                        
                        # Check if label is valid
                        status = "✓" if predicted_label != "ERROR" else "✗"
                        match = "✓" if predicted_label == true_label else "✗"
                        print(f"      {status} Predicted: {predicted_label} | True: {true_label} {match}")
                        
                        results.append({
                            "thread_id": thread_id,
                            "message_id": msg_id,
                            "text": text[:100],
                            "category": category,
                            "strategy": strategy,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "role": msg_role,
                        })
                        
                        y_true.append(true_label)
                        y_pred.append(predicted_label if predicted_label != "ERROR" else None)
                        classified_count += 1
                        
                        # Stop after N messages per (category, strategy)
                        if classified_count >= messages_per_category:
                            break
                    
                    except Exception as e:
                        print(f"      ✗ Error: {e}")
                    
                if classified_count >= messages_per_category:
                    break
            
            # Calculate metrics for this strategy
            if y_true and y_pred:
                try:
                    # Filter out None predictions
                    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
                    if valid_pairs:
                        y_true_valid, y_pred_valid = zip(*valid_pairs)
                        
                        accuracy = accuracy_score(y_true_valid, y_pred_valid)
                        macro_f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
                        
                        category_metrics[strategy] = {
                            "count": classified_count,
                            "accuracy": accuracy,
                            "macro_f1": macro_f1,
                        }
                        
                        print(f"      Accuracy: {accuracy:.3f} | Macro F1: {macro_f1:.3f}")
                    else:
                        print(f"      No valid predictions")
                except Exception as e:
                    print(f"      Error calculating metrics: {e}")
            else:
                print(f"      No classifications")

        metrics[category] = category_metrics
    
    # Save results
    print(f"\n{'='*70}")
    print(f"  SAVING RESULTS")
    print(f"{'='*70}\n")

    output_dir = Path("src/results/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"test_results_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Results saved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  Metrics")
    print(f"{'='*70}\n")
    
    for category, strategies in metrics.items():
        print(f"{category}:")
        for strategy, metric_data in strategies.items():
            count = metric_data.get("count", 0)
            accuracy = metric_data.get("accuracy", 0)
            macro_f1 = metric_data.get("macro_f1", 0)
            print(f"   • {strategy}: {count} msgs | Accuracy: {accuracy:.3f} | F1: {macro_f1:.3f}")
    
    print(f"{'='*70}\n")
    
    return results, metrics

if __name__ == "__main__":
    results, metrics = run_evaluation()