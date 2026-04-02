from pathlib import Path
import json

from src.llm.gpt_4o import LLMClassifierGpt4o
from src.llm.prompt_loader import PromptLoader
from src.experiments.config import MODELS, CATEGORIES, STRATEGIES

def run_evaluation():
    loader = PromptLoader()
    classifier = LLMClassifierGpt4o(model="gpt-4o")
    # truths = loadTruths()
    
    # Extract interactions
    # all_interactions = [i for thread in truths.values() for i in thread]
    
    results = {}
    messages_per_category = 2
    
    with open("data/process_data/test_interactions.json") as f:
        data = json.load(f)

    print(f" Small test of GPT-4 Classification with Role-Based Filtering\n")

    for category, role_filter in CATEGORIES.items():
        role_label = {0: "user", 1: "chatbot", None: "all"}.get(role_filter, role_filter)
        print(f"📁 Category: {category}")
        print(f"   Role filter: {role_label}")

        category_results = {}

        for strategy in STRATEGIES:
            print(f"\n   📋 Strategy: {strategy}")
            try:
                prompt = loader.load_prompt(category, strategy)
            except FileNotFoundError as e:
                print(f"      ❌ Prompt not found: {e}")
                continue
            
            strategy_results = []
            classified_count = 0
            
            for thread_id, messages in data.items():
                for msg in messages:
                    msg_role = msg.get("role")
                    
                    # APPLY ROLE FILTER FOR THIS CATEGORY
                    if role_filter is not None:
                        if msg_role != role_filter:
                            continue
                    text = msg.get("text", "").strip()
                    msg_id = msg.get("id")
                    
                    # Skip empty text
                    if not text or len(text) < 10:
                        continue

                    # Classify
                    try:
                        print(f"      Classifying msg {msg_id} (role={msg_role})...")
                        prediction = classifier.classify(prompt, text)
                        predicted_label = prediction.get("label", "ERROR")
                        
                        # Check if label is valid
                        status = "✓" if predicted_label != "ERROR" else "✗"
                        print(f"      {status} Predicted: {predicted_label}")
                        
                        strategy_results.append({
                            "message_id": msg_id,
                            "role": msg_role,
                            "predicted_label": predicted_label,
                            "response": prediction
                        })
                        
                        classified_count += 1
                        
                        # Stop after N messages per (category, strategy)
                        if classified_count >= messages_per_category:
                            break
                    
                    except Exception as e:
                        print(f"    ✗ Error: {e}")
                    
                if classified_count >= messages_per_category:
                    break
            
            category_results[strategy] = {
                "count": classified_count,
                "results": strategy_results
            }
            print(f"      ✅ Classified {classified_count} messages")

        results[category] = category_results
    
    # Print summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}\n")
    
    total_classifications = 0
    for category, strategies in results.items():
        role_info = CATEGORIES[category]
        role_label = {0: "user", 1: "chatbot", None: "all"}.get(role_info, role_info)
        print(f"📁 {category} (role={role_label}):")
        for strategy, data in strategies.items():
            count = data["count"]
            total_classifications += count
            print(f"   • {strategy}: {count} messages")
    
    print(f"\nTotal classifications: {total_classifications}")
    print(f"{'='*70}\n")
    
    return results

if __name__ == "__main__":
    run_evaluation()