# Predicts a interaction in a choosen category

from src.llm.prompt_loader import PromptLoader
from src.llm.gpt_4o import LLMClassifierGpt4o
from src.llm.claude_sonnet import LLMClassifierClaudeSonnet
from src.llm.llama_3 import LLMClassifierLlama3

class Classifier:

    def __init__(self, model) :
        self.model = model


    def predict(self, promt, text):

        try:
            prediction = self.model.classify(promt, text) 
            predicted_label = prediction.get("label")

            if predicted_label.isValid() :




    def isValid(self, label) -> bool:
        
        if label is None or label == "ERROR":
            return False
        
        return True



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


                            result_row = {
                                "thread_id": threadxe_id,
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
       