# Evaluate experiment results with metrics

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd


class MetricsEvaluator:
    """Evaluate classification results"""
    
    def __init__(self, results_path: str):
        """Load results from file"""
        self.results_path = results_path
        self.results = self._load_results()
    
    def _load_results(self) -> List[Dict]:
        """Load results from JSON file"""
        if not Path(self.results_path).exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        
        with open(self.results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def evaluate_overall(self):
        """Overall accuracy and F1"""
        print(f"\n{'='*70}")
        print(f"  OVERALL METRICS")
        print(f"{'='*70}\n")
        
        y_true = [r["true_label"] for r in self.results]
        y_pred = [r["predicted_label"] for r in self.results if r["predicted_label"]]
        
        # Filter to same length
        valid_results = [(t, p) for t, p in zip(y_true, y_pred) if p]
        if valid_results:
            y_true_valid, y_pred_valid = zip(*valid_results)
            
            accuracy = accuracy_score(y_true_valid, y_pred_valid)
            macro_f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
            
            print(f"  Total predictions:     {len(self.results)}")
            print(f"  Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Macro F1:              {macro_f1:.4f}")
            print(f"  Weighted F1:           {weighted_f1:.4f}")
            print(f"\n{'='*70}\n")
    
    def evaluate_per_category(self):
        """Metrics per category"""
        print(f"\n{'='*70}")
        print(f"  PER CATEGORY METRICS")
        print(f"{'='*70}\n")
        
        grouped = defaultdict(list)
        for r in self.results:
            grouped[r["category"]].append(r)
        
        for category in sorted(grouped.keys()):
            items = grouped[category]
            y_true = [i["true_label"] for i in items]
            y_pred = [i["predicted_label"] for i in items if i["predicted_label"]]
            
            valid = [(t, p) for t, p in zip(y_true, y_pred) if p]
            if valid:
                y_true_valid, y_pred_valid = zip(*valid)
                
                accuracy = accuracy_score(y_true_valid, y_pred_valid)
                macro_f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
                
                print(f"\n  {category}:")
                print(f"    Samples:    {len(items)}")
                print(f"    Accuracy:   {accuracy:.4f}")
                print(f"    Macro F1:   {macro_f1:.4f}")
                print(f"\n    Classification Report:")
                print(classification_report(y_true_valid, y_pred_valid, zero_division=0))
    
    def evaluate_per_model(self):
        """Metrics per model"""
        print(f"\n{'='*70}")
        print(f"  PER MODEL METRICS")
        print(f"{'='*70}\n")
        
        grouped = defaultdict(list)
        for r in self.results:
            key = (r["model"], r["strategy"])
            grouped[key].append(r)
        
        for (model, strategy), items in sorted(grouped.items()):
            y_true = [i["true_label"] for i in items]
            y_pred = [i["predicted_label"] for i in items if i["predicted_label"]]
            
            valid = [(t, p) for t, p in zip(y_true, y_pred) if p]
            if valid:
                y_true_valid, y_pred_valid = zip(*valid)
                
                accuracy = accuracy_score(y_true_valid, y_pred_valid)
                macro_f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
                
                print(f"\n  {model} ({strategy}):")
                print(f"    Samples:    {len(items)}")
                print(f"    Accuracy:   {accuracy:.4f}")
                print(f"    Macro F1:   {macro_f1:.4f}")
                print(f"\n    Classification Report:")
                print(classification_report(y_true_valid, y_pred_valid, zero_division=0))
    
    def evaluate_per_model_category(self):
        """Metrics per model+category combination"""
        print(f"\n{'='*70}")
        print(f"  PER MODEL+CATEGORY METRICS")
        print(f"{'='*70}\n")
        
        grouped = defaultdict(list)
        for r in self.results:
            key = (r["model"], r["category"], r["strategy"])
            grouped[key].append(r)
        
        # Create summary table
        summary_data = []
        for (model, category, strategy), items in sorted(grouped.items()):
            y_true = [i["true_label"] for i in items]
            y_pred = [i["predicted_label"] for i in items if i["predicted_label"]]
            
            valid = [(t, p) for t, p in zip(y_true, y_pred) if p]
            if valid:
                y_true_valid, y_pred_valid = zip(*valid)
                
                accuracy = accuracy_score(y_true_valid, y_pred_valid)
                macro_f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
                
                summary_data.append({
                    "Model": model,
                    "Category": category,
                    "Strategy": strategy,
                    "Samples": len(items),
                    "Accuracy": accuracy,
                    "F1": macro_f1,
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            print(df.to_string(index=False))
            print()
    
    def run_all_evaluations(self):
        """Run all evaluation types"""
        self.evaluate_overall()
        self.evaluate_per_category()
        self.evaluate_per_model()
        self.evaluate_per_model_category()
        
        print(f"{'='*70}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate experiment results"
    )
    
    parser.add_argument(
        "--results",
        required=True,
        help="Path to results JSON file (e.g., src/results/raw/results_20260409_143522.json)"
    )
    
    args = parser.parse_args()
    
    try:
        evaluator = MetricsEvaluator(args.results)
        evaluator.run_all_evaluations()
        return 0
    
    except FileNotFoundError as e:
        print(f"{e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())