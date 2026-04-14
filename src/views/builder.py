# View builder - handles all console output and formatting

from datetime import datetime
from typing import Dict, List, Optional
from src.experiments.stats import ExperimentStats


class ExperimentViewBuilder:
    """Handles all console output and formatting for the experiment"""
    
    def print_experiment_header(self, models: List[str], categories: List[str], timestamp: Optional[datetime] = None):
        """Print the main experiment header"""
        if timestamp is None:
            timestamp = datetime.now()
        
        print(f"\n{'='*70}")
        print(f"  STARTING EXPERIMENT")
        print(f"  {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Models: {', '.join(models)}")
        print(f"  Categories: {', '.join(categories)}")
        print(f"{'='*70}\n")
    
    def print_model_header(self, model_name: str, model_idx: int, total_models: int):
        """Print header for a model"""
        print(f"\n{'─'*70}")
        print(f"  [{model_idx}/{total_models}] Model: {model_name}")
        print(f"{'─'*70}")
    
    def print_category_header(self, category: str, category_idx: int, total_categories: int, role_label: str):
        """Print header for a category"""
        print(f"\n  [{category_idx}/{total_categories}] {category} (role={role_label})")
    
    def print_strategy_header(self, strategy: str, strategy_idx: int, total_strategies: int):
        """Print header for a strategy"""
        print(f"    [{strategy_idx}/{total_strategies}] {strategy}")
    
    def print_classification_progress(self, message_id: int, predicted_label: str, match: bool):
        """Print progress for a single classification"""
        match_sym = "✓" if match else "✗"
        print(f"      ✓ msg {message_id}: {predicted_label} {match_sym}")
    
    def print_category_summary(self, count: int):
        """Print summary for a category+strategy"""
        print(f"      → {count} predictions")
    
    def print_saving_header(self):
        """Print header for saving results"""
        print(f"\n{'='*70}")
        print(f"  SAVING RESULTS")
        print(f"{'='*70}\n")
    
    def print_save_complete(self, result_count: int, output_path: str):
        """Print save completion message"""
        print(f"  Saved {result_count} results")
        print(f"      {output_path}\n")
    
    def print_statistics(self, stats: ExperimentStats):
        """Print overall statistics"""
        print(f"{'='*70}")
        print(f"STATISTICS")
        print(f"{'='*70}\n")
        
        print(f"  Total predictions:     {stats.total_predictions}")
        print(f"  ✓ Successful:          {stats.successful_predictions}")
        print(f"  ✗ Failed:              {stats.failed_predictions}")
        print(f"  ⊘ Skipped:             {stats.skipped_messages}")
        
        accuracy = stats.successful_predictions / stats.total_predictions if stats.total_predictions > 0 else 0
        print(f"  Overall accuracy:      {accuracy:6.2%}")
    
    def print_per_model_accuracy(self, stats: ExperimentStats):
        """Print per-model accuracy breakdown"""
        print(f"\n{'─'*70}")
        print(f"  PER-MODEL ACCURACY")
        print(f"{'─'*70}\n")
        
        for model, model_stats in stats.per_model.items():
            accuracy = model_stats["correct"] / model_stats["total"] if model_stats["total"] > 0 else 0
            print(f"  {model:15} {accuracy:6.2%} ({model_stats['correct']}/{model_stats['total']})")
    
    def print_per_category_accuracy(self, stats: ExperimentStats):
        """Print per-category accuracy breakdown"""
        print(f"\n{'─'*70}")
        print(f"  PER-CATEGORY ACCURACY")
        print(f"{'─'*70}\n")
        
        for category, cat_stats in stats.per_category.items():
            accuracy = cat_stats["correct"] / cat_stats["total"] if cat_stats["total"] > 0 else 0
            print(f"  {category:20} {accuracy:6.2%} ({cat_stats['correct']}/{cat_stats['total']})")
    
    def print_completion_summary(self, output_path: str):
        """Print final completion summary"""
        print(f"\n{'='*70}\n")
        print(f"Experiment completed successfully!")
        print(f"   Results: {output_path}")