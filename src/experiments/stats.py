# Statistics tracking

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ExperimentStats:
    """Track experiment statistics"""
    
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    skipped_messages: int = 0
    
    # Per-model stats
    per_model: Dict[str, dict] = field(default_factory=dict)
    
    # Per-category stats
    per_category: Dict[str, dict] = field(default_factory=dict)
    
    def increment_success(self):
        self.successful_predictions += 1
        self.total_predictions += 1
    
    def increment_failure(self):
        self.failed_predictions += 1
        self.total_predictions += 1
    
    def increment_skip(self):
        self.skipped_messages += 1
    
    def add_model_result(self, model: str, category: str, match: bool):
        """Track result for model+category"""
        if model not in self.per_model:
            self.per_model[model] = {"correct": 0, "total": 0}
        
        self.per_model[model]["total"] += 1
        if match:
            self.per_model[model]["correct"] += 1
        
        if category not in self.per_category:
            self.per_category[category] = {"correct": 0, "total": 0}
        
        self.per_category[category]["total"] += 1
        if match:
            self.per_category[category]["correct"] += 1
    
    def get_overall_accuracy(self) -> float:
        """Get overall accuracy"""
        if self.total_predictions == 0:
            return 0.0
        return self.successful_predictions / self.total_predictions
    
    def print_summary(self):
        """Print summary statistics"""
        print(f"\n{'='*70}")
        print(f"STATISTICS")
        print(f"{'='*70}\n")
        print(f"  Total predictions:     {self.total_predictions}")
        print(f"  ✓ Successful:          {self.successful_predictions}")
        print(f"  ✗ Failed:              {self.failed_predictions}")
        print(f"  ⊘ Skipped:             {self.skipped_messages}")
        print(f"  Overall accuracy:      {self.get_overall_accuracy():.2%}")
        print(f"\n{'='*70}\n")