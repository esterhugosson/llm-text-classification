# Statistics tracking

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ExperimentStats:
    """Track experiment statistics"""
    
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    
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
    
