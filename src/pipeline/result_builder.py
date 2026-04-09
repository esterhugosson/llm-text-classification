# Build and aggregate results

import json
from pathlib import Path
from datetime import datetime
from typing import List
from src.data.models.data_models import PredictionResult


class ResultBuilder:
    """Build and save experiment results"""
    
    def __init__(self, output_dir: str = "src/results/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[PredictionResult] = []
    
    def add_result(self, result: PredictionResult):
        """Add a single result"""
        self.results.append(result)
    
    def add_results(self, results: List[PredictionResult]):
        """Add multiple results"""
        self.results.extend(results)
    
    def save(self, filename: str = None) -> str:
        """
        Save results to JSON file
        
        Args:
            filename: Custom filename (default: results_TIMESTAMP.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert results to dicts
        data = [r.to_dict() for r in self.results]
        
        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def get_stats(self) -> dict:
        """Get statistics about results"""
        total = len(self.results)
        matches = sum(1 for r in self.results if r.match)
        accuracy = matches / total if total > 0 else 0
        
        return {
            "total": total,
            "matches": matches,
            "accuracy": accuracy,
        }