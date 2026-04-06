# Load ground truth labels
import json
from typing import Any

def load_ground_truths(path:str) -> Any:

    if not path:
        raise ValueError("No ground truth path available")
    
    # Load ground truth data (for true_label)
    try:
        with open(path, "r", encoding="utf-8") as f:
            ground_truths = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in file: {path}")

    return ground_truths