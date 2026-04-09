# Load interactions
import json

def load_interactions(path:str):

    if not path:
        raise ValueError("No interactions path avaliable")

    # Load ground truth data (for true_label)
    with open(path, 'r', encoding='utf-8') as f:
        interactions_data = json.load(f)

    return interactions_data

