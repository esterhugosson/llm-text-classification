# Load interactions
import json

def load_interactions(path:str):

    if path is None:
        raise "No interactions path avaliable"
        return
    
    # Load ground truth data (for true_label)
    with open(path) as f:
        interactions_data = json.load(f)

    return interactions_data