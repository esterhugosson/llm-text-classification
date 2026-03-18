from pathlib import Path
import csv
from typing import Dict, List

interactions_data_path = Path(__file__).parent / 'dataset' / 'interactions.csv'

def loadInteractions() -> Dict[str, List[Dict]]:
    data = {}
    with open(interactions_data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            thread_id = row['ThreadId']
            if thread_id not in data:
                data[thread_id] = []
            interaction = {
                'id': int(row['Id']),
                'text': row['Text'].strip(),
                'created': row['Created'],
                'role': int(row['Role'])
            }
            data[thread_id].append(interaction)
    
    # Sort each thread's interactions by created time
    for thread in data:
        data[thread].sort(key=lambda x: x['created'])
    
    return data

if __name__ == "__main__":
    import json
    data = loadInteractions()
    output_path = Path(__file__).parent / 'processed_interactions.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {output_path}")


