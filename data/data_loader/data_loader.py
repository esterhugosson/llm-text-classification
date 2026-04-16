from pathlib import Path
import csv
from typing import Dict, List

interactions_data_path = Path(__file__).parent.parent / 'dataset' / 'interactions.csv'
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


# user_id;thread_id;Title;assistant_name;message_id;Text;Created;Created_dt;turn_index_in_thread;speaker;cps_behavior;response_substance;response_stance;interactional_move;prompt_type;is_followup

truth_data_path = Path(__file__).parent.parent / 'ground_truth' / 'Manually_coded_plus_2.csv'

def loadTruths() -> Dict[str, List[Dict]]:
    data = {}
    with open(truth_data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            thread_id = row['thread_id']
            if thread_id not in data:
                data[thread_id] = []
            interaction = {
                'user_id': row.get('user_id', '').strip(),
                'title': row.get('Title', '').strip(),
                'assistant_name': row.get('assistant_name', '').strip(),
                'message_id': int(row.get('message_id', -1)) if row.get('message_id') else None,
                'text': row.get('Text', '').strip(),
                'created': row.get('Created', '').strip(),
                'created_dt': row.get('Created_dt', '').strip(),
                'turn_index_in_thread': int(row.get('turn_index_in_thread', -1)) if row.get('turn_index_in_thread') else None,
                'speaker': row.get('speaker', '').strip(),
                'cps_behavior': row.get('cps_behavior', '').strip(),
                'response_substance': row.get('response_substance', '').strip(),
                'response_stance': row.get('response_stance', '').strip(),
                'interactional_move': row.get('interactional_move', '').strip(),
                'prompt_type': row.get('prompt_type', '').strip(),
                'OLD_is_followup': row.get('OLD_is_followup', '').strip().upper() == 'SANT',
                'is_followup': row.get('is_followup', '').strip().upper() == 'SANT'
            }
            data[thread_id].append(interaction)
    
    # Sort each thread's interactions by created time (by Created field)
    for thread in data:
        data[thread].sort(key=lambda x: x['created'])
    
    return data






if __name__ == "__main__":
    import json

    interactions = loadInteractions()
    output_path = Path(__file__).parent / 'processed_interactions_plus.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(interactions, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {output_path}") 

    """ truths = loadTruths()
    output_truth_path = Path(__file__).parent / 'processed_ground_truths_plus.json'
    with open(output_truth_path, 'w', encoding='utf-8') as f:
        json.dump(truths, f, indent=2, ensure_ascii=False)
    print(f"Truth data saved to {output_truth_path}") """


