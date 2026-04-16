import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

# Load the processed data
data_path = Path(__file__).parent.parent / 'process_data' / 'processed_ground_truths_plus.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Flatten interactions into a list
all_interactions = [interaction for thread in data.values() for interaction in thread]

# Define which labels apply to which speaker roles
LABEL_ROLE_MAPPING = {
    'cps_behavior': ['teacher', 'user'],
    'response_substance': ['chatbot'],
    'response_stance': ['chatbot'],
    'interactional_move': ['chatbot'],
    'prompt_type': ['teacher', 'user'],
    'is_followup': ['teacher', 'user', 'chatbot'],  # all roles
}

# Categories to analyze
categories = list(LABEL_ROLE_MAPPING.keys())

def filter_interactions_for_category(interactions, category):
    """Filter interactions to only include relevant speaker roles for this category"""
    relevant_roles = LABEL_ROLE_MAPPING[category]
    return [i for i in interactions if i.get('speaker', '').strip() in relevant_roles]

results_filtered = {}

print('=' * 80)
print('Role-Filtered Ground Truth Distribution')
print('=' * 80)

for cat in categories:
    print(f'\n{cat.upper()}')
    print('-' * 80)
    
    # Filtered analysis (only relevant speaker roles)
    counter_filtered = Counter()
    filtered_interactions = filter_interactions_for_category(all_interactions, cat)
    for interaction in filtered_interactions:
        if cat not in interaction:
            continue
        value = interaction[cat]
        if cat == 'is_followup':
            if isinstance(value, bool):
                value = 'True' if value else 'False'
            elif isinstance(value, str):
                value = value.strip().title() if value.strip() else 'Unknown'
        if value is None or (isinstance(value, str) and value.strip() == ''):
            value = 'Unknown'
        counter_filtered[value] += 1
    
    results_filtered[cat] = counter_filtered
    
    # Print filtered summary
    total_filtered = sum(counter_filtered.values())
    relevant_roles = LABEL_ROLE_MAPPING[cat]
    print(f'{total_filtered} records, {len(counter_filtered)} unique values (roles: {", ".join(relevant_roles)})')
    for val, count in counter_filtered.most_common(10):
        pct = count / total_filtered * 100 if total_filtered else 0
        print(f'    {val!r}: {count} ({pct:.1f}%)')

# Generate filtered charts
print(f'\n{"=" * 80}')
print('Generating charts...')
print('=' * 80)

for cat in categories:
    labels_filtered = list(results_filtered[cat].keys())
    counts_filtered = [results_filtered[cat][k] for k in labels_filtered]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels_filtered, counts_filtered, color='tab:blue', alpha=0.8)
    ax.set_title(f'Ground truth category distribution: {cat}')
    ax.set_xlabel(cat)
    ax.set_ylabel('Count')
    ax.set_xticklabels(labels_filtered, rotation=45, ha='right')
    
    plt.tight_layout()
    
    out_path = Path(__file__).parent.parent / 'process_data' / f'NEW_ground_truth_{cat}_distribution_filtered.png'
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')

print('\nAnalysis complete!')