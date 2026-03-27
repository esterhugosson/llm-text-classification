import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

# Load the processed data
data_path = Path(__file__).parent /'process_data'/ 'processed_ground_truths.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Flatten interactions into a list
all_interactions = [interaction for thread in data.values() for interaction in thread]

# Categories to analyze
categories = [
    'cps_behavior',
    'response_substance',
    'response_stance',
    'interactional_move',
    'prompt_type',
    'is_followup',
]

results = {}
for cat in categories:
    counter = Counter()
    for interaction in all_interactions:
        if cat not in interaction:
            continue
        value = interaction[cat]
        if cat == 'is_followup':
            # normalize boolean-like values into strings
            if isinstance(value, bool):
                value = 'True' if value else 'False'
            elif isinstance(value, str):
                value = value.strip().title() if value.strip() else 'Unknown'
        if value is None or (isinstance(value, str) and value.strip() == ''):
            value = 'Unknown'
        counter[value] += 1
    results[cat] = counter

# Print summary stats
print('Ground truth category distribution:')
for cat, counter in results.items():
    total = sum(counter.values())
    print(f'- {cat}: {total} records, {len(counter)} unique values')
    for val, count in counter.most_common(10):
        pct = count / total * 100 if total else 0
        print(f'    {val!r}: {count} ({pct:.1f}%)')

# Plot one chart per category
for cat, counter in results.items():
    labels = list(counter.keys())
    counts = [counter[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, counts, color='tab:blue', alpha=0.8)

    ax.set_title(f'Ground truth category distribution: {cat}')
    ax.set_xlabel(cat)
    ax.set_ylabel('Count')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()

    out_path = Path(__file__).parent / f'ground_truth_{cat}_distribution.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f'Saved chart: {out_path}')
