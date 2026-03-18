import json
from pathlib import Path
import matplotlib.pyplot as plt

# Load the processed data
data_path = Path(__file__).parent / 'processed_interactions.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Calculate statistics
num_threads = len(data)
total_messages = sum(len(interactions) for interactions in data.values())
avg_messages_per_thread = total_messages / num_threads if num_threads > 0 else 0

print(f"Number of threads: {num_threads}")
print(f"Total messages: {total_messages}")
print(f"Average messages per thread: {avg_messages_per_thread:.2f}")

# Prepare data for stacked bar chart: messages per thread by role
threads = list(data.keys())
role_0_counts = []
role_1_counts = []

for thread in threads:
    interactions = data[thread]
    role_0 = sum(1 for msg in interactions if msg['role'] == 0)
    role_1 = sum(1 for msg in interactions if msg['role'] == 1)
    role_0_counts.append(role_0)
    role_1_counts.append(role_1)

# Create stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(threads, role_0_counts, label='Role 0 (User)', color='skyblue')
ax.bar(threads, role_1_counts, bottom=role_0_counts, label='Role 1 (Assistant)', color='salmon')

ax.set_xlabel('Thread ID')
ax.set_ylabel('Number of Messages')
ax.set_title('Messages per Thread by Role')
ax.legend()
plt.xticks(rotation=90)
plt.tight_layout()

# Save the chart
chart_path = Path(__file__).parent / 'messages_per_thread_stacked.png'
plt.savefig(chart_path)
print(f"Stacked bar chart saved to {chart_path}")

# Also, create a histogram of total messages per thread
message_counts = [len(interactions) for interactions in data.values()]

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.hist(message_counts, bins=20, edgecolor='black', alpha=0.7)
ax2.set_xlabel('Number of Messages per Thread')
ax2.set_ylabel('Number of Threads')
ax2.set_title('Distribution of Messages per Thread')
plt.tight_layout()

hist_path = Path(__file__).parent / 'messages_distribution.png'
plt.savefig(hist_path)
print(f"Histogram saved to {hist_path}")