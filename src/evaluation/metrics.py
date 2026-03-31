import json
from collections import defaultdict
from sklearn.metrics import classification_report


def load_results(path):
    with open(path) as f:
        return json.load(f)


def evaluate_per_category(results):

    grouped = defaultdict(list)

    for r in results:
        grouped[r["category"]].append(r)

    for category, items in grouped.items():

        y_true = [i["true_label"] for i in items]
        y_pred = [i["predicted_label"] for i in items]

        print(f"\n=== {category} ===")
        print(classification_report(y_true, y_pred, zero_division=0))


def evaluate_per_model(results):

    grouped = defaultdict(list)

    for r in results:
        key = (r["model"], r["strategy"])
        grouped[key].append(r)

    for (model, strategy), items in grouped.items():

        y_true = [i["true_label"] for i in items]
        y_pred = [i["predicted_label"] for i in items]

        print(f"\n=== {model} | {strategy} ===")
        print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":

    path = "src/results/raw/results_latest.json"  # ändra

    results = load_results(path)

    evaluate_per_category(results)
    evaluate_per_model(results)