from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    cohen_kappa_score
)


def evaluate(resultsJson):

    y_true = resultsJson["true_label"]
    y_pred = resultsJson["predicted_label"]

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, output_dict=False)
    }

    return results
