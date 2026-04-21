from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    cohen_kappa_score
)

def overall_accuracy(df):
    return df["match"].mean()

def accuracy_by(df, column):
    return df.groupby(column)["match"].mean()

def accuracy_by_two(df, column, secondColumn):
    return df.groupby(column, secondColumn)["match"].mean().trim()

def accuracy_pivot(df, row, col):
    return df.pivot_table(
        values="match",
        index=row,
        columns=col,
        aggfunc="mean"
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

    return y_true, y_pred
