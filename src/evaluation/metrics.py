from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    cohen_kappa_score
)

class Metrics:

    def __init__(self, df):
        self.df = df


    def filter(self, **kwargs):
        filtered = self.df.copy()

        for column, value in kwargs.items():
            filtered = filtered[filtered[column] == value]

        return filtered


    def overall_accuracy(self, df=None):
        df = self.df if df is None else df
        return df["match"].mean()

    def accuracy_by(self, column, df=None):
        df = self.df if df is None else df
        return df.groupby(column)["match"].mean()

    def accuracy_by_two(self, column, secondColumn, df=None):
        df = self.df if df is None else df
        return df.groupby([column, secondColumn])["match"].mean()
    
    def accuracy_by_three(self, column, secondColumn, col3, df=None):
        df = self.df if df is None else df
        return df.groupby([column, secondColumn, col3])["match"].mean()

    def accuracy_pivot(self, row, col, df=None):
        df = self.df if df is None else df

        return df.pivot_table(
            values="match",
            index=row,
            columns=col,
            aggfunc="mean"
        )

    def evaluate(self, df=None):
        df = self.df if df is None else df

        y_true = df["true_label"].astype(str).str.strip().str.lower().tolist()
        y_pred = df["predicted_label"].astype(str).str.strip().str.lower().tolist()


        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            ),
            "weighted_f1": f1_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0
            ),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            "report": classification_report(
                y_true,
                y_pred,
                zero_division=0
            )
        }
