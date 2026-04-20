from src.data.loaders.results_loader import load_results
from src.evaluation.metrics import evaluate

df = load_results("src/results/raw/results_20260416_091650.json")

def overall_accuracy(df):
    return df["match"].mean()

def accuracy_by(df, column):
    return df.groupby(column)["match"].mean()


def main():

    print(overall_accuracy(df))

    print(accuracy_by(df, "model")
    ,accuracy_by(df, "strategy")
    ,accuracy_by(df, "category"))
    



main()