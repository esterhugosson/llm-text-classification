from src.data.loaders.results_loader import load_results
from src.evaluation.metrics import evaluate, overall_accuracy, accuracy_by, accuracy_by_two, accuracy_pivot

df = load_results("src/results/raw/results_20260416_091650.json")




def main():

    print("Overall Accuracy:")
    print(overall_accuracy(df))

    print("------")
    print(accuracy_by(df, "model"))
    print("------")
    print(accuracy_by(df, "strategy"))
    print("------")
    print(accuracy_by(df, "category"))

    print("------")

    print(accuracy_pivot(df, "model", "strategy"))
    print("------")

    print(accuracy_pivot(df, "model", "category"))
    print("------")

    print(accuracy_pivot(df, "category", "strategy"))
    

def test():
    ytrue, ypred = evaluate(df)

    print(ytrue, ypred)

test()
#main()