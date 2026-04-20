import json
import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data)