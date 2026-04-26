import os
from datetime import datetime

import pandas as pd
from sklearn.datasets import load_iris

os.makedirs("data/raw", exist_ok=True)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

def add_some_data(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление необходимых признаков."""
    df["target"] = iris.target
    df["flower_id"] = range(1, len(df) + 1)
    df["event_timestamp"] = datetime.now()
    return df

res_data = add_some_data(df)

res_data.to_csv("data/raw/iris.csv", index=False)
res_data.to_parquet("data/raw/iris.parquet", index=False)
