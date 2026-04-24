import os

import pandas as pd
from sklearn.datasets import load_iris

os.makedirs("data/raw", exist_ok=True)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

df.to_csv("data/raw/iris.csv", index=False)
