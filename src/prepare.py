import os
import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/raw/iris.csv")

X = df.drop("target", axis=1).values
y = df["target"].values

# TODO: добавить простую предподготовку

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["prepare"]["test_size"],
    random_state=params["prepare"]["random_state"],
    stratify=y,
)

os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)
