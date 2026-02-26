import os
import pandas as pd

def load_and_prepare_data(data_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, data_path)

    df = pd.read_csv(full_path)

    y = df["Machine failure"]
    X = df.drop("Machine failure", axis=1)

    X = X.drop(["Product ID"], axis=1, errors="ignore")
    X_encoded = pd.get_dummies(X, columns=["Type"], drop_first=True)

    return X_encoded, y