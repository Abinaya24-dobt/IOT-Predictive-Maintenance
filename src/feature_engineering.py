import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Drop non-numeric / ID / categorical columns
    drop_cols = ["UDI", "Product ID", "Type"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Encode target column
    df["Machine failure"] = df["Machine failure"].astype(str).str.strip()

    #Map target safely
    df["Machine failure"] = df["Machine failure"].map({"No":0,"Yes":1,"0":0,"1":1})
    df = df.dropna(subset=["Machine failure"])

    # Separate features and target
    X = df.drop("Machine failure", axis=1)
    y = df["Machine failure"]

    # Feature Scaling (numeric only)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y