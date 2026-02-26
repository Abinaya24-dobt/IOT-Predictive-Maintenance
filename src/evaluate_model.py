import joblib
from sklearn.metrics import classification_report
from feature_engineering import load_and_prepare_data
from sklearn.model_selection import train_test_split

def evaluate():
    # Load data
    X, y = load_and_prepare_data("data/data.csv")

    # Same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load trained model
    model = joblib.load("../model/logistic_baseline.joblib")

    # Predict
    y_pred = model.predict(X_test)

    # Report
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate()