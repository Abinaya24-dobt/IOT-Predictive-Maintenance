import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from feature_engineering import load_and_prepare_data


def train():
    # Load & prepare data
    X, y = load_and_prepare_data("data/data.csv")
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model (baseline)
    model = LogisticRegression(max_iter=1000)

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    #Random Forest Model (Improved)
    rf_model = RandomForestClassifier(n_estimators=100,random_state=42)

    #Train Random Forest
    rf_model.fit(X_train,y_train)

    #Evaluate Random Forest
    rf_pred = rf_model.predict(X_test)
    print("Random Forest Classification Report")
    print(classification_report(y_test,rf_pred))

    # Save model
    import os

    os.makedirs("../model", exist_ok=True)
    joblib.dump(model, "../model/logistic_baseline.joblib")
    print(" Model saved successfully")

    joblib.dump(rf_model, "../model/random_forest.joblib")
    print(" Random Forest model saved successfully")


if __name__ == "__main__":
    train()
