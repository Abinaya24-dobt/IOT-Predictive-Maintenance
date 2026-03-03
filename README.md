# IoT Predictive Maintenance Engine – FactoryGuard AI

##  Project Overview
FactoryGuard AI is a machine learning–based predictive maintenance system
designed for manufacturing environments.

The goal is to predict machine failures in advance using sensor data,
so maintenance can be scheduled before breakdowns happen.

---

##  Business Use Case
Manufacturing machines are equipped with sensors that record:
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

Using this historical data, the system predicts whether a machine
is likely to fail.

---

##  Machine Learning Approach
- Problem Type: Binary Classification
- Target Variable: Machine failure (0 = No Failure, 1 = Failure)
- Baseline Model: Logistic Regression
- Evaluation Metrics:
  - Precision
  - Recall
  - F1-score

---

##  Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Joblib
- Flask (API)
- Git & GitHub

---

##  Project Structure

IOT_Predictive_Maintenance_Project/
│
├── api/
│   └── app.py                 # Flask API for model inference
│
├── data/
│   └── data.csv               # Raw dataset
│
├── model/
│   └── logistic_baseline.joblib  # Trained ML model
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── config.py              # Configuration settings
│   ├── feature_engineering.py # Data preparation logic
│   ├── train_model.py         # Model training script
│   └── evaluate_model.py      # Model evaluation script
│
├── .gitignore                 # Files ignored by Git
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation