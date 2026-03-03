"""
IoT Predictive Maintenance API
Flask API for machine failure prediction
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'model')

try:
    model = joblib.load(os.path.join(MODEL_PATH, 'random_forest_model.joblib'))
    scaler = joblib.load(os.path.join(SCALER_PATH, 'scaler.joblib'))
    print(" Models loaded successfully!")
except Exception as e:
    print(f" Error loading models: {e}")
    model = None
    scaler = None


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "IoT Predictive Maintenance API",
        "version": "1.0.0",
        "endpoints": {
            "/": "This message",
            "/predict": "POST - Make predictions",
            "/health": "GET - Health check"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        # Expected features (must match training data)
        required_features = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Type'
        ]
        
        # Check for required fields
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing required field: {feature}"}), 400
        
        # Prepare input data
        input_data = np.array([[
            data['Air temperature [K]'],
            data['Process temperature [K]'],
            data['Rotational speed [rpm]'],
            data['Torque [Nm]'],
            data['Tool wear [min]']
        ]])
        
        # Encode Type (L=0, M=1, H=2)
        type_encoding = {'L': 0, 'M': 1, 'H': 2}
        type_val = type_encoding.get(data['Type'].upper(), 0)
        
        # Add engineered features
        temp_diff = data['Process temperature [K]'] - data['Air temperature [K]']
        power = (data['Torque [Nm]'] * data['Rotational speed [rpm]']) / 1000
        high_wear = 1 if data['Tool wear [min]'] > 200 else 0
        high_stress = 1 if (data['Torque [Nm]'] > 50 and data['Rotational speed [rpm]'] < 1500) else 0
        temp_stress = 1 if temp_diff > 11 else 0
        risk_score = high_wear + high_stress + temp_stress
        
        # Combine features
        features = np.column_stack([
            input_data,
            type_val,
            temp_diff,
            power,
            high_wear,
            high_stress,
            temp_stress,
            risk_score
        ])
        
        # Scale features
        # Note: For Random Forest, scaling is not required but keeping for consistency
        # features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return jsonify({
            "prediction": int(prediction),
            "prediction_label": "Failure" if prediction == 1 else "No Failure",
            "confidence": {
                "no_failure": float(probability[0]),
                "failure": float(probability[1])
            },
            "risk_factors": {
                "high_wear": bool(high_wear),
                "high_stress": bool(high_stress),
                "temp_stress": bool(temp_stress),
                "risk_score": int(risk_score)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
