from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('health_risk_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Validate required fields
        heart_rate = float(data.get('heart_rate', 0))
        spo2 = float(data.get('spo2', 0))
        temperature = float(data.get('temperature', 0))

        # Reject if sensor not ready
        if heart_rate == 0 or spo2 == 0:
            return jsonify({
                'risk': 0,
                'risk_probability': 0,
                'status': 'Sensor warming up',
                'prediction': 'Normal',
                'message': 'Place finger on sensor and wait.',
                'heart_rate': heart_rate,
                'temperature': temperature,
                'spo2': spo2
            }), 200

        # Use defaults for age and sex if not provided
        age = int(data.get('age', 30))
        sex = int(data.get('sex', 1))

        features = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'temperature': temperature,
            'heart_rate': heart_rate,
            'spo2': spo2
        }])

        risk = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            'risk': int(risk),
            'risk_probability': round(float(probability) * 100, 2),
            'status': 'HIGH RISK - Consult Doctor!' if risk == 1 else 'Normal',
            'prediction': 'Alert' if risk == 1 else 'Normal',
            'message': 'Alert! Abnormal vitals detected.' if risk == 1 else 'All vitals are normal.',
            'heart_rate': heart_rate,
            'temperature': temperature,
            'spo2': spo2
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return """
    <h1>Smart Health AI Prediction API</h1>
    <p>Send POST request to /predict with JSON data.</p>
    <p>Required: heart_rate, temperature, spo2</p>
    <p>Optional: age (default 30), sex (default 1)</p>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
