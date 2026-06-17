from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained AI model
model = joblib.load('health_risk_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        features = pd.DataFrame([{
            'age': data['age'],
            'sex': data['sex'],
            'temperature': data['temperature'],
            'heart_rate': data['heart_rate'],
            'spo2': data['spo2']
        }])
        
        risk = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        return jsonify({
            'risk': int(risk),
            'risk_probability': round(float(probability) * 100, 2),
            'status': 'HIGH RISK - Consult Doctor!' if risk == 1 else 'Normal',
            'message': 'Alert! Abnormal vitals detected.' if risk == 1 else 'All vitals are normal.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# For testing in browser
@app.route('/')
def home():
    return """
    <h1>Smart Health AI Prediction API</h1>
    <p>Send POST request to /predict with JSON data.</p>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)