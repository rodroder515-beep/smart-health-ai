import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate Synthetic Data
np.random.seed(42)
n = 2000

data = {
    'age': np.random.randint(20, 85, n),
    'sex': np.random.randint(0, 2, n),
    'temperature': np.round(np.random.normal(37.0, 1.0, n), 1),
    'heart_rate': np.random.randint(50, 130, n),
    'spo2': np.random.randint(85, 100, n)
}

df = pd.DataFrame(data)

# Simple Rule-based Target (1 = High Risk)
df['target'] = ((df['age'] > 60) | 
                (df['heart_rate'] > 100) | 
                (df['spo2'] < 94) | 
                (df['temperature'] > 38.0)).astype(int)

# Train Model
X = df[['age', 'sex', 'temperature', 'heart_rate', 'spo2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, 'health_risk_model.pkl')

print("✅ Model trained and saved successfully!")
print("Accuracy on test data:", model.score(X_test, y_test))