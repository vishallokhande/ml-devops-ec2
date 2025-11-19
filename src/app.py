# src/app.py
from flask import Flask, request, jsonify
import joblib
import os

MODEL_FILE = os.environ.get('MODEL_FILE', 'model.pkl')

app = Flask(__name__)

# Load model at startup
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file {MODEL_FILE} not found. Run train.py first.")

model = joblib.load(MODEL_FILE)

@app.route('/health')
def health():
    return 'OK', 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    instances = data.get('instances')
    if instances is None:
        return jsonify({'error': 'No instances provided'}), 400
    preds = model.predict(instances)
    return jsonify({'predictions': preds.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
