
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('salary_model.pkl')

@app.route('/')
def home():
    return "Salary Prediction Application is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)

    return jsonify({
        'Predicted Salary': float(prediction[0])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
