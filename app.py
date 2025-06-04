from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
from src.mlProject import logging
from src.mlProject.exception import CustomException
import sys
import joblib
from pathlib import Path

app = Flask(__name__)

# Load the model and preprocessor
model_path = Path('artifacts/model_trainer/model.joblib')
preprocessor_path = Path('artifacts/data_transformation/preprocessor.pkl')

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            'fixed acidity': float(request.form['fixed_acidity']),
            'volatile acidity': float(request.form['volatile_acidity']),
            'citric acid': float(request.form['citric_acid']),
            'residual sugar': float(request.form['residual_sugar']),
            'chlorides': float(request.form['chlorides']),
            'free sulfur dioxide': float(request.form['free_sulfur_dioxide']),
            'total sulfur dioxide': float(request.form['total_sulfur_dioxide']),
            'density': float(request.form['density']),
            'pH': float(request.form['ph']),
            'sulphates': float(request.form['sulphates']),
            'alcohol': float(request.form['alcohol'])
        }

        # Convert to DataFrame
        features = pd.DataFrame([data])
        logging.info("Created features DataFrame")

        # Preprocess features
        scaled_features = preprocessor.transform(features)
        logging.info("Preprocessed features")

        # Make prediction
        prediction = model.predict(scaled_features)
        logging.info("Made prediction")

        return render_template('index.html', 
                             prediction=round(prediction[0], 2))

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('index.html', 
                             error="Error making prediction. Please try again.")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)