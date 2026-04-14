from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from tensorflow import keras
from PIL import Image
import os

app = Flask(__name__)

# Load models
print("Loading models...")
bank_model = joblib.load('bank_marketing_model.pkl')
bank_scaler = joblib.load('bank_scaler.pkl')
flower_model = keras.models.load_model('flower_model.h5')
print("✅ Ready!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/bank')
def bank():
    return render_template('bank.html')

@app.route('/flower')
def flower():
    return render_template('flower.html')

# Bank prediction
@app.route('/predict_bank', methods=['POST'])
def predict_bank():
    data = request.form
    features = [[
        float(data['age']), float(data['job']), float(data['marital']),
        float(data['education']), float(data['balance']), float(data['housing']),
        float(data['loan']), float(data['duration']), float(data['campaign']),
        float(data['pdays']), float(data['previous'])
    ]]
    
    scaled = bank_scaler.transform(features)
    pred = bank_model.predict(scaled)[0]
    prob = bank_model.predict_proba(scaled)[0][pred] * 100
    
    result = "✅ Subscribe" if pred == 1 else "❌ Not Subscribe"
    return jsonify({'result': result, 'confidence': f'{prob:.1f}%'})

# Flower prediction
@app.route('/predict_flower', methods=['POST'])
def predict_flower():
    file = request.files['file']
    path = 'temp.jpg'
    file.save(path)
    
    img = Image.open(path).resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prob = float(flower_model.predict(img_array)[0][0])
    os.remove(path)
    
    if prob > 0.5:
        return jsonify({'result': '🌷 Tulip', 'confidence': f'{prob*100:.1f}%'})
    else:
        return jsonify({'result': '🌻 Daisy', 'confidence': f'{(1-prob)*100:.1f}%'})

if __name__ == '__main__':
    app.run(debug=True)