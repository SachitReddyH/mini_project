from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from ..model.sign_language_model import SignLanguageTranslator, TextToSignGenerator

app = Flask(__name__)

# Load models
model_dir = Path("models")
sign_language_model = SignLanguageTranslator.load_model(str(model_dir / "sign_language_model.h5"))
text_to_sign_model = TextToSignGenerator.load_model(str(model_dir / "text_to_sign_model.h5"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate_text', methods=['POST'])
def translate_text():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Generate sign sequence from text
        sign_sequence = text_to_sign_model.generate_sign_sequence(text)
        return jsonify({'sign_sequence': sign_sequence.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recognize_sign', methods=['POST'])
def recognize_sign():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read and preprocess image
        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict sign
        prediction = sign_language_model.predict(img)
        predicted_class = np.argmax(prediction[0])
        
        return jsonify({'predicted_sign': int(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 