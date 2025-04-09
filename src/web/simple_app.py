import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our models
from src.models.sign_recognition import SignLanguageRecognizer
from src.models.text_to_sign import TextToSignTranslator

app = Flask(__name__)

# Initialize models
sign_recognizer = SignLanguageRecognizer()
text_translator = TextToSignTranslator()

# Get the absolute path to the templates directory
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.template_folder = template_dir

print(f"Template directory: {template_dir}")
print("Available templates:")
for template in os.listdir(template_dir):
    print(f"  - {template}")

@app.route('/')
def index():
    return render_template('simple_index.html')

@app.route('/recognize_sign', methods=['POST'])
def recognize_sign():
    try:
        # Check if the image is in the request files
        if 'image' in request.files:
            file = request.files['image']
            # Read the image from the file
            img_array = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # Check if the image is in the request JSON
        elif request.is_json and 'image' in request.json:
            # Get the base64 image data
            img_data = request.json['image'].split(',')[1]
            # Decode the base64 image data
            img_array = np.frombuffer(base64.b64decode(img_data), np.uint8)
            # Decode the image
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Use our sign recognizer to recognize the sign
        sign, confidence = sign_recognizer.recognize_sign(img)
        
        return jsonify({
            'sign': sign,
            'confidence': confidence
        })
    except Exception as e:
        print(f"Error in recognize_sign: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/translate_text', methods=['POST'])
def translate_text():
    try:
        # Get the text from the request
        text = request.json.get('text', '')
        
        # Use our text translator to translate the text
        signs = text_translator.translate(text)
        
        return jsonify({
            'signs': signs
        })
    except Exception as e:
        print(f"Error in translate_text: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 