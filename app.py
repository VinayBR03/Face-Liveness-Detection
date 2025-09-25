# /app.py

import base64
import io
import numpy as np
import torch
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image

# --- Configuration ---
MODEL_PATH = "liveness_model.tflite"

# --- Initialize Flask App and Model ---
app = Flask(__name__)

# 1. Load TFLite Model and allocate tensors.
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded successfully.")
    # Print input shapes for debugging
    print("Expected Input Shapes:", [detail['shape'] for detail in input_details])
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'. The app will not work.")
    interpreter = None
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None

# 2. Define Clip Preprocessing for TFLite
def preprocess_clip(image_b64_list):
    """Preprocesses a list of base64 encoded images into a clip tensor for TFLite."""
    clip_frames = []
    for image_b64 in image_b64_list:
        image_bytes = base64.b64decode(image_b64.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize to [0, 1]
        image_np = np.array(image, dtype=np.float32) / 255.0
        clip_frames.append(image_np)

    # Stack frames to form a clip and add a batch dimension.
    # The expected shape is (1, clip_length, height, width, channels).
    # Note: The TFLite model expects channels-last format (H, W, C).
    image_clip_np = np.stack(clip_frames, axis=0)
    return np.expand_dims(image_clip_np, axis=0).astype(np.float32)

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction for a clip of frames and sensor data."""
    if interpreter is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    image_clip_b64 = data.get('image_clip')
    sensor_clip_data = data.get('sensor_clip')

    if not image_clip_b64:
        return jsonify({'error': 'Image clip data is missing.'}), 400

    # Preprocess the full image clip
    image_clip_tensor = preprocess_clip(image_clip_b64)

    # Preprocess the full sensor clip
    if sensor_clip_data and len(sensor_clip_data) > 0:
        # Add a batch dimension to match model's expected shape (1, 5, 8)
        sensor_clip_tensor = np.expand_dims(np.array(sensor_clip_data, dtype=np.float32), axis=0)
    else:
        # If no sensor data, create a zero-tensor
        sensor_clip_tensor = np.zeros(input_details[0]['shape'], dtype=np.float32)

    # Set tensors for the TFLite interpreter
    # IMPORTANT: Match the tensor to the correct input detail index.
    # Based on the logs, input 0 is sensor data and input 1 is image data.
    sensor_input_index = input_details[0]['index']
    image_input_index = input_details[1]['index']

    # Check which input is which based on shape
    if len(input_details[0]['shape']) == 3: # Sensor shape is (1, 5, 8)
        sensor_input_index = input_details[0]['index']
        image_input_index = input_details[1]['index']
    else: # Image shape is (1, 5, 224, 224, 3)
        image_input_index = input_details[0]['index']
        sensor_input_index = input_details[1]['index']
    
    # The TFLite model expects a channels-first format (N, D, C, H, W).
    # We must transpose the image clip from (1, 5, 224, 224, 3) to (1, 5, 3, 224, 224).
    image_clip_tensor = np.transpose(image_clip_tensor, (0, 1, 4, 2, 3))

    interpreter.set_tensor(sensor_input_index, sensor_clip_tensor)
    interpreter.set_tensor(image_input_index, image_clip_tensor)

    # Run inference
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    liveness_score = 1 / (1 + np.exp(-output[0][0])) # Apply sigmoid manually
    return jsonify({'liveness_score': liveness_score})

if __name__ == '__main__':
    app.run(debug=True)