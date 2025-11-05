# predict.py
import numpy as np
from PIL import Image
import argparse

from tensorflow.lite.python.interpreter import Interpreter

# --- Configuration ---
MODEL_PATH = "liveness_model_int8.tflite"
CLIP_LENGTH = 25
SENSOR_DIM = 8

def load_model():
    """Loads the TFLite model and allocates tensors."""
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image_clip(image_paths):
    """Preprocesses a list of image paths into a model-ready tensor."""
    frames = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # Normalization consistent with training
        arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        frames.append(arr)
    
    # Stack and add batch dimension
    clip = np.stack(frames, axis=0)[np.newaxis, ...]
    return clip.astype(np.float32)

def run_inference(interpreter, image_paths, sensor_data):
    """Runs inference on the TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare inputs
    img_clip = preprocess_image_clip(image_paths)
    sensor_clip = np.array(sensor_data, dtype=np.float32)[np.newaxis, ...]

    # Set tensors and invoke
    interpreter.set_tensor(input_details[0]["index"], img_clip)
    interpreter.set_tensor(input_details[1]["index"], sensor_clip)
    interpreter.invoke()

    # Get output and calculate liveness
    raw_score = interpreter.get_tensor(output_details[0]["index"])[0][0]
    liveness_score = 1 / (1 + np.exp(-raw_score)) # Sigmoid activation
    decision = "Real Face ✅" if liveness_score > 0.5 else "Fake Face ❌"
    
    return decision, liveness_score

if __name__ == "__main__":
    interpreter = load_model()
    example_images = ["data/sample.jpg"] * 25
    example_sensors = np.random.rand(CLIP_LENGTH, SENSOR_DIM).tolist()
    
    decision, score = run_inference(interpreter, example_images, example_sensors)
    print(f"Prediction: {decision}, Liveness Score: {score:.4f}")
