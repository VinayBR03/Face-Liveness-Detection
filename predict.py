# /liveness_detection/predict.py

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import MultiModalLivenessModel

# --- Configuration ---
MODEL_PATH = "liveness_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path, device):
    """Loads the model and sets it to evaluation mode."""
    model = MultiModalLivenessModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_clip(image_paths):
    """Preprocesses a list of image paths into a clip tensor."""
    clip_frames = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        clip_frames.append(image_tensor)
    
    # Stack frames to form a clip and add a batch dimension
    return torch.stack(clip_frames, dim=0).unsqueeze(0)

def run_inference(model, image_clip_paths, sensor_clip_data):
    """
    Runs liveness detection on a clip of data.
    
    Returns:
        A tuple of (decision_str, liveness_score)
    """
    # 1. Preprocess Inputs
    image_clip_tensor = preprocess_clip(image_clip_paths).to(DEVICE)
    sensor_clip_tensor = torch.tensor(sensor_clip_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 2. Run Inference
    with torch.no_grad():
        output = model(image_clip_tensor, sensor_clip_tensor)
        
    # 3. Get Score and Decision
    liveness_score = torch.sigmoid(output).item()
    decision = "Live" if liveness_score > 0.5 else "Spoof"
    
    return decision, liveness_score

if __name__ == '__main__':
    try:
        # --- Example Usage ---
        # This demonstrates how to run inference on a clip of 100 frames.
        model = load_model(MODEL_PATH, DEVICE)
        print(f"Model loaded on {DEVICE} for inference.")

        # Create a dummy clip of 100 identical images for demonstration
        example_image_paths = ["data/train/live_train_0000.jpg"] * 100
        
        # Dummy sensor data for a clip of 5 frames (5x8 values)
        example_sensor_clip = np.random.rand(5, 8).tolist()

        decision, score = run_inference(model, example_image_paths, example_sensor_clip)
        print(f"Prediction Result: {decision}")
        print(f"Liveness Score: {score:.4f}")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please train the model first.")
        print(f"Or, ensure the example image '{example_image_paths[0]}' exists.")
    except Exception as e:
        print(f"An error occurred: {e}")
