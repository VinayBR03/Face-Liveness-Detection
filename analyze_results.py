import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import argparse

# --- Model & Dataset ---
from model import MultiModalLivenessModel
from dataset import LivenessDataset

# --- Runtimes for other models ---
import onnxruntime as ort
import tensorflow as tf

# --- Configuration ---
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = "images"

def ensure_output_dir():
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created output directory: {OUTPUT_DIR}")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def save_plot(filename):
    """Helper to save plots to the specific images folder."""
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    print(f"✅ Saved '{path}'")
    plt.close() # Close plot to free memory

# ==========================================
# 1. Data Distribution Visualization
# ==========================================
def plot_data_distribution(data_dir, split_name):
    """Scans a directory and plots the distribution of real vs. fake samples."""
    ensure_output_dir()
    real_path = os.path.join(data_dir, 'real')
    fake_path = os.path.join(data_dir, 'fake')

    # Count based on unique video clips (directories)
    real_count = len(os.listdir(real_path)) if os.path.exists(real_path) else 0
    fake_count = len(os.listdir(fake_path)) if os.path.exists(fake_path) else 0

    if real_count == 0 and fake_count == 0:
        print(f"⚠️ No data found in {data_dir}. Skipping distribution plot.")
        return

    labels = ['Real', 'Fake']
    counts = [real_count, fake_count]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=['#4CAF50', '#F44336'])
    title = f"{split_name} Data Distribution"
    plt.title(title, fontsize=16)
    plt.ylabel('Number of Video Clips', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add count labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', fontsize=12)
    
    plt.tight_layout()
    save_plot(f"distribution_{split_name.lower()}.png")

# ==========================================
# 2. Training History Visualization
# ==========================================
def plot_training_history(history_path):
    """Loads training history and plots loss and performance metrics in separate files."""
    if not os.path.exists(history_path):
        print(f"⚠️ History file not found: {history_path}. Skipping history plot.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    if not history.get('train_loss'):
        print("⚠️ Training history is empty. Skipping plot.")
        return

    epochs = range(1, len(history['train_loss']) + 1)
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    accuracy = history.get('accuracy', [])
    precision = history.get('precision', [])
    f1_score = history.get('f1_score', [])
    roc_auc = history.get('roc_auc', [])

    # --- Plot 1: Loss History ---
    plt.figure(figsize=(8, 6))
    if train_loss: plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    if val_loss: plt.plot(epochs, val_loss, 'r-', label='Validation Loss')

    plt.title('Loss vs. Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot("training_loss_history.png")

    # --- Plot 2: Performance Metrics History ---
    plt.figure(figsize=(8, 6))
    accuracy_scaled = [acc / 100.0 for acc in accuracy] # Scale accuracy to [0, 1]
    if accuracy_scaled: plt.plot(epochs, accuracy_scaled, 'g-o', label='Accuracy', markersize=4)
    if precision: plt.plot(epochs, precision, 'c-^', label='Precision', markersize=4)
    if f1_score: plt.plot(epochs, f1_score, 'm-s', label='F1-Score', markersize=4)
    if roc_auc: plt.plot(epochs, roc_auc, 'y-d', label='ROC AUC', markersize=4)

    plt.title('Performance Metrics vs. Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0.8, 1.01) # Zoom in on the high-performance range
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot("training_metrics_history.png")

# ==========================================
# 3. Model Evaluation & Confusion Matrices
# ==========================================
def plot_cm(y_true, y_pred, title, filename):
    """Helper function to plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Fake', 'Real']

    plt.figure(figsize=(8, 7))
    # annot=True writes the data value in each cell
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    save_plot(filename)

def plot_confusion_matrix_torch(model_path, test_dir, clip_length):
    """Evaluates the original PyTorch model."""
    if not os.path.exists(model_path):
        print(f"⚠️ PyTorch model not found: {model_path}. Skipping.")
        return

    print(f"Processing PyTorch Model: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model structure (Must match training structure)
    model = MultiModalLivenessModel().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ Error loading PyTorch model: {e}")
        return

    test_dataset = LivenessDataset(root_dir=test_dir, clip_length=clip_length)
    if len(test_dataset) == 0:
        print("⚠️ Test dataset is empty. Skipping.")
        return

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for image_clip, sensor_clip, labels in tqdm(test_loader, desc="PyTorch Inference"):
            outputs = model(image_clip.to(device), sensor_clip.to(device))
            predicted = torch.sigmoid(outputs) > 0.5
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    plot_cm(all_labels, all_preds, 'Confusion Matrix (PyTorch)', 'confusion_matrix_torch.png')

def plot_confusion_matrix_onnx(model_path, test_dir, clip_length):
    """Evaluates the ONNX model."""
    if not os.path.exists(model_path):
        print(f"⚠️ ONNX model not found: {model_path}. Skipping.")
        return

    print(f"Processing ONNX Model: {model_path}")
    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"❌ Error loading ONNX model: {e}")
        return

    img_name = sess.get_inputs()[0].name
    sen_name = sess.get_inputs()[1].name
    out_name = sess.get_outputs()[0].name

    test_dataset = LivenessDataset(root_dir=test_dir, clip_length=clip_length)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_preds, all_labels = [], []
    for image_clip, sensor_clip, labels in tqdm(test_loader, desc="ONNX Inference"):
        # ONNX Runtime expects numpy arrays
        feeds = {img_name: image_clip.numpy(), sen_name: sensor_clip.numpy()}
        out = sess.run([out_name], feeds)[0]
        score = sigmoid(out.item())
        all_preds.append(score > 0.5)
        all_labels.append(labels.item() > 0.5)

    plot_cm(all_labels, all_preds, 'Confusion Matrix (ONNX)', 'confusion_matrix_onnx.png')

def plot_confusion_matrix_tflite(model_path, test_dir, clip_length):
    """Evaluates a TFLite model (FP32 or INT8)."""
    if not os.path.exists(model_path):
        print(f"⚠️ TFLite model not found: {model_path}. Skipping.")
        return
    
    print(f"Processing TFLite Model: {model_path}")

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"❌ Error loading TFLite model: {e}")
        return
    
    # --- Robustly identify inputs by checking their shape ---
    input_details = interpreter.get_input_details()
    img_in, sen_in = None, None
    for detail in input_details:
        if len(detail['shape']) == 5: # Image clip has 5 dimensions (B, T, C, H, W)
            img_in = detail
        elif len(detail['shape']) == 3: # Sensor clip has 3 dimensions (B, T, D)
            sen_in = detail
    
    if img_in is None or sen_in is None:
        print("❌ Could not identify image/sensor inputs in TFLite model. Skipping.")
        return

    out_details = interpreter.get_output_details()[0]

    test_dataset = LivenessDataset(root_dir=test_dir, clip_length=clip_length)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_preds, all_labels = [], []
    for image_clip, sensor_clip, labels in tqdm(test_loader, desc=f"TFLite Inference"):
        img_data = image_clip.numpy()
        sen_data = sensor_clip.numpy()

        # Handle batch dimension mismatch if model expects strictly fixed batch size
        if img_data.shape[0] != img_in['shape'][0]:
             # If model expects batch 1 but loader gave something else (unlikely with batch_size=1)
             pass 

        # Quantize inputs if model is INT8
        if img_in['dtype'] == np.int8:
            img_scale, img_zero = img_in['quantization']
            img_data = (img_data / img_scale + img_zero).astype(np.int8)
        
        if sen_in['dtype'] == np.int8:
            sen_scale, sen_zero = sen_in['quantization']
            sen_data = (sen_data / sen_scale + sen_zero).astype(np.int8)

        # Set tensors
        interpreter.set_tensor(img_in["index"], img_data)
        interpreter.set_tensor(sen_in["index"], sen_data)
        
        # Run inference
        interpreter.invoke()
        
        raw_output = interpreter.get_tensor(out_details["index"])[0][0]

        # Dequantize output if necessary
        if out_details['dtype'] == np.int8:
            out_scale, out_zero = out_details['quantization']
            logit = (float(raw_output) - out_zero) * out_scale
        else:
            logit = float(raw_output)
        
        score = sigmoid(logit)
        all_preds.append(score > 0.5)
        all_labels.append(labels.item() > 0.5)

    model_name = "int8" if "int8" in model_path else "fp32"
    plot_cm(all_labels, all_preds, f'Confusion Matrix (TFLite {model_name.upper()})', f'confusion_matrix_tflite_{model_name}.png')

# ==========================================
# Main Execution
# ==========================================
def main(args):
    print("🚀 Starting Analysis Pipeline...")
    ensure_output_dir()

    # 1. Plot Data Distributions
    print("\n--- 📊 Analyzing Data Distribution ---")
    plot_data_distribution(TRAIN_DIR, "Training")
    plot_data_distribution(TEST_DIR, "Testing")

    # 2. Plot Training History
    print("\n--- 📈 Analyzing Training History ---")
    plot_training_history(args.history_path)

    # 3. Plot Confusion Matrices
    print("\n--- 🧮 Analyzing Model Performance on Test Set ---")
    
    if not os.path.exists(TEST_DIR) or not os.listdir(TEST_DIR):
        print(f"❌ Test directory '{TEST_DIR}' is empty or not found. Skipping confusion matrix generation.")
        return

    # Evaluate PyTorch
    if args.torch_model:
        plot_confusion_matrix_torch(args.torch_model, TEST_DIR, args.clip_length)
    
    # Evaluate ONNX
    if args.onnx_model:
        plot_confusion_matrix_onnx(args.onnx_model, TEST_DIR, args.clip_length)

    # Evaluate TFLite (FP32)
    if args.tflite_fp32_model:
        plot_confusion_matrix_tflite(args.tflite_fp32_model, TEST_DIR, args.clip_length)

    # Evaluate TFLite (INT8)
    if args.tflite_int8_model:
        plot_confusion_matrix_tflite(args.tflite_int8_model, TEST_DIR, args.clip_length)

    print(f"\n✨ Analysis Complete. Check the '{OUTPUT_DIR}' folder for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model training, data distribution, and conversion results.")
    
    # Paths
    parser.add_argument('--torch-model', type=str, default='liveness_model.pth', help='Path to PyTorch .pth model')
    parser.add_argument('--onnx-model', type=str, default='liveness_model.onnx', help='Path to ONNX .onnx model')
    parser.add_argument('--tflite-fp32-model', type=str, default='liveness_model_fp32.tflite', help='Path to TFLite FP32 model')
    parser.add_argument('--tflite-int8-model', type=str, default='liveness_model_int8.tflite', help='Path to TFLite INT8 model')
    parser.add_argument('--history-path', type=str, default='training_history.json', help='Path to training history JSON')
    
    # Parameters
    parser.add_argument('--clip-length', type=int, default=10, help='Number of frames per clip (must match training)')
    
    args = parser.parse_args()
    main(args)