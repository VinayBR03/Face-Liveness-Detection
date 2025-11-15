# analyze_results.py
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

from model import MultiModalLivenessModel
from dataset import LivenessDataset

# --- Configuration ---
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

def plot_data_distribution(data_dir, title):
    """Scans a directory and plots the distribution of real vs. fake samples."""
    real_path = os.path.join(data_dir, 'real')
    fake_path = os.path.join(data_dir, 'fake')

    # Count based on unique video clips (directories)
    real_count = len(os.listdir(real_path)) if os.path.exists(real_path) else 0
    fake_count = len(os.listdir(fake_path)) if os.path.exists(fake_path) else 0

    if real_count == 0 and fake_count == 0:
        print(f"No data found in {data_dir}. Skipping distribution plot.")
        return

    labels = ['Real', 'Fake']
    counts = [real_count, fake_count]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=['#4CAF50', '#F44336'])
    plt.title(title, fontsize=16)
    plt.ylabel('Number of Video Clips', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    print(f"✅ Saved '{title.lower().replace(' ', '_')}.png'")
    plt.show()

def plot_training_history(history_path):
    """Loads training history and plots accuracy and F1 score vs. epochs."""
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}. Please run train.py first.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(14, 6))

    # Accuracy vs. Epoch
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'bo-', label='Validation Accuracy')
    plt.title('Accuracy vs. Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True)

    # F1 Score vs. Epoch
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['f1_score'], 'ro-', label='Validation F1 Score')
    plt.title('F1 Score vs. Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_performance_vs_epoch.png")
    print("✅ Saved 'training_performance_vs_epoch.png'")
    plt.show()

def plot_confusion_matrix(model_path, test_dir, clip_length):
    """Loads the best model, evaluates on the test set, and plots the confusion matrix."""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}. Please run train.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MultiModalLivenessModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test data
    test_dataset = LivenessDataset(root_dir=test_dir, clip_length=clip_length, is_train=False)
    if len(test_dataset) == 0:
        print(f"No data found in {test_dir}. Skipping confusion matrix.")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for image_clip, sensor_clip, labels in tqdm(test_loader, desc="Generating Predictions for Confusion Matrix"):
            image_clip = image_clip.to(device)
            sensor_clip = sensor_clip.to(device)
            
            outputs = model(image_clip, sensor_clip)
            predicted = torch.sigmoid(outputs) > 0.5
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['Fake', 'Real']

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.title('Confusion Matrix on Test Data', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("✅ Saved 'confusion_matrix.png'")
    plt.show()

def main(args):
    # 1. Plot Data Distributions
    print("\n--- 📊 Analyzing Data Distribution ---")
    plot_data_distribution(TRAIN_DIR, "Training Data Distribution")
    plot_data_distribution(TEST_DIR, "Testing Data Distribution")

    # 2. Plot Training History
    print("\n--- 📈 Analyzing Training History ---")
    plot_training_history(args.history_path)

    # 3. Plot Confusion Matrix
    print("\n--- 🧮 Analyzing Model Performance on Test Set ---")
    plot_confusion_matrix(args.model_path, TEST_DIR, args.clip_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model training results.")
    parser.add_argument('--model-path', type=str, default='liveness_model.pth', help='Path to the saved model file.')
    parser.add_argument('--history-path', type=str, default='training_history.json', help='Path to the training history file.')
    parser.add_argument('--clip-length', type=int, default=10, help='Number of frames per clip (must match training).')
    
    args = parser.parse_args()
    main(args)