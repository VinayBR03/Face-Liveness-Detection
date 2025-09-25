# /liveness_detection/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MultiModalLivenessModel
from dataset import LivenessDataset

# --- Hyperparameters & Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4 # Reduced batch size to handle longer clips and avoid memory issues
NUM_EPOCHS = 25
CLIP_LENGTH = 100 # 100 frames at 5 FPS = 20 seconds
# NOTE: You need to create these files/directories
TRAIN_CSV = "data/train_annotations.csv"
VAL_CSV = "data/val_annotations.csv"
DATA_ROOT = "data/"
MODEL_SAVE_PATH = "liveness_model.pth"

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    """Runs a single training epoch."""
    model.train()
    loop = tqdm(loader, leave=True)
    running_loss = 0.0

    for (images, sensors, labels) in loop:
        images, sensors, labels = images.to(device), sensors.to(device), labels.to(device)

        # Forward pass
        outputs = model(images, sensors)
        loss = loss_fn(outputs.squeeze(-1), labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return running_loss / len(loader)

def validate(loader, model, loss_fn, device):
    """Evaluates the model on the validation set."""
    model.eval()
    total_correct = 0
    total_samples = 0
    val_loss = 0.0

    with torch.no_grad():
        for (images, sensors, labels) in loader:
            images, sensors, labels = images.to(device), sensors.to(device), labels.to(device)
            
            outputs = model(images, sensors)
            loss = loss_fn(outputs.squeeze(-1), labels)
            val_loss += loss.item()

            # Calculate accuracy
            preds = torch.sigmoid(outputs.squeeze(-1)) > 0.5
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100
    avg_loss = val_loss / len(loader)
    print(f"Validation Accuracy: {accuracy:.2f}%, Avg Loss: {avg_loss:.4f}")
    return accuracy

def main():
    print(f"Using device: {DEVICE}")
    
    # Initialize model
    model = MultiModalLivenessModel().to(DEVICE)

    # Loss and optimizer
    # BCEWithLogitsLoss is numerically stable and expects raw logits
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Data loaders
    train_dataset = LivenessDataset(csv_file=TRAIN_CSV, root_dir=DATA_ROOT, clip_length=CLIP_LENGTH)
    val_dataset = LivenessDataset(csv_file=VAL_CSV, root_dir=DATA_ROOT, clip_length=CLIP_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, DEVICE)
        print(f"Average Training Loss: {train_loss:.4f}")
        
        val_accuracy = validate(val_loader, model, loss_fn, DEVICE)

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with accuracy: {val_accuracy:.2f}%")

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    main()
