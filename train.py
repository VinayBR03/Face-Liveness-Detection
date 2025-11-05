# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

from model import MultiModalLivenessModel
from dataset import LivenessDataset

# --- Configuration ---
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "test")

def main(args):
    # Determine if the OS is Windows, as it affects num_workers
    is_windows = os.name == 'nt'
    num_workers = 0 if is_windows else 4
    print(f"OS: {os.name}, Using {num_workers} workers for DataLoader.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Datasets and Dataloaders ---
    print("Loading datasets...")
    # Note: We no longer pass a CSV file path
    train_dataset = LivenessDataset(root_dir=TRAIN_DIR, clip_length=args.clip_length)
    val_dataset = LivenessDataset(root_dir=VAL_DIR, clip_length=args.clip_length)

    # Check if datasets are empty
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Training or validation dataset is empty.")
        print(f"Please check the following directories for data: '{TRAIN_DIR}' and '{VAL_DIR}'")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if not is_windows else False # pin_memory can be problematic on Windows
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"Found {len(train_dataset)} training clips and {len(val_dataset)} validation clips.")

    # --- Model, Loss, and Optimizer ---
    model = MultiModalLivenessModel().to(device)
    criterion = nn.BCEWithLogitsLoss() # Numerically stable
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        # Training phase
        model.train()
        train_loss = 0.0
        for image_clip, sensor_clip, labels in tqdm(train_loader, desc="Training"):
            image_clip = image_clip.to(device)
            sensor_clip = sensor_clip.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(image_clip, sensor_clip)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for image_clip, sensor_clip, labels in tqdm(val_loader, desc="Validating"):
                image_clip = image_clip.to(device)
                sensor_clip = sensor_clip.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(image_clip, sensor_clip)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = torch.sigmoid(outputs) > 0.5
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"✅ Model saved to {args.model_save_path} (Val Loss: {best_val_loss:.4f})")

    print("\n--- Training Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-modal liveness detection model.")
    parser.add_argument('--model-save-path', type=str, default='liveness_model.pth', help='Path to save the trained model.')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--clip-length', type=int, default=25, help='Number of frames per clip.')
    
    args = parser.parse_args()
    main(args)