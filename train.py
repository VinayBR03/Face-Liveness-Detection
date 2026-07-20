# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import json
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score

# Make sure these match your actual file names
from model import MultiModalLivenessModel
from dataset import LivenessDataset

# --- Configuration ---
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "test")

def main(args):
    # 1. Setup Device & Workers
    is_windows = os.name == 'nt'
    num_workers = 0 if is_windows else 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  OS: {os.name} | Workers: {num_workers} | Device: {device}")

    # 2. Load Datasets
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Error: Training directory '{TRAIN_DIR}' not found.")
        return

    train_dataset = LivenessDataset(root_dir=TRAIN_DIR, clip_length=args.clip_length)
    val_dataset = LivenessDataset(root_dir=VAL_DIR, clip_length=args.clip_length)
    
    print(f"📂 Found {len(train_dataset)} training clips and {len(val_dataset)} validation clips.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=not is_windows
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )

    # 3. Initialize Model (Unrolled LSTM version)
    model = MultiModalLivenessModel(lstm_hidden_dim=128).to(device)
    
    # BCEWithLogitsLoss is standard for binary classification (Real vs Fake)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = args.patience

    history = {
        "train_loss": [], 
        "val_loss": [], 
        "accuracy": [], 
        "precision": [], 
        "f1_score": [], 
        "roc_auc": []
    }

    # 4. Training Loop
    for epoch in range(args.epochs):
        print(f"\n--- ⏳ Epoch {epoch+1}/{args.epochs} ---")

        # --- Train ---
        model.train()
        train_loss = 0.0
        for image_clip, sensor_clip, labels in tqdm(train_loader, desc="Training"):
            image_clip = image_clip.to(device)
            sensor_clip = sensor_clip.to(device)
            labels = labels.to(device).unsqueeze(1) # [Batch, 1]

            optimizer.zero_grad()
            outputs = model(image_clip, sensor_clip)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_scores = []

        with torch.no_grad():
            for image_clip, sensor_clip, labels in tqdm(val_loader, desc="Validating"):
                image_clip = image_clip.to(device)
                sensor_clip = sensor_clip.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(image_clip, sensor_clip)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Get probabilities (scores) and binary predictions
                scores = torch.sigmoid(outputs)
                predicted = scores > 0.5

                # Collect all labels, predictions, and scores for metric calculation
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics using sklearn
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(all_labels, all_scores)
        except ValueError: # Only one class present in y_true. ROC AUC score is not defined in that case.
            roc_auc = 0.0

        print(f"📉 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | ✅ Accuracy: {accuracy:.2f}%")
        print(f"   📊 Precision: {precision:.4f} | F1-Score: {f1:.4f} | ROC AUC: {roc_auc:.4f}")

        # Save Checkpoint
        # --- Early Stopping & Checkpoint Logic ---
        if avg_val_loss < best_val_loss:
            print(f"✅ Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}.")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"💾 Best model saved to {args.model_save_path}")
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1
            print(f"⚠️ Validation loss did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"❌ Early stopping triggered after {patience} epochs with no improvement.")
            break # Exit training loop

        # --- Update History & Scheduler ---
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["accuracy"].append(accuracy)
        history["precision"].append(precision)
        history["f1_score"].append(f1)
        history["roc_auc"].append(roc_auc)

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

    with open('training_history.json', 'w') as f:
        json.dump(history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-save-path', type=str, default=os.path.join("models", 'liveness_model.pth'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--clip-length', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5, help="Epochs to wait for improvement before early stopping.")
    args = parser.parse_args()
    main(args)