# c:\Users\Vinay B R\Desktop\FACE LIVENESS DETECTION\create_dummy_dataset.py

import os
import numpy as np
import pandas as pd
from PIL import Image

def create_dummy_dataset(base_dir="data", num_train_videos=10, num_val_videos=4, frames_per_video=100):
    """
    Creates a dummy dataset structure with placeholder images and sensor files
    that simulates video clips.
    """
    train_samples = num_train_videos * frames_per_video
    val_samples = num_val_videos * frames_per_video
    print("Creating dummy dataset...")

    # Create directories
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_annotations = []
    val_annotations = []

    # --- Generate Training Data ---
    print(f"Generating {train_samples} training samples...")
    frame_counter = 0
    for i in range(num_train_videos):
        video_id = f"train_vid_{i:02d}"
        label = i % 2 # Each video is either all live or all spoof
        prefix = "live" if label == 1 else "spoof"
        for j in range(frames_per_video):
            # Create a dummy image (224x224 RGB)
            img_filename = f"{prefix}_train_{frame_counter:04d}.jpg"
            img_path = os.path.join(train_dir, img_filename)
            color = (100, 150, 200) if label == 1 else (200, 150, 100)
            Image.new('RGB', (224, 224), color).save(img_path)

            # Create dummy sensor data (8 random float values)
            sensor_filename = f"{prefix}_train_{frame_counter:04d}.txt"
            sensor_path = os.path.join(train_dir, sensor_filename)
            sensor_data = np.random.rand(8).astype(np.float32)
            np.savetxt(sensor_path, sensor_data, delimiter=',')

            train_annotations.append([video_id, j, os.path.join("train", img_filename), os.path.join("train", sensor_filename), label])
            frame_counter += 1

    # --- Generate Validation Data ---
    print(f"Generating {val_samples} validation samples...")
    frame_counter = 0
    for i in range(num_val_videos):
        video_id = f"val_vid_{i:02d}"
        label = i % 2
        prefix = "live" if label == 1 else "spoof"
        for j in range(frames_per_video):
            img_filename = f"{prefix}_val_{frame_counter:04d}.jpg"
            img_path = os.path.join(val_dir, img_filename)
            color = (100, 150, 200) if label == 1 else (200, 150, 100)
            Image.new('RGB', (224, 224), color).save(img_path)

            sensor_filename = f"{prefix}_val_{frame_counter:04d}.txt"
            sensor_path = os.path.join(val_dir, sensor_filename)
            sensor_data = np.random.rand(8).astype(np.float32)
            np.savetxt(sensor_path, sensor_data, delimiter=',')
            
            val_annotations.append([video_id, j, os.path.join("val", img_filename), os.path.join("val", sensor_filename), label])
            frame_counter += 1

    # --- Create CSV files ---
    train_df = pd.DataFrame(train_annotations, columns=["video_id", "frame_id", "image_path", "sensor_path", "label"])
    train_df.to_csv(os.path.join(base_dir, "train_annotations.csv"), index=False)

    val_df = pd.DataFrame(val_annotations, columns=["video_id", "frame_id", "image_path", "sensor_path", "label"])
    val_df.to_csv(os.path.join(base_dir, "val_annotations.csv"), index=False)

    print("\nDummy dataset created successfully!")
    print(f"You can now run 'python train.py' to test your training pipeline.")

if __name__ == "__main__":
    # You might need to install Pillow: pip install Pillow pandas
    create_dummy_dataset()