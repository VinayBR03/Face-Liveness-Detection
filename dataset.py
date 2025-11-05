# /liveness_detection/dataset.py

import os
from glob import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LivenessDataset(Dataset):
    """
    Custom Dataset for loading multi-modal liveness data by scanning directories.
    It expects a structure like:
    - {root_dir}/live/{video_id}/frame_xxx.jpg
    - {root_dir}/spoof/{video_id}/frame_xxx.txt
    """
    def __init__(self, root_dir, clip_length=25, transform=None, is_train=True):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.valid_indices = self._precompute_valid_indices()
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if is_train:
            # Augmentation for the training set
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # No augmentation for the validation/test set
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        # Return the number of valid clips we can create.
        return len(self.valid_indices)

    def _precompute_valid_indices(self):
        """Scans the directory to find all valid video clips."""
        clips = []
        # Use 'real' and 'fake' as per the new data structure
        for label_str in ['real', 'fake']:
            label = 1 if label_str == 'real' else 0
            # The folder itself contains all frames for that class
            class_dir = os.path.join(self.root_dir, label_str)
            if not os.path.isdir(class_dir):
                continue
            
            # Find all image files and sort them to ensure temporal order
            frame_paths = sorted(glob(os.path.join(class_dir, '*.jpg')))
            if len(frame_paths) >= self.clip_length:
                # Create clips using a sliding window over the sorted file list
                for i in range(len(frame_paths) - self.clip_length + 1):
                    clip_info = {"frames": frame_paths[i : i + self.clip_length], "label": label}
                    clips.append(clip_info)
        return clips

    def __getitem__(self, index):
        clip_info = self.valid_indices[index]
        clip_frames, clip_sensors = [], []

        for img_path in clip_info['frames']:
            # Image data
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            clip_frames.append(image)

            # Sensor data
            # Assumes sensor file has the same name but with .txt extension
            sensor_path = os.path.splitext(img_path)[0] + '.txt'
            # Handle cases where a sensor file might be missing
            if os.path.exists(sensor_path):
                sensor_data = np.loadtxt(sensor_path, delimiter=',', dtype=np.float32)
            else:
                # If no sensor file, create a zero-filled array. Assumes sensor_dim is 8.
                sensor_data = np.zeros(8, dtype=np.float32)
            clip_sensors.append(sensor_data)

        # Stack the frames and sensors to create clip tensors
        image_clip = torch.stack(clip_frames, dim=0)
        sensor_clip = torch.from_numpy(np.array(clip_sensors, dtype=np.float32))
        label = torch.tensor(clip_info['label'], dtype=torch.float32)

        return image_clip, sensor_clip, label
