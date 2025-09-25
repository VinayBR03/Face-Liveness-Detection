# /liveness_detection/dataset.py

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LivenessDataset(Dataset):
    """
    Custom Dataset for loading multi-modal liveness data.
    Assumes a root directory with images and a CSV file mapping them
    to sensor data files and labels. Now handles clips of frames.
    
    CSV format:
    video_id,frame_id,image_path,sensor_path,label
    vid01,0,frame_001.jpg,sensors_001.txt,1
    vid01,1,frame_002.jpg,sensors_002.txt,1
    """
    def __init__(self, csv_file, root_dir, clip_length=5, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        # Return the number of possible start frames for a full clip
        return len(self.annotations) - self.clip_length + 1

    def __getitem__(self, index):
        # Get a clip of frames and sensor data
        start_frame_info = self.annotations.iloc[index]
        end_frame_index = index + self.clip_length - 1

        # Ensure the clip does not cross over to a different video
        # This is a simple check; a more robust implementation would pre-calculate valid indices
        if end_frame_index >= len(self.annotations) or \
           start_frame_info['video_id'] != self.annotations.iloc[end_frame_index]['video_id']:
            # If at the end, recursively get the previous item to ensure a full clip
            return self.__getitem__(index - 1)

        clip_frames = []
        clip_sensors = []

        for i in range(self.clip_length):
            frame_index = index + i
            frame_info = self.annotations.iloc[frame_index]

            # Image data
            img_path = os.path.join(self.root_dir, frame_info['image_path'])
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            clip_frames.append(image)

            # Sensor data
            sensor_path = os.path.join(self.root_dir, frame_info['sensor_path'])
            sensor_data = np.loadtxt(sensor_path, delimiter=',', dtype=np.float32)
            clip_sensors.append(sensor_data)

        # Stack the frames and sensors to create clip tensors
        image_clip = torch.stack(clip_frames, dim=0)
        sensor_clip = torch.from_numpy(np.array(clip_sensors, dtype=np.float32))

        # Label
        # The label for the clip is the label of its first frame
        label = torch.tensor(int(start_frame_info['label']), dtype=torch.float32)

        return image_clip, sensor_clip, label
