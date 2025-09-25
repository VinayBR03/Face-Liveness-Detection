# /liveness_detection/model.py

import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention block."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_cat = self.conv1(x_cat)
        return self.sigmoid(x_cat)

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class MultiModalLivenessModel(nn.Module):
    """
    A lightweight, multi-modal model for face liveness detection.
    
    Inputs (Modalities):
    - Camera (Image): Processed by MobileNetV3 with CBAM attention.
    - IMU + Ambient Sensors: Processed by a simple MLP.
    
    Fusion & Temporal Analysis:
    - A CNN extracts features from each frame in a clip.
    - An LSTM processes the sequence of image features and sensor data over time.
    """
    def __init__(self, image_feature_dim=576, sensor_input_dim=8, lstm_hidden_dim=128, dropout_rate=0.5):
        super(MultiModalLivenessModel, self).__init__()

        # 1. Image Backbone (Lightweight CNN)
        # This will process each frame of the clip individually.
        # Using MobileNetV3-Small as the backbone
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Remove the final classifier to use it as a feature extractor
        self.image_features = nn.Sequential(*list(mobilenet.children())[:-1])
        
        # 2. Attention Mechanism
        # Applying CBAM to the output of the image backbone
        self.attention = CBAM(in_planes=image_feature_dim)

        # 3. Temporal Fusion (LSTM)
        # The LSTM will process the sequence of features from all modalities.
        # Input to LSTM will be concatenated image features and sensor data.
        self.lstm = nn.LSTM(
            input_size=image_feature_dim + sensor_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True, # Crucial for (batch, seq_len, features) input
            dropout=dropout_rate
        )

        # 4. Fusion and Classifier Head
        # Takes the final hidden state of the LSTM as input.
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 1) # Final output: a single liveness score
        )
        
        self.adap_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, image_clip, sensor_clip):
        # image_clip shape: (batch_size, clip_length, channels, height, width)
        # sensor_clip shape: (batch_size, clip_length, num_sensors)
        batch_size, clip_length, C, H, W = image_clip.shape

        # Reshape to process all frames in the batch at once
        image_input = image_clip.view(batch_size * clip_length, C, H, W)

        # Process Image Stream frame by frame
        img_feat = self.image_features(image_input)
        img_feat_attended = self.attention(img_feat)
        img_feat_flat = self.adap_pool(img_feat_attended)
        img_feat_flat = img_feat_flat.view(batch_size, clip_length, -1)

        # Early Fusion: Concatenate image features and sensor data along the feature dimension
        fused_features = torch.cat((img_feat_flat, sensor_clip), dim=2)

        # Temporal Analysis with LSTM
        # lstm_out shape: (batch, seq_len, hidden_dim)
        # hidden shape: (num_layers, batch, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(fused_features)
        
        # Use the last hidden state of the last layer for classification
        final_hidden_state = hidden[-1]
        output = self.classifier(final_hidden_state)
        
        return output

def export_to_onnx(model, onnx_file_path, input_shapes):
    """
    Exports the PyTorch model to ONNX format.
    Args:
        model: The PyTorch model to export.
        onnx_file_path: Path to save the ONNX file.
        input_shapes: Tuple of input shapes for the model (image_clip, sensor_clip).
    """
    model.eval()
    dummy_image_clip = torch.randn(*input_shapes[0])  # Example input for image_clip
    dummy_sensor_clip = torch.randn(*input_shapes[1])  # Example input for sensor_clip

    torch.onnx.export(
        model,
        (dummy_image_clip, dummy_sensor_clip),
        onnx_file_path,
        export_params=True,
        opset_version=11,
        input_names=["image_clip", "sensor_clip"],
        output_names=["output"],
        dynamic_axes={
            "image_clip": {0: "batch_size", 1: "clip_length"},
            "sensor_clip": {0: "batch_size", 1: "clip_length"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model exported to ONNX format at {onnx_file_path}")
