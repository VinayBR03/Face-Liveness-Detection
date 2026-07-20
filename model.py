# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(self.avg_pool(x))
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class MultiModalLivenessModel(nn.Module):
    def __init__(self, image_feature_dim=576, sensor_input_dim=8, lstm_hidden_dim=128, dropout_rate=0.5):
        super(MultiModalLivenessModel, self).__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.image_features = nn.Sequential(*list(mobilenet.children())[:-1])
        self.attention = CBAM(image_feature_dim)
        self.image_feature_dim = image_feature_dim
        
        # First reshape the combined features
        self.pre_lstm = nn.Sequential(
            nn.Linear(image_feature_dim + sensor_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, lstm_hidden_dim),
            nn.ReLU()
        )
        
        # --- Use LSTMCell for manual unrolling ---
        self.lstm_cell = nn.LSTMCell(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
        self.adap_pool = nn.AdaptiveAvgPool2d(1)
        self.lstm_hidden_dim = lstm_hidden_dim

    def forward(self, image_clip, sensor_clip):
        batch_size, clip_length, C, H, W = image_clip.shape

        # Merge batch and time dimensions
        image_input = image_clip.flatten(0, 1)

        # CNN feature extraction
        img_feat = self.image_features(image_input)
        img_feat = self.attention(img_feat)
        img_feat = self.adap_pool(img_feat)

        # Restore sequence dimension
        img_feat = img_feat.contiguous().view(batch_size, clip_length, -1)

        # Concatenate image and sensor features
        combined = torch.cat((img_feat, sensor_clip), dim=2)

        # Feature projection before LSTM
        lstm_input = self.pre_lstm(combined)

        # Initialize hidden and cell states
        h = torch.zeros(
            batch_size,
            self.lstm_hidden_dim,
            device=lstm_input.device,
            dtype=lstm_input.dtype,
        )

        c = torch.zeros_like(h)

        # Manual LSTM unrolling
        for t in range(clip_length):
            h, c = self.lstm_cell(lstm_input[:, t], (h, c))

        # Final classifier
        output = self.classifier(h)

        return output