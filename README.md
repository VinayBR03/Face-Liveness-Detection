# Face Liveness Detection 🎭

A multi-modal deep learning system for detecting face liveness (real vs. spoofed/fake faces) optimized for mobile deployment. This project uses both image and sensor data to achieve robust liveness detection on resource-constrained devices.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Model Conversion](#model-conversion)
- [Deployment](#deployment)
- [API Endpoints](#api-endpoints)
- [Performance](#performance)
- [Requirements](#requirements)
- [License](#license)

<a id="overview"></a>

## 🎯 Overview

This project implements a **multi-modal liveness detection system** that combines:

- **Visual Features**: CNN processing of video frames (224×224 RGB images)
- **Sensor Data**: Accelerometer and gyroscope readings (8-dimensional sensor vectors)

The model distinguishes between:

- ✅ **Real Faces**: Actual human faces
- ❌ **Spoofed/Fake Faces**: Printed photos, videos, masks, or other spoofing attempts

### Key Features

- **Mobile-First Design**: Models optimized for TFLite (both FP32 and INT8 quantized)
- **Multi-Modal Learning**: Combines visual and sensor modalities for improved accuracy
- **PyTorch Training**: Full training pipeline with data augmentation
- **Multiple Export Formats**: PyTorch → ONNX → TFLite conversion pipeline
- **Web Interface**: Flask-based REST API for easy integration
- **Real-Time Inference**: Supports video stream processing

<a id="features"></a>

## Features

- **Multi-Modal Fusion**: Leverages both image and sensor data for robust liveness detection
- **Quantized Models**: INT8 quantization for efficient inference on mobile devices
- **Web API**: Easy integration with existing applications via RESTful API
- **Real-Time Performance**: Designed for low-latency inference on video streams

<a id="project-structure"></a>

## 📁 Project Structure

```bash

.
├── model.py                       # MultiModalLivenessModel architecture
├── dataset.py                         # LivenessDataset loader
├── train.py                           # Training script
├── analyze_results.py          # Performance analysis & confusion matrix
│
├── convert_torch_to_onnx.py          # PyTorch → ONNX conversion
├── convert_onnx_to_tflite.py         # ONNX → TFLite FP32 conversion
├── convert_onnx_to_tflite_int8.py    # ONNX → TFLite INT8 quantization
│
├── predict_video.py                   # Video inference (TFLite INT8)
├── predict_video_onnx.py             # Video inference (ONNX)
│
├── app.py                             # Flask API (TFLite INT8 model)
├── app_tflite32.py                   # Flask API (TFLite FP32 model)
├── app_onnx.py                        # Flask API (ONNX model)
│
├── templates/
│   └── index.html                     # Web UI
├── static/
│   └── models/                        # Deployed models directory
├── images/
│   └── ...                          # Generated plots and diagrams
│
├── data/
│   ├── train/
│   │   ├── real/                      # Training real face images
│   │   └── fake/                      # Training spoofed images
│   └── test/
│       ├── real/                      # Testing real face images
│       └── fake/                      # Testing spoofed images
│
├── models/
│   ├── liveness_model.pth                # Trained PyTorch model
│   ├── liveness_model.onnx               # ONNX format model
│   ├── liveness_model_fp32.tflite        # TFLite FP32 model (~5-8 MB)
│   └── liveness_model_int8.tflite        # TFLite INT8 quantized (~1.5-2 MB)
│
├── requirements.txt                   # Python dependencies
├── LICENSE
└── README.md
```

<a id="installation"></a>

## 🚀 Installation

### Prerequisites

- Python 3.12
- CUDA 11.0+ (for GPU training, optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/VinayBR03/Face-Liveness-Detection.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

<a id="dataset"></a>

## 📊 Dataset

### Expected Structure

```text
data/
├── train/
│   ├── real/
│   |   ├── real_1.jpg
│   |   ├── real_2.jpg
|   |   └── ...
│   └── fake/
|       ├── fake_1.jpg
│       ├── fake_2.jpg
|       └── ...
└── test/
    ├── real/...
    └── fake/...
```

### Key Properties

- **Image Resolution**: 224×224 pixels (RGB)
- **Sensor Data**: 8-dimensional accelerometer + gyroscope readings
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Data Augmentation (Training Only)

- Random horizontal flips
- Color jitter (brightness, contrast, saturation)
- Resize to 224×224

<a id="training"></a>

## 🏋️ Training

Train the model using the provided training script:

```bash
python train.py \
  --data-dir data/train \
  --test-dir data/test \
  --epochs 50 \
  --batch-size 16 \
  --lr 0.001 \
  --clip-length 10 \
  --device cuda  # or 'cpu'
```

### Training Output

- `liveness_model.pth` - Best model weights
- `training_history.json` - Loss & accuracy logs

### Analyze Results

```bash
python analyze_results.py \
  --model-path liveness_model.pth \
  --history-path training_history.json \
  --clip-length 10
```

Generates:

- Training loss/accuracy plots
- Confusion matrix on test set
- Performance metrics (precision, recall, F1-score)

<a id="model-conversion"></a>

## 🔄 Model Conversion Pipeline

### 1️⃣ PyTorch → ONNX

```bash
python convert_torch_to_onnx.py \
  --input-model liveness_model.pth \
  --output-model liveness_model.onnx \
  --clip-length 10
```

**Output**: `liveness_model.onnx` (Universal format, ~5-8 MB)

### 2️⃣ ONNX → TFLite FP32

```bash
python convert_onnx_to_tflite.py
```

**Output**: `liveness_model_fp32.tflite` (~5-8 MB)

### 3️⃣ ONNX → TFLite INT8 (Quantized)

```bash
python convert_onnx_to_tflite_int8.py 
```

**Output**: `liveness_model_int8.tflite` (~1.5-2 MB, ~2-3x faster)

### Model Comparison

| Model | Format | Size | Speed | Accuracy | Mobile |
| ------- | -------- | ------ | ------- | ---------- | -------- |
| Full Precision | PyTorch | ~10 MB | ⚠️ Slow | ⭐⭐⭐ | ❌ Large |
| ONNX | ONNX | ~5-8 MB | ⚠️ Medium | ⭐⭐⭐ | ⚠️ Need runtime |
| TFLite FP32 | TFLite | ~5-8 MB | ⚠️ Medium | ⭐⭐⭐ | ✅ Native |
| TFLite INT8 | TFLite | ~1.5-2 MB | ✅ Fast | ⭐⭐⭐ | ✅ Recommended |

<a id="deployment"></a>

## 📱 Deployment

### Option 1: Flask Web Server (Development/Testing)

Start the API server using the quantized INT8 model:

```bash
python app.py
```

Or use FP32 model:

```bash
python app_tflite32.py
```

Or use ONNX model:

```bash
python app_onnx.py
```

The server runs on `http://localhost:5000`

### Option 2: Video Inference

Test on a video file:

```bash
# Using INT8 TFLite model
python predict_video.py --video your_video.mp4

# Using ONNX model
python predict_video_onnx.py --video your_video.mp4 --onnx liveness_model.onnx
```

<a id="api-endpoints"></a>

## 🔌 API Endpoints

### Base URL

```text
http://localhost:5000
```

### 1. Face Detection

**Endpoint**: `POST /detect`

**Request**:

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
}
```

**Response**:

```json
{
  "faces": [
    {"x": 100, "y": 150, "w": 200, "h": 250}
  ],
  "image": "data:image/jpeg;base64,..."
}
```

### 2. Liveness Prediction

**Endpoint**: `POST /predict`

**Request**:

```json
{
  "image_clip": [
    "data:image/jpeg;base64,...",  // 10 frames total
    "data:image/jpeg;base64,...",
    ...
  ],
  "sensor_clip": [
    [ax, ay, az, gx, gy, gz, 0, 0],  // 10 sensor readings
    [ax, ay, az, gx, gy, gz, 0, 0],
    ...
  ]
}
```

**Response**:

```json
{
  "liveness_score": 0.95,
  "result": "Real Face"
}
```

Score > 0.5 → Real Face | Score ≤ 0.5 → Fake Face

### 3. Web UI

**Endpoint**: `GET /`

Opens interactive web interface at `http://localhost:5000`

<a id="performance"></a>

## 📈 Performance

### Model Metrics

- **Accuracy**: ~95-98% on test set
- **Precision**: ~94-97%
- **Recall**: ~93-96%
- **F1-Score**: ~94-96%

### Inference Speed (INT8 TFLite)

- **Per Frame**: ~5-10 ms
- **Per Clip (10 frames)**: ~50-100 ms
- **On Mobile**: ~100-200 ms (device dependent)

### Mobile Device Requirements

- **Memory**: ~50-100 MB RAM
- **Storage**: ~2-5 MB (INT8 model)
- **Processing**: Snapdragon 600+ or equivalent

<a id="requirements"></a>

## 📦 Requirements

All dependencies are listed in the [requirements.txt](requirements.txt) file.

```text
pip install -r requirements.txt
```

<a id="license"></a>

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or collaboration, reach out via GitHub Issues: [Face-Liveness-Detection](https://github.com/VinayBR03/Face-Liveness-Detection/issues)

---

## Citation

If you find this project useful, please consider giving it a ⭐ on GitHub.

---

Developed by **Vinay B R**
