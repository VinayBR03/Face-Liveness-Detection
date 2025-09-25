# Multi-Modal Face Liveness Detection

This project is an end-to-end system for detecting face liveness to prevent spoofing attacks (e.g., using a photo or video of a person). It uses a multi-modal deep learning model built with PyTorch that analyzes a short video clip and corresponding sensor data (from a mobile device's IMU) to make a prediction.

The final trained model is converted to TensorFlow Lite (`.tflite`) for efficient deployment and served via a Flask web application.

## Features

- **Multi-Modal Input**: Fuses video frames with sensor data for more robust detection.
- **Temporal Analysis**: Uses an LSTM network to analyze patterns over a sequence of frames, making it difficult to fool with static images.
- **Lightweight Architecture**: Built on a MobileNetV3 backbone for efficient feature extraction.
- **Attention Mechanism**: Incorporates a CBAM (Convolutional Block Attention Module) to help the model focus on relevant features.
- **End-to-End Pipeline**: Includes scripts for data generation, training, model conversion, and web deployment.
- **Web Interface**: A simple HTML/JavaScript frontend to interact with the deployed model in real-time.

## Project Structure

```
FACE LIVENESS DETECTION
├── data/                     # Holds dummy data and annotations
├── templates/
│   └── index.html            # Frontend for the web application
├── app.py                    # Flask web server to deploy the TFLite model
├── convert_to_tflite.py      # Converts the trained PyTorch model to TFLite
├── create_dummy_dataset.py   # Generates a placeholder dataset for testing
├── dataset.py                # PyTorch custom Dataset for loading clips
├── model.py                  # Defines the MultiModalLivenessModel architecture
├── predict.py                # Standalone script for quick inference tests
├── train.py                  # Script to train the PyTorch model
└── requirements.txt          # Python package dependencies
```

## Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/VinayBR03/Face-Liveness-Detection.git
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Install all the required packages using the provided `requirements.txt` file. This specific set of versions is known to be stable and compatible.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Full Pipeline

Follow these steps in order to generate data, train a model, and deploy the application.

### Step 1: Create a Dummy Dataset

This script generates placeholder images, sensor data, and the necessary `.csv` annotation files for training and validation.

```bash
python create_dummy_dataset.py
```

### Step 2: Train the Liveness Model

This script trains the `MultiModalLivenessModel` on the dummy dataset and saves the best-performing model weights as `liveness_model.pth`.

```bash
python train.py
```

### Step 3: Convert the Model to TensorFlow Lite

After training, convert the saved PyTorch model (`.pth`) into the efficient TensorFlow Lite format (`.tflite`) for deployment.

```bash
python convert_to_tflite.py
```

### Step 4: Run the Web Application

Start the Flask server, which will load the `liveness_model.tflite` file and serve the web interface.

```bash
python app.py
```

Open your web browser and navigate to `http://127.0.0.1:5000`. You can now test the liveness detection system using your webcam.

## Technologies Used

- **Deep Learning**: PyTorch
- **Model Deployment**: TensorFlow Lite, ONNX
- **Web Framework**: Flask
- **Data Handling**: NumPy, Pandas, Pillow

---