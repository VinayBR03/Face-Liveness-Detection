import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import cv2

from cvzone.FaceDetectionModule import FaceDetector
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter


app = Flask(__name__)

# ------------------ Config ------------------
MODEL_PATH = "liveness_model_int8.tflite"
CLIP_LENGTH = 10
SENSOR_DIM = 8

# ------------------ Load TFLite with XNNPack ------------------
try:
    # Enable XNNPack delegate for better performance
    interpreter = Interpreter(
        model_path=MODEL_PATH,
        num_threads=4  # Use multiple threads for parallel processing
    )
    print("INFO: TFLite interpreter created with XNNPack support")
except Exception as e:
    print(f"WARNING: Could not enable XNNPack: {e}")
    interpreter = Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def _rank(detail):
    shape = detail.get("shape", None)
    return len(shape) if isinstance(shape, np.ndarray) and shape.size > 0 else 0

# Identify image vs sensor input
img_input_detail = None
sensor_input_detail = None
for det in input_details:
    r = _rank(det)
    if r == 5:
        img_input_detail = det
    elif r == 3:
        sensor_input_detail = det

if img_input_detail is None or sensor_input_detail is None:
    for det in input_details:
        name = det.get("name", "").lower()
        if img_input_detail is None and "image" in name:
            img_input_detail = det
        if sensor_input_detail is None and "sensor" in name:
            sensor_input_detail = det

assert img_input_detail is not None, "Could not find image input"
assert sensor_input_detail is not None, "Could not find sensor input"

# Cache quantization parameters
img_scale, img_zero = img_input_detail.get("quantization", (0.0, 0))
sen_scale, sen_zero = sensor_input_detail.get("quantization", (0.0, 0))
out_scale, out_zero = output_details[0].get("quantization", (0.0, 0))

IS_IMG_QUANTIZED = img_scale > 0.0
IS_SEN_QUANTIZED = sen_scale > 0.0
IS_OUT_QUANTIZED = out_scale > 0.0

img_shape = img_input_detail["shape"]
EXPECTS_NCHW_TIME = (len(img_shape) == 5 and img_shape[2] == 3)

print(f"Image quantized: {IS_IMG_QUANTIZED}, Sensor quantized: {IS_SEN_QUANTIZED}")

# ------------------ Detector ------------------
detector = FaceDetector(minDetectionCon=0.7)

# ------------------ Optimized Preprocessing ------------------
# Pre-compute normalization constants
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Pre-compute quantization scaling for images
if IS_IMG_QUANTIZED:
    IMG_QUANT_SCALE = 1.0 / img_scale
    IMG_QUANT_ZERO = img_zero
    
# Pre-compute quantization scaling for sensors
if IS_SEN_QUANTIZED:
    SEN_QUANT_SCALE = 1.0 / sen_scale
    SEN_QUANT_ZERO = sen_zero

def preprocess_image_clip_optimized(image_b64_list):
    """
    Optimized preprocessing that minimizes conversions and combines operations.
    Returns quantized INT8 directly if model expects it, otherwise FP32.
    """
    frames = []
    for b64_img in image_b64_list:
        try:
            raw = b64_img.split(",")[-1]
            img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
            img = img.resize((224, 224), Image.BILINEAR)  # Faster than default
            
            # Convert to numpy and normalize in one go
            arr = np.asarray(img, dtype=np.float32)
            arr = (arr / 255.0 - IM_MEAN) / IM_STD
            frames.append(arr)
        except Exception as e:
            print(f"Frame decode error: {e}")
            continue

    if not frames:
        return None

    # Stack frames efficiently
    clip = np.stack(frames, axis=0)

    # Transpose if needed
    if EXPECTS_NCHW_TIME:
        clip = np.transpose(clip, (0, 3, 1, 2))
        clip = clip[np.newaxis, ...]
    else:
        clip = clip[np.newaxis, ...]
    
    # Quantize directly if needed (avoid separate step later)
    if IS_IMG_QUANTIZED:
        clip = np.clip(
            np.round(clip * IMG_QUANT_SCALE + IMG_QUANT_ZERO),
            -128, 127
        ).astype(np.int8)
    
    return clip

def prepare_sensor_clip_optimized(sensor_clip):
    """
    Optimized sensor preprocessing with direct quantization if needed.
    """
    if not sensor_clip:
        sensor_data = np.zeros((CLIP_LENGTH, SENSOR_DIM), dtype=np.float32)
    else:
        sensor_data = np.array(sensor_clip, dtype=np.float32)
        if sensor_data.shape != (CLIP_LENGTH, SENSOR_DIM):
            raise ValueError(f"sensor_clip must be shape ({CLIP_LENGTH},{SENSOR_DIM})")
    
    sensor_np = np.expand_dims(sensor_data, axis=0)
    
    # Quantize directly if model expects INT8
    if IS_SEN_QUANTIZED:
        sensor_np = np.clip(
            np.round(sensor_np * SEN_QUANT_SCALE + SEN_QUANT_ZERO),
            -128, 127
        ).astype(np.int8)
    
    return sensor_np

def safe_sigmoid(x):
    """Numerically stable sigmoid."""
    x = float(x)
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)

# ------------------ Routes ------------------
@app.route("/")
def index():
    print("\n=== TFLite Model Details ===")
    print(f"Inputs: {len(input_details)}")
    for d in input_details:
        print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}, "
              f"quant={d.get('quantization', 'none')}")
    print(f"Outputs: {len(output_details)}")
    for d in output_details:
        print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}, "
              f"quant={d.get('quantization', 'none')}")
    print("===========================\n")
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing frame data"}), 400

        frame_b64 = data["image"].split(",")[-1]
        img_bytes = base64.b64decode(frame_b64)
        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        img, bboxs = detector.findFaces(frame, draw=False)
        faces = []
        if bboxs:
            for bbox in bboxs:
                x, y, w, h = bbox["bbox"]
                faces.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

        out = frame.copy()
        for f in faces:
            cv2.rectangle(out, (f["x"], f["y"]), 
                         (f["x"] + f["w"], f["y"] + f["h"]), (0, 255, 0), 2)
        _, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        encoded = base64.b64encode(buffer).decode("utf-8")
        return jsonify({"faces": faces, "image": f"data:image/jpeg;base64,{encoded}"})
    except Exception as e:
        print(f"Detection Error: {e}")
        return jsonify({"error": "Detection failed"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_clip = data.get("image_clip", [])
        sensor_clip = data.get("sensor_clip", [])

        if len(image_clip) != CLIP_LENGTH:
            return jsonify({"error": f"Expected {CLIP_LENGTH} frames"}), 400

        # Use optimized preprocessing (quantizes internally if needed)
        img_clip = preprocess_image_clip_optimized(image_clip)
        if img_clip is None:
            return jsonify({"error": "Failed to preprocess images"}), 400

        sensor_np = prepare_sensor_clip_optimized(sensor_clip)

        # Set tensors directly (already quantized if needed)
        interpreter.set_tensor(img_input_detail["index"], img_clip)
        interpreter.set_tensor(sensor_input_detail["index"], sensor_np)
        
        # Run inference
        interpreter.invoke()

        # Get output and dequantize if needed
        raw_output = interpreter.get_tensor(output_details[0]["index"])[0][0]
        
        if IS_OUT_QUANTIZED:
            logit = (float(raw_output) - out_zero) * out_scale
        else:
            logit = float(raw_output)

        liveness = safe_sigmoid(logit)
        result = "Real Face" if liveness > 0.5 else "Fake Face"

        return jsonify({
            "liveness_score": float(liveness), 
            "result": result
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
