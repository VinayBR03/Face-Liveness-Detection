import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import cv2

from cvzone.FaceDetectionModule import FaceDetector
from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

# ------------------ Config ------------------
MODEL_PATH = os.path.join("models", "liveness_model_fp32.tflite")  # keep this name even if the model has FP16 weights
CLIP_LENGTH = 10
SENSOR_DIM = 8

# ------------------ Load TFLite ------------------
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
        img_input_detail = det  # image: [1, 10, ?, 224, 224]
    elif r == 3:
        sensor_input_detail = det  # sensor: [1, 10, 8]

# Fallback by name if needed
if img_input_detail is None or sensor_input_detail is None:
    for det in input_details:
        name = det.get("name", "").lower()
        if img_input_detail is None and "image" in name:
            img_input_detail = det
        if sensor_input_detail is None and "sensor" in name:
            sensor_input_detail = det

assert img_input_detail is not None, "Could not find image input in TFLite model"
assert sensor_input_detail is not None, "Could not find sensor input in TFLite model"

# Does the image input expect N T C H W?
img_shape = img_input_detail["shape"]

if img_shape[1] == 3:
    IMAGE_LAYOUT = "BCHWT"      # (B,C,H,W,T)
elif img_shape[4] == 3:
    IMAGE_LAYOUT = "BTHWC"      # (B,T,H,W,C)
else:
    raise RuntimeError(f"Unknown image layout: {img_shape}")

# ------------------ Detector ------------------
detector = FaceDetector(minDetectionCon=0.7)

# ------------------ Helpers ------------------
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image_clip(image_b64_list):
    frames = []

    for b64_img in image_b64_list:
        try:
            raw = b64_img.split(",")[-1]

            img = Image.open(
                io.BytesIO(base64.b64decode(raw))
            ).convert("RGB")

            img = img.resize((224, 224), Image.BILINEAR)

            arr = np.asarray(img, dtype=np.float32)
            arr = (arr / 255.0 - IM_MEAN) / IM_STD

            frames.append(arr)

        except Exception as e:
            print("Frame decode error:", e)

    if len(frames) == 0:
        return None

    clip = np.stack(frames, axis=0)

    if IMAGE_LAYOUT == "BCHWT":
        clip = np.transpose(clip, (3, 1, 2, 0))
        clip = np.expand_dims(clip, axis=0)
    else:
        clip = np.expand_dims(clip, axis=0)

    return clip.astype(np.float32)

def prepare_sensor_clip(sensor_clip):
    """Returns (1, T, 8) float32. If missing, fills zeros."""
    if not sensor_clip:
        sensor_data = np.zeros((CLIP_LENGTH, SENSOR_DIM), dtype=np.float32)
    else:
        sensor_data = np.array(sensor_clip, dtype=np.float32)
        if sensor_data.shape != (CLIP_LENGTH, SENSOR_DIM):
            raise ValueError(f"sensor_clip must be shape ({CLIP_LENGTH},{SENSOR_DIM})")
    return np.expand_dims(sensor_data, axis=0).astype(np.float32)

def safe_sigmoid(x):
    x = float(x)
    if x > 60:
        return 1.0
    if x < -60:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))

# ------------------ Routes ------------------
@app.route("/")
def index():
    # Just to see bindings once in console
    print("TFLite inputs:")
    for d in input_details:
        print({k: d[k] for k in ["index", "name", "shape", "dtype", "quantization"]})
    print("TFLite outputs:")
    for d in output_details:
        print({k: d[k] for k in ["index", "name", "shape", "dtype", "quantization"]})
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
            cv2.rectangle(out, (f["x"], f["y"]), (f["x"] + f["w"], f["y"] + f["h"]), (0, 255, 0), 2)
        _, buffer = cv2.imencode(".jpg", out)
        encoded = base64.b64encode(buffer).decode("utf-8")
        return jsonify({"faces": faces, "image": f"data:image/jpeg;base64,{encoded}"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Prediction Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_clip = data.get("image_clip", [])
        sensor_clip = data.get("sensor_clip", [])

        if len(image_clip) != CLIP_LENGTH:
            return jsonify({"error": f"Expected {CLIP_LENGTH} frames, got {len(image_clip)}"}), 400
        
        img_clip = preprocess_image_clip(image_clip)
        print("img_clip:", None if img_clip is None else img_clip.shape)
        if img_clip is None:
            return jsonify({"error": "Failed to preprocess images"}), 400

        sensor_np = prepare_sensor_clip(sensor_clip)

        # Debug
        print(f"[DEBUG] prepared img_clip {img_clip.shape}, sensor_clip {sensor_np.shape}")
        print(f"[DEBUG] model expects img {img_input_detail['shape']}, sensor {sensor_input_detail['shape']}")

        # Set tensors by the discovered indices (do NOT assume order)
        interpreter.set_tensor(sensor_input_detail["index"], sensor_np)
        interpreter.set_tensor(img_input_detail["index"], img_clip)
        interpreter.invoke()

        score = interpreter.get_tensor(output_details[0]["index"])[0][0]
        liveness = safe_sigmoid(score)
        result = "Real Face" if liveness > 0.5 else "Fake Face"

        return jsonify({"liveness_score": float(liveness), "result": result})
    except Exception as e:
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)
