import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import cv2
from cvzone.FaceDetectionModule import FaceDetector
import onnxruntime as ort

app = Flask(__name__)

# ------------------ Config ------------------
ONNX_MODEL_PATH = "liveness_model.onnx"   # your working ONNX
CLIP_LENGTH = 10                          # must match model/export
SENSOR_DIM = 8

# ------------------ Load ONNX ------------------
# Expecting input_names: ["image_clip", "sensor_clip"], output_names: ["output"]
# Shapes: image_clip (1, 10, 3, 224, 224), sensor_clip (1, 10, 8)
providers = ["CPUExecutionProvider"]
sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
io_bindings = {i.name: i for i in sess.get_inputs()}
out_bindings = {o.name: o for o in sess.get_outputs()}

IMG_INPUT_NAME = "image_clip" if "image_clip" in io_bindings else list(io_bindings.keys())[0]
SENSOR_INPUT_NAME = "sensor_clip" if "sensor_clip" in io_bindings else list(io_bindings.keys())[1]
OUTPUT_NAME = "output" if "output" in out_bindings else list(out_bindings.keys())[0]

print("ONNX inputs:")
for i in sess.get_inputs():
    print({"name": i.name, "shape": i.shape, "type": i.type})
print("ONNX outputs:")
for o in sess.get_outputs():
    print({"name": o.name, "shape": o.shape, "type": o.type})

# ------------------ Detector ------------------
detector = FaceDetector(minDetectionCon=0.7)

# ------------------ Helpers ------------------
def preprocess_image_clip(image_b64_list):
    """
    Decodes a list of base64 JPEG face crops, builds a float32 tensor:
      ONNX export used channels-first: (1, T, 3, 224, 224)
    Normalization: ImageNet mean/std (same as your video test).
    """
    frames = []
    for b64_img in image_b64_list:
        try:
            raw = b64_img.split(",")[-1]
            img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
            img = img.resize((224, 224))
            arr = np.asarray(img, dtype=np.float32) / 255.0        # (H, W, C)
            # normalize
            arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            # to CHW
            arr = np.transpose(arr, (2, 0, 1))                     # (C, H, W)
            frames.append(arr)
        except Exception as e:
            print("Frame decode error:", e)
            continue

    if not frames:
        return None

    clip = np.stack(frames, axis=0)                                # (T, C, H, W)
    clip = clip[np.newaxis, ...].astype(np.float32)                # (1, T, C, H, W)
    return clip

def prepare_sensor_clip(sensor_clip):
    """
    Returns a (1, CLIP_LENGTH, SENSOR_DIM) float32 array.
    - If missing/invalid: zeros.
    - If shorter than CLIP_LENGTH: pad by repeating the last row (or zeros).
    - If longer: keep the most recent CLIP_LENGTH rows.
    """
    try:
        arr = np.asarray(sensor_clip, dtype=np.float32)
    except Exception:
        arr = np.zeros((0, SENSOR_DIM), dtype=np.float32)

    if arr.ndim != 2 or (arr.shape[0] > 0 and arr.shape[1] != SENSOR_DIM):
        arr = np.zeros((0, SENSOR_DIM), dtype=np.float32)

    T = arr.shape[0]
    if T == 0:
        arr = np.zeros((CLIP_LENGTH, SENSOR_DIM), dtype=np.float32)
    elif T < CLIP_LENGTH:
        pad_len = CLIP_LENGTH - T
        last = arr[-1:] if T > 0 else np.zeros((1, SENSOR_DIM), dtype=np.float32)
        pad = np.repeat(last, pad_len, axis=0)
        arr = np.vstack([arr, pad]).astype(np.float32)
    elif T > CLIP_LENGTH:
        arr = arr[-CLIP_LENGTH:, :].astype(np.float32)

    return np.expand_dims(arr, axis=0)  # (1, T, 8)

def safe_sigmoid(x):
    x = float(x)
    if x > 60:  # avoid overflow
        return 1.0
    if x < -60:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))

# ------------------ Routes ------------------
@app.route("/")
def index():
    # Render your existing index.html (unchanged CSS + FPS counter)
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

        # Draw preview box for the client (optional, keeps your overlay logic simple)
        out = frame.copy()
        for f in faces:
            cv2.rectangle(out, (f["x"], f["y"]), (f["x"] + f["w"], f["y"] + f["h"]), (0, 255, 0), 2)
        _, buffer = cv2.imencode(".jpg", out)
        encoded = base64.b64encode(buffer).decode("utf-8")
        return jsonify({"faces": faces, "image": f"data:image/jpeg;base64,{encoded}"})
    except Exception as e:
        print("Detection Error:", e)
        return jsonify({"error": "Detection failed"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_clip = data.get("image_clip", [])
        sensor_clip = data.get("sensor_clip", [])

        # Enforce exactly CLIP_LENGTH frames by padding/trimming here too
        if len(image_clip) == 0:
            return jsonify({"error": "No frames received"}), 400
        if len(image_clip) < CLIP_LENGTH:
            last = image_clip[-1]
            image_clip = image_clip + [last] * (CLIP_LENGTH - len(image_clip))
        elif len(image_clip) > CLIP_LENGTH:
            image_clip = image_clip[-CLIP_LENGTH:]

        img_np = preprocess_image_clip(image_clip)
        if img_np is None:
            return jsonify({"error": "Failed to preprocess images"}), 400

        sensor_np = prepare_sensor_clip(sensor_clip)

        # Debug
        print(f"[DEBUG] prepared img_clip {img_np.shape}, sensor_clip {sensor_np.shape}")

        # ONNX inference
        ort_inputs = {
            IMG_INPUT_NAME: img_np,           # (1, 10, 3, 224, 224)
            SENSOR_INPUT_NAME: sensor_np      # (1, 10, 8)
        }
        outputs = sess.run([OUTPUT_NAME], ort_inputs)
        raw_score = float(outputs[0][0][0])
        liveness = safe_sigmoid(raw_score)
        result = "Real Face" if liveness > 0.5 else "Fake Face"

        return jsonify({"liveness_score": liveness, "result": result})
    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    # If you want LAN/mobile testing without port-forward, you can do:
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(debug=True)
