import cv2
import base64
import io
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from cvzone.FaceDetectionModule import FaceDetector

from tensorflow.lite.python.interpreter import Interpreter

# ------------------ Flask App ------------------
app = Flask(__name__)

# ------------------ Config ------------------
MODEL_PATH = "liveness_model_int8.tflite"
CLIP_LENGTH = 25
SENSOR_DIM = 8  # adjust if your model expects a different sensor dimension

# ------------------ Load TFLite Model ------------------
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------ Face Detector ------------------
detector = FaceDetector(minDetectionCon=0.7)

# ------------------ Preprocessing ------------------
def preprocess_image_clip(image_b64_list):
    """Convert base64-encoded frames into normalized model-ready tensor."""
    frames = []
    for b64_img in image_b64_list:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64_img.split(",")[1]))).convert("RGB")
            img = img.resize((224, 224))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            frames.append(arr)
        except Exception as e:
            print("Frame decode error:", e)
            continue
    if not frames:
        return None
    clip = np.stack(frames, axis=0)[np.newaxis, ...]  # shape (1, clip_len, h, w, c)
    return clip.astype(np.float32)

# ------------------ Routes ------------------
@app.route("/")
def index():
    """Serve the webcam interface."""
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """Detect faces from webcam frame."""
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
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", img)
        encoded = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"faces": faces, "image": f"data:image/jpeg;base64,{encoded}"})
    except Exception as e:
        print("Detection Error:", e)
        return jsonify({"error": "Detection failed"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Predict liveness score from image clip."""
    try:
        data = request.get_json()
        image_clip = data.get("image_clip", [])
        sensor_clip = data.get("sensor_clip", [])

        if len(image_clip) != CLIP_LENGTH:
            return jsonify({"error": f"Expected {CLIP_LENGTH} frames, got {len(image_clip)}"}), 400

        img_clip = preprocess_image_clip(image_clip)
        if img_clip is None:
            return jsonify({"error": "Failed to preprocess images"}), 400

        # Handle sensor data
        if not sensor_clip:
            sensor_data = np.zeros((CLIP_LENGTH, SENSOR_DIM), dtype=np.float32)
        else:
            sensor_data = np.array(sensor_clip, dtype=np.float32)
        sensor_clip = np.expand_dims(sensor_data, axis=0)

        # Feed to model
        interpreter.set_tensor(input_details[0]["index"], img_clip)
        interpreter.set_tensor(input_details[1]["index"], sensor_clip)
        interpreter.invoke()

        score = interpreter.get_tensor(output_details[0]["index"])[0][0]
        liveness = 1 / (1 + np.exp(-score))
        result = "Real Face ✅" if liveness > 0.5 else "Fake Face ❌"

        return jsonify({
            "liveness_score": float(liveness),
            "result": result
        })
    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": "Prediction failed"}), 500


# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(debug=True)
