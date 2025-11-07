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

# Liveness decision threshold (stricter than 0.5 to reduce false accepts)
LIVENESS_THRESHOLD = 0.80

# Motion/box gates (tune as needed)
# Mean absolute difference between consecutive grayscale crops (0..255 space)
MOTION_MIN = 2.5          # require at least this much average per-frame change
BOX_SHIFT_MIN = 1.0       # in pixels; average center shift across clip
BOX_SCALE_MIN = 0.002     # relative area change (e.g., 0.2% of frame area on avg)

# ------------------ Load ONNX ------------------
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
def decode_b64_to_rgb(b64_img, size=None):
    """Decode base64 data URL to RGB ndarray; optionally resize to (w,h)."""
    raw = b64_img.split(",")[-1]
    img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
    if size is not None:
        img = img.resize(size)
    return np.asarray(img, dtype=np.uint8)

def preprocess_image_clip(image_b64_list):
    """
    Build CHW-normalized tensor for ONNX:
      output: (1, T, 3, 224, 224), float32, ImageNet normalized
    Also return grayscale crops (for motion gate).
    """
    frames_chw = []
    frames_gray_small = []  # for motion gate
    for b64_img in image_b64_list:
        try:
            # model crop 224x224
            rgb = decode_b64_to_rgb(b64_img, size=(224, 224))       # (224,224,3) uint8
            arr = rgb.astype(np.float32) / 255.0
            arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            chw = np.transpose(arr, (2, 0, 1))                      # (3,224,224)
            frames_chw.append(chw)

            # small grayscale for cheap motion measure
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)  # (64,64) uint8-ish
            frames_gray_small.append(small.astype(np.float32))
        except Exception as e:
            print("Frame decode error:", e)
            continue

    if not frames_chw:
        return None, None

    clip = np.stack(frames_chw, axis=0)                             # (T,3,224,224)
    clip = clip[np.newaxis, ...].astype(np.float32)                 # (1,T,3,224,224)

    gray_stack = np.stack(frames_gray_small, axis=0)                # (T,64,64), float32 (0..255)
    return clip, gray_stack

def prepare_sensor_clip(sensor_clip):
    """
    Returns a (1, CLIP_LENGTH, SENSOR_DIM) float32 array.
    Pads/trims as needed; zeros if missing.
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

def compute_motion_score(gray_stack):
    """
    Cheap motion score: mean absolute difference between consecutive grayscale frames.
    gray_stack: (T, 64, 64), float32 in 0..255 space.
    Returns mean MAD over T-1 transitions.
    """
    if gray_stack is None or gray_stack.shape[0] < 2:
        return 0.0
    diffs = []
    for t in range(1, gray_stack.shape[0]):
        diff = np.mean(np.abs(gray_stack[t] - gray_stack[t-1]))
        diffs.append(diff)
    return float(np.mean(diffs)) if diffs else 0.0

def compute_box_gates(face_boxes):
    """
    Given list of face boxes across the clip: [{'x','y','w','h'}, ...] length T (or <=T),
    compute average center shift (pixels) and average relative area change.
    If boxes are missing for some frames, we treat those as zeros (which tends to lower movement).
    """
    if not face_boxes:
        return 0.0, 0.0
    xs, ys, ws, hs = [], [], [], []
    for f in face_boxes:
        xs.append(f.get("x", 0))
        ys.append(f.get("y", 0))
        ws.append(max(1, f.get("w", 0)))
        hs.append(max(1, f.get("h", 0)))

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    ws = np.array(ws, dtype=np.float32)
    hs = np.array(hs, dtype=np.float32)

    centers = np.stack([xs + ws/2.0, ys + hs/2.0], axis=1)  # (N,2)
    shifts = np.linalg.norm(centers[1:] - centers[:-1], axis=1) if len(centers) > 1 else np.array([0.0])
    avg_shift = float(np.mean(shifts)) if shifts.size else 0.0

    areas = ws * hs
    if len(areas) > 1:
        rel_changes = np.abs(areas[1:] - areas[:-1]) / np.maximum(areas[:-1], 1.0)
        avg_rel_area = float(np.mean(rel_changes))
    else:
        avg_rel_area = 0.0

    return avg_shift, avg_rel_area

# ------------------ Routes ------------------
@app.route("/")
def index():
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

        # Draw preview box for the client (helps overlay)
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
        face_boxes = data.get("face_boxes", [])  # optional: if you send boxes per frame from the client

        # Enforce exactly CLIP_LENGTH frames
        if len(image_clip) == 0:
            return jsonify({"error": "No frames received"}), 400
        if len(image_clip) < CLIP_LENGTH:
            last = image_clip[-1]
            image_clip = image_clip + [last] * (CLIP_LENGTH - len(image_clip))
        elif len(image_clip) > CLIP_LENGTH:
            image_clip = image_clip[-CLIP_LENGTH:]

        # Align box list length with frames (optional; improves box gate)
        if face_boxes and len(face_boxes) != len(image_clip):
            if len(face_boxes) < len(image_clip):
                face_boxes = face_boxes + [face_boxes[-1]] * (len(image_clip) - len(face_boxes))
            else:
                face_boxes = face_boxes[-len(image_clip):]

        img_np, gray_stack = preprocess_image_clip(image_clip)
        if img_np is None:
            return jsonify({"error": "Failed to preprocess images"}), 400

        sensor_np = prepare_sensor_clip(sensor_clip)

        # --- Anti-spoof gates ---
        motion_score = compute_motion_score(gray_stack)                 # avg MAD in 0..255
        shift_px, area_rel = compute_box_gates(face_boxes) if face_boxes else (0.0, 0.0)

        # If motion is too low OR box movement/scale is too low, mark likely spoof
        motion_ok = motion_score >= MOTION_MIN
        box_ok = (shift_px >= BOX_SHIFT_MIN) or (area_rel >= BOX_SCALE_MIN)

        # If gates fail, short-circuit to Fake without calling the model
        if not (motion_ok and box_ok):
            print(f"[GATE] motion_ok={motion_ok} (motion={motion_score:.3f}), "
                  f"box_ok={box_ok} (shift={shift_px:.3f}, area_rel={area_rel:.4f}) -> reject")
            return jsonify({
                "liveness_score": 0.0,
                "result": "Fake Face",
                "reason": "Insufficient motion/variation"
            })

        # --- ONNX inference ---
        ort_inputs = {
            IMG_INPUT_NAME: img_np,           # (1, 10, 3, 224, 224)
            SENSOR_INPUT_NAME: sensor_np      # (1, 10, 8)
        }
        outputs = sess.run([OUTPUT_NAME], ort_inputs)
        raw_score = float(outputs[0][0][0])
        liveness = safe_sigmoid(raw_score)

        result = "Real Face" if liveness >= LIVENESS_THRESHOLD else "Fake Face"
        return jsonify({
            "liveness_score": liveness,
            "result": result,
            "motion": motion_score,
            "box_shift_px": shift_px,
            "box_area_rel": area_rel
        })
    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True)  # use this for mobile over LAN if you want
    app.run(debug=True)
