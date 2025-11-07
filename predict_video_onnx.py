# predict_video_onnx.py
import os
import cv2
import math
import argparse
import numpy as np
import onnxruntime as ort

CLIP_LEN = 10
TARGET = 224
SENSOR_DIM = 8

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def center_crop_resize_bgr_to_rgb(img, size=TARGET):
    h, w = img.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    crop = img[y1:y1+side, x1:x1+side]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return rgb

def preprocess_frame(rgb):
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - IM_MEAN) / IM_STD
    chw = np.transpose(arr, (2, 0, 1))  # [3,H,W]
    return chw

def build_clip_from_video(path, need_len=CLIP_LEN):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        idxs = list(range(need_len))
    else:
        step = max(total // need_len, 1)
        idxs = [min(i*step, total-1) for i in range(need_len)]

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frm = cap.read()
        if not ok or frm is None:
            break
        rgb = center_crop_resize_bgr_to_rgb(frm, TARGET)
        chw = preprocess_frame(rgb)
        frames.append(chw)
    cap.release()

    if not frames:
        frames = [np.zeros((3, TARGET, TARGET), np.float32)] * need_len
    if len(frames) < need_len:
        frames += [frames[-1]] * (need_len - len(frames))
    elif len(frames) > need_len:
        frames = frames[:need_len]

    clip = np.stack(frames, axis=0)[np.newaxis, ...]  # [1,10,3,224,224]
    return clip.astype(np.float32)

def build_zero_sensor(need_len=CLIP_LEN):
    return np.zeros((1, need_len, SENSOR_DIM), dtype=np.float32)

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def run(video_path, onnx_path):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # Infer real input/output names from model
    names = [i.name for i in sess.get_inputs()]
    # Expect something like ['image_clip','sensor_clip'] (order may vary)
    img_name = next((n for n in names if "image" in n), names[0])
    sen_name = next((n for n in names if "sensor" in n), names[1 if len(names) > 1 else 0])

    out_name = sess.get_outputs()[0].name

    img = build_clip_from_video(video_path)
    sen = build_zero_sensor()

    feeds = {img_name: img, sen_name: sen}
    out = sess.run([out_name], feeds)[0]
    raw = float(np.array(out).reshape(-1)[0])
    score = sigmoid(raw)
    decision = "Real Face" if score >= 0.5 else "Fake Face"

    print(f"Raw logit: {raw:.4f}")
    print(f"Liveness score: {score:.4f}")
    print(f"Decision: {decision}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--onnx", default="liveness_model.onnx")
    args = ap.parse_args()
    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(f"ONNX not found: {args.onnx}")
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    run(args.video, args.onnx)
