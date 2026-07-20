import argparse, cv2, numpy as np, math
from tensorflow.lite.python.interpreter import Interpreter

CLIP = 10
H, W, C = 224, 224, 3
SENSOR_DIM = 8

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def center_crop_resize_bgr_to_rgb(img, size=224):
    h, w = img.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    crop = img[y1:y1+side, x1:x1+side]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

def preprocess_rgb(rgb):
    x = rgb.astype(np.float32) / 255.0
    x = (x - IM_MEAN) / IM_STD
    return x  # HWC

def load_interpreter(path):
    itp = Interpreter(model_path=path)
    itp.allocate_tensors()
    return itp

def get_bindings(interpreter):
    ins = interpreter.get_input_details()
    img_in = None
    sen_in = None
    for d in ins:
        name = d["name"].lower()
        if "image" in name:
            img_in = d
        if "sensor" in name:
            sen_in = d
    if img_in is None or sen_in is None:
        for d in ins:
            if len(d["shape"]) == 5:
                img_in = img_in or d
            if len(d["shape"]) == 3:
                sen_in = sen_in or d
    assert img_in is not None and sen_in is not None, "Could not map inputs"
    # Keras TFLite models are typically NTHWC, PyTorch ONNX are NTCHW
    # Check the channel dimension.
    img_shape = img_in["shape"]

    if img_shape[1] == 3:
        image_layout = "BCHWT"
    elif img_shape[4] == 3:
        image_layout = "BTHWC"
    else:
        raise RuntimeError(f"Unknown image layout: {img_shape}")

    return img_in, sen_in, image_layout

def build_clip_from_video(path, image_layout):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample CLIP frames uniformly
    if total <= 0:
        idxs = list(range(CLIP))
    else:
        step = max(total // CLIP, 1)
        idxs = [min(i * step, total - 1) for i in range(CLIP)]

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()

        if not ok:
            continue

        # BGR -> RGB + center crop + resize
        rgb = center_crop_resize_bgr_to_rgb(frame, H)

        # Normalize (H,W,C)
        rgb = preprocess_rgb(rgb)
        frames.append(rgb)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames could be read from the video.")

    # Pad if fewer than CLIP frames
    while len(frames) < CLIP:
        frames.append(frames[-1])

    # Trim if somehow more than CLIP
    frames = frames[:CLIP]

    # Stack to (T,H,W,C)
    clip = np.stack(frames, axis=0).astype(np.float32)

    if image_layout == "BCHWT":
        # (T,H,W,C) -> (1,C,H,W,T)
        clip = np.transpose(clip, (3, 1, 2, 0))
        clip = np.expand_dims(clip, axis=0)
    elif image_layout == "BTHWC":
        # (T,H,W,C) -> (1,T,H,W,C)
        clip = np.expand_dims(clip, axis=0)
    else:
        raise RuntimeError(f"Unsupported image layout: {image_layout}")
    return clip.astype(np.float32)

def build_sensor():
    return np.zeros((1, CLIP, SENSOR_DIM), dtype=np.float32)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="models/liveness_model_int8.tflite", help="Path to the TFLite model file.")
    args = ap.parse_args()

    interpreter = load_interpreter(args.model)
    img_in, sen_in, image_layout = get_bindings(interpreter)
    out = interpreter.get_output_details()[0]

    img = build_clip_from_video(args.video, image_layout)
    sen = build_sensor()

    # Sanity
    print("TFLite inputs:")
    for d in interpreter.get_input_details():
        print({k: d[k] for k in ["index", "name", "shape", "dtype", "quantization"]})
    print("TFLite outputs:")
    for d in interpreter.get_output_details():
        print({k: d[k] for k in ["index", "name", "shape", "dtype", "quantization"]})
    print(f"[DEBUG] Prepared image_clip {img.shape}, sensor_clip {sen.shape}")
    print(f"[DEBUG] Model expects image {img_in['shape']}, sensor {sen_in['shape']}")

    # --- Quantize inputs if the model is INT8 ---
    img_scale, img_zero = img_in.get("quantization", (0.0, 0))
    if img_scale > 0.0:
        img = (img / img_scale + img_zero).astype(np.int8)

    sen_scale, sen_zero = sen_in.get("quantization", (0.0, 0))
    if sen_scale > 0.0:
        sen = (sen / sen_scale + sen_zero).astype(np.int8)

    # Set tensors
    interpreter.set_tensor(img_in["index"], img)
    interpreter.set_tensor(sen_in["index"], sen)

    # Run inference
    interpreter.invoke()

    raw_output = interpreter.get_tensor(out["index"]).reshape(-1)[0]

    # Dequantize the output if the model is quantized (INT8/UINT8)
    scale, zero_point = out.get("quantization", (0.0, 0))
    if scale > 0.0:
        logit = (float(raw_output) - zero_point) * scale
        print(f"[DEBUG] Raw quantized output: {raw_output}, Dequantized: {logit:.4f}")
    else:
        logit = float(raw_output)

    score = sigmoid(logit)
    print(f"Raw logit: {logit:.4f}")
    print(f"Liveness score: {score:.4f}")
    print("Decision:", "Real Face" if score >= 0.5 else "Fake Face")

if __name__ == "__main__":
    main()
