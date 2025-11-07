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
    expects_nchw_time = (len(img_in["shape"]) == 5 and img_in["shape"][2] == 3)
    return img_in, sen_in, expects_nchw_time

def build_clip_from_video(path, expects_nchw_time):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        idxs = list(range(CLIP))
    else:
        step = max(total // CLIP, 1)
        idxs = [min(i*step, total-1) for i in range(CLIP)]

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frm = cap.read()
        if not ok:
            break
        rgb = center_crop_resize_bgr_to_rgb(frm, H)
        nhwc = preprocess_rgb(rgb)
        frames.append(nhwc)
    cap.release()

    if not frames:
        frames = [np.zeros((H, W, C), np.float32)] * CLIP
    if len(frames) < CLIP:
        frames += [frames[-1]] * (CLIP - len(frames))
    if len(frames) > CLIP:
        frames = frames[:CLIP]

    clip = np.stack(frames, axis=0).astype(np.float32)  # T H W C
    if expects_nchw_time:
        clip = np.transpose(clip, (0, 3, 1, 2))          # T C H W
        clip = clip[np.newaxis, ...]                     # 1 T C H W
    else:
        clip = clip[np.newaxis, ...]                     # 1 T H W C
    return clip

def build_sensor():
    return np.zeros((1, CLIP, SENSOR_DIM), dtype=np.float32)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="liveness_model_int8.tflite")
    args = ap.parse_args()

    interpreter = load_interpreter(args.model)
    img_in, sen_in, expects_nchw_time = get_bindings(interpreter)
    out = interpreter.get_output_details()[0]

    img = build_clip_from_video(args.video, expects_nchw_time)
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

    interpreter.set_tensor(sen_in["index"], sen)
    interpreter.set_tensor(img_in["index"], img)
    interpreter.invoke()

    y = interpreter.get_tensor(out["index"]).reshape(-1)[0]
    score = sigmoid(y)
    print(f"Raw logit: {y:.4f}")
    print(f"Liveness score: {score:.4f}")
    print("Decision:", "Real Face" if score >= 0.5 else "Fake Face")

if __name__ == "__main__":
    main()
