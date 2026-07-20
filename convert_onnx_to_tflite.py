import argparse
import os
import shutil
import subprocess


def convert_onnx_to_tflite(onnx_path, output_dir, models_dir):
    print("=" * 60)
    print("Converting ONNX → TensorFlow SavedModel + TFLite")
    print("=" * 60)

    os.makedirs(models_dir, exist_ok=True)

    subprocess.run(
        [
            "onnx2tf",
            "-i", onnx_path,
            "-o", output_dir,
            "-kat", "sensor_clip",
        ],
        check=True,
    )

    fp32_src = os.path.join(output_dir, "liveness_model_float32.tflite")
    fp16_src = os.path.join(output_dir, "liveness_model_float16.tflite")

    fp32_dst = os.path.join(models_dir, "liveness_model_fp32.tflite")
    fp16_dst = os.path.join(models_dir, "liveness_model_fp16.tflite")

    if os.path.exists(fp32_src):
        shutil.move(fp32_src, fp32_dst)

    if os.path.exists(fp16_src):
        shutil.move(fp16_src, fp16_dst)

    print("\nConversion completed successfully!")
    print(f"SavedModel : {output_dir}")
    print(f"FP32 TFLite : {fp32_dst}")
    print(f"FP16 TFLite : {fp16_dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-model",
        default=os.path.join("models", "liveness_model.onnx"),
    )

    parser.add_argument(
        "--output-dir",
        default="temp_saved_model",
    )

    parser.add_argument(
        "--models-dir",
        default="models",
    )

    args = parser.parse_args()

    convert_onnx_to_tflite(
        args.input_model,
        args.output_dir,
        args.models_dir,
    )