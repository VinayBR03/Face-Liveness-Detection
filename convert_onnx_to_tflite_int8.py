import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from tqdm import tqdm

# Import your existing dataset loader
from dataset import LivenessDataset
from torch.utils.data import DataLoader

def representative_dataset_gen(data_dir, num_samples=100):
    """
    A generator function that provides a small number of samples
    to the TFLite converter for calibrating quantization.
    """
    print(f"Loading representative dataset from: {data_dir}")
    dataset = LivenessDataset(root_dir=data_dir, clip_length=10, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"Providing {min(num_samples, len(dataset))} samples for quantization calibration...")
    
    for i, (image_clip, sensor_clip, _) in enumerate(tqdm(loader, desc="Calibration")):
        if i >= num_samples:
            break
        
        # The converter expects a dictionary mapping input names to numpy arrays.
        yield {
            "image_clip": image_clip.numpy().astype(np.float32),
            "sensor_clip": sensor_clip.numpy().astype(np.float32)
        }

def convert_onnx_to_int8_tflite(onnx_path, tflite_path, data_dir):
    """
    Converts an ONNX model to a fully quantized INT8 TFLite model using onnx-tf.
    """
    print(f"📥 Loading ONNX model from: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model checked.")

    # Create a temporary directory for the intermediate SavedModel
    tf_model_path = "temp_tf_model_for_quant"
    if os.path.exists(tf_model_path):
        shutil.rmtree(tf_model_path)

    # 1. Convert ONNX to TensorFlow SavedModel using onnx-tf
    print(f"⚙️ Exporting to intermediate SavedModel format at {tf_model_path}")
    
    # --- THE CRITICAL FIX ---
    # By explicitly providing the input names, we ensure the SavedModel has a
    # static, well-defined input signature that the TFLite converter can understand.
    input_names = [node.name for node in onnx_model.graph.input]
    tf_rep = prepare(onnx_model, input_names=input_names)
    
    tf_rep.export_graph(tf_model_path)
    print("✅ Successfully created intermediate SavedModel.")

    # 2. Convert the generated SavedModel to INT8 TFLite
    print("🔄 Converting SavedModel to INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)

    # --- This is where the quantization magic happens ---
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(data_dir)
    
    # Enforce full integer quantization.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Set the input and output tensors to INT8 for the final model.
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    # 3. Save the quantized model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_quant_model)

    # 4. Clean up the temporary directory
    shutil.rmtree(tf_model_path)
    
    print("-" * 50)
    print(f"✅ Model successfully converted and saved to {tflite_path}")
    print(f"   Original ONNX size: {os.path.getsize(onnx_path) / 1e6:.2f} MB")
    print(f"   Quantized TFLite size: {os.path.getsize(tflite_path) / 1e6:.2f} MB")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to a fully quantized INT8 TFLite model.")
    parser.add_argument('--onnx-model', type=str, default='liveness_model.onnx', help='Path to the input ONNX model.')
    parser.add_argument('--tflite-model', type=str, default='liveness_model_int8.tflite', help='Path for the output INT8 TFLite model.')
    parser.add_argument('--data-dir', type=str, default='data/train', help='Path to the directory containing representative data.')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}.")
    if not os.path.exists(args.onnx_model):
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_model}.")

    convert_onnx_to_int8_tflite(args.onnx_model, args.tflite_model, args.data_dir)
