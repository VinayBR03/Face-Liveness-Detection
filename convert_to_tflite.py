import os
import shutil
import argparse
import logging
from pathlib import Path
import tempfile

import torch
import onnx
import numpy as np
from onnx_tf.backend import prepare  # onnx-tf is deprecated but required for this ONNX->TF step
import tensorflow as tf

from model import MultiModalLivenessModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_tflite_model(tflite_model_path, pytorch_model, dummy_inputs):
    """Compares the output of the TFLite model and the PyTorch model."""
    logging.info("Validating converted TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set TFLite inputs
    interpreter.set_tensor(input_details[0]['index'], dummy_inputs[0].numpy())
    interpreter.set_tensor(input_details[1]['index'], dummy_inputs[1].numpy())
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_model(*dummy_inputs).numpy()

    # Compare outputs
    if np.allclose(pytorch_output, tflite_output, atol=1e-2):
        logging.info("Validation successful: TFLite model output matches PyTorch model output.")
    else:
        logging.warning("Validation failed: TFLite and PyTorch model outputs do not match closely.")
        logging.warning(f"PyTorch output: {pytorch_output}")
        logging.warning(f"TFLite output:  {tflite_output}")

def export_pytorch_to_tflite(
    pytorch_model_path: Path,
    tflite_model_path: Path,
    input_shapes: tuple,
    opset_version: int = 11,
    quantization: str = "DEFAULT"
):
    """
    Converts a PyTorch model to TensorFlow Lite format via ONNX.

    Args:
        pytorch_model_path: Path to the PyTorch model file.
        tflite_model_path: Path to save the TFLite model file.
        input_shapes: Tuple of input shapes for the model (image_clip, sensor_clip).
        opset_version: The ONNX opset version to use for exporting.
        quantization: The type of TFLite quantization to apply ('DEFAULT', 'Float16', or 'None').
    """
    # Use a temporary directory to handle intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        onnx_model_path = temp_dir_path / "model.onnx"
        tf_model_path = temp_dir_path / "tf_model"

        # --- 1. Load PyTorch Model ---
        if not pytorch_model_path.exists():
            logging.error(f"PyTorch model not found at: {pytorch_model_path}")
            raise FileNotFoundError(f"PyTorch model not found at: {pytorch_model_path}")

        # Load the PyTorch model
        model = MultiModalLivenessModel()
        model.load_state_dict(torch.load(pytorch_model_path, map_location=torch.device('cpu')))
        model.eval()
        logging.info(f"Successfully loaded PyTorch model from {pytorch_model_path}")

        # --- 2. Convert PyTorch to ONNX ---
        dummy_image_clip = torch.randn(*input_shapes[0])
        dummy_sensor_clip = torch.randn(*input_shapes[1])

        torch.onnx.export(
            model,
            (dummy_image_clip, dummy_sensor_clip),
            onnx_model_path,
            export_params=True,
            opset_version=opset_version,
            input_names=["image_clip", "sensor_clip"],
            output_names=["output"],
            dynamic_axes=None,  # Use static shapes for TFLite conversion
        )
        logging.info(f"Model exported to ONNX format at {onnx_model_path}")

        # --- 3. Convert ONNX to TensorFlow SavedModel ---
        onnx_model = onnx.load(onnx_model_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(tf_model_path))
        logging.info(f"Model converted to TensorFlow SavedModel format at {tf_model_path}")

        # --- 4. Convert TensorFlow SavedModel to TFLite ---
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))

        if quantization == "DEFAULT":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            logging.info("Applying default TFLite quantization (weight quantization).")
        elif quantization == "Float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            logging.info("Applying Float16 TFLite quantization.")

        tflite_model = converter.convert()

        # --- 5. Save the TFLite Model ---
        tflite_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        logging.info(f"TFLite model saved successfully at {tflite_model_path}")
        logging.info(f"Final model size: {tflite_model_path.stat().st_size / 1024:.2f} KB")

        # --- 6. Validate the TFLite Model ---
        validate_tflite_model(tflite_model_path, model, (dummy_image_clip, dummy_sensor_clip))

def main():
    parser = argparse.ArgumentParser(description="Convert a PyTorch liveness model to TFLite.")
    parser.add_argument(
        "--input-model",
        type=Path,
        default=Path("liveness_model.pth"),
        help="Path to the input PyTorch model (.pth file)."
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("liveness_model.tflite"),
        help="Path to save the output TFLite model (.tflite file)."
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=100,
        help="Number of frames in the input clip."
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["DEFAULT", "Float16", "None"],
        default="DEFAULT",
        help="Type of TFLite quantization to apply."
    )
    args = parser.parse_args()

    # Define input shapes based on clip length (batch_size=1)
    # (image_clip, sensor_clip)
    input_shapes = [
        (1, args.clip_length, 3, 224, 224),
        (1, args.clip_length, 8)
    ]

    try:
        export_pytorch_to_tflite(
            pytorch_model_path=args.input_model,
            tflite_model_path=args.output_model,
            input_shapes=input_shapes,
            quantization=args.quantization
        )
    except Exception as e:
        logging.error(f"An error occurred during conversion: {e}", exc_info=True)

if __name__ == "__main__":
    main()