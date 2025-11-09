# convert_onnx_to_tflite.py
import onnx
from onnx import numpy_helper
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import os
import argparse
import tempfile
from tabulate import tabulate

# -------------------------------------
# 🧩 Utility: print readable ONNX graph
# -------------------------------------
def print_model_summary(model, max_nodes=50):
    print("\n🧠 ONNX Model Summary:")
    rows = []
    for i, node in enumerate(model.graph.node):
        if i >= max_nodes:
            rows.append(["...", "...", "...", "..."])
            break
        inputs = ", ".join(node.input)
        outputs = ", ".join(node.output)
        rows.append([i, node.op_type, inputs, outputs])
    print(tabulate(rows, headers=["#", "OpType", "Inputs", "Outputs"], tablefmt="fancy_grid"))

    print(f"\nTotal Nodes: {len(model.graph.node)}")
    print(f"Total Initializers: {len(model.graph.initializer)}")
    print(f"Inputs: {[i.name for i in model.graph.input]}")
    print(f"Outputs: {[o.name for o in model.graph.output]}\n")

# -------------------------------------
# 🧠 Fix bad reshape ops
# -------------------------------------
def fix_bad_reshapes(model):
    fixed = 0
    for node in model.graph.node:
        if node.op_type == "Reshape":
            if len(node.input) > 1:
                shape_name = node.input[1]
                for init in model.graph.initializer:
                    if init.name == shape_name:
                        shape_array = numpy_helper.to_array(init)
                        if not -1 in shape_array:
                            init.CopyFrom(
                                numpy_helper.from_array(np.array([-1], dtype=np.int64), name=shape_name)
                            )
                            print(f"⚙️ Fixed Reshape '{node.name}' → [-1]")
                            fixed += 1
    if fixed == 0:
        print("✅ No reshape issues detected.")
    else:
        print(f"🔧 Fixed {fixed} reshape nodes.")
    return model

# -------------------------------------
# 🔄 Convert ONNX → TFLite
# -------------------------------------
def convert_onnx_to_tflite(onnx_path, tflite_path, calib_samples=50):
    print(f"📥 Loading ONNX model from: {onnx_path}")
    model = onnx.load(onnx_path)
    print_model_summary(model)
    model = fix_bad_reshapes(model)
    tf_rep = prepare(model, auto_cast=True)
    print("✅ Created TensorFlow representation from ONNX model.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_model_dir = os.path.join(tmp_dir, "saved_model")
        tf_rep.export_graph(saved_model_dir)
        print(f"✅ Exported intermediate SavedModel → {saved_model_dir}")

        # Representative dataset
        def representative_data_gen():
            for _ in range(calib_samples):
                # The onnx-tf converter expects the TensorFlow format (NDHWC) for its
                # The onnx-tf converter expects the TensorFlow format (NHWC) for its
                # internal graph representation, even when converting from an ONNX model
                # that used NCDHW.
                # that used NCHW.
                image_clip = np.random.rand(1, 10, 224, 224, 3).astype(np.float32)
                sensor_clip = np.random.rand(1, 10, 8).astype(np.float32)
                yield {"image_clip": image_clip, "sensor_clip": sensor_clip}
                yield [image_clip, sensor_clip]

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter._experimental_lower_tensor_list_ops = False

        try:
            print("🔄 Converting to INT8 TFLite (full quantization)...")
            tflite_model = converter.convert()
            print("✅ Successfully created INT8 TFLite model.")
        except Exception as e:
            print(f"❌ INT8 quantization failed: {e}")
            print("🔁 Retrying with FP16 fallback...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.target_spec.supported_types = [tf.float16]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
            tflite_model = converter.convert()
            print("✅ FP16 TFLite model created successfully.")

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"💾 Saved final TFLite model: {tflite_path}")

# -------------------------------------
# 🚀 CLI entry
# -------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert ONNX → TFLite with summary + reshape fix")
    parser.add_argument("--input-model", required=True, help="Path to ONNX model")
    parser.add_argument("--output-model", required=True, help="Path to save TFLite model")
    parser.add_argument("--calib_samples", type=int, default=50, help="Number of representative samples")
    args = parser.parse_args()
    convert_onnx_to_tflite(args.input_model, args.output_model, args.calib_samples)

if __name__ == "__main__":
    main()
