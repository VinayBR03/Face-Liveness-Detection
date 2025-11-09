import os
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

def convert_onnx_to_tflite(onnx_path, tflite_path):
    """
    Convert ONNX model to TFLite format without quantization
    """
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert ONNX to TensorFlow
    tf_model_path = "temp_tf_model"
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)
    
    # Convert SavedModel to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # Configure the converter
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # Configure the converter for LSTM support
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    
    # Optimize for inference
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    
    # Ensure input shapes are fixed
    converter.experimental_enable_resource_variables = True
    
    # Convert the model with reduced operations
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Clean up temporary files
    if os.path.exists(tf_model_path):
        import shutil
        shutil.rmtree(tf_model_path)
    
    print(f"Model successfully converted and saved to {tflite_path}")

if __name__ == "__main__":
    # Paths configuration
    onnx_model_path = "liveness_model.onnx"
    tflite_model_path = "liveness_model_fp32.tflite"
    
    # Convert the model
    convert_onnx_to_tflite(onnx_model_path, tflite_model_path)
