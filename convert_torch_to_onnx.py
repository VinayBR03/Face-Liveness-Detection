# convert_torch_to_onnx.py
import torch
from model import MultiModalLivenessModel
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", type=str, default="liveness_model.pth", help="Path to the input .pth model file.")
    parser.add_argument("--output-model", type=str, default="liveness_model.onnx", help="Path for the output .onnx model file.")
    parser.add_argument("--clip-length", type=int, default=10, help="Length of the input clips for the model.")
    args = parser.parse_args()

    print(f"🔄 Converting {args.input_model} to {args.output_model}...")

    model = MultiModalLivenessModel()
    model.load_state_dict(torch.load(args.input_model, map_location=torch.device('cpu')))
    model.eval()

    # Fix all dimensions for ONNX export
    batch_size = 1
    clip_length = args.clip_length
    dummy_image = torch.randn(batch_size, clip_length, 3, 224, 224)
    dummy_sensor = torch.randn(batch_size, clip_length, 8)

    # Create initial hidden and cell states for LSTM
    h0 = torch.zeros(2, batch_size, 128)
    c0 = torch.zeros(2, batch_size, 128)

    # Export with fixed shapes and simplified settings
    torch.onnx.export(
        model,
        (dummy_image, dummy_sensor),
        args.output_model,
        input_names=["image_clip", "sensor_clip"],
        output_names=["output"],
        opset_version=11,  # Using lower opset version for better compatibility
        dynamic_axes=None,
        do_constant_folding=True,
        export_params=True,
        verbose=True
    )

    print(f"✅ Successfully exported to ONNX: {args.output_model}")

if __name__ == "__main__":
    main()
