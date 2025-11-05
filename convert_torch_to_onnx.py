# convert_torch_to_onnx.py
import torch
from model import MultiModalLivenessModel
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", type=str, required=True, help="Path to the input .pth model file.")
    parser.add_argument("--output-model", type=str, required=True, help="Path for the output .onnx model file.")
    parser.add_argument("--clip-length", type=int, default=25, help="Length of the input clips for the model.")
    args = parser.parse_args()

    print(f"🔄 Converting {args.input_model} to {args.output_model}...")

    model = MultiModalLivenessModel()
    model.load_state_dict(torch.load(args.input_model, map_location=torch.device('cpu')))
    model.eval()

    dummy_image = torch.randn(1, args.clip_length, 3, 224, 224)
    dummy_sensor = torch.randn(1, args.clip_length, 8)
    
    torch.onnx.export(
        model,
        (dummy_image, dummy_sensor),
        args.output_model,
        input_names=["image_clip", "sensor_clip"],
        output_names=["output"],
        opset_version=12,
    )
    print(f"✅ Successfully exported to ONNX: {args.output_model}")

if __name__ == "__main__":
    main()