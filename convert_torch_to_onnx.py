import os
import argparse
import torch
import torch.nn as nn

from model import MultiModalLivenessModel


class StaticExportModel(nn.Module):
    """
    Export wrapper that removes dynamic reshape operations from ONNX.
    This wrapper is used ONLY for ONNX export.
    """

    def __init__(self, model):
        super().__init__()

        self.image_features = model.image_features
        self.attention = model.attention
        self.adap_pool = model.adap_pool
        self.pre_lstm = model.pre_lstm
        self.lstm_cell = model.lstm_cell
        self.classifier = model.classifier
        self.lstm_hidden_dim = model.lstm_hidden_dim
        self.image_feature_dim = model.image_feature_dim

    def forward(self, image_clip, sensor_clip):

        # image_clip is ALWAYS (1,10,3,224,224)
        image_input = image_clip.reshape(10, 3, 224, 224)

        img_feat = self.image_features(image_input)
        img_feat = self.attention(img_feat)
        img_feat = self.adap_pool(img_feat)

        img_feat = img_feat.reshape(1, 10, self.image_feature_dim)

        combined = torch.cat((img_feat, sensor_clip), dim=2)

        lstm_input = self.pre_lstm(combined)

        h = torch.zeros(
            1,
            self.lstm_hidden_dim,
            dtype=lstm_input.dtype,
            device=lstm_input.device,
        )

        c = torch.zeros_like(h)

        for t in range(10):
            h, c = self.lstm_cell(lstm_input[:, t], (h, c))

        return self.classifier(h)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-model",
        default=os.path.join("models", "liveness_model.pth"),
    )

    parser.add_argument(
        "--output-model",
        default=os.path.join("models", "liveness_model.onnx"),
    )

    args = parser.parse_args()

    print("Loading PyTorch model...")

    model = MultiModalLivenessModel()

    model.load_state_dict(
        torch.load(
            args.input_model,
            map_location="cpu",
        )
    )

    model.eval()

    export_model = StaticExportModel(model)
    export_model.eval()

    dummy_image = torch.randn(1, 10, 3, 224, 224)
    dummy_sensor = torch.randn(1, 10, 8)

    print("Exporting ONNX...")

    torch.onnx.export(
        export_model,
        (dummy_image, dummy_sensor),
        args.output_model,
        input_names=[
            "image_clip",
            "sensor_clip",
        ],
        output_names=[
            "output",
        ],
        opset_version=11,
        export_params=True,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
    )

    print(f"\nONNX saved to\n{args.output_model}")


if __name__ == "__main__":
    main()