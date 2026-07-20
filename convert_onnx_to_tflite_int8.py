import os
import argparse
import numpy as np
import tensorflow as tf

from torch.utils.data import DataLoader
from dataset import LivenessDataset


def representative_dataset(saved_model_dir, data_dir, clip_length=10, num_samples=100):
    """
    Representative dataset generator.
    Automatically matches the SavedModel input names and shapes.
    """

    # Read SavedModel signature
    loaded = tf.saved_model.load(saved_model_dir)
    infer = loaded.signatures["serving_default"]
    input_specs = infer.structured_input_signature[1]

    print("\nSavedModel Inputs:")
    for name, spec in input_specs.items():
        print(f"  {name}: {spec.shape}")

    dataset = LivenessDataset(
        root_dir=data_dir,
        clip_length=clip_length,
        is_train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )

    count = 0

    for image_clip, sensor_clip, _ in loader:

        image = image_clip.numpy().astype(np.float32)
        sensor = sensor_clip.numpy().astype(np.float32)

        feed_dict = {}

        for name, spec in input_specs.items():

            shape = tuple(spec.shape)

            # image input
            if len(shape) == 5:

                # SavedModel expects BCHWT
                if shape == (1, 3, 224, 224, clip_length):
                    image = np.transpose(image, (0, 2, 3, 4, 1))

                # SavedModel expects BTCHW
                elif shape == (1, clip_length, 3, 224, 224):
                    pass

                else:
                    raise RuntimeError(
                        f"Unsupported image shape from SavedModel: {shape}"
                    )

                feed_dict[name] = image

            # sensor input
            elif len(shape) == 3:
                feed_dict[name] = sensor

        if count == 0:
            print("\nRepresentative Sample")
            for k, v in feed_dict.items():
                print(k, v.shape)

        yield feed_dict

        count += 1

        if count >= num_samples:
            break


def convert(saved_model_dir, output_path, data_dir):

    print("=" * 60)
    print("Converting SavedModel -> INT8 TFLite")
    print("=" * 60)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = lambda: representative_dataset(
        saved_model_dir=saved_model_dir,
        data_dir=data_dir,
        clip_length=10,
        num_samples=100,
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print("\nConversion completed successfully!")
    print(f"Saved : {output_path}")
    print(f"Size  : {os.path.getsize(output_path)/1024/1024:.2f} MB")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--saved-model",
        default="temp_saved_model",
    )

    parser.add_argument(
        "--data-dir",
        default="data/train",
    )

    parser.add_argument(
        "--output-model",
        default=os.path.join("models", "liveness_model_int8.tflite"),
    )

    args = parser.parse_args()

    convert(
        saved_model_dir=args.saved_model,
        output_path=args.output_model,
        data_dir=args.data_dir,
    )