import argparse
from pathlib import Path

from fastai.vision.all import (
    ImageDataLoaders,
    Normalize,
    accuracy,
    aug_transforms,
    imagenet_stats,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    vision_learner,
    Resize,
)


BACKBONES = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a FastAI vehicle classifier from local train/validation folders."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Root directory containing the train and validation folders.",
    )
    parser.add_argument(
        "--train-dir",
        default="train",
        help="Training folder name inside --data-dir.",
    )
    parser.add_argument(
        "--valid-dir",
        default="test",
        help="Validation folder name inside --data-dir.",
    )
    parser.add_argument(
        "--arch",
        choices=sorted(BACKBONES),
        default="resnet50",
        help="Backbone architecture to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=299,
        help="Resize dimension for training images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .pkl path. Defaults to simple_cnn_fastai_<arch>.pkl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or Path(f"simple_cnn_fastai_{args.arch}.pkl")

    batch_tfms = [
        *aug_transforms(size=args.image_size, min_scale=0.75),
        Normalize.from_stats(*imagenet_stats),
    ]

    dls = ImageDataLoaders.from_folder(
        args.data_dir,
        train=args.train_dir,
        valid=args.valid_dir,
        item_tfms=Resize(args.image_size),
        batch_tfms=batch_tfms,
        bs=args.batch_size,
    )

    learner = vision_learner(dls, BACKBONES[args.arch], metrics=accuracy)
    learner.fine_tune(args.epochs)
    learner.export(output_path)

    print(f"Finished training. Model saved to {output_path}")


if __name__ == "__main__":
    main()
