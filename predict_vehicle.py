import argparse
from io import BytesIO
from pathlib import Path

import requests
from fastai.vision.all import PILImage, load_learner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predictions against one or more exported FastAI models."
    )
    parser.add_argument(
        "--image-url",
        required=True,
        help="Remote image URL to classify.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Path to an exported .pkl model. Repeat for multiple models.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds for the image download.",
    )
    return parser.parse_args()


def load_image(image_url: str, timeout: int) -> PILImage:
    response = requests.get(image_url, timeout=timeout)
    response.raise_for_status()
    return PILImage.create(BytesIO(response.content))


def main() -> None:
    args = parse_args()
    image = load_image(args.image_url, args.timeout)

    for model_path_str in args.model:
        model_path = Path(model_path_str)
        learner = load_learner(model_path)
        prediction, prediction_idx, probabilities = learner.predict(image)
        confidence = probabilities[prediction_idx].item() * 100

        print(f"Model: {model_path}")
        print(f"Predicted class: {prediction}")
        print(f"Probability: {confidence:.2f}%")
        print()


if __name__ == "__main__":
    main()
