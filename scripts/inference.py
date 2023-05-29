import adddeps

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T

from src.models import get_multilabel_model
from src.utils import inference
from src.dataset import CLASSES


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image-path',
        type=str,
        required=True,
        help='Path to an image',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help='Name of the model',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the model',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device name to use when training model'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for prediction',
    )
    parser.add_argument(
        '--width',
        type=int,
        default=256,
        help='Width of a single image before processing',
    )
    parser.add_argument(
        '--height',
        type=int,
        default=256,
        help='Height of a single image before processing',
    )

    return parser.parse_args()


def main(args):
    image_path = Path(args.image_path)
    h, w = args.height, args.width

    transform = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    model_name = args.model_name
    if not model_name:
        model_name = args.model_path.split('_')[0].split('/')[-1]

    num_classes = len(CLASSES)

    model = get_multilabel_model(model_name, num_classes=num_classes)

    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    labels = inference(image_path, model, transform, args.threshold)
    print(labels)


if __name__ == '__main__':
    args = parse_args()
    main(args)
