import adddeps

import argparse
import torch
import torchvision.transforms as T

from pathlib import Path

from src.resnet import get_multilabel_resnet
from src.utils import inference
from src.dataset import CLASSES

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image-path',
        type=str,
        default='./data/UCMerced_LandUse_processed/Images/000a431a.jpg',
        help='Path to an image',
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='./models/resnet18_2023-05-27_18:26:05.712432_best_weights.pt',
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
        T.ToTensor()
    ])

    model_name = args.model_path.split('_')[0].split('/')[-1]
    num_classes = len(CLASSES)
    model = get_multilabel_resnet(model_name, num_classes=num_classes, weights=None)

    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    labels = inference(image_path, model, transform, args.threshold)
    print(labels)


if __name__ == '__main__':
    args = parse_args()
    main(args)
