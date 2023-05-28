import os
import argparse
import json
import math
import torch
import uuid
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm

from pathlib import Path
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-path',
        type=str,
        default='./data/UCMerced_LandUse/Images',
        help='Path to dataset',
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default='./data/UCMerced_LandUse_processed',
        help='Path to processed dataset',
    )

    parser.add_argument(
        '--extension',
        type=str,
        default='jpg',
        help='The extension of processed images. One of jpg, jpeg, png, tif',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
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

    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Number of images to be combined',
    )

    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffle images before processing',
    )

    return parser.parse_args()


def main(args):
    if args.extension not in ['jpg', 'jpeg', 'png', 'tif']:
        raise NotImplementedError

    dataset_path = Path(args.data_path)
    h, w = args.height, args.width

    transform = T.Compose([
        T.Resize((h, w)),
        T.ToTensor()
    ])

    dataset = ImageFolder(dataset_path, transform=transform)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, generator=generator, drop_last=True)

    class_to_idx = dataset.class_to_idx
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    toPIL = T.ToPILImage()

    output_path = Path(args.output_path)
    output_path_images = output_path / 'Images'
    os.makedirs(output_path_images, exist_ok=True)

    images_labels_array = []

    image_count_on_edge = int(math.sqrt(args.batch_size))

    for idx, (batch_images, batch_labels) in tqdm(enumerate(dataloader), total=len(dataset)):
        batch_images = [toPIL(batch_image) for batch_image in batch_images]

        combined_image = Image.new('RGB', (image_count_on_edge * w, image_count_on_edge * h))

        image_id = uuid.uuid4().hex[:8]

        for i in range(args.batch_size):
            combined_image.paste(batch_images[i], (i % image_count_on_edge * w, i // image_count_on_edge * h))

        one_hot_labels = np.zeros(len(dataset.classes))
        one_hot_labels[batch_labels] = 1
        images_labels_array.append({image_id: list(map(int, one_hot_labels))})

        combined_image.save(output_path_images / f'{image_id}.{args.extension}')

    json_data = {
        'mapping': idx_to_class,
        'extension': args.extension,
        'images': images_labels_array
    }

    with open(output_path / 'image_labels.json', 'w') as json_file:
        json.dump(json_data, json_file) # indent = 4


if __name__ == '__main__':
    args = parse_args()
    main(args)
