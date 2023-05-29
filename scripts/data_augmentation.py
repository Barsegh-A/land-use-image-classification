import adddeps

import os
import json
import torch
import uuid
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import cv2
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.dataset import LandUseImagesDataset
from src.transformations import randomAffine, randomPerspective, blend_with_background


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-path',
        type=str,
        default='./data/UCMerced_LandUse_processed',
        help='Path to dataset',
    )
    parser.add_argument(
        '--label-path',
        type=str,
        default='./data/UCMerced_LandUse_processed/image_labels.json',
        help='Path to labels',
    )
    parser.add_argument(
        '--bg-data-path',
        type=str,
        default='./data/EuroSAT_RGB_processed',
        help='Path to dataset containing background images',
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./data/UCMerced_LandUse_augmented',
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
        '--affine-prop',
        type=float,
        default=0.3,
        help='Proportion of images for affine transformation',
    )
    parser.add_argument(
        '--perspective-prop',
        type=float,
        default=0.3,
        help='Proportion of images for perspective transformation',
    )

    return parser.parse_args()


def main(args):
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    transform = T.Compose([
        lambda x: np.array(x),
    ])

    dataset = LandUseImagesDataset(args.data_path, transform=transform)
    bg_dataset = LandUseImagesDataset(args.bg_data_path, transform=transform)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, generator=generator)
    bg_dataloader = DataLoader(bg_dataset, batch_size=1, shuffle=True, generator=generator)

    output_path = Path(args.output_path)
    aug_images_path = output_path / 'Images'
    os.makedirs(aug_images_path, exist_ok=True)

    dataset_lenght = len(dataset)
    affine_idx = min(int(args.affine_prop * dataset_lenght), dataset_lenght)
    persp_idx = dataset_lenght - min(int(args.perspective_prop * dataset_lenght), dataset_lenght)

    label_path = Path(args.label_path)
    with open(label_path, 'r') as f:
        labels_json_data = json.load(f)

    toPIL = T.ToPILImage()
    image_labels = []

    for idx, ((batch_images, batch_lables), (bg_batch_images, _)) in tqdm(enumerate(zip(dataloader, bg_dataloader)),
                                                                          total=dataset_lenght):
        image_id = uuid.uuid4().hex[:8]
        image = batch_images[0].numpy()
        bg_image = bg_batch_images[0].numpy()

        blended_image = blend_with_background(image, bg_image)
        toPIL(blended_image).save(aug_images_path / f'{image_id}.{args.extension}')

        curr_labels = batch_lables[0].tolist()
        image_labels.append({f'{image_id}': curr_labels})

        if persp_idx < idx < affine_idx:
            transfored_image = randomAffine(blended_image, bg_image, idx)
            transfored_image = randomPerspective(transfored_image, idx)
            toPIL(transfored_image).save(aug_images_path / f'{image_id}_aff_persp.{args.extension}')
            image_labels.append({f'{image_id}_aff_persp': curr_labels})
        elif idx < affine_idx:
            transfored_image = randomAffine(blended_image, bg_image, idx)
            toPIL(transfored_image).save(aug_images_path / f'{image_id}_aff.{args.extension}')
            image_labels.append({f'{image_id}_aff': curr_labels})
        elif idx > persp_idx:
            transfored_image = randomPerspective(blended_image, idx)
            toPIL(transfored_image).save(aug_images_path / f'{image_id}_persp.{args.extension}')
            image_labels.append({f'{image_id}_persp': curr_labels})

    aug_json_data = {
        'mapping': labels_json_data['mapping'],
        'extension': args.extension,
        'images': image_labels
    }
    with open(output_path / label_path.name, 'w') as f:
        json.dump(aug_json_data, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)
