import json
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class LandUseImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        json_path = self.root_dir / 'image_labels.json'
        with open(json_path, 'r') as f:
            image_labels = json.load(f)

        self.idx_to_class = image_labels['mapping']
        self.image_labels = image_labels['images']
        self.extension = image_labels['extension']

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        image_id, label = next(iter(self.image_labels[index].items()))

        image_path = self.root_dir / 'Images' / f'{image_id}.{self.extension}'
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label).type(torch.float)
