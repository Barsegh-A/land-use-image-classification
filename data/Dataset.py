import torch
import PIL
class LandUseDataset(torch.utils.data.Dataset):
    def __init__(self, image_labels, images_path, transform=None):
        self.image_labels = image_labels
        self.images_path = images_path
        self.transform = transform
    def __len__(self):
        return len(self.image_labels)
    def __getitem__(self, idx):
        image_id, labels = self.image_labels[idx]
        # Load the JPG image containing 4 sub-pictures
        image_filename = f'{image_id}.jpg'
        image_filepath = f'{self.images_path}/{image_filename}'
        image = PIL.Image.open(image_filepath)
        # Preprocess the image if a transform is provided
        if self.transform is not None:
            image = self.transform(image)
        # Convert labels to tensors
        labels = torch.nn.functional.one_hot(torch.tensor(self.image_labels),21)
        return image,labels