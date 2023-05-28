import adddeps

import torch
import argparse
from src.models import get_multilabel_model
import torchvision.transforms as T
from src.utils import TrainEval
from src.dataset import LandUseImagesDataset

from torch.utils.data import DataLoader, Dataset, random_split


def get_objects_for_training(model_name, learning_rate=None, num_classes=None):
    """
    Constructs model, optimizer and loss function objects.
    :param model_name: Model name(currently only resnet models)
    :param learning_rate: Learning rate of the optimizer
    :param num_classes: Number of classes in classification
    :return: Objects of (model, optimizer, loss function)
    """
    model = get_multilabel_model(model_name, num_classes=num_classes)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    return model, optimizer, criterion


def get_train_val_loader(data_path, train_portion=0.8, batch_size=64, seed=42, width=256, height=256):
    """
    Constructs train and validation loaders
    :param data_path: Path to data
    :param train_portion: Proportion of train data size from 0 to 1
    :param batch_size: Batch size
    :param seed: Seed for random generator
    :return: (Train data loader, Validation data loader)
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = LandUseImagesDataset(data_path,
                                   transform=T.Compose([T.Resize((height, width)), T.ToTensor()]))

    train, val = random_split(dataset, [train_portion, 1.0 - train_portion], generator=generator)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)
    val_loader = DataLoader(val, batch_size=batch_size, generator=generator)

    return train_loader, val_loader


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Name of the deployment'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='batch size for training and validation'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate of the optimizer'
    )
    parser.add_argument(
        '--train-portion',
        type=float,
        default=0.8,
        help='Training data size portion in train-validation split'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device name to use when training model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs to train model'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=21,
        help='Number of classes in data'
    )

    parser.add_argument(
        '--width',
        type=int,
        default=256,
        help='Image width'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=256,
        help='Image height'
    )

    return parser.parse_args(*argument_array)


def main():
    args = parse_args()

    train_loader, validation_loader = get_train_val_loader(args.data,
                                                           train_portion=args.train_portion,
                                                           batch_size=args.batch_size,
                                                           seed=args.seed,
                                                           width=args.width,
                                                           height=args.height)
    model, optimizer, criterion = get_objects_for_training(args.model,
                                                           learning_rate=args.lr,
                                                           num_classes=args.num_classes)

    tr_eval = TrainEval(model=model,
                        model_name=args.model,
                        optimizer=optimizer,
                        criterion=criterion,
                        train_dataloader=train_loader,
                        val_dataloader=validation_loader,
                        epochs=args.epochs,
                        device=torch.device(args.device))

    tr_eval.train()


if __name__ == '__main__':
    main()
