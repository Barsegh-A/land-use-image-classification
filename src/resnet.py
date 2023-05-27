import torch
import torch.nn as nn


def get_multilabel_resnet(resnet_version='resnet18', num_classes=10, weights=None):
    """
     Function which returns modified ResNet network for multilabel classification
    :param resnet_version: Version of ResNet, ex. 'resnet18'
    :param num_classes: Length of all available labels
    :param pretrained: Boolean argument
    :return: Torch model
    """

    model = torch.hub.load('pytorch/vision',
                           resnet_version,
                           weights=weights)
    model.fc = nn.Linear(in_features=model.fc.in_features,
                         out_features=num_classes,
                         bias=True)

    return nn.Sequential(model, nn.Sigmoid())
