import torch
import torch.nn as nn


def get_multilabel_resnet(resnet_version='resnet18', num_classes=21, weights=None):
    """
     Function which returns modified ResNet network for multilabel classification
    :param resnet_version: Version of ResNet, ex. 'resnet18'
    :param num_classes: Length of all available labels
    :param pretrained: Boolean argument
    :return: Torch model
    """

    model = torch.hub.load('pytorch/vision:v0.14.0',
                           resnet_version,
                           weights=weights)
    model.fc = nn.Linear(in_features=model.fc.in_features,
                         out_features=num_classes,
                         bias=True)

    return nn.Sequential(model, nn.Sigmoid())


def get_multilabel_densenet(densenet_version='densenet121', num_classes=21, weights=None):
    """
     Function which returns modified DenseNet network for multilabel classification
    :param densenet_version: Version of DenseNet, ex. 'densenet121'
    :param num_classes: Length of all available labels
    :param pretrained: Boolean argument
    :return: Torch model
    """

    model = torch.hub.load('pytorch/vision:v0.14.0',
                           densenet_version,
                           weights=weights)

    num_ftrs = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    return nn.Sequential(model, nn.Sigmoid())


def get_multilabel_model(model_name='resnet34', num_classes=21, weights=None):
    """
     Function which returns the corresponding model for multilabel classification
    :param model_name: Version of model, ex. 'densenet121', 'resnet18'
    :param num_classes: Length of all available labels
    :param pretrained: Boolean argument
    :return: Torch model
    """

    if 'resnet' in model_name:
        model = get_multilabel_resnet(model_name, num_classes=num_classes, weights=weights)
    elif 'densenet' in model_name:
        model = get_multilabel_densenet(model_name, num_classes=num_classes, weights=weights)
    else:
        raise 'Please, specify a valid model'

    return model
