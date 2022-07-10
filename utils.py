import json
import os

import torch
from torch import nn

from models import ResNet18, ResNet50


def load_config():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")) as f:
        config = json.load(f)

    return config


def load_model(config):
    model_name = config["model"]

    if model_name == "resnet18":
        return ResNet18(num_classes=1, **config["model_config"])

    if model_name == "resnet50":
        return ResNet50(num_classes=1, **config["model_config"])

    elif model_name == "dummy":
        return nn.Sequential(
            nn.MaxPool2d(256),
            nn.Flatten(),
            nn.Linear(2 * 2 * 3, 1)
        )


def load_optimizer(model, config):
    optimizer_name = config["optimizer"]

    if optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), **config["optimizer_config"])


def load_criterion(config):
    criterion_name = config["criterion"]

    if criterion_name == "MSE":
        return nn.MSELoss(**config["criterion_config"])
