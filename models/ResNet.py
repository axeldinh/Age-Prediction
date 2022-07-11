from torch import nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights


class ResNet18(nn.Module):
    """
    ResNet18 model
    """

    def __init__(self, num_classes, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)


class ResNet50(nn.Module):
    """
    ResNet50 model
    """

    def __init__(self, num_classes, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)
