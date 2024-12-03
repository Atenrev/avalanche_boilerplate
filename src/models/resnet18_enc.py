import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision.models import resnet18


class ResNet18(nn.Module):
    """
    Custom ResNet18 with a projector head.
    
    Args:
        projector (tuple): size of the projector head. Default: (4096, 1024)
        mini_version (bool): whether to use the mini version of the model. Default: True
    """

    def __init__(self, projector: tuple = (4096, 1024), mini_version: bool = True, p_norm: float = 2):
        super().__init__()
        self.p_norm = p_norm
        self.backbone = resnet18()
        self.backbone.fc = nn.Identity()

        if mini_version:
            self.backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()

        # projector
        sizes = [512] + list(projector)
        layers = []

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.projector(x)
        return F.normalize(x, p=self.p_norm)
        
