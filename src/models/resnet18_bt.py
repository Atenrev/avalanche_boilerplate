import torch.nn as nn
from avalanche.models.resnet32 import resnet32


class Resnet32sBT(nn.Module):
    """
    ResNet18 for Barlow Twins for 32x32 images.

    Reference: https://github.com/facebookresearch/barlowtwins/
    """

    def __init__(self, projector='32-32'):
        super().__init__()
        self.backbone = resnet32()
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [64] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x):
        return self.bn(self.projector(self.backbone(x)))