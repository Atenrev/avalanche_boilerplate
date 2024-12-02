import torch
import torch.nn as nn

from ..common.utils import off_diagonal


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins Loss

    Reference: https://github.com/facebookresearch/barlowtwins/
    """

    def __init__(self, lambda_param=5e-3):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, z):
        z_a, z_b = z
        c = z_a.T @ z_b
        c = c / z_a.size(0)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambda_param * off_diag
        return loss