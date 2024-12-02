import random
import torch

from PIL import Image
from torchvision import transforms


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.solarize(img)
        else:
            return img

    def solarize(self, img):
        return torch.where(img < 0.5, img, 1 - img)


class EMPSSLTrainingAugmentations(object):
    def __init__(self, image_size: int = 32, num_patch=100):
        self.image_size = image_size
        self.num_patch = num_patch

    def __call__(self, x):
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.image_size, scale=(0.25, 0.25), ratio=(1, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            Solarization(0.1),
        ])
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]

        return augmented_x


def EMPSSLTransform(image_size: int = 32):
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])
    return train_transform, eval_transform


__all__ = [
    "EMPSSLTrainingAugmentations",
    "EMPSSLTransform"
]
