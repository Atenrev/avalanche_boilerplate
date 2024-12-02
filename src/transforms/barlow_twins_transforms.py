import random
import torch
import torch.nn.functional as F

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


class BTTrainingAugmentations:
    def __init__(self, image_size: int = 224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            Solarization(p=0.0),
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            Solarization(p=0.2),
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


def BarlowTwinsTransform(image_size: int = 224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])
    return train_transform, eval_transform


__all__ = [
    "BarlowTwinsTransform",
    "BTTrainingAugmentations",
]
