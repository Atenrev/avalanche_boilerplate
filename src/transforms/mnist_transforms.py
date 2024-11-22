from torchvision import transforms


def MNISTTransform(image_size=32):
    standardize = transforms.Normalize((0.5,), (0.5,))

    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            standardize,
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

    return train_transform, train_transform