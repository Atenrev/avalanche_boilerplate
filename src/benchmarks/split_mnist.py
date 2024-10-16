from torchvision import transforms
from avalanche.benchmarks.classic import SplitMNIST


def SplitMNISTBenchmark(n_experiences, shuffle=True, seed=None, train_transform=None, eval_transform=None, image_size=32):
    standardize = transforms.Normalize((0.5,), (0.5,))

    if train_transform is None:
        train_transform = transforms.Compose(
            [
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
                standardize,
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ]
        )

    if eval_transform is None:
        eval_transform = train_transform

    return SplitMNIST(
        n_experiences=n_experiences,
        shuffle=shuffle,
        seed=seed,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )