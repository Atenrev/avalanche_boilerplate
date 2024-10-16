from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR10


def SplitCIFAR10Benchmark(n_experiences, shuffle=True, seed=None, train_transform=None, eval_transform=None, image_size=32):
    standardize = transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                     (0.2023, 0.1994, 0.2010))
    
    if train_transform is None:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                standardize,
            ]
        )   
    
    if eval_transform is None:
        eval_transform = transforms.Compose(
            [
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                standardize,
            ]
        )

    return SplitCIFAR10(
        n_experiences=n_experiences,
        shuffle=shuffle,
        seed=seed,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )