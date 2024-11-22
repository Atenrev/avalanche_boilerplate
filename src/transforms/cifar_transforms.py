from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR10


def CIFARTransform(image_size=32):
    standardize = transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                     (0.2023, 0.1994, 0.2010))
    
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            standardize,
        ]
    )   
    
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            standardize,
        ]
    )

    return train_transform, eval_transform