from pathlib import Path
from typing import Union

from PIL import Image


class ConConDataset:
    """
    ConConDataset represents a continual learning task with two classes: positive and negative.
    All data instances are images based on the CLEVR framework. A ground truth rule can be used
    to determine the binary class affiliation of any image. The dataset is designed to be used
    in a continual learning setting with three sequential tasks, each confounded by a task-specific
    confounder. The challenge arises from the fact that task-specific confounders change across tasks.
    There are two dataset variants:
    
    - Disjoint: Task-specific confounders never appear in other tasks.
    - Strict: Task-specific confounders may appear in other tasks as random features in both positive
      and negative samples.
    - Unconf: No task-specific confounders.
    
    Reference: 
    Busch, Florian Peter, et al. "Where is the Truth? The Risk of Getting Confounded in a Continual World." 
    arXiv preprint arXiv:2402.06434 (2024).
    
    Args:
        root (Union[str, Path]): The root directory of the dataset.
        variant (str): The variant of the dataset, must be one of 'strict', 'disjoint', 'unconf'.
        scenario (int): The scenario number, must be between 0 and 2.
        train (bool): If True, use the training set, otherwise use the test set.
    """
    
    def __init__(self, root: Union[str, Path], variant: str, scenario: int, train: bool = True):
        assert variant in ["strict", "disjoint", "unconfounded"], "Invalid variant, must be one of 'strict', 'disjoint', 'unconf'"
        assert scenario in range(0, 3), "Invalid scenario, must be between 0 and 2"
        assert variant != "unconfounded" or scenario == 0, "Unconfounded scenario only has one variant"
        
        self.root = Path(root) / variant
        
        if train:
            self.root = self.root / "train"
        else:
            self.root = self.root / "test"
            
        self.root = self.root / "images" / f"t{scenario}"
        
        self.image_paths = []
        self.targets = []
        
        for class_id, class_dir in enumerate(self.root.iterdir()):
            for image_path in class_dir.iterdir():
                self.image_paths.append(image_path)
                self.targets.append(class_id)
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        target = self.targets[idx]
        return image, target


__all__ = ["ConConDataset"]