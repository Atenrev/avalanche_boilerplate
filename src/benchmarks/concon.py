from pathlib import Path
from typing import Optional, Union, Any

from avalanche.benchmarks.utils.classification_dataset import _as_taskaware_supervised_classification_dataset
from avalanche.benchmarks import NIScenario, ni_benchmark

from ..datasets import ConConDataset


def ConConDisjoint(
    dataset_root: Union[str, Path],
    n_experiences: int,
    *,
    seed: Optional[int] = None,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    **kwargs,
) -> NIScenario:
    """
    Creates a ConCon Disjoint benchmark.
    """

    list_train_dataset = []
    list_test_dataset = []

    for i in range(3):
        train_dataset = ConConDataset(dataset_root, "disjoint", i, train=True)
        test_dataset = ConConDataset(dataset_root, "disjoint", i, train=False)
        train_dataset = _as_taskaware_supervised_classification_dataset(train_dataset)
        test_dataset = _as_taskaware_supervised_classification_dataset(test_dataset)
        list_train_dataset.append(train_dataset)
        list_test_dataset.append(test_dataset)

    return ni_benchmark(
        list_train_dataset,
        list_test_dataset,
        n_experiences=n_experiences,
        seed=seed,
        shuffle=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


def ConConStrict(
    dataset_root: Union[str, Path],
    n_experiences: int,
    *,
    seed: Optional[int] = None,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    **kwargs,
) -> NIScenario:
    """
    Creates a ConCon Strict benchmark.
    """

    list_train_dataset = []
    list_test_dataset = []

    for i in range(3):
        train_dataset = ConConDataset(dataset_root, "strict", i, train=True)
        test_dataset = ConConDataset(dataset_root, "strict", i, train=False)
        list_train_dataset.append(train_dataset)
        list_test_dataset.append(test_dataset)

    return ni_benchmark(
        list_train_dataset,
        list_test_dataset,
        n_experiences=n_experiences,
        seed=seed,
        shuffle=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


def ConConUnconfounded(
    dataset_root: Union[str, Path],
    *,
    seed: Optional[int] = None,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    **kwargs,
) -> NIScenario:
    """
    Creates a ConCon Strict benchmark.
    """
    train_dataset = []
    test_dataset = []
    
    train_dataset.append(ConConDataset(dataset_root, "unconf", 0, train=True))
    test_dataset.append(ConConDataset(dataset_root, "unconf", 0, train=False))

    return ni_benchmark(
        train_dataset,
        test_dataset,
        n_experiences=1,
        seed=seed,
        shuffle=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


__all__ = [
    "ConConDisjoint",
    "ConConStrict",
    "ConConUnconfounded",
]
