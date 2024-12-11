import random

from pathlib import Path
from typing import Optional, Union, Any, List, TypeVar

from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.classification_dataset import _as_taskaware_supervised_classification_dataset
from avalanche.benchmarks import benchmark_from_datasets, CLScenario

from ..datasets import ConConDataset


TCLDataset = TypeVar("TCLDataset", bound="AvalancheDataset")


def build_concon_scenario(
    list_train_dataset: List[TCLDataset],
    list_test_dataset: List[TCLDataset],
    seed: Optional[int] = None,
    n_experiences: int = 3,
    shuffle_order: bool = False,
):
    if shuffle_order and not n_experiences == 1:
        random.seed(seed)
        random.shuffle(list_train_dataset)
        random.seed(seed)
        random.shuffle(list_test_dataset)

    if n_experiences == 1:
        new_list_train_dataset = []
        new_list_train_dataset.append(list_train_dataset[0])

        for i in range(1, len(list_train_dataset)):
            new_list_train_dataset[0] = new_list_train_dataset[0].concat(
                list_train_dataset[i])

        list_train_dataset = new_list_train_dataset

        new_list_test_dataset = []
        new_list_test_dataset.append(list_test_dataset[0])

        for i in range(1, len(list_test_dataset)):
            new_list_test_dataset[0] = new_list_test_dataset[0].concat(
                list_test_dataset[i])

        list_test_dataset = new_list_test_dataset

    return benchmark_from_datasets(
        train=list_train_dataset,
        test=list_test_dataset
    )


def ConConDisjoint(
    dataset_root: Union[str, Path],
    n_experiences: int,
    *,
    seed: Optional[int] = None,
    shuffle_order: bool = False,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    **kwargs,
) -> CLScenario:
    """
    Creates a ConCon Disjoint benchmark.

    Args:
        dataset_root: The root directory of the dataset.
        n_experiences: The number of experiences to use.
        seed: The seed to use.
        shuffle_order: Whether to shuffle the order of the experiences.
        train_transform: The training transform to use.
        eval_transform: The evaluation transform to use.
        **kwargs: Additional keyword

    Returns:
        The ConCon Disjoint benchmark.
    """
    assert n_experiences == 3 or n_experiences == 1, "n_experiences must be 1 or 3 for ConCon Disjoint"
    list_train_dataset = []
    list_test_dataset = []

    for i in range(3):
        train_dataset = ConConDataset(dataset_root, "disjoint", i, train=True)
        test_dataset = ConConDataset(dataset_root, "disjoint", i, train=False)
        train_dataset = _as_taskaware_supervised_classification_dataset(
            train_dataset,
            transform=train_transform
        )
        test_dataset = _as_taskaware_supervised_classification_dataset(
            test_dataset,
            transform=eval_transform
        )
        list_train_dataset.append(train_dataset)
        list_test_dataset.append(test_dataset)

    return build_concon_scenario(
        list_train_dataset,
        list_test_dataset,
        seed=seed,
        n_experiences=n_experiences,
        shuffle_order=shuffle_order
    )


def ConConStrict(
    dataset_root: Union[str, Path],
    n_experiences: int,
    *,
    seed: Optional[int] = None,
    shuffle_order: bool = False,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    **kwargs,
) -> CLScenario:
    """
    Creates a ConCon Strict benchmark.

    Args:
        dataset_root: The root directory of the dataset.
        n_experiences: The number of experiences to use.
        seed: The seed to use.
        shuffle_order: Whether to shuffle the order of the experiences.
        train_transform: The training transform to use.
        eval_transform: The evaluation transform to use.
        **kwargs: Additional keyword

    Returns:
        The ConCon Strict benchmark.
    """
    assert n_experiences == 3 or n_experiences == 1, "n_experiences must be 1 or 3 for ConCon Disjoint"
    list_train_dataset = []
    list_test_dataset = []

    for i in range(3):
        train_dataset = ConConDataset(dataset_root, "strict", i, train=True)
        test_dataset = ConConDataset(dataset_root, "strict", i, train=False)
        train_dataset = _as_taskaware_supervised_classification_dataset(
            train_dataset,
            transform=train_transform
        )
        test_dataset = _as_taskaware_supervised_classification_dataset(
            test_dataset,
            transform=eval_transform
        )
        list_train_dataset.append(train_dataset)
        list_test_dataset.append(test_dataset)

    return build_concon_scenario(
        list_train_dataset,
        list_test_dataset,
        seed=seed,
        n_experiences=n_experiences,
        shuffle_order=shuffle_order
    )


def ConConUnconfounded(
    dataset_root: Union[str, Path],
    *,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    **kwargs,
) -> CLScenario:
    """
    Creates a ConCon Strict benchmark.
    """
    train_dataset = []
    test_dataset = []

    train_dataset.append(ConConDataset(
        dataset_root, "unconfounded", 0, train=True))
    test_dataset.append(ConConDataset(
        dataset_root, "unconfounded", 0, train=False))

    train_dataset[0] = _as_taskaware_supervised_classification_dataset(
        train_dataset[0],
        transform=train_transform
    )

    test_dataset[0] = _as_taskaware_supervised_classification_dataset(
        test_dataset[0],
        transform=eval_transform
    )

    return benchmark_from_datasets(
        train=train_dataset,
        test=test_dataset
    )


__all__ = [
    "ConConDisjoint",
    "ConConStrict",
    "ConConUnconfounded",
]
