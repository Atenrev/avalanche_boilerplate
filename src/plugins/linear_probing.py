import torch
from torch import nn

from typing import Union

from avalanche.benchmarks import NCScenario, NIScenario
from avalanche.core import SelfSupervisedPlugin
from avalanche.training.supervised import Naive


class LinearProbingPlugin(SelfSupervisedPlugin):
    """
    A plugin that trains a linear probing classifier after the training of the
    current experience. During evaluation, the linear probing classifier is
    used to compute the accuracy.
    """

    def __init__(self, benchmark: Union[NCScenario, NIScenario], num_classes: int, epochs: int = 1, lr: float = 1e-3):
        super().__init__()
        self.benchmark = benchmark
        self.num_classes = num_classes
        self.train_epochs = epochs
        self.lr = lr
        self.probe_model = None
        self.probe_strategy = None

    def after_training_exp(self, strategy, **kwargs):
        """
        After the training of the current experience, 
        train the linear probing classifier.
        """
        last_layer_dim = None
        last_module = list(strategy.model.modules())[-1]

        if hasattr(last_module, "out_features"):
            last_layer_dim = last_module.out_features
        elif hasattr(last_module, "num_features"):
            last_layer_dim = last_module.num_features
        else:
            raise ValueError("Could not determine the last layer dimension.")

        self.probe_model = nn.Sequential(
            strategy.model,
            nn.Linear(last_layer_dim, self.num_classes)
        )

        # Freeze the feature extractor
        for param in self.probe_model[0].parameters():
            param.requires_grad = False

        self.probe_strategy = Naive(
            model=self.probe_model,
            optimizer=torch.optim.Adam(self.probe_model.parameters(), lr=self.lr),
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=strategy.train_mb_size,
            eval_mb_size=strategy.eval_mb_size,
            train_epochs=self.train_epochs,
            device=strategy.device,
            evaluator=strategy.evaluator,
        )    
        
        self.probe_strategy.train(self.benchmark.train_stream)

        for param in self.probe_model[0].parameters():
            param.requires_grad = True

    @torch.no_grad()
    def eval_representations(self, strategy, exp_list, **kwargs):
        """
        During evaluation, use the linear probing classifier to compute the accuracy.
        """
        if self.probe_strategy is not None:
            return self.probe_strategy.eval(exp_list)
        else:
            return {}
        

__all__ = [
    "LinearProbingPlugin",
]