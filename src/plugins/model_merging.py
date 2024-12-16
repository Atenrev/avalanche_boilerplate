from torch import nn
from copy import deepcopy
from avalanche.core import SupervisedPlugin


class VanillaModelMergingPlugin(SupervisedPlugin):
    """
    A plugin that merges the model after each experience or epoch.
    """

    def __init__(self, merge_coeff: float = 0.5, every: str = "exp"):
        self.merge_coeff = merge_coeff
        self.every = every

        self.old_model = None

    def after_training_exp(self, strategy, *args, **kwargs):
        if self.every == "exp":
            if self.old_model:
                self._merge_models(strategy.model)
            
            self.old_model = deepcopy(strategy.model)

    def after_training_epoch(self, strategy, *args, **kwargs):
        if self.every == "epoch":
            if self.old_model:
                self._merge_models(strategy.model)
            
            self.old_model = deepcopy(strategy.model)

    def _merge_models(self, model):
        for param, old_param in zip(model.parameters(), self.old_model.parameters()):
            param.data = deepcopy(self.merge_coeff * old_param.data + \
                (1 - self.merge_coeff) * param.data)
            
        for m1, m2 in zip(model.modules(), self.old_model.modules()):
            if isinstance(m1, nn.BatchNorm2d):
                m1.running_mean = deepcopy(self.merge_coeff * m2.running_mean + \
                    (1 - self.merge_coeff) * m1.running_mean)
                m1.running_var = deepcopy(self.merge_coeff * m2.running_var + \
                    (1 - self.merge_coeff) * m1.running_var)
                
            
        # Should we use strategy.make_optimizer(**kwargs) here?


__all__ = [
    "VanillaModelMergingPlugin",
]
