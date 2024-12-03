from torch import nn
from copy import deepcopy
from avalanche.core import SupervisedPlugin


class ShrinkAndPerturbPlugin(SupervisedPlugin):
    """
    A plugin that shrinks and perturbs the model after each experience or epoch.
    
    Reference: https://github.com/JordanAsh/warm_start/
    """

    def __init__(self, shrink: float = 0.4, perturb: float = 0.1, every_epoch: bool = False, every_experience: bool = True):
        self.shrink = shrink
        self.perturb = perturb
        self.every_epoch = every_epoch
        self.every_experience = every_experience

    def shrink_perturb(self, model):
        def reinitialize_weights(model):
            for layer in model.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    # Apply Xavier initialization
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
                
        new_init = deepcopy(model)
        reinitialize_weights(new_init)

        params1 = new_init.parameters()
        params2 = model.parameters()

        for p1, p2 in zip(*[params1, params2]):
            p1.data = deepcopy(self.shrink * p2.data + self.perturb * p1.data)

        return new_init

    def after_training_exp(self, strategy, *args, **kwargs):
        if self.every_experience:
            self.shrink_perturb(strategy.model)
            
    def after_training_epoch(self, strategy, *args, **kwargs):
        if self.every_epoch:
            self.shrink_perturb(strategy.model)


__all__ = [
    "ShrinkAndPerturbPlugin",
]