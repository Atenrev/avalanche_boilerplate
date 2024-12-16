import torch

from copy import deepcopy
from avalanche.core import SupervisedPlugin


class RandomPerturbPlugin(SupervisedPlugin):
    """
    A plugin that perturbs the model after each experience or epoch.
    """

    def __init__(self, perturb_std_ratio: float = 0.1, magnitude_sensitivity: float = 0.1, every: str = "exp"):
        """
        Initializes the plugin.
        
        Args:
            perturb_std_ratio (float): Ratio of the largest weight magnitude to be used as the standard deviation of the perturbation.
            magnitude_sensitivity (float): Controls how weight magnitude affects the probability of perturbation. Higher values make perturbation less likely for large weights.
            every (str): Specifies when to perturb the model. Can be either "exp" or "epoch".
        """
        self.perturb_std_ratio = perturb_std_ratio
        self.magnitude_sensitivity = magnitude_sensitivity
        self.every = every
        self.epoch_counter = 0
        self.experience_counter = 0

    def perturb_model_weights(self, model: torch.nn.Module):
        """
        Perturb the weights of a PyTorch model with the following properties:
        - Perturbation is sampled from a normal distribution centered at 0.
        - The covariance of the normal distribution is based on the largest weight magnitude in the model.
        - Higher magnitude weights have a lower probability of being perturbed.

        Args:
            model (nn.Module): The PyTorch model to be perturbed.
            
        Returns:
            None: The model weights are modified in-place.
        """
        max_weight_magnitude = 0

        # Find the largest weight magnitude in the model
        for param in model.parameters():
            if param.requires_grad:
                max_weight_magnitude = max(
                    max_weight_magnitude, torch.abs(param.data).max().item())

        # Define the standard deviation for perturbation
        perturb_std = self.perturb_std_ratio * max_weight_magnitude

        for param in model.parameters():
            if param.requires_grad:
                weight_magnitude = torch.abs(param.data)

                # Probability of perturbation inversely proportional to weight magnitude
                prob_perturb = torch.exp(-self.magnitude_sensitivity *
                                         weight_magnitude / max_weight_magnitude)

                # Generate a mask for perturbation
                mask = torch.bernoulli(prob_perturb).to(param.device)

                # Generate perturbation
                perturbation = torch.normal(
                    mean=0, std=perturb_std, size=param.data.size(), device=param.device)

                # Apply perturbation based on the mask
                param.data = deepcopy(param.data + mask * perturbation)

        # Should we use strategy.make_optimizer(**kwargs) here?

    def before_training_exp(self, strategy, *args, **kwargs):
        if self.every == "exp" and self.experience_counter > 0:
            self.perturb_model_weights(strategy.model)

    def after_training_exp(self, strategy, *args, **kwargs):
        self.experience_counter += 1

    def before_training_epoch(self, strategy, *args, **kwargs):
        if self.every == "epoch" and self.epoch_counter > 0:
            self.perturb_model_weights(strategy.model)

    def after_training_epoch(self, strategy, *args, **kwargs):
        self.epoch_counter += 1


__all__ = [
    "RandomPerturbPlugin",
]
