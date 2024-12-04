import torch
import os
import tempfile

from tqdm import tqdm
from torch import nn

from typing import Union

from avalanche.benchmarks import NCScenario, NIScenario
from avalanche.core import SelfSupervisedPlugin
from avalanche.training.supervised import Naive


class FeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
    
class SimpleClassifier(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


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
        self.classifier = None
        self.original_model = None
        self.probe_strategy = None

    def extract_features(self, model, dataloader, device):
        """
        Extract features from the current experience 
        and stores them in a temporary folder.
        """
        print("Extracting features...")
        features = []
        targets = []

        model.eval()
        for mb in tqdm(dataloader):
            x, y = mb[0].to(device), mb[1].to(device)
            with torch.no_grad():
                feats = model(x)
            features.append(feats.view(feats.size(0), -1).detach().cpu())
            targets.append(y.cpu())

        features = torch.cat(features)
        targets = torch.cat(targets)

        return features, targets

    @torch.enable_grad()
    def train_linear_probing(self, dataloader, device):
        """
        Train the linear probing classifier.
        """
        self.classifier.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)

        self.classifier.train()
        for epoch in range(self.train_epochs):
            bar = tqdm(dataloader)
            for features, targets in bar:
                features = features.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = self.classifier(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                acc = (outputs.argmax(dim=1) == targets).float().mean()
                bar.set_description(
                    f"Epoch {epoch + 1}/{self.train_epochs} - Loss: {loss.item():.4f} - Acc: {acc.item():.4f}")

    def before_eval(self, strategy, *args, **kwargs):
        """
        After the training of the current experience, 
        train the linear probing classifier.
        """
        dataloader = torch.utils.data.DataLoader(
            self.benchmark.train_stream[0].dataset,
            batch_size=strategy.train_mb_size,
        )
        
        model_backbone = strategy.model
        
        # If the model has a fc layer, remove it
        if hasattr(model_backbone, "fc"):
            model_backbone = nn.Sequential(*list(model_backbone.children())[:-1])
            
        feats, targets = self.extract_features(model_backbone, dataloader, strategy.device)
        
        # Get the last layer dimension by passing a dummy input
        shape = self.benchmark.train_stream[0].dataset[0][0].shape
        dummy_input = torch.zeros(1, *shape)
        dummy_input = dummy_input.to(strategy.device)
        last_layer_dim = model_backbone(dummy_input).squeeze().shape[0]

        self.classifier = nn.Sequential(
            nn.Linear(last_layer_dim, self.num_classes, bias=False),
        )

        features_dataset = FeaturesDataset(feats, targets)
        features_dataloader = torch.utils.data.DataLoader(
            features_dataset, batch_size=strategy.train_mb_size, shuffle=True)

        self.train_linear_probing(features_dataloader, strategy.device)
        
        self.classifier = SimpleClassifier(model_backbone, self.classifier)
        self.original_model = strategy.model
        strategy.model = self.classifier
        
    def after_eval(self, strategy, *args, **kwargs):
        super().after_eval(strategy, *args, **kwargs)
        strategy.model = self.original_model
        self.classifier = None
        self.original_model = None


__all__ = [
    "LinearProbingPlugin",
]
