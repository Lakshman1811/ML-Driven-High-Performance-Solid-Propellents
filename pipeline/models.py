import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FORMULATION_DIM


class FeaturesFormulation(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(FORMULATION_DIM, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.hidden3 = nn.Linear(256, 256)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return F.relu(self.hidden3(x))


class FeaturesEMS(nn.Module):
    def __init__(self, n_ems: int):
        super().__init__()
        self.hidden1 = nn.Linear(n_ems, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.hidden3 = nn.Linear(256, 256)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return F.relu(self.hidden3(x))


class MLPHead(nn.Module):
    def __init__(self, layer_sizes, dropout_prob=0.2):
        super().__init__()
        layers = []
        prev = 512
        for i, size in enumerate(layer_sizes):
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            if i < 1:
                layers.append(nn.Dropout(dropout_prob))
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class DualBranchMLP(nn.Module):
    """Paper MLP: formulation branch + EC-descriptor branch, then fused head."""

    def __init__(self, n_features: int, layer_sizes):
        super().__init__()
        n_ems = n_features - FORMULATION_DIM
        self.features_formulation = FeaturesFormulation()
        self.features_ems = FeaturesEMS(n_ems)
        self.head = MLPHead(layer_sizes)

    def forward(self, x):
        formulation = x[:, :FORMULATION_DIM]
        ems = x[:, FORMULATION_DIM:]
        fused = torch.cat(
            (self.features_formulation(formulation), self.features_ems(ems)), dim=1
        )
        return self.head(fused)
