import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import math
import pdb

class DCNN1D(nn.Module):
    def __init__(self, num_classes: int = 18, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(50, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2,2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2, 2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x.squeeze())
        feature = torch.flatten(x, 1)
        x = self.classifier(feature)
        return feature, x

def dcnn1d(**kwargs) -> DCNN1D:
    model = DCNN1D(**kwargs)
    return model
