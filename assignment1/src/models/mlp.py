import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 256, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, X):
        X = self.flatten(X)
        return self.classifier(X)
