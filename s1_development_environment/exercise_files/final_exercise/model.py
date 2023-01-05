from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "corrupmnist model"
        self.fc = nn.Sequential(
            nn.Linear(784, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            
            nn.LogSoftmax(dim=1)
        )
        self.double()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x