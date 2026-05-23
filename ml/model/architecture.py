import torch.nn as nn

class ModelV1(nn.Module):
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, y_num)
        )

    def forward(self, x):
        return self.net(x)
