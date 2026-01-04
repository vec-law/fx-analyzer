import torch.nn as nn

class ModelV1(nn.Module):
    def __init__(self, features_num, target_num):
        super(ModelV1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(features_num, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, target_num)
        )

    def forward(self, x):
        return self.net(x)
