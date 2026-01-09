import torch.nn as nn

class ModelV1(nn.Module):
    def __init__(self, x_num, y_num):
        super(ModelV1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, y_num)
        )

    def forward(self, x):
        return self.net(x)
