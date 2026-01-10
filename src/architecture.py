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
    
class ModelV2(nn.Module):
    """Głębsza sieć z większą liczbą warstw."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV3(nn.Module):
    """Szeroka sieć (jedna duża warstwa ukryta)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 128),
            nn.ReLU(),
            nn.Linear(128, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV4(nn.Module):
    """Model z Batch Normalization."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV5(nn.Module):
    """Model z Dropoutem (regularyzacja)."""
    def __init__(self, x_num, y_num, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV6(nn.Module):
    """Model z funkcją LeakyReLU."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV7(nn.Module):
    """Model z połączeniem rezydualnym (Skip Connection)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, y_num)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        identity = x
        x = self.relu(self.fc2(x)) + identity
        return self.fc3(x)


class ModelV8(nn.Module):
    """Model typu Bottleneck."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV9(nn.Module):
    """Model z funkcją SELU."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 48),
            nn.SELU(),
            nn.Linear(48, 24),
            nn.SELU(),
            nn.Linear(24, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV10(nn.Module):
    """Model o strukturze piramidy."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV11(nn.Module):
    """Model z funkcją Tanh."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, y_num)
        )

    def forward(self, x):
        return self.net(x)
