import torch.nn as nn
import torch

class ModelV1(nn.Module):
    def __init__(self, x_num, y_num):
        super().__init__()
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

class ModelV12(nn.Module):
    """Model z funkcją ELU (Exponential Linear Unit)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 40),
            nn.ELU(),
            nn.Linear(40, 20),
            nn.ELU(),
            nn.Linear(20, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV13(nn.Module):
    """Model typu 'Inverted Pyramid' (rozszerzający się)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV14(nn.Module):
    """Model z wagami inicjalizowanymi metodą He (dla ReLU)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, y_num)
        self.relu = nn.ReLU()
        
        # Inicjalizacja wag
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ModelV15(nn.Module):
    """Model z funkcją Softplus (gładkie ReLU)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 32),
            nn.Softplus(),
            nn.Linear(32, 16),
            nn.Softplus(),
            nn.Linear(16, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV16(nn.Module):
    """Głęboki model z agresywnym Dropoutem na wejściu."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.1),  # Szum na wejściu
            nn.Linear(x_num, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV17(nn.Module):
    """Model z połączeniem Residual i BatchNorm."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.pre = nn.Linear(x_num, 32)
        self.block = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32)
        )
        self.post = nn.Linear(32, y_num)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.pre(x))
        identity = x
        x = self.block(x)
        x = self.relu(x + identity)
        return self.post(x)


class ModelV18(nn.Module):
    """Model z Sigmoidą (klasyczna architektura)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 48),
            nn.Sigmoid(),
            nn.Linear(48, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV19(nn.Module):
    """Model o dużej kompresji (Encoder style)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # Bardzo wąskie gardło
            nn.ReLU(),
            nn.Linear(4, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV20(nn.Module):
    """Model z wieloma warstwami o stałej szerokości."""
    def __init__(self, x_num, y_num):
        super().__init__()
        layers = []
        for _ in range(5):
            layers.append(nn.Linear(64 if _ > 0 else x_num, 64))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(64, y_num))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class ModelV21(nn.Module):
    """1D CNN - wyłapywanie lokalnych wzorców (trendów)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        # Zakładamy input (batch, 1, x_num)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * x_num, 32),
            nn.ReLU(),
            nn.Linear(32, y_num)
        )

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.conv(x)


class ModelV22(nn.Module):
    """Model z warstwą Gated Linear Unit (GLU)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 64)
        self.glu = nn.GLU()  # Dzieli input na pół: połowa to sygnał, połowa to brama
        self.fc2 = nn.Linear(32, y_num)

    def forward(self, x):
        x = self.fc1(x)
        x = self.glu(x)
        return self.fc2(x)


class ModelV23(nn.Module):
    """Model z aktywacją PReLU (uczalny parametr nachylenia)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 64)
        self.prelu = nn.PReLU()
        self.fc2 = nn.Linear(64, y_num)

    def forward(self, x):
        return self.fc2(self.prelu(self.fc1(x)))


class ModelV24(nn.Module):
    """Model z rzadką warstwą (L1-ready) - duża liczba neuronów, mała gęstość."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 256),
            nn.ReLU(),
            nn.Linear(256, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV25(nn.Module):
    """Model typu 'Alpha' - połączenie ścieżki liniowej i nieliniowej."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.linear_path = nn.Linear(x_num, y_num)
        self.nonlinear_path = nn.Sequential(
            nn.Linear(x_num, 32),
            nn.ReLU(),
            nn.Linear(32, y_num)
        )

    def forward(self, x):
        return self.linear_path(x) + self.nonlinear_path(x)


class ModelV26(nn.Module):
    """Model z aktywacją CELU - płynniejsza niż ReLU, lepiej radzi sobie z regresją."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.CELU(alpha=1.0),
            nn.Linear(64, 32),
            nn.CELU(alpha=1.0),
            nn.Linear(32, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV27(nn.Module):
    """Model z warstwami grupowanymi (GroupNorm)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 64)
        self.gn = nn.GroupNorm(4, 64) # 4 grupy po 16 kanałów
        self.fc2 = nn.Linear(64, y_num)

    def forward(self, x):
        x = torch.relu(self.gn(self.fc1(x)))
        return self.fc2(x)


class ModelV28(nn.Module):
    """Model z funkcją Mish (często lepsza niż ReLU w głębokich sieciach)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.Mish(),
            nn.Linear(64, 32),
            nn.Mish(),
            nn.Linear(32, y_num)
        )

    def forward(self, x):
        return self.net(x)


class ModelV29(nn.Module):
    """Model 'Siamese-like' - przetwarzanie wejścia dwiema różnymi głowicami."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.head1 = nn.Linear(x_num, 32)
        self.head2 = nn.Linear(x_num, 32)
        self.tail = nn.Linear(64, y_num)

    def forward(self, x):
        h1 = torch.relu(self.head1(x))
        h2 = torch.sigmoid(self.head2(x))
        return self.tail(torch.cat([h1, h2], dim=1))


class ModelV30(nn.Module):
    """Model z kaskadowym łączeniem cech."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 16)
        self.fc2 = nn.Linear(16 + x_num, 16)
        self.fc3 = nn.Linear(16 + 16, y_num)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(torch.cat([x1, x], dim=1)))
        return self.fc3(torch.cat([x1, x2], dim=1))