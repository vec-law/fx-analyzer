import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class ModelV31(nn.Module):
    """Model z prostym mechanizmem Attention na wejściach."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(x_num, x_num),
            nn.Softmax(dim=1)
        )
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.ReLU(),
            nn.Linear(64, y_num)
        )

    def forward(self, x):
        # Mnożymy wejścia przez wyuczone wagi istotności
        attn = self.attention_weights(x)
        x = x * attn
        return self.net(x)
    
class ModelV32(nn.Module):
    """Model z funkcją Hardswish (używana w MobileNetV3)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.Hardswish(),
            nn.Linear(64, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV33(nn.Module):
    """Model z Layer Normalization (lepsza dla małych batchy niż BatchNorm)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV34(nn.Module):
    """Model typu 'Dense Block' - konkatenacja wejścia z warstwą ukrytą."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 32)
        self.fc2 = nn.Linear(32 + x_num, y_num)
    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc2(torch.cat([h1, x], dim=1))

class ModelV35(nn.Module):
    """Model z adaptacyjnym Dropoutem (AlphaDropout - dla SELU)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.SELU(),
            nn.AlphaDropout(p=0.1),
            nn.Linear(64, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV36(nn.Module):
    """Model o architekturze 'Hourglass' (64 -> 16 -> 64)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV37(nn.Module):
    """Model z funkcją LogSigmoid."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 48),
            nn.LogSigmoid(),
            nn.Linear(48, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV38(nn.Module):
    """Model z bramkowaniem Rezero (inicjalizacja warstwy zerem)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, x_num)
        self.res_weight = nn.Parameter(torch.zeros(1))
        self.out = nn.Linear(x_num, y_num)
    def forward(self, x):
        x = x + self.res_weight * torch.tanh(self.fc(x))
        return self.out(x)

class ModelV39(nn.Module):
    """Model z RReLU (Randomized LeakyReLU)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.RReLU(),
            nn.Linear(64, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV40(nn.Module):
    """Model z dwiema równoległymi ścieżkami o różnej szerokości."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.path1 = nn.Linear(x_num, 16)
        self.path2 = nn.Linear(x_num, 64)
        self.out = nn.Linear(80, y_num)
    def forward(self, x):
        x1, x2 = torch.relu(self.path1(x)), torch.relu(self.path2(x))
        return self.out(torch.cat([x1, x2], dim=1))

class ModelV41(nn.Module):
    """Model z mechanizmem Squeeze-and-Excitation (uproszczony)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.se = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 64), nn.Sigmoid())
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = torch.relu(self.fc(x))
        h = h * self.se(h)
        return self.out(h)

class ModelV42(nn.Module):
    """Model 'Deep Bottleneck' (128 -> 8 -> 8 -> 128)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 128), nn.ReLU(),
            nn.Linear(128, 8), nn.ReLU(),
            nn.Linear(8, 8), nn.ReLU(),
            nn.Linear(8, 128), nn.ReLU(),
            nn.Linear(128, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV43(nn.Module):
    """Model z funkcją Softsign."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.Softsign(),
            nn.Linear(64, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV44(nn.Module):
    """Model z kaskadowym Dropoutem (zwiększający się p)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 64)
        self.dr1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 64)
        self.dr2 = nn.Dropout(0.4)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        x = self.dr1(torch.relu(self.fc1(x)))
        x = self.dr2(torch.relu(self.fc2(x)))
        return self.out(x)

class ModelV45(nn.Module):
    """Model z wagami inicjalizowanymi metodą Xaviera (dla Tanh)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        nn.init.xavier_uniform_(self.fc.weight)
        self.out = nn.Linear(64, y_num)
    def forward(self, x): return self.out(torch.tanh(self.fc(x)))

class ModelV46(nn.Module):
    """Model z 'Input Scaling' - uczalny parametr skali dla każdego wejścia."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(x_num))
        self.net = nn.Linear(x_num, y_num)
    def forward(self, x): return self.net(x * self.scale)

class ModelV47(nn.Module):
    """Model z Gated Residual Connection."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, x_num)
        self.gate = nn.Linear(x_num, x_num)
        # Warstwa wyjściowa musi być tu, żeby wagi były trwałe i mogle się uczyć
        self.output_layer = nn.Linear(x_num, y_num)

    def forward(self, x):
        h = torch.relu(self.fc(x))
        g = torch.sigmoid(self.gate(x))
        # Połączenie rezydualne z bramkowaniem, potem projekcja na wyjście
        combined = (h * g) + x
        return self.output_layer(combined)

class ModelV48(nn.Module):
    """Model z funkcją GELU (Gaussian Error Linear Unit - jak w BERT)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64),
            nn.GELU(),
            nn.Linear(64, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV49(nn.Module):
    """Model typu 'Expanding Bottleneck'."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 8), nn.ReLU(),
            nn.Linear(8, 256), nn.ReLU(),
            nn.Linear(256, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV50(nn.Module):
    """Model z uśrednianiem dwóch aktywacji (ReLU + Tanh)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = self.fc(x)
        h = (torch.relu(h) + torch.tanh(h)) / 2
        return self.out(h)

class ModelV51(nn.Module):
    """Linear Bottleneck - rzadka projekcja."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 128),
            nn.Linear(128, 16, bias=False),
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV52(nn.Module):
    """Pyramid ELU - stabilne zwężanie cech."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, 16), nn.ELU(),
            nn.Linear(16, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV53(nn.Module):
    """Poprawiony Residual - gwarantowana zgodność wymiarów."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.proj = nn.Linear(x_num, 64)
        self.block = nn.Linear(64, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = torch.relu(self.proj(x))
        return self.out(h + torch.relu(self.block(h)))

class ModelV54(nn.Module):
    """Gated Linear Unit - stabilne bramkowanie."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.gate = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        return self.out(torch.tanh(self.fc(x)) * torch.sigmoid(self.gate(x)))

class ModelV55(nn.Module):
    """Hourglass - LayerNorm w wąskim gardle."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV56(nn.Module):
    """Deep Tanh - klasyczna głęboka sieć."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV57(nn.Module):
    """Skip-Input - bezpośrednie łączenie wejścia z cechami."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.out = nn.Linear(64 + x_num, y_num)
    def forward(self, x):
        h = torch.relu(self.fc(x))
        return self.out(torch.cat([h, x], dim=1))

class ModelV58(nn.Module):
    """Softplus Bottleneck - gładka nieliniowość."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 128), nn.Softplus(),
            nn.Linear(128, 32), nn.Softplus(),
            nn.Linear(32, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV59(nn.Module):
    """Deep ReLU Narrow - 4 warstwy po 32 jednostki."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV60(nn.Module):
    """Bilinear Interaction - bezpieczny iloczyn dwóch ścieżek."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 32)
        self.fc2 = nn.Linear(x_num, 32)
        self.out = nn.Linear(32, y_num)
    def forward(self, x):
        return self.out(torch.sigmoid(self.fc1(x)) * torch.relu(self.fc2(x)))

class ModelV61(nn.Module):
    """Expanding MLP - 128 -> 256."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV62(nn.Module):
    """Chebyshev Polynomials - aproksymacja bazą wielomianów ortogonalnych."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num * 3, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        t0 = torch.ones_like(x)
        t1 = x
        t2 = 2 * x**2 - 1
        x_poly = torch.cat([t0, t1, t2], dim=1)
        return self.out(torch.relu(self.fc(x_poly)))

class ModelV63(nn.Module):
    """Residual Block with projection - bezpieczny dla każdego x_num."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.proj = nn.Linear(x_num, 64)
        self.res = nn.Linear(64, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = torch.relu(self.proj(x))
        h = h + torch.relu(self.res(h))
        return self.out(h)

class ModelV64(nn.Module):
    """LeakyReLU BottleNeck - ściskanie do 8 cech."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 8), nn.LeakyReLU(0.1),
            nn.Linear(8, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV65(nn.Module):
    """Parallel Concatenation - ReLU i Tanh razem."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.p1 = nn.Linear(x_num, 32)
        self.p2 = nn.Linear(x_num, 32)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h1 = torch.relu(self.p1(x))
        h2 = torch.tanh(self.p2(x))
        return self.out(torch.cat([h1, h2], dim=1))

class ModelV66(nn.Module):
    """Sigmoid-Gated MLP - bramkowanie nieliniowe."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.gate = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        return self.out(torch.relu(self.fc(x)) * torch.sigmoid(self.gate(x)))

class ModelV67(nn.Module):
    """Adaptive Gated Multiplier - szybki, stabilny i świetny do oscylatorów."""
    def __init__(self, x_num, y_num):
        super().__init__()
        # Bramka decydująca o sile sygnału
        self.gate = nn.Sequential(
            nn.Linear(x_num, x_num),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(x_num, x_num)
        self.out = nn.Linear(x_num, y_num)
        self.norm = nn.LayerNorm(x_num)

    def forward(self, x):
        g = self.gate(x)
        # Mnożenie sygnału przez bramkę - mechanizm uwagi (attention)
        refined = self.transform(x) * g 
        return self.out(self.norm(refined))

class ModelV68(nn.Module):
    """Hardtanh Bounded - ograniczenie aktywacji do [-1, 1]."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 64), nn.Hardtanh(-1, 1),
            nn.Linear(64, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV69(nn.Module):
    """Fuzzy Logic Logic (T-Norm) - warstwa bramek logicznych Łukaszewicza."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 32)
        self.fc2 = nn.Linear(x_num, 32)
        self.out = nn.Linear(32, y_num)
    def forward(self, x):
        a = torch.sigmoid(self.fc1(x))
        b = torch.sigmoid(self.fc2(x))
        h = torch.clamp(a + b - 1, min=0)
        return self.out(h)

class ModelV70(nn.Module):
    """Wide Single Layer - 512 jednostek."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 512), nn.ReLU(),
            nn.Linear(512, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV71(nn.Module):
    """Asymmetric Pyramid - gwałtowne zwężenie, wolne rozszerzanie."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 16), nn.ReLU(),
            nn.Linear(16, 128), nn.ReLU(),
            nn.Linear(128, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV72(nn.Module):
    def __init__(self, x_num, y_num):
        super().__init__()
        
        # Architektura oparta wyłącznie na proporcjach x_num i y_num
        self.w1 = nn.Linear(x_num, x_num * 2)
        self.b1 = nn.BatchNorm1d(x_num * 2)
        
        self.w2 = nn.Linear(x_num * 2, y_num * 2)
        self.b2 = nn.BatchNorm1d(y_num * 2)
        
        self.w_out = nn.Linear(y_num * 2, y_num)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Bezpośrednie operacje na warstwach
        x = self.w1(x)
        x = self.b1(x)
        x = self.relu(x)
        
        x = self.w2(x)
        x = self.b2(x)
        x = self.relu(x)
        
        return self.w_out(x)

class ModelV73(nn.Module):
    """Feature Dropout MLP - wysoka regularyzacja na szerokiej warstwie."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 256)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(256, y_num)
    def forward(self, x):
        return self.out(self.drop(torch.relu(self.fc(x))))

class ModelV74(nn.Module):
    """Mirror Symmetric MLP - wymuszenie symetrii względem osi 0."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = self.fc(x)
        h_sym = (torch.relu(h) + torch.relu(-h)) / 2
        return self.out(h_sym)

class ModelV75(nn.Module):
    """ISTA Block - jeden krok Sparse Codingu (Soft-thresholding)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn(x_num, x_num))
        self.lambd = nn.Parameter(torch.tensor(0.01))
        self.out = nn.Linear(x_num, y_num)
    def forward(self, x):
        res = torch.matmul(x, self.w)
        h = torch.sign(res) * torch.relu(torch.abs(res) - self.lambd)
        return self.out(h)

class ModelV76(nn.Module):
    """Direct Linear Projection - bez warstw ukrytych (Baseline)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, y_num)
    def forward(self, x): return self.fc(x)

class ModelV77(nn.Module):
    """Log-Output Scaler - logarytmowanie cech przed wyjściem."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 64)
        self.fc2 = nn.Linear(64, y_num)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(torch.log1p(h))

class ModelV78(nn.Module):
    """RReLU BottleNeck - Randomized LeakyReLU z kompresją."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 128), nn.RReLU(),
            nn.Linear(128, 16), nn.RReLU(),
            nn.Linear(16, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV79(nn.Module):
    """Sorted-Input MLP - wymusza niezmienniczość względem permutacji wejść."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, y_num)

    def forward(self, x):
        # Sortowanie cech w każdym wierszu batcha (dim=1)
        x_sorted, _ = torch.sort(x, dim=1)
        h = torch.relu(self.fc1(x_sorted))
        h = torch.relu(self.fc2(h))
        return self.out(h)

class ModelV80(nn.Module):
    """Double Bottleneck - 32 -> 8 -> 32."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV81(nn.Module):
    """Wavelet Derivative - analiza różnicowa (pochodna Haar)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num - 1, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        diff = x[:, 1:] - x[:, :-1]
        return self.out(torch.relu(self.fc(diff)))

class ModelV82(nn.Module):
    """Bypass Linear - suma ścieżki surowej i przetworzonej."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.bypass = nn.Linear(x_num, y_num)
        self.proc = nn.Sequential(nn.Linear(x_num, 64), nn.ReLU(), nn.Linear(64, y_num))
    def forward(self, x): return self.bypass(x) + self.proc(x)

class ModelV83(nn.Module):
    """Tanh-GELU Hybrid - alternatywne nieliniowości."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 64)
        self.fc2 = nn.Linear(64, y_num)
    def forward(self, x):
        h = torch.tanh(self.fc1(x)) * torch.sigmoid(self.fc1(x))
        return self.fc2(h)

class ModelV84(nn.Module):
    """Gradual Expansion MLP - 16 -> 32 -> 64 -> 128."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV85(nn.Module):
    """Stochastic Depth Bypass (Training Only)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.proj = nn.Linear(x_num, 64)
        self.res = nn.Linear(64, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = torch.relu(self.proj(x))
        if self.training:
            h = h + torch.relu(self.res(h))
        return self.out(h)

class ModelV86(nn.Module):
    """Hardswish Narrow MLP."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 32), nn.Hardswish(),
            nn.Linear(32, 32), nn.Hardswish(),
            nn.Linear(32, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV87(nn.Module):
    """Abs-Activation Transform - wymusza dodatnie cechy ukryte."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x): return self.out(torch.abs(self.fc(x)))

class ModelV88(nn.Module):
    """Attention-Based Feature Gating - dynamiczne skalowanie wejść."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.feature_attention = nn.Sequential(
            nn.Linear(x_num, x_num),
            nn.Sigmoid()
        )
        self.fc1 = nn.Linear(x_num, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, y_num)

    def forward(self, x):
        # Generowanie wag istotności dla każdej cechy wejściowej
        weights = self.feature_attention(x)
        x_gated = x * weights
        
        h = torch.relu(self.fc1(x_gated))
        h = torch.relu(self.fc2(h))
        return self.out(h)

class ModelV89(nn.Module):
    """Adaptive Bit-Shift - uczalne przesunięcie potęgowe."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.shifts = nn.Parameter(torch.zeros(x_num))
        self.fc = nn.Linear(x_num, y_num)
    def forward(self, x):
        x_shifted = x * torch.pow(2.0, torch.round(self.shifts))
        return self.fc(x_shifted)

class ModelV90(nn.Module):
    """Deep Squeeze-and-Excitation MLP."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 64)
        self.se = nn.Sequential(nn.Linear(64, 8), nn.ReLU(), nn.Linear(8, 64), nn.Sigmoid())
        self.fc2 = nn.Linear(64, y_num)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h * self.se(h))

class ModelV91(nn.Module):
    """Square-Activation MLP - modelowanie kwadratowe cech."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = self.fc(x)
        return self.out(h * h)

class ModelV92(nn.Module):
    """CELU BottleNeck with Dropout."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 128), nn.CELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32), nn.CELU(),
            nn.Linear(32, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV93(nn.Module):
    """Triple-Layer Tanh with small width (16)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV94(nn.Module):
    """GELU Wide Bottleneck (256 -> 16 -> 256)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 256), nn.GELU(),
            nn.Linear(256, 16), nn.GELU(),
            nn.Linear(16, 256), nn.GELU(),
            nn.Linear(256, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV95(nn.Module):
    """Learnable Input Scale with Tanh."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(x_num))
        self.fc = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        return self.out(torch.tanh(self.fc(x * self.scale)))

class ModelV96(nn.Module):
    """Sigmoid Gated Residual (Alternative)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.gate = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = torch.relu(self.fc(x))
        g = torch.sigmoid(self.gate(x))
        return self.out(h * g)

class ModelV97(nn.Module):
    """Deep ReLU with 5 narrow layers (24 nodes)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        layers = []
        for _ in range(5):
            layers.append(nn.Linear(x_num if _ == 0 else 24, 24))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(24, y_num))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ModelV98(nn.Module):
    """Sine-Activation Block (Periodic bias)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x): return self.out(torch.sin(self.fc(x)))

class ModelV99(nn.Module):
    """Log-Sigmoid Narrow MLP."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_num, 32), nn.LogSigmoid(),
            nn.Linear(32, y_num)
        )
    def forward(self, x): return self.net(x)

class ModelV100(nn.Module):
    """Deep Linear Bottleneck with Residual Rezero."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 128)
        self.bottleneck = nn.Linear(128, 4, bias=False)
        self.expand = nn.Linear(4, 128)
        self.res_weight = nn.Parameter(torch.zeros(1))
        self.out = nn.Linear(128, y_num)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        # Ścieżka rezydualna z mechanizmem ReZero
        bottleneck = self.expand(self.bottleneck(h1))
        h2 = h1 + self.res_weight * torch.relu(bottleneck)
        return self.out(h2)
    
class ModelV101(nn.Module):
    """Fourier Feature Mapping - rzutowanie na bazę sinus/cosinus."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.b = nn.Parameter(torch.randn(x_num, 32) * 10.0, requires_grad=False)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        proj = torch.matmul(x, self.b)
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        return self.out(ff)

class ModelV102(nn.Module):
    """Polynomial Interaction - iloczyn cech (x^2) przed warstwą liniową."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num * 2, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        x_poly = torch.cat([x, x * x], dim=1)
        return self.out(torch.relu(self.fc(x_poly)))

class ModelV103(nn.Module):
    """Simple Mixture of Experts (MoE) - 3 ekspertów + router."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(x_num, 32) for _ in range(3)])
        self.gate = nn.Sequential(nn.Linear(x_num, 3), nn.Softmax(dim=1))
        self.out = nn.Linear(32, y_num)
    def forward(self, x):
        weights = self.gate(x)
        expert_outputs = torch.stack([torch.relu(e(x)) for e in self.experts], dim=1)
        combined = torch.sum(weights.unsqueeze(2) * expert_outputs, dim=1)
        return self.out(combined)

class ModelV104(nn.Module):
    """Statistical Moment Augmentation - dokleja średnią i wariancję wektora x."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num + 2, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x_aug = torch.cat([x, mean, std], dim=1)
        return self.out(torch.relu(self.fc(x_aug)))

class ModelV105(nn.Module):
    """Maxout Network - wybiera max z dwóch liniowych projekcji."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 64)
        self.fc2 = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = torch.max(self.fc1(x), self.fc2(x))
        return self.out(h)

class ModelV106(nn.Module):
    """Spectral Gate - stabilna wersja po pocięciu danych."""
    def __init__(self, x_num, y_num):
        super().__init__()
        # Zakomentowany stary kod Gabora (zachowujemy standardy)
        # self.freq = nn.Parameter(torch.randn(1, 8, 1))
        # self.sigma = nn.Parameter(torch.ones(1, 8, 1))
        # self.out = nn.Linear(8 * x_num, y_num)
        
        self.filter_bank = nn.Linear(x_num, x_num * 4)
        self.gate = nn.Sigmoid()
        self.out = nn.Linear(x_num * 4, y_num)
        self.norm = nn.LayerNorm(x_num * 4)

    def forward(self, x):
        features = self.filter_bank(x)
        activated = features * self.gate(features)
        return self.out(self.norm(activated))
    
class ModelV107(nn.Module):
    """Soft-Thresholding MLP - odszumianie aktywacji."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.threshold = nn.Parameter(torch.tensor(0.1))
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = self.fc(x)
        h = torch.sign(h) * torch.relu(torch.abs(h) - self.threshold)
        return self.out(h)

# ModelV108 = ModelV108_Original # Zakomentowane: podobne do V100 (ReZero)

class ModelV108(nn.Module):
    """Signed Log Transform MLP - nieliniowe skalowanie wejścia przed siecią."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)

    def forward(self, x):
        # Logarytm z zachowaniem znaku: sign(x) * log(|x| + 1)
        x_log = torch.sign(x) * torch.log1p(torch.abs(x))
        return self.out(torch.relu(self.fc(x_log)))

# ModelV109 = ModelV109_Original # Zakomentowane: zbyt podobne do V98 i V101

class ModelV109(nn.Module):
    """Activation Mixture - dynamiczny wybór między Tanh a ReLU."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.gate = nn.Linear(x_num, 1)
        self.out = nn.Linear(64, y_num)

    def forward(self, x):
        h = self.fc(x)
        g = torch.sigmoid(self.gate(x))
        # Mieszanie dwóch różnych światów aktywacji
        h_mixed = g * torch.tanh(h) + (1 - g) * torch.relu(h)
        return self.out(h_mixed)

class ModelV110(nn.Module):
    """Recurrent-style MLP - wielokrotne przepuszczenie przez tę samą warstwę."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.proj = nn.Linear(x_num, 64)
        self.recurrent = nn.Linear(64, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = torch.relu(self.proj(x))
        for _ in range(3):
            h = torch.relu(self.recurrent(h) + h)
        return self.out(h)

class ModelV111(nn.Module):
    """Bilinear Pooling (Simplified) - interakcje między cechami ukrytymi."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 16)
        self.out = nn.Linear(16 * 16, y_num)
    def forward(self, x):
        h = torch.relu(self.fc(x))
        # Outer product b x 16 x 1 * b x 1 x 16 -> b x 16 x 16
        bilinear = torch.bmm(h.unsqueeze(2), h.unsqueeze(1))
        return self.out(bilinear.view(x.size(0), -1))

class ModelV112(nn.Module):
    """LayerNorm + Sigmoid Gating."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 64)
        self.ln = nn.LayerNorm(64)
        self.gate = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = self.ln(self.fc1(x))
        g = torch.sigmoid(self.gate(x))
        return self.out(h * g)

class ModelV113(nn.Module):
    """Log-Log Linear - modelowanie relacji potęgowych (Power Law)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        x_log = torch.log1p(torch.abs(x))
        h = torch.relu(self.fc(x_log))
        return self.out(torch.log1p(torch.abs(h)))
    
class ModelV114(nn.Module):
    """Grouped Linear Layers - niezależne przetwarzanie podzbiorów cech."""
    def __init__(self, x_num, y_num):
        super().__init__()
        half = x_num // 2
        self.fc1 = nn.Linear(half, 32)
        self.fc2 = nn.Linear(x_num - half, 32)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        half = x.size(1) // 2
        h1 = torch.relu(self.fc1(x[:, :half]))
        h2 = torch.relu(self.fc2(x[:, half:]))
        return self.out(torch.cat([h1, h2], dim=1))

class ModelV115(nn.Module):
    """Piecewise Linear Activation (Manual) - ręczny model progowy."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h = self.fc(x)
        return self.out(torch.where(h > 0, h, 0.1 * h))

class ModelV116(nn.Module):
    """Min-Max Path Concatenation - cechy ekstremalne."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 32)
        self.fc2 = nn.Linear(x_num, 32)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        return self.out(torch.cat([torch.max(h1, h2), torch.min(h1, h2)], dim=1))

class ModelV117(nn.Module):
    """Householder Transform MLP - rotacja cech bez zmiany normy."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.v = nn.Parameter(torch.randn(x_num, 1))
        self.fc = nn.Linear(x_num, 64)
        self.out = nn.Linear(64, y_num)
    def forward(self, x):
        norm_v = torch.norm(self.v)
        v = self.v / norm_v
        # H = I - 2vv^T
        h_transform = x - 2 * torch.matmul(x, v) * v.t()
        return self.out(torch.relu(self.fc(h_transform)))

class ModelV118(nn.Module):
    """Feature Shuffle MLP - wymusza komunikację między kanałami poprzez permutację."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.fc1 = nn.Linear(x_num, 64)
        self.fc2 = nn.Linear(64, 64)
        # Stały indeks permutacji
        self.register_buffer('perm_idx', torch.randperm(64))
        self.out = nn.Linear(64, y_num)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        # Shuffle: zmiana kolejności cech przed drugą warstwą
        h = h[:, self.perm_idx]
        h = torch.relu(self.fc2(h))
        return self.out(h)

class ModelV119(nn.Module):
    """Inverse Residual - zwężenie przed skip-connection."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.proj = nn.Linear(x_num, 16)
        self.expand = nn.Linear(16, x_num)
        self.out = nn.Linear(x_num, y_num)
    def forward(self, x):
        h = torch.relu(self.proj(x))
        h = torch.relu(self.expand(h)) + x
        return self.out(h)

# ModelV120 = ModelV120_Original # Zakomentowane ze względu na duplikat z V88

class ModelV120(nn.Module):
    """Rank-1 Factorized Linear - aproksymacja macierzy wag przez wektory (LoRA style)."""
    def __init__(self, x_num, y_num):
        super().__init__()
        self.u = nn.Parameter(torch.randn(x_num, 1))
        self.v = nn.Parameter(torch.randn(1, 64))
        self.bias = nn.Parameter(torch.zeros(64))
        self.out = nn.Linear(64, y_num)

    def forward(self, x):
        # Symulacja warstwy liniowej o niskim rzędzie: x * (u * v)
        w = torch.matmul(self.u, self.v)
        h = torch.relu(torch.matmul(x, w) + self.bias)
        return self.out(h)