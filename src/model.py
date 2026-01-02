import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.utils import get_path

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

def load_model(mod_dict, instrument, interval):
    path = get_path('mod', f"{instrument}_{interval}_model.pt")

    if not os.path.exists(path):
        print(f"   [load_model] Błąd: Nie znaleziono pliku {path}")
        return mod_dict

    saved_data = torch.load(path)
    mod_dict['model'].load_state_dict(saved_data['model_state_dict'])
  
    if 'optimizer_state_dict' in saved_data:
        mod_dict['optimizer'].load_state_dict(saved_data['optimizer_state_dict'])
        
    mod_dict['train_losses'] = saved_data.get('train_losses', [])

    print(f"   [load_model] Model wczytany z: {path}")
    
    return mod_dict

def save_model(mod_dict, instrument, interval):
    path = get_path('mod', f"{instrument}_{interval}_model.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        'model_state_dict': mod_dict['model'].state_dict(),
        'optimizer_state_dict': mod_dict['optimizer'].state_dict(),
        'train_losses': mod_dict.get('train_losses', [])
    }

    torch.save(state, path)
    print(f"   [save_model] Zapisano model w pliku {path}")

    return True

def evaluate_model(mod_dict, df_dict, ten_dict):

    model = mod_dict['model']
    loss_function = mod_dict['loss_function']
    device = ten_dict['device']

    mean_val = df_dict['stats']['mean']['target']
    std_val = df_dict['stats']['std']['target']

    model.eval()
    
    with torch.no_grad():

        ten_test_features_norm = torch.tensor(df_dict['test_norm'].drop(columns=['target']).to_numpy(), dtype=torch.float32).to(device)
        ten_test_target_norm = torch.tensor(df_dict['test_norm']['target'].to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)
        ten_test_predictions_norm = model(ten_test_features_norm)

        loss = loss_function(ten_test_predictions_norm, ten_test_target_norm)
    
    ten_test_predictions = (ten_test_predictions_norm * std_val) + mean_val
    ten_test_target = torch.tensor(df_dict['test']['target'].to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)

    ten_test_diff = ten_test_predictions - ten_test_target

    ten_dict['test_features_norm'] = ten_test_features_norm
    print(f"   [evaluate_model] Zapis ten_dict[test_features_norm] = ten_test_features_norm")

    ten_dict['test_target'] = ten_test_target
    print(f"   [evaluate_model] Zapis ten_dict[test_target] = ten_test_target")
    
    ten_dict['test_predictions'] = ten_test_predictions
    print(f"   [evaluate_model] Zapis ten_dict[test_predictions] = ten_test_predictions")

    df_dict['test'].loc[:, 'predictions'] = ten_dict['test_predictions'].cpu().numpy().flatten()
    print(f"   [evaluate_model] Zapis df_dict[test][predictions] z ten_dict[test_predictions]")

    mod_dict['test_loss'] = loss.item()
    mod_dict['test_mae'] = ten_test_diff.abs().mean().item()
    print(f"   [evaluate_model] Zapis mod_dict[test_loss], mod_dict[test_mae]")

    print(f"   [evaluate_model] Błąd MSE: {mod_dict['test_loss']:.6f}")
    print(f"   [evaluate_model] Błąd MAE: {mod_dict['test_mae']:.6f}")
    
    return df_dict, ten_dict, mod_dict

def train_model(ten_dict, mod_dict, epochs):

    if not isinstance(epochs, int) or epochs < 10:
        print("[train_model] Nieprawidłowa liczba kroków")
        return None

    model = mod_dict['model']
    optimizer = mod_dict['optimizer']
    loss_function = mod_dict['loss_function']

    instrument = ten_dict['instrument']
    interval = ten_dict['interval']

    ten_train_features_norm = ten_dict['train_features_norm']
    ten_train_target_norm = ten_dict['train_target_norm']

    model.train()
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        noise = torch.randn_like(ten_train_features_norm) * 0.01
        ten_train_predictions_norm = model(ten_train_features_norm + noise)
        loss = loss_function(ten_train_predictions_norm, ten_train_target_norm)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0: print(f"   [train_model] Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    mod_dict['train_losses'] = losses
    print(f"   [train_model] Zapisano mod_dict['train_losses'] = losses")

    save_model(mod_dict, instrument, interval)
            
    return mod_dict

def prepare_model_params(ten_dict, seed):
    features_num = ten_dict['train_features_norm'].shape[1]
    target_num = ten_dict['train_target_norm'].shape[1]

    if features_num < 1 or target_num < 1: 
        print("  [prepare_model_params] Nieporawidłowa ilość target i/lub features")
        return None

    torch.manual_seed(seed)
    device = ten_dict['device']
    print(f"  [torch.manual_seed] Ustawiono seed = {seed}")
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        print(f"  [torch.cuda.manual_seed] Ustawiono seed = {seed} na {device}")

    model = ModelV1(features_num, target_num)
    print(f"  [prepare_model_params] Ustawiono model ModelV1")

    model = model.to(device)
    print(f"  [prepare_model_params] Przeniesiono model na {device}")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
    print(f"  [prepare_model_params] Ustawiono optimizer na optim.Adam")

    loss_function = nn.MSELoss()
    print(f"  [prepare_model_params] Ustawiono funkcję loss_function na nn.MSELoss")

    mod_dict = {
        'model': model,
        'optimizer': optimizer,
        'loss_function': loss_function
    }

    print(f"  [prepare_model_params] Utworzono słownik mod_dict[model], mod_dict[optimizer], mod_dict[loss_function]")
    
    return mod_dict

def prepare_tensors(df_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ten_train_target_norm = torch.tensor(df_dict['train_norm']['target'].to_numpy(), dtype=torch.float32).unsqueeze(1)
    ten_train_features_norm = torch.tensor(df_dict['train_norm'].drop(columns=['target']).to_numpy(), dtype=torch.float32)

    ten_dict = {}
    ten_dict['train_target_norm'] = ten_train_target_norm
    ten_dict['train_features_norm'] = ten_train_features_norm
    print(f"  [prepare_tensors] Utworzono tensory ten_dict[train_target_norm] i ten_dict[train_features_norm]")

    ten_dict['instrument'] = df_dict['train']['instrument'].iloc[0]
    ten_dict['interval'] = df_dict['train']['interval'].iloc[0]
    print(f"  [prepare_tensors] Zapisano ten_dict[instrument] = {ten_dict['instrument']}, ten_dict[interval] = {ten_dict['interval']}")

    ten_dict['device'] = device
    print(f"  [prepare_tensors] Zapisano obecne urządzenie ({device}) w ten_dict[device]")

    instrument = df_dict['train']['instrument'].iloc[0]
    interval = df_dict['train']['interval'].iloc[0]

    path = get_path('ten', f"{instrument}_{interval}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ten_dict, path)
    print(f"  [prepare_tensors] Zapisano słownik ten_dict do pliku {path}")

    ten_dict['train_target_norm'] = ten_dict['train_target_norm'].to(device)
    ten_dict['train_features_norm'] = ten_dict['train_features_norm'].to(device)
    print(f"  [prepare_tensors] Przeniesiono tensory ten_dict[train_target_norm] i ten_dict[train_features_norm] na {device}")

    print(f"  [prepare_tensors] Rozmiar ten_dict[train_target_norm]: {list(ten_dict['train_target_norm'].shape)}, rozmiar ten_dict[train_features_norm]: {list(ten_dict['train_features_norm'].shape)}")
    
    return ten_dict
