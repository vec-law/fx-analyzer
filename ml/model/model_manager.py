import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save
from safetensors.torch import load
import ml.model.architecture as archs
from db.queries.trainings import add_training_log
from db.queries.trainings import get_training_status

def create_model(x_num, y_num, params, arch, device):
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(params['seed'])

    model = _get_model(arch, x_num, y_num)
    if model is None:
        return None, None, None
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-2)
    loss_function = nn.MSELoss()

    return model, optimizer, loss_function

def _get_model(class_name, x_num, y_num):
    try:
        model_class = getattr(archs, class_name)
        if isinstance(model_class, type) and issubclass(model_class, nn.Module):
            return model_class(x_num, y_num)
        return None
    except AttributeError:
        return None
    
def train_model(model, optimizer, loss_function, ten_train_norm_x, ten_train_norm_y, params, device, train_uuid):
    try:
        if model is None or optimizer is None or loss_function is None:
            return None
        if ten_train_norm_x is None or ten_train_norm_y is None:
            return None
        if params is None or device is None:
            return None

        model.train()

        for epoch in range(params['epochs']):
            if (epoch + 1) % 100 == 0:
                status = get_training_status(train_uuid)
                if status == 'stopping':
                    return None
            
            optimizer.zero_grad()
            noise = (torch.randn_like(ten_train_norm_x) * params['train_noise']).to(device)
            ten_train_norm_p = model(ten_train_norm_x + noise)
            loss = loss_function(ten_train_norm_p, ten_train_norm_y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                add_training_log(train_uuid, f"[train_uuid: {str(train_uuid)[:6]}] Epoch: {epoch+1}/{params['epochs']}, Loss: {loss.item():.6f}")

        return model
    
    except Exception as e:
        raise Exception(f"Błąd: {e}")
    
def predict(model, ten_norm_x):
    try:
        if model is None or ten_norm_x is None:
            return None
        
        model.eval()
        with torch.no_grad():
            return model(ten_norm_x)
        
    except Exception as e:
        raise Exception(f"Błąd: {e}")
    
def evaluate_model(model, loss_function, ten_test_norm_x, ten_test_norm_y):
    try:
        if model is None or loss_function is None:
            return None, None
        if ten_test_norm_x is None or ten_test_norm_y is None:
            return None, None

        ten_test_norm_p = predict(model, ten_test_norm_x)
        mse_loss = loss_function(ten_test_norm_p, ten_test_norm_y).item()
        mae_loss = F.l1_loss(ten_test_norm_p, ten_test_norm_y).item()
        
        return mse_loss, mae_loss

    except Exception as e:
        raise Exception(f"Błąd: {e}")

def get_model_weights(model):
    try:
        if model is None:
            return None
        
        state_dict = model.state_dict()
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        return save(cpu_state_dict)

    except Exception as e:
        raise Exception(f"Błąd: {e}")
    
def set_model_weights(model, weights):
    try:
        if model is None or weights is None:
            return False
        state_dict = load(weights)
        model.load_state_dict(state_dict)
        return True

    except Exception as e:
        raise Exception(f"Błąd: {e}")
