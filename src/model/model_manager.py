import inspect
import torch
import torch.optim as optim
import torch.nn as nn
import src.model.architecture as archs

class ModelManager:
    def __init__(self, config: dict, log_signal):
        self.config = config
        self.log_signal = log_signal

    def create_model(self, x_num, y_num, params, arch, device):
        f_name = inspect.currentframe().f_code.co_name
        self.log_signal.emit(f"[{f_name}] Tworzenie modelu...")

        torch.manual_seed(params['seed'])
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(params['seed'])

        model = self._get_model(arch, x_num, y_num)

        if model is None:
            self.log_signal.emit(f"[{f_name}] Nie utworzono modelu: {arch}")
            return None, None, None
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-2)
        loss_function = nn.MSELoss()

        self.log_signal.emit(f"[{f_name}] Utworzono model: {arch}")
        return model, optimizer, loss_function

    def _get_model(self, class_name, x_num, y_num):
        f_name = inspect.currentframe().f_code.co_name
        try:
            model_class = getattr(archs, class_name)

            if isinstance(model_class, type) and issubclass(model_class, nn.Module):
                return model_class(x_num, y_num)
            
            error_msg = f"nie jest klasą nn.Module" if isinstance(model_class, type) else "nie jest klasą"
            self.log_signal.emit(f"[{f_name}] Błąd: {class_name} {error_msg}")
            return None
                
        except AttributeError:
            self.log_signal.emit(f"[{f_name}] Błąd: Model {class_name} nie istnieje w architecture.py")
            return None
        
    def train_model(
            self, model,
            optimizer,
            loss_function,
            ten_train_norm_x,
            ten_train_norm_y,
            params,
            device
        ):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if model is None or optimizer is None or loss_function is None:
                self.log_signal.emit(f"[{f_name}] Błąd: brak modelu, optymalizatora lub funkcji kosztu")
                return None
            if ten_train_norm_x is None or ten_train_norm_y is None:
                self.log_signal.emit(f"[{f_name}] Błąd: brak tensorów")
                return None
            if params is None:
                self.log_signal.emit(f"[{f_name}] Błąd: brak parametrów")
                return None
            if device is None:
                self.log_signal.emit(f"[{f_name}] Błąd: nie określono urządzenia")
                return None

            model.train()

            self.log_signal.emit(f"[{f_name}] Rozpoczęto uczenie modelu")

            for epoch in range(params['epochs']):
                optimizer.zero_grad()
                
                noise = (torch.randn_like(ten_train_norm_x) * params['train_noise']).to(device)
                ten_train_norm_p = model(ten_train_norm_x + noise)
                
                loss = loss_function(ten_train_norm_p, ten_train_norm_y)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 100 == 0:
                    self.log_signal.emit(f"[{f_name}] Epoch: {epoch+1}/{params['epochs']}, Loss: {loss.item():.6f}")

            self.log_signal.emit(f"[{f_name}] Zakończono uczenie modelu")

            return model
        
        except Exception as e:
            self.log_signal.emit(f"[{f_name}] Błąd: {e}")
            return None