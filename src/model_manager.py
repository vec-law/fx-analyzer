from .container import Container
from . import architecture as arch
import torch
import torch.optim as optim
import torch.nn as nn


class ModelManager:
    @staticmethod
    def get_model(model_id, x_num, y_num):
        class_name = f"ModelV{model_id}"
        model_class = getattr(arch, class_name, None)
        
        if model_class is None:
            print(f"   [get_model] {class_name} nie istnieje w architecture.py")
            return None
            
        return model_class(x_num, y_num)
    
    @staticmethod
    def create_model_and_params(container: Container):
        try:
            x_num = container.ten_dict['x_train_norm'].shape[1]
            y_num = container.ten_dict['y_train_norm'].shape[1]
            
            params = container.params_set['p']['params']
            model_id = params['model_id']
            seed = params['seed']
            device = container.ten_dict['device']
            
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(seed)

            model = ModelManager.get_model(model_id, x_num, y_num)
            if model is None:
                return False
            model.to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)

            loss_function = nn.MSELoss()

            container.mod_dict = {
                'model': model,
                'optimizer': optimizer,
                'loss_function': loss_function
            }

            print(f"  [create_model_and_params] Zainicjalizowano {model.__class__.__name__} na {device}")
            return True

        except Exception as e:
            print(f"  [create_model_and_params] Błąd: {e}")
            return False