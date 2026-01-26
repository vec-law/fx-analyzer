import inspect

class ModelManager:
    def __init__(self, config: dict, log_signal):
        self.config = config
        self.log_signal = log_signal

    def create_model(self, x_num, y_num):
        pass


# class ModelManager:
#     @staticmethod
#     def get_model(model_id, x_num, y_num):
#         class_name = f"ModelV{model_id}"
#         model_class = getattr(arch, class_name, None)
        
#         if model_class is None:
#             print(f"   [get_model] {class_name} nie istnieje w architecture.py")
#             return None
            
#         return model_class(x_num, y_num)
    
#     @staticmethod
#     def create_model_and_params(container: Container):
#         try:
#             ten_train = container.ten_dict.get('norm', {}).get('train', {})
            
#             if 'x' not in ten_train or 'y' not in ten_train:
#                 print("  [create_model_and_params] Błąd: Brak wymaganych tensorów x, y w train")
#                 return False

#             x_num = ten_train['x'].shape[1]
#             y_num = ten_train['y'].shape[1]
            
#             params = container.params_set['p']['params']
#             model_id = params['model_id']
#             seed = params['seed']
#             device = container.ten_dict['device']
            
#             torch.manual_seed(seed)
#             if device.type == 'cuda':
#                 torch.cuda.manual_seed_all(seed)

#             model = ModelManager.get_model(model_id, x_num, y_num)
#             if model is None:
#                 return False
#             model.to(device)
            
#             optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)

#             loss_function = nn.MSELoss()

#             container.mod_dict = {
#                 'model': model,
#                 'optimizer': optimizer,
#                 'loss_function': loss_function
#             }

#             print(f"  [create_model_and_params] Zainicjalizowano {model.__class__.__name__} na {device}")
#             return True

#         except Exception as e:
#             print(f"  [create_model_and_params] Błąd: {e}")
#             return False