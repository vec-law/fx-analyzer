from .container import Container
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt


class Trainer:
    @staticmethod
    def train_model(container: Container):
        try:
            req_components = ['model', 'optimizer', 'loss_function']
            if not all(k in container.mod_dict for k in req_components):
                print("  [train_model] Błąd: Brak modelu, optymalizatora lub funkcji kosztu")
                return False

            ten_dict = container.ten_dict
            ten_norm = ten_dict.get('norm', {})
            ten_norm_train = ten_norm.get('train', {})

            if 'x' not in ten_norm_train or 'y' not in ten_norm_train:
                print("  [train_model] Błąd: Brak tensorów ten_dict[norm][train][x] i ten_dict[norm][train][y]")
                return False
            
            ten_norm_train_x = ten_norm_train['x']
            ten_norm_train_y = ten_norm_train['y']

            params = container.params_set['p']['params']

            epochs = params.get('epochs', 1000)
            if epochs < 10:
                epochs = 1000

            train_noise = params.get('train_noise', 0.01)
            
            model = container.mod_dict['model']
            optimizer = container.mod_dict['optimizer']
            loss_function = container.mod_dict['loss_function']

            model.train()
            losses = []

            for epoch in range(epochs):
                optimizer.zero_grad()
                
                device = ten_norm_train_x.device
                noise = (torch.randn_like(ten_norm_train_x) * train_noise).to(device)
                ten_norm_train_p = model(ten_norm_train_x + noise)
                
                loss = loss_function(ten_norm_train_p, ten_norm_train_y)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                
                if (epoch + 1) % 100 == 0:
                    print(f"  [train_model] Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

            container.mod_dict['train_losses'] = losses
            print(f"  [train_model] Zakończono uczenie modelu")

            return True
        
        except Exception as e:
            print(f"   [train_model] Błąd: {e}")
            return False

    @staticmethod
    def evaluate_model(container: Container):
        try:
            ten_dict = container.ten_dict
            ten_norm = ten_dict.get('norm', {})
            ten_norm_test = ten_norm.get('test', {})

            if 'x' not in ten_norm_test or 'y' not in ten_norm_test:
                print("  [train_model] Błąd: Brak tensorów ten_dict[norm][test][x] i ten_dict[norm][test][y]")
                return False
            
            ten_norm_test_x = ten_norm_test['x']
            ten_norm_test_y = ten_norm_test['y']

            model = container.mod_dict['model']
            loss_function = container.mod_dict['loss_function']

            model.eval()
            with torch.no_grad():
                ten_norm_test_p = model(ten_norm_test_x)

                mse_loss = loss_function(ten_norm_test_p, ten_norm_test_y)
                mae_loss = F.l1_loss(ten_norm_test_p, ten_norm_test_y)

            container.mod_dict['test_mse'] = mse_loss.item()
            container.mod_dict['test_mae'] = mae_loss.item()

            if 'p' not in ten_norm_test:
                container.ten_dict['norm']['test']['p'] = ten_norm_test_p

            print(f"  [evaluate_model] Błąd MSE: {mse_loss.item():.6f}")
            print(f"  [evaluate_model] Błąd MAE: {mae_loss.item():.6f}")
            
            return True

        except Exception as e:
            print(f"  [evaluate_model] Błąd: {e}")
            return False
        
    @staticmethod
    def predict(container: 'Container'):
        try:
            ten_norm = container.ten_dict.get('norm', {})
            ten_norm_train = ten_norm.get('train', {})
            ten_norm_test = ten_norm.get('test', {})

            if 'x' not in ten_norm_train or 'x' not in ten_norm_test:
                print("  [predict] Błąd: Brak tensorów x w ten_dict['norm']['test']")
                return False
            
            ten_norm_train_x = ten_norm_train['x']
            ten_norm_test_x = ten_norm_test['x']
            model = container.mod_dict['model']

            model.eval()
            
            with torch.no_grad():
                ten_norm_train_p = model(ten_norm_train_x)
                ten_norm_test_p = model(ten_norm_test_x)

            container.ten_dict['norm']['train']['p'] = ten_norm_train_p
            container.ten_dict['norm']['test']['p'] = ten_norm_test_p

            print(f"  [predict] Obliczono tensory p")
            return True

        except Exception as e:
            print(f"  [predict] Błąd: {e}")
            return False

