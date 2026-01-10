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

            if 'x_train_norm' not in container.ten_dict or 'y_train_norm' not in container.ten_dict:
                print("  [train_model] Błąd: Brak kluczy 'x_train_norm', 'y_train_norm' w ten_dict")
                return False

            params = container.params_set['p']['params']

            epochs = params.get('epochs', 1000)
            if epochs < 10:
                epochs = 1000

            train_noise = params.get('train_noise', 0.01)
            
            model = container.mod_dict['model']
            optimizer = container.mod_dict['optimizer']
            loss_function = container.mod_dict['loss_function']

            x_train_norm = container.ten_dict['x_train_norm']
            y_train_norm = container.ten_dict['y_train_norm']

            model.train()
            losses = []

            for epoch in range(epochs):
                optimizer.zero_grad()
                
                noise = torch.randn_like(x_train_norm) * train_noise
                p_train_norm = model(x_train_norm + noise)
                
                loss = loss_function(p_train_norm, y_train_norm)
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
            if 'model' not in container.mod_dict or 'x_test_norm' not in container.ten_dict:
                print("  [evaluate_model] Błąd: Brak modelu lub danych testowych")
                return False

            model = container.mod_dict['model']
            loss_function = container.mod_dict['loss_function']
            
            x_test_norm = container.ten_dict['x_test_norm']
            y_test_norm = container.ten_dict['y_test_norm']

            model.eval()
            with torch.no_grad():
                p_test_norm = model(x_test_norm)

                mse_loss = loss_function(p_test_norm, y_test_norm)
                mae_loss = F.l1_loss(p_test_norm, y_test_norm)

            container.mod_dict['test_mse'] = mse_loss.item()
            container.mod_dict['test_mae'] = mae_loss.item()
            container.ten_dict['p_test_norm'] = p_test_norm

            print(f"  [evaluate_model] Obliczono błędy")
            print(f"  [evaluate_model] Błąd MSE: {mse_loss.item():.6f}")
            print(f"  [evaluate_model] Błąd MAE: {mae_loss.item():.6f}")

            # plt.plot(range(len(y_test_norm)), y_test_norm[:, 0])
            # plt.plot(range(len(y_test_norm)), p_test_norm[:, 0])
            # plt.show()
            
            return True

        except Exception as e:
            print(f"  [evaluate_model] Błąd: {e}")
            return False
