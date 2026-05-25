from ml.loader import get_loader
from ml.cleaner import clean_data
from ml.data_extractor import add_calculated_columns, add_features, add_targets, dropna_and_cut
from ml.preprocessor import split_data, calculate_stats, scale_data, create_tensors
import torch
from ml.model.model_manager import create_model, train_model, evaluate_model, get_model_weights
from db.queries.models import save_model_weights
from db.queries.trainings import update_training_status, save_training_stats, add_training_log, get_training_status

class TrainingPipeline:
    def __init__(self, config: dict, train_uuid):
        self.config = config
        self.train_uuid = train_uuid
        self.df = None
        self.df_train = None
        self.df_train_norm = None
        self.df_test = None
        self.df_test_norm = None
        self.ser_mean = None
        self.ser_std = None
        self.ten_train_norm_x = None
        self.ten_train_norm_y = None
        self.ten_test_norm_x = None
        self.ten_test_norm_y = None
        self.device = None

    def run(self):
        try:
            update_training_status(self.train_uuid, 'running')
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Rozpoczynanie treningu")

            loader = get_loader(self.config)
            self.df = loader.load_data()
            if self.df is None or self.df.empty:
                raise ValueError("Loader nie zwrócił danych")
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Załadowano dane")
            if self._on_stop(): return

            self.df = clean_data(self.df, self.config)
            if self.df is None or self.df.empty:
                raise ValueError("Cleaner usunął wszystkie dane")
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Oczyszczono dane")
            if self._on_stop(): return

            self.df = add_calculated_columns(self.df, self.config)
            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano obliczanych kolumn")
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Dodano obliczane kolumny")
            if self._on_stop(): return

            self.df = add_features(self.df, self.config)
            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano cech")
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Dodano cechy")
            if self._on_stop(): return
            
            self.df = add_targets(self.df, self.config)
            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano wartości docelowych")
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Dodano wartości docelowe")
            if self._on_stop(): return
            
            self.df = dropna_and_cut(self.df, self.config['all_samples'])
            if self.df is None or self.df.empty:
                raise ValueError("Nie ucięto df")
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Ucięto df")
            if self._on_stop(): return
            
            self.df_train, self.df_test = split_data(
                self.df,
                self.config['test_samples'],
                self.config['features'] + self.config['targets']
            )
            if self.df_train is None:
                raise ValueError("Nie wykonano splitu")
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Wykonano split")
            if self._on_stop(): return
            
            self.ser_mean, self.ser_std = calculate_stats(
                self.df_train,
                self.config['features'] + self.config['targets']
            )
            if self.ser_mean is None or self.ser_std is None:
                raise ValueError("Nie obliczono statystyk")
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Obliczono statystyki")
            if self._on_stop(): return
        
            save_training_stats(self.train_uuid, self.ser_mean, self.ser_std)
            if self._on_stop(): return

            self.df_train_norm = scale_data(
                self.df_train, self.ser_mean, self.ser_std,
                self.config['features'] + self.config['targets']
            )
            if self.df_train_norm is None or self.df_train_norm.empty:
                raise ValueError("Nie znormalizowano df_train")
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Znormalizowano df_train")
            if self._on_stop(): return
            
            if self.df_test is not None and not self.df_test.empty:
                self.df_test_norm = scale_data(
                    self.df_test, self.ser_mean, self.ser_std,
                    self.config['features'] + self.config['targets']
                )
                if self.df_test_norm is None or self.df_test_norm.empty:
                    raise ValueError("Nie znormalizowano df_test")
                add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Znormalizowano df_test")
                if self._on_stop(): return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            if self._on_stop(): return

            self.ten_train_norm_x = create_tensors(
                self.df_train_norm, self.config['features'], self.device)
            self.ten_train_norm_y = create_tensors(
                self.df_train_norm, self.config['targets'], self.device)
            if self.ten_train_norm_x is None or self.ten_train_norm_y is None:
                raise ValueError("Nie utworzono ten_train_norm")
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Utworzono ten_train_norm")
            if self._on_stop(): return
            
            if self.df_test_norm is not None and not self.df_test_norm.empty:
                self.ten_test_norm_x = create_tensors(
                    self.df_test_norm, self.config['features'], self.device)
                self.ten_test_norm_y = create_tensors(
                    self.df_test_norm, self.config['targets'], self.device)
                if self.ten_test_norm_x is None or self.ten_test_norm_y is None:
                    raise ValueError("Nie utworzono ten_test_norm")
                add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Utworzono ten_test_norm")
                if self._on_stop(): return

            for arch in self.config['architectures']:
                if self._on_stop(): return

                model, optimizer, loss_function = create_model(
                    len(self.config['features']),
                    len(self.config['targets']),
                    self.config,
                    arch, self.device
                )
                if model is None or optimizer is None or loss_function is None:
                    raise ValueError("Nie utworzono modelu")
                add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Utworzono model")
                if self._on_stop(): return
                
                model = train_model(
                    model, optimizer, loss_function,
                    self.ten_train_norm_x, self.ten_train_norm_y,
                    self.config, self.device, self.train_uuid
                )
                if model is None:
                    raise ValueError("Nie wykonano uczenia modelu")
                add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Wykonano uczenie modelu")
                if self._on_stop(): return

                mae_loss = None
                mse_loss = None
                
                if self.ten_test_norm_x is not None and self.ten_test_norm_y is not None:
                    mse_loss, mae_loss = evaluate_model(
                        model, loss_function,
                        self.ten_test_norm_x, self.ten_test_norm_y,
                    )
                    if mse_loss is None or mae_loss is None:
                        raise ValueError("Nie wykonano ewaluacji modelu")
                    add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Wykonano ewaluację modelu")
                    if self._on_stop(): return
                    
                weights = get_model_weights(model)
                if weights is None:
                    raise ValueError("Nie odczytano wag modelu")
                add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Odczytano wagi modelu")
                if self._on_stop(): return
                
                if not save_model_weights(self.train_uuid, arch, weights, mse_loss, mae_loss):
                    raise ValueError("Nie zapisano wag modelu")
                add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Zapisano wagi modelu")
                if self._on_stop(): return

            if self._on_stop(): return

            update_training_status(self.train_uuid, "completed")
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Koniec treningu")

        except Exception as e:
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Błąd: {e}")
            try:
                update_training_status(self.train_uuid, 'failed')
            except Exception as db_err:
                add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Błąd bazy danych: {db_err}")

    def _on_stop(self):
        status = get_training_status(self.train_uuid)
        if status == 'stopping':
            update_training_status(self.train_uuid, 'stopped')
            add_training_log(self.train_uuid, f"[train_uuid: {str(self.train_uuid)[:6]}] Proces przerwany przez użytkownika")
            return True
        return False