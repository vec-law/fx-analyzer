import inspect
from ml.loader import Loader
from ml.cleaner import Cleaner
from ml.data_extractor import DataExtractor
from ml.preprocessor import Preprocessor
import torch
from ml.model.model_manager import ModelManager
from db.queries.models import save_model_weights
from db.queries.trainings import update_training_status, save_training_stats, add_training_log, get_training_status

class TrainingPipeline:
    def __init__(self, config: dict, train_uuid):
        self.config = config
        self.model_manager = None
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
        f_name = inspect.currentframe().f_code.co_name
        try:
            update_training_status(self.train_uuid, 'running')
            add_training_log(self.train_uuid, f"[{f_name}] Rozpoczynanie treningu")

            loader = Loader(self.config)
            self.df = loader.load_data()
            if self.df is None or self.df.empty:
                raise ValueError("Loader nie zwrócił danych")
            add_training_log(self.train_uuid, f"[{f_name}] Załadowano dane")
            if self._on_stop(f_name): return
            
            cleaner = Cleaner(self.config)
            self.df = cleaner.clean_data(self.df)
            if self.df is None or self.df.empty:
                raise ValueError("Cleaner usunął wszystkie dane")
            add_training_log(self.train_uuid, f"[{f_name}] Oczyszczono dane")
            if self._on_stop(f_name): return

            data_extractor = DataExtractor(self.config)
            self.df = data_extractor.add_calculated_columns(self.df)
            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano obliczanych kolumn")
            add_training_log(self.train_uuid, f"[{f_name}] Dodano obliczane kolumny")
            if self._on_stop(f_name): return

            self.df = data_extractor.add_features(self.df)
            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano cech")
            add_training_log(self.train_uuid, f"[{f_name}] Dodano cechy")
            if self._on_stop(f_name): return
            
            self.df = data_extractor.add_targets(self.df)
            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano wartości docelowych")
            add_training_log(self.train_uuid, f"[{f_name}] Dodano wartości docelowe")
            if self._on_stop(f_name): return
            
            self.df = data_extractor.dropna_and_cut(self.df, self.config['all_samples'])
            if self.df is None or self.df.empty:
                raise ValueError("Nie ucięto df")
            add_training_log(self.train_uuid, f"[{f_name}] Ucięto df")
            if self._on_stop(f_name): return
            
            preprocessor = Preprocessor()
            self.df_train, self.df_test = preprocessor.split_data(
                self.df,
                self.config['test_samples'],
                self.config['features'] + self.config['targets']
            )
            if self.df_train is None:
                raise ValueError("Nie wykonano splitu")
            add_training_log(self.train_uuid, f"[{f_name}] Wykonano split")
            if self._on_stop(f_name): return
            
            self.ser_mean, self.ser_std = preprocessor.calculate_stats(
                self.df_train,
                self.config['features'] + self.config['targets']
            )
            if self.ser_mean is None or self.ser_std is None:
                raise ValueError("Nie obliczono statystyk")
            add_training_log(self.train_uuid, f"[{f_name}] Obliczono statystyki")
            if self._on_stop(f_name): return
        
            save_training_stats(self.train_uuid, self.ser_mean, self.ser_std)
            if self._on_stop(f_name): return

            self.df_train_norm = preprocessor.scale_data(
                self.df_train, self.ser_mean, self.ser_std,
                self.config['features'] + self.config['targets']
            )
            if self.df_train_norm is None or self.df_train_norm.empty:
                raise ValueError("Nie znormalizowano df_train")
            add_training_log(self.train_uuid, f"[{f_name}] Znormalizowano df_train")
            if self._on_stop(f_name): return
            
            if self.df_test is not None and not self.df_test.empty:
                self.df_test_norm = preprocessor.scale_data(
                    self.df_test, self.ser_mean, self.ser_std,
                    self.config['features'] + self.config['targets']
                )
                if self.df_test_norm is None or self.df_test_norm.empty:
                    raise ValueError("Nie znormalizowano df_test")
                add_training_log(self.train_uuid, f"[{f_name}] Znormalizowano df_test")
                if self._on_stop(f_name): return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            if self._on_stop(f_name): return

            self.ten_train_norm_x = preprocessor.create_tensors(
                self.df_train_norm, self.config['features'], self.device)
            self.ten_train_norm_y = preprocessor.create_tensors(
                self.df_train_norm, self.config['targets'], self.device)
            if self.ten_train_norm_x is None or self.ten_train_norm_y is None:
                raise ValueError("Nie utworzono ten_train_norm")
            add_training_log(self.train_uuid, f"[{f_name}] Utworzono ten_train_norm")
            if self._on_stop(f_name): return
            
            if self.df_test_norm is not None and not self.df_test_norm.empty:
                self.ten_test_norm_x = preprocessor.create_tensors(
                    self.df_test_norm, self.config['features'], self.device)
                self.ten_test_norm_y = preprocessor.create_tensors(
                    self.df_test_norm, self.config['targets'], self.device)
                if self.ten_test_norm_x is None or self.ten_test_norm_y is None:
                    raise ValueError("Nie utworzono ten_test_norm")
                add_training_log(self.train_uuid, f"[{f_name}] Utworzono ten_test_norm")
                if self._on_stop(f_name): return

            self.model_manager = ModelManager()

            for arch in self.config['architectures']:
                if self._on_stop(f_name): return

                model, optimizer, loss_function = self.model_manager.create_model(
                    len(self.config['features']),
                    len(self.config['targets']),
                    self.config,
                    arch, self.device
                )
                if model is None or optimizer is None or loss_function is None:
                    raise ValueError("Nie utworzono modelu")
                add_training_log(self.train_uuid, f"[{f_name}] Utworzono model")
                if self._on_stop(f_name): return
                
                model = self.model_manager.train_model(
                    model, optimizer, loss_function,
                    self.ten_train_norm_x, self.ten_train_norm_y,
                    self.config, self.device, self.train_uuid
                )
                if model is None:
                    raise ValueError("Nie wykonano uczenia modelu")
                add_training_log(self.train_uuid, f"[{f_name}] Wykonano uczenie modelu")
                if self._on_stop(f_name): return

                mae_loss = None
                mse_loss = None
                
                if self.ten_test_norm_x is not None and self.ten_test_norm_y is not None:
                    mse_loss, mae_loss = self.model_manager.evaluate_model(
                        model, loss_function,
                        self.ten_test_norm_x, self.ten_test_norm_y,
                    )
                    if mse_loss is None or mae_loss is None:
                        raise ValueError("Nie wykonano ewaluacji modelu")
                    add_training_log(self.train_uuid, f"[{f_name}] Wykonano ewaluację modelu")
                    if self._on_stop(f_name): return
                    
                weights = self.model_manager.get_model_weights(model)
                if weights is None:
                    raise ValueError("Nie odczytano wag modelu")
                add_training_log(self.train_uuid, f"[{f_name}] Odczytano wagi modelu")
                if self._on_stop(f_name): return
                
                if not save_model_weights(self.train_uuid, arch, weights, mse_loss, mae_loss):
                    raise ValueError("Nie zapisano wag modelu")
                add_training_log(self.train_uuid, f"[{f_name}] Zapisano wagi modelu")
                if self._on_stop(f_name): return

            if self._on_stop(f_name): return

            update_training_status(self.train_uuid, "completed")
            add_training_log(self.train_uuid, f"[{f_name}] Koniec treningu")

        except Exception as e:
            add_training_log(self.train_uuid, f"[{f_name}] Błąd: {e}")
            try:
                update_training_status(self.train_uuid, 'failed')
            except Exception as db_err:
                add_training_log(self.train_uuid, f"[{f_name}] Błąd bazy danych: {db_err}")

    def _on_stop(self, f_name):
        status = get_training_status(self.train_uuid)
        if status == 'stopping':
            update_training_status(self.train_uuid, 'failed')
            add_training_log(self.train_uuid, f"[{f_name}] Proces przerwany przez użytkownika")
            return True
        return False