import inspect
from src.loader import Loader
from src.cleaner import Cleaner
from src.data_extractor import DataExtractor
from src.preprocessor import Preprocessor
import torch
import io
from src.model.model_manager import ModelManager

class PredictionPipeline:
    def __init__(self, pred_config: dict, log_signal, db_manager, pred_uuid):
        self.pred_config = pred_config
        self.log_signal = log_signal
        self.db_manager = db_manager
        self.pred_uuid = pred_uuid
        self._is_stopped = False
        self.train_config = None
        self.df = None
        self.df_pred = None
        self.df_pred_norm = None
        self.ser_mean = None
        self.ser_std = None
        self.device = None

    def run(self):
        f_name = inspect.currentframe().f_code.co_name
        try:
            self.db_manager.update_prediction_status(self.pred_uuid, 'running')
            self.log_signal.emit(f"[{f_name}] Rozpoczynanie predykcji")

            self.train_config = self.db_manager.get_training_config(self.pred_config['train_uuid'])
            if self.train_config is None:
                raise ValueError("Nie pobrano konfiguracji treningu")
            if self._handle_stop(f_name): return

            loader = Loader(self.train_config, log_signal=self.log_signal)

            self.df = loader.load_data()
            if self.df is None or self.df.empty:
                raise ValueError("Loader nie zwrócił danych")
            if self._handle_stop(f_name): return
            
            cleaner = Cleaner(self.train_config, log_signal=self.log_signal)

            self.df = cleaner.clean_data(self.df)
            if self.df is None or self.df.empty:
                raise ValueError("Cleaner usunął wszystkie dane")
            if self._handle_stop(f_name): return

            data_extractor = DataExtractor(self.train_config, log_signal=self.log_signal)

            self.df = data_extractor.add_calculated_columns(self.df)
            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano obliczanych kolumn")
            if self._handle_stop(f_name): return

            self.df = data_extractor.add_features(self.df)
            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano cech")
            if self._handle_stop(f_name): return
            
            self.df = data_extractor.dropna_and_cut(self.df, self.pred_config['all_samples'])
            if self.df is None or self.df.empty:
                raise ValueError("Nie ucięto df")
            if self._handle_stop(f_name): return
            
            preprocessor = Preprocessor(self.log_signal)

            _, self.df_pred = preprocessor.split_data(
                self.df,
                self.pred_config['predicted_samples'],
                self.train_config['feature_names']
            )
            if self.df_pred is None:
                raise ValueError("Nie wykonano splitu")
            if self._handle_stop(f_name): return

            self.ser_mean, self.ser_std = self.db_manager.load_training_stats(self.pred_config['train_uuid'])
            if self.ser_mean is None or self.ser_std is None:
                raise ValueError("Nie załadowano statystyk")
            if self._handle_stop(f_name): return

            self.df_pred_norm = preprocessor.scale_data(
                self.df_pred,
                self.ser_mean,
                self.ser_std,
                self.train_config['feature_names']
            )
            if self.df_pred_norm is None or self.df_pred_norm.empty:
                raise ValueError("Nie znormalizowano df_pred")
            if self._handle_stop(f_name): return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            if self._handle_stop(f_name): return

            self.ten_pred_norm_x = preprocessor.create_tensors(
                self.df_pred_norm,
                self.train_config['feature_names'],
                self.device
            )
            if self.ten_pred_norm_x is None:
                raise ValueError("Nie utworzono ten_pred_norm")
            if self._handle_stop(f_name): return

            model_manager = ModelManager(self.log_signal)

            for arch in self.train_config['architectures']:
                if self._handle_stop(f_name): return

                model, _, _ = model_manager.create_model(
                    len(self.train_config['feature_names']),
                    len(self.train_config['target_names']),
                    self.train_config['parameter_set'],
                    arch,
                    self.device
                )
                if model is None:
                    raise ValueError("Nie utworzono modelu")
                if self._handle_stop(f_name): return

                weights = self.db_manager.load_model_weights(self.pred_config['train_uuid'], arch)

                if weights is None:
                    raise ValueError("Nie pobrano wag modelu")
                if self._handle_stop(f_name): return

                if not model_manager.set_model_weights(model, weights):
                    raise ValueError("Nie załadowano wag modelu")
                if self._handle_stop(f_name): return

                ten_pred_norm_y = model_manager.predict(model, self.ten_pred_norm_x)
                if ten_pred_norm_y is None:
                    raise ValueError("Nie policzono ten_pred_norm_y")
                if self._handle_stop(f_name): return

                df_pred = preprocessor.descale_data(
                    ten_pred_norm_y,
                    self.ser_mean,
                    self.ser_std,
                    self.train_config['target_names']
                )
                if df_pred is None or df_pred.empty:
                    raise ValueError("Nie zdenormalizowano ten_pred_norm_y")
                if self._handle_stop(f_name): return

                df = data_extractor.join_at_end(self.df, df_pred)
                
                df_buffer = io.BytesIO()
                df.to_parquet(df_buffer, engine='pyarrow', index=True)
                df_parquet = df_buffer.getvalue()
                if not self.db_manager.save_prediction_result(
                    self.pred_uuid,
                    arch,
                    df_parquet
                    ):
                    raise ValueError("Nie zapisano wyników do db")

            self.db_manager.update_prediction_status(self.pred_uuid, "completed")
            self.log_signal.emit(f"[{f_name}] Koniec predykcji")

        except Exception as e:
            self.log_signal.emit(f"[{f_name}] Przerwano z powodu błędu: {e}")
            try:
                self.db_manager.update_prediction_status(self.pred_uuid, 'failed')
            except Exception as db_err:
                self.log_signal.emit(f"[{f_name}] Błąd bazy danych: {db_err}")

    def stop(self):
        f_name = inspect.currentframe().f_code.co_name
        self._is_stopped = True
        self.log_signal.emit(f"[{f_name}] Zatrzymywanie...")

    def _handle_stop(self, f_name):
        if self._is_stopped:
            self.db_manager.update_prediction_status(self.pred_uuid, 'failed')
            self.log_signal.emit(f"[{f_name}] Proces przerwany przez użytkownika")
            return True
        return False
