import io
import torch
from ml.loader import get_loader
from ml.cleaner import clean_data
from ml.data_extractor import add_calculated_columns, add_features, dropna_and_cut, join_at_end, add_diff
from ml.preprocessor import split_data, descale_data, scale_data, create_tensors
from ml.model.model_manager import create_model, predict, set_model_weights
from db.queries.models import load_model_weights
from db.queries.trainings import load_training_stats
from db.queries.predictions import update_prediction_status, add_prediction_log, save_prediction_result, get_prediction_status

class PredictionPipeline:
    def __init__(self, user_id, train_uuid, pred_uuid, train_config: dict, pred_config: dict):
        self.user_id = user_id
        self.train_uuid = train_uuid
        self.pred_uuid = pred_uuid
        self.train_config = train_config
        self.pred_config = pred_config
        self.df = None
        self.df_pred = None
        self.df_pred_norm = None
        self.ser_mean = None
        self.ser_std = None
        self.device = None

    def run(self):
        try:
            update_prediction_status(self.user_id, self.pred_uuid, 'running')
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Rozpoczynanie predykcji")

            if self.train_config is None:
                raise ValueError("Nie pobrano konfiguracji treningu")
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Pobrano konfigurację treningu")
            if self._on_stop(): return
            
            loader = get_loader(self.train_config)

            self.df = loader.load_data()
            if self.df is None or self.df.empty:
                raise ValueError("Loader nie zwrócił danych")
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Załadowano dane")
            if self._on_stop(): return
            
            self.df = clean_data(self.df, self.train_config)
            if self.df is None or self.df.empty:
                raise ValueError("Cleaner usunął wszystkie dane")
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Oczyszczono dane")
            if self._on_stop(): return

            self.df = add_calculated_columns(self.df, self.train_config)
            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano obliczanych kolumn")
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Dodano obliczane kolumny")
            if self._on_stop(): return

            self.df = add_features(self.df, self.train_config)
            if self.df is None or self.df.empty:
                raise ValueError("Nie dodano cech")
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Dodano cechy")
            if self._on_stop(): return

            self.df = dropna_and_cut(self.df, self.pred_config['all_samples'])
            if self.df is None or self.df.empty:
                raise ValueError("Nie ucięto df")
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Ucięto df")
            if self._on_stop(): return

            _, self.df_pred = split_data(
                self.df,
                self.pred_config['predicted_samples'],
                self.train_config['features']
            )
            if self.df_pred is None:
                raise ValueError("Nie wykonano splitu")
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Wykonano split")
            if self._on_stop(): return

            self.ser_mean, self.ser_std = load_training_stats(self.train_uuid)
            if self.ser_mean is None or self.ser_std is None:
                raise ValueError("Nie obliczono statystyk")
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Załadowano statystyki")
            if self._on_stop(): return

            self.df_pred_norm = scale_data(
                self.df_pred, self.ser_mean, self.ser_std,
                self.train_config['features']
            )
            if self.df_pred_norm is None or self.df_pred_norm.empty:
                raise ValueError("Nie znormalizowano df_pred")
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Znormalizowano df_pred")
            if self._on_stop(): return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            if self._on_stop(): return

            self.ten_pred_norm_x = create_tensors(
                self.df_pred_norm, self.train_config['features'], self.device)
            if self.ten_pred_norm_x is None:
                raise ValueError("Nie utworzono ten_pred_norm")
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Utworzono ten_pred_norm")
            if self._on_stop(): return

            for arch in self.train_config['architectures']:
                if self._on_stop(): return

                model, _, _ = create_model(
                    len(self.train_config['features']),
                    len(self.train_config['targets']),
                    self.train_config,
                    arch, self.device
                )
                if model is None:
                    raise ValueError("Nie utworzono modelu")
                add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Utworzono model")
                if self._on_stop(): return
                
                weights = load_model_weights(self.train_uuid, arch)
                if weights is None:
                    raise ValueError("Nie odczytano wag modelu")
                add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Odczytano wagi modelu")
                if self._on_stop(): return

                if not set_model_weights(model, weights):
                    raise ValueError("Nie ustawiono wag modelu")
                add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Ustawiono wagi modelu")
                if self._on_stop(): return

                ten_pred_norm_y = predict(model, self.ten_pred_norm_x)
                if ten_pred_norm_y is None:
                    raise ValueError("Nie policzono ten_pred_norm_y")
                add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Policzono ten_pred_norm_y")
                if self._on_stop(): return

                df_pred = descale_data(
                    ten_pred_norm_y,
                    self.ser_mean,
                    self.ser_std,
                    self.train_config['targets']
                )
                if df_pred is None or df_pred.empty:
                    raise ValueError("Nie zdenormalizowano ten_pred_norm_y")
                add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Zdenormalizowano ten_pred_norm_y")
                if self._on_stop(): return

                df = join_at_end(self.df, df_pred)
                if df is None or df.empty:
                    raise ValueError("Nie dołączono df_pred")
                add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Dołączono df_pred")
                if self._on_stop(): return

                df = add_diff(df, self.train_config)
                if df is None or df.empty:
                    raise ValueError("Nie dodano diff")
                add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] dodano diff")
                if self._on_stop(): return
                
                df_buffer = io.BytesIO()
                df.to_parquet(df_buffer, engine='pyarrow', index=True)
                df_parquet = df_buffer.getvalue()
                if not save_prediction_result(
                    self.pred_uuid,
                    arch,
                    df_parquet
                    ):
                    raise ValueError("Nie zapisano wyników do db")
                add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Zapisano wyniki do db")
                if self._on_stop(): return

            update_prediction_status(self.user_id, self.pred_uuid, "completed")
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Koniec predykcji")

        except Exception as e:
            add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Błąd: {e}")
            try:
                update_prediction_status(self.user_id, self.pred_uuid, 'failed')
            except Exception as db_err:
                add_prediction_log(self.pred_uuid, f"[pred_uuid: {str(self.pred_uuid)[:6]}] Błąd bazy danych: {db_err}")

    def _on_stop(self):
        status = get_prediction_status(self.user_id, self.pred_uuid)
        if status == 'stopping':
            update_prediction_status(self.user_id, self.pred_uuid, 'stopped')
            add_prediction_log(
                self.pred_uuid,
                f"[pred_uuid: {str(self.pred_uuid)[:6]}] Proces przerwany przez użytkownika"
            )
            return True
        return False