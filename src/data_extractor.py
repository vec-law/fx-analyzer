import inspect
import pandas as pd

class DataExtractor:
    def __init__(self, config: dict, log_signal):
        self.config = config
        self.log_signal = log_signal

    def add_features(self, df):
        f_name = inspect.currentframe().f_code.co_name
        try:
            self.log_signal.emit(f"[{f_name}] Rozpoczynanie dodawania cech")
            if df is None or df.empty:
                self.log_signal.emit(f"[{f_name}] Brak danych w df")
                return None
            
            f_cols = []
            
            for f in self.config["features"]:
                if f["feature_type"] == 'sma':
                    series = df[f["base_column"]].rolling(window=f["feature_period"]).mean().shift(f["shift"])
                    series.name = f"{f['feature_type']}:{f['feature_period']}:{f['base_column']}:{f['shift']}"
                    f_cols.append(series)

                elif f["feature_type"] == 'ema':
                    series = df[f["base_column"]].ewm(span=f["feature_period"], adjust=False).mean().shift(f["shift"])
                    series.name = f"{f['feature_type']}:{f['feature_period']}:{f['base_column']}:{f['shift']}"
                    f_cols.append(series)

                elif f["feature_type"] == 'rsi':
                    delta = df[f["base_column"]].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=f["feature_period"]).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=f["feature_period"]).mean()
                    rs = gain / loss
                    series = (100 - (100 / (1 + rs))).shift(f["shift"])
                    series.name = f"{f['feature_type']}:{f['feature_period']}:{f['base_column']}:{f['shift']}"
                    f_cols.append(series)

            if not f_cols:
                self.log_signal.emit(f"[{f_name}] Brak cech do dodania")
                return None
            
            df = pd.concat([df] + f_cols, axis=1).copy()

            self.log_signal.emit(f"[{f_name}] Dodano cechy")            
            return df

        except Exception as e:
            self.log_signal.emit(f"[{f_name}] Błąd: {e}")
            return None
