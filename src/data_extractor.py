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
                periods_str = "-".join(map(str, f["feature_periods"]))
                col_name = f"{f['feature_type']}:{periods_str}:{f['shift']}"

                if f["feature_type"] == 'sma':
                    series = df['close'].rolling(window=f["feature_periods"][0]).mean().shift(f["shift"])
                    series.name = col_name
                    f_cols.append(series)

                elif f["feature_type"] == 'ema':
                    series = df['close'].ewm(span=f["feature_periods"][0], adjust=False).mean().shift(f["shift"])
                    series.name = col_name
                    f_cols.append(series)

                elif f["feature_type"] == 'rsi':
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=f["feature_periods"][0]).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=f["feature_periods"][0]).mean()
                    rs = gain / loss
                    series = (100 - (100 / (1 + rs))).shift(f["shift"])
                    series.name = col_name
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