import inspect
import pandas as pd

class DataExtractor:
    def __init__(self, config: dict, log_signal):
        self.config = config
        self.log_signal = log_signal

    def add_features(self, df):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df.empty:
                self.log_signal.emit(f"[{f_name}] Brak danych w df")
                return None
            
            f_cols = []
            
            for f, col_name in zip(self.config["features"], self.config["feature_names"]):
                f_type = f["feature_type"]
                periods = f["feature_periods"]
                shift = f["shift"]

                if f_type == 'sma':
                    series = df['close'].rolling(window=periods[0]).mean().shift(shift)
                    series.name = col_name
                    f_cols.append(series)

                elif f_type == 'ema':
                    series = df['close'].ewm(span=periods[0], adjust=False).mean().shift(shift)
                    series.name = col_name
                    f_cols.append(series)

                elif f_type == 'rsi':
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=periods[0]).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=periods[0]).mean()
                    rs = gain / loss
                    series = (100 - (100 / (1 + rs))).shift(shift)
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
        
    def add_targets(self, df):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df.empty:
                self.log_signal.emit(f"[{f_name}] Brak danych w df")
                return None
            
            t_cols = []

            for t, col_name in zip(self.config["targets"], self.config["target_names"]):
                series = df[t['base_column']].shift(t['shift'])
                series.name = col_name
                t_cols.append(series)
                
            if not t_cols:
                self.log_signal.emit(f"[{f_name}] Brak wartości docelowych do dodania")
                return None
            
            df = pd.concat([df] + t_cols, axis=1).copy()
            self.log_signal.emit(f"[{f_name}] Dodano wartości docelowe")
            return df

        except Exception as e:
            self.log_signal.emit(f"[{f_name}] Błąd: {e}")
            return None

    def dropna_and_cut(self, df, limit):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df.empty:
                self.log_signal.emit(f"[{f_name}] Brak danych w df")
                return None

            res_df = df.dropna()

            if res_df.empty:
                self.log_signal.emit(f"[{f_name}] Brak danych w df po dropna")
                return None
            
            if limit:
                res_df = res_df.tail(limit).reset_index(drop=True)
            
            len_res_df = len(res_df)

            if limit and len_res_df != limit:
                self.log_signal.emit(f"[{f_name}] Zbyt mało rekordów do ucięcia, len(df) = {len_res_df}")
                return None

            self.log_signal.emit(f"[{f_name}] Usunięto NaN i ucięto df, len(df) = {len_res_df}")
            return res_df

        except Exception as e:
            self.log_signal.emit(f"[{f_name}] Błąd: {e}")
            return None

    def join_at_end(self, df, df_append):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df_append is None:
                self.log_signal.emit(f"[{f_name}] Błąd: Brak danych wejściowych")
                return None

            if len(df) < len(df_append):
                self.log_signal.emit(f"[{f_name}] Błąd: df jest krótszy niż dane do dołączenia")
                return None

            df_aligned = df_append.copy()
            df_aligned.index = df.index[-len(df_append):]

            df_result = df.join(df_aligned)

            self.log_signal.emit(f"[{f_name}] Pomyślnie dołączono kolumny (wyrównanie do końca)")
            return df_result

        except Exception as e:
            self.log_signal.emit(f"[{f_name}] Błąd podczas operacji join_at_end: {e}")
            return None