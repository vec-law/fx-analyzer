import inspect
import pandas as pd

class DataExtractor:
    def __init__(self, config: dict):
        self.config = config

    def add_calculated_columns(self, df):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df.empty:
                return None
            
            c_cols = []
            
            for col_name in self.config["calculated_columns"]:
                if col_name == 'typical':
                    series = (df['high'] + df['low'] + df['close']) / 3
                    series.name = col_name
                    c_cols.append(series)

                elif col_name == 'median':
                    series = (df['high'] + df['low']) / 2
                    series.name = col_name
                    c_cols.append(series)

                elif col_name == 'weighted':
                    series = (df['high'] + df['low'] + 2 * df['close']) / 4
                    series.name = col_name
                    c_cols.append(series)

                elif col_name == 'ohlc':
                    series = (df['open'] + df['high'] + df['low'] + df['close']) / 4
                    series.name = col_name
                    c_cols.append(series)

                else:
                    raise ValueError(f"Brak wzoru dla obliczanej kolumny: {col_name}")
            
            df = pd.concat([df] + c_cols, axis=1).copy()
            return df

        except Exception as e:
            raise Exception(f"[{f_name}] Błąd: {e}")

    def add_features(self, df):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df.empty:
                return None
            
            feautures = []
            for feature in self.config["features"]:
                f_str = feature.strip()
                if not f_str:
                    continue
                parts = f_str.split(":")
                if len(parts) != 3:
                    raise ValueError(f"Format feature: typ:parametry:shift")
                feautures.append({
                    "feature_type": parts[0].strip(),
                    "feature_periods": [int(p.strip()) for p in parts[1].split("-") if p.strip()],
                    "shift": int(parts[2].strip())
                })
            
            f_cols = []
            
            for f, col_name in zip(feautures, self.config["features"]):
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

                elif f_type == 'pct':
                    series = df['close'].pct_change(periods[0]).shift(shift)
                    series.name = col_name
                    f_cols.append(series)

                else:
                    raise ValueError(f"Brak wzoru dla cechy: {f_type}")
            
            df = pd.concat([df] + f_cols, axis=1).copy()
            return df

        except Exception as e:
            raise Exception(f"[{f_name}] Błąd: {e}")
        
    def add_targets(self, df):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df.empty:
                return None
            
            targets = []
            for target in self.config["targets"]:
                target_str = target.strip()
                if not target_str:
                    continue
                parts = target_str.split(":")
                if len(parts) != 2:
                    raise ValueError("Format targetu to 'nazwa_kolumny:shift'")
                targets.append({
                    "column": parts[0].strip(),
                    "shift": int(parts[1].strip())
                })
            
            t_cols = []

            for t, col_name in zip(targets, self.config["targets"]):
                series = df[t['column']].shift(t['shift'])
                series.name = col_name
                t_cols.append(series)
                
            if not t_cols:
                return None
            
            df = pd.concat([df] + t_cols, axis=1).copy()
            return df

        except Exception as e:
            raise Exception(f"[{f_name}] Błąd: {e}")
        
    def dropna_and_cut(self, df, limit):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df.empty:
                return None

            res_df = df.dropna()

            if res_df.empty:
                return None
            
            if limit:
                res_df = res_df.tail(limit).reset_index(drop=True)
            
            len_res_df = len(res_df)

            if limit and len_res_df != limit:
                return None

            return res_df

        except Exception as e:
            raise Exception(f"[{f_name}] Błąd: {e}")

    def join_at_end(self, df, df_append):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df_append is None:
                return None

            if len(df) < len(df_append):
                return None

            df_aligned = df_append.copy()
            df_aligned.index = df.index[-len(df_append):]
            df_result = df.join(df_aligned)
            return df_result

        except Exception as e:
            raise Exception(f"[{f_name}] Błąd: {e}")
        
    def add_diff(self, df):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df.empty:
                return None
            
            cols_to_diff = self.config["targets"]
            diff_df = df[cols_to_diff].diff().add_suffix('_diff')
            diff_df = diff_df.fillna(0)
            df = df.join(diff_df)
            return df

        except Exception as e:
            raise Exception(f"[{f_name}] Błąd: {e}")
