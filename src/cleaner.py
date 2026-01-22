import inspect

class Cleaner:
    def __init__(self, config: dict, log_callback=None):
        self.config = config
        self.log = log_callback or (lambda msg: None)

    def clean_data(self, df):
        f_name = inspect.currentframe().f_code.co_name
        try:
            self.log(f"[{f_name}] Rozpoczynanie czyszczenia danych")
            if df is None or df.empty:
                self.log(f"[{f_name}] Brak danych w df")
                return None
            
            df = df.dropna().copy()
            df = df[df['datetime'].dt.dayofweek < 5]
            
            if df.empty:
                self.log(f"[{f_name}] Brak danych po usunięciu weekendów")
                return None

            if self.config['timeframe']['check_period'] is not None and self.config['timeframe']['min_count'] is not None:
                periods = df['datetime'].dt.to_period(self.config['timeframe']['check_period'])
                counts = periods.value_counts()
                current_period = periods.iloc[-1]

                invalid_periods = counts[
                    (counts < self.config['timeframe']['min_count']) & (counts.index != current_period)
                ].index

                if not invalid_periods.empty:
                    is_invalid = periods.isin(invalid_periods)
                    last_invalid_date = df[is_invalid]['datetime'].iloc[-1]
                    last_invalid_idx = df[is_invalid].index[-1]
                    
                    df = df.loc[last_invalid_idx:].iloc[1:].copy()
                    self.log(f"[{f_name}] Odcięto historię do {last_invalid_date} do indeksu {last_invalid_idx}")

            if df.empty:
                self.log(f"[{f_name}] Brak danych po usunięciu luk")
                return None

            self.log(f"[{f_name}] Pozostawiono {df.shape[0]} rekordów")            
            return df

        except Exception as e:
            self.log(f"[{f_name}] Błąd: {e}")
            return None
