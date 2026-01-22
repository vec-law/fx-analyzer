import inspect
import yfinance as yf

class Loader:
    def __init__(self, config: dict, log_callback=None):
        self.config = config
        self.log = log_callback or (lambda msg: None)

    def load_data(self):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if self.config['data_source'] == 'YF':
                self.log(f"[{f_name}] Rozpoczynanie ładowania danych z YF")

                df = yf.download(
                    tickers=self.config['instrument']['ticker'], 
                    period=self.config['timeframe']['range'], 
                    interval=self.config['timeframe']['name'], 
                    auto_adjust=False,
                    progress=False,
                    multi_level_index=False
                )

                if df is not None and not df.empty:
                    df.columns = [col.lower() for col in df.columns]
                    df = df.reset_index()
                    df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
                    selected_columns = ['datetime'] + self.config['target_types']
                    df = df[selected_columns]

                self.log(f"[{f_name}] Załadowano {len(df)} rekordów")
                return df
            else:
                self.log(f"[{f_name}] Nieobsługiwane źródło danych: {self.config['data_source']}")
                return None

        except Exception as e:
            self.log(f"[{f_name}] Błąd: {e}")
            return None
