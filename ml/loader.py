import inspect
import yfinance as yf

class Loader:
    def __init__(self, config: dict):
        self.config = config

    def load_data(self):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if self.config['data_source_name'] == 'YF':
                df = yf.download(
                    tickers=self.config['instrument_ticker'], 
                    period=self.config['timeframe_range'], 
                    interval=self.config['timeframe_name'], 
                    auto_adjust=False,
                    progress=False,
                    multi_level_index=False
                )

                if df is not None and not df.empty:
                    df.columns = [col.lower() for col in df.columns]
                    df = df.reset_index()
                    df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
                    selected_columns = ['datetime'] + self.config['base_columns']
                    df = df[selected_columns]

                return df
            else:
                return None

        except Exception as e:
            raise Exception(f"[{f_name}] Błąd: {e}")
