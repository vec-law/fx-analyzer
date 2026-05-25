from ml.loader.loader import Loader
import yfinance as yf

class YFLoader(Loader):
    def __init__(self, config: dict):
        super().__init__(config)

    def load_data(self):
        try:
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
                raise ValueError("Brak danych z YF")

        except Exception as e:
            raise Exception(f"Błąd: {e}")
