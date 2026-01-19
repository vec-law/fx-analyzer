import pandas as pd
import os
import yfinance as yf
from .utils import get_path, save_df
from .instrument import Instrument


class Loader:        
    @staticmethod
    def load_data(container):
        try:
            instrument = container.instrument
            params = container.params_set['p']['params']
            mode = params.get('mode')

            if not isinstance(instrument, Instrument) or mode not in ['auto', 'local', 'server']:
                print(f"  [load_data] Niepoprawne dane wejściowe: mode={mode}")
                return False

            path = get_path(instrument.name, instrument.interval, 'raw')
            
            if mode == 'local':
                return Loader.load_csv(container, path)
            elif mode == 'server':
                return Loader.load_yf(container)
            else:
                success = Loader.load_csv(container, path)
                if not success:
                    success = Loader.load_yf(container)
                return success

        except Exception as e:
            print(f"  [load_data] Nieoczekiwany błąd: {e}")
            return False
    
    @staticmethod
    def load_csv(container, path):
        if not os.path.exists(path):
            print(f"  [load_csv] Plik {path} nie istnieje")
            return False
        
        try:
            print(f"  [load_csv] Pobieranie danych z {path}")
            df = pd.read_csv(
                path, 
                header=None, 
                names=container.config['BROKER_COLUMNS'], 
                sep=','
            )
            
            df.columns = [col.lower() for col in df.columns]
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            df = df.drop(columns=['date', 'time'])
            df = df.sort_values('datetime').reset_index(drop=True)
            df = df[container.config['FINAL_COLUMNS']]

            container.df = df

            print(f"  [load_csv] Pobrano {container.df.shape[0]} rekordów")
            
            save_df(container, 'raw', 'csv')
            return True

        except Exception as e:
            print(f"  [load_csv] Błąd przy wczytywaniu pliku {path}: {e}")
            return False

    @staticmethod
    def load_yf(container):
        try:
            inst = container.instrument
            print(f"  [load_yf] Pobieranie danych z serwera YF")
            
            df = yf.download(
                tickers=inst.ticker, 
                period=inst.history_range, 
                interval=inst.interval, 
                auto_adjust=False,
                progress=False
            )
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if df.empty:
                print("  [load_yf] Brak danych na serwerze YF")
                return False

            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]
            df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            df = df[container.config['FINAL_COLUMNS']]

            container.df = df

            print(f"  [load_yf] Pobrano {container.df.shape[0]} rekordów")

            save_df(container, 'raw', 'yf')
            return True

        except Exception as e:
            print(f"  [load_yf] Błąd przy pobieraniu danych serwera YF: {e}")
            return False
