from .instrument import Instrument
import pandas as pd
import os
import yfinance as yf
from src.utils import get_path, save_df

def check_and_repair_data(instrument, repair_gaps):
    if instrument.df is None or instrument.df.empty:
        return False
    
    instrument.df = instrument.df.dropna().copy()
    instrument.df = instrument.df[instrument.df['datetime'].dt.dayofweek < 5]
    
    if instrument.df.empty:
        print(f"  [check_and_repair_data] Brak danych po usunięciu wekendów")
        return False

    if repair_gaps:
        periods = instrument.df['datetime'].dt.to_period(instrument.check_period)
        counts = periods.value_counts()
        current_period = periods.iloc[-1]

        invalid_periods = counts[(counts < instrument.min_count) & (counts.index != current_period)].index

        if not invalid_periods.empty:
            is_invalid = periods.isin(invalid_periods)
            last_invalid_date = instrument.df[is_invalid]['datetime'].iloc[-1]
            last_invalid_idx = instrument.df[is_invalid].index[-1]
            
            instrument.df = instrument.df.loc[last_invalid_idx:].iloc[1:].copy()
            print(f"  [check_and_repair_data] Odcięto historię do {last_invalid_date}, do indeksu {last_invalid_idx}")

    if instrument.df.empty:
        print(f"  [check_and_repair_data] Brak danych po usunięciu luk")
        return False

    save_df(instrument, 'raw', 'crd')
    print(f"  [check_and_repair_data] Pozostawiono {instrument.df.shape[0]} rekordów")
    
    return True

def load_csv(instrument, config, path):
    if not os.path.exists(path):
        print(f"  [load_csv] Plik {path} nie istnieje")
        return False
    
    if not config or 'BROKER_COLUMNS' not in config:
        raise KeyError("Błąd: instrument_data jest pusty lub brak BROKER_COLUMNS")

    if 'FINAL_COLUMNS' not in config:
        raise KeyError("Błąd: brak FINAL_COLUMNS w instrument_data")
    
    try:
        print(f"  [load_yf] Pobieranie danych z {path}")
        df = pd.read_csv(
            path, 
            header=None, 
            names=config['BROKER_COLUMNS'], 
            sep=','
        )
        
        df.columns = [col.lower() for col in df.columns]
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.drop(columns=['date', 'time'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        df = df[config['FINAL_COLUMNS']]

        instrument.df = df
        instrument.source = 'local'

        print(f"  [load_csv] Pobrano {instrument.df.shape[0]} rekordów")
        save_df(instrument, 'raw', 'csv')

        return True

    except Exception as e:
        print(f"Błąd przy wczytywaniu pliku {path}: {e}")
        return False


def load_yf(instrument, config):
    if 'FINAL_COLUMNS' not in config:
        raise KeyError(f"  [load_yf] Brak klucza FINAL_COLUMNS w konfiguracji")

    try:
        print(f"  [load_yf] Pobieranie danych z serwera")
        df = yf.download(
            tickers=instrument.ticker, 
            period=instrument.history_range, 
            interval=instrument.interval, 
            auto_adjust=False,
            progress=False
        )
        df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            print(f"  [load_yf] Brak danych na serwerze")
            return False

        df = df.reset_index()

        df.columns = [col.lower() for col in df.columns]
        df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
        
        df = df[config['FINAL_COLUMNS']]

        instrument.df = df
        instrument.source = 'server'

        save_df(instrument, 'raw', 'yf')
        print(f"  [load_yf] Pobrano {instrument.df.shape[0]} rekordów")

        return True

    except Exception as e:
        print(f"  [load_yf] Błąd przy pobieraniu danych serwera")
        return False

def load_data(instrument, mode, repair_gaps, config):
    if not isinstance(instrument, Instrument):
        raise ValueError("Błąd: Parametr 'instrument' musi być obiektem klasy Instrument.")
    
    if mode not in ['auto', 'local', 'server']:
        raise ValueError(f"Błąd: Nieobsługiwany tryb mode: '{mode}'. Dozwolone tryby to: 'auto', 'local', 'server'")

    if not isinstance(repair_gaps, bool):
        raise ValueError(f"Błąd: Parametr repair_gaps musi być typem bool (True/False).")
        
    path = get_path(instrument.name, instrument.interval, 'raw')
    success = False

    if mode == 'local':
        success = load_csv(instrument, config, path)
    elif mode == 'server':
        success = load_yf(instrument, config)
    else:
        success = load_csv(instrument, config, path) or load_yf(instrument, config)

    if not success:
        return False

    if not check_and_repair_data(instrument, repair_gaps):
        return False
        
    print(f"  [load_data] Pobrano {instrument.df.shape[0]} rekordów")
    return True
