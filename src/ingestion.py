import pandas as pd
import os
import yfinance as yf
from src.utils import get_path, save_df

MIN_WORKDAYS = 18
SUPPORTED_INTERVALS = ['1d']

def check_and_repair_data(df):
    df = df.dropna().copy()
    if df.empty:
        print(f"  [check_and_repair_data] df nie zawiera użytecznych danych")
        return None

    interval = df['interval'].iloc[0]
    instrument = df['instrument'].iloc[0]

    if interval == '1d':        
        df = df[df['datetime'].dt.dayofweek < 5].copy()
        if df.empty:
            print(f"  [check_and_repair_data] df nie zawiera danych z dni roboczych")
            return None

        month_periods = df['datetime'].dt.to_period('M')
        workdays_number = month_periods.value_counts()
        df['workdays_per_month'] = month_periods.map(workdays_number)

        current_month = df['datetime'].iloc[-1].to_period('M')
        too_few_workdays = (df['workdays_per_month'] < MIN_WORKDAYS) & (month_periods != current_month)

        last_invalid_idx = df[too_few_workdays].index.max()

        if pd.notna(last_invalid_idx):
            df_err = df[too_few_workdays].copy()
            if not df_err.empty:
                save_df(df_err, 'raw', f"{instrument}_{interval}_err.csv")
                print(f"  [check_and_repair_data] Znaleziono nieprawidłowe wiersze w df")
                print(f"  [save_df] Nieprawidłowe wiersze zapisano w pliku data/raw/{instrument}_{interval}_err.csv")
            
            if last_invalid_idx >= df.index.max():
                print(f"  [check_and_repair_data] df nie zawiera pawidłowych danych z dni roboczych")
                return None
            
            df = df.loc[last_invalid_idx + 1:].copy()
            
        if len(df) < MIN_WORKDAYS:
            print(f"  [check_and_repair_data] df nie zawiera odpowiedniej liczby danych z dni roboczych")
            return None
        
        print(f"  [check_and_repair_data] Pozostawiono w df ostatnich {len(df)} prawidłowych wierszy")

        save_df(df, 'raw', f"{instrument}_{interval}_crd.csv")
        print(f"  [save_df] df zapisano w pliku data/raw/{instrument}_{interval}_crd.csv")

    return df

def load_csv(target_column, path):
    try:
        all_columns = ['date', 'time', 'open', 'high', 'low', 'close', 'vol']

        print(f"  [load_csv] Pobieranie danych z pliku {path}")
        
        df = pd.read_csv(path, header=None, names=all_columns)
         
        df = pd.DataFrame({
            'original_idx': df.index,
            'datetime': pd.to_datetime(
                df['date'] + ' ' + df['time'],
                format='%Y.%m.%d %H:%M',
                errors='coerce'
            ),
            'target': df[target_column.lower()].values
        })

        df = df.dropna(subset=['datetime']).copy()

        print(f"  [load_csv] Pobrano {len(df)} wierszy")
        
        return df
    
    except Exception as e:
        print(f"  [load_csv] Błąd: {e}")
        return None


def load_yf(instrument, interval, target_column):
    try:
        ticker = f"{instrument}=X"
        print(f"  [load_yf] Pobieranie danych z YF")

        df = yf.download(ticker, period="max", interval=interval, auto_adjust=True, progress=False)

        if df.empty:
            print(f"  [load_yf] Brak danych dla {instrument}")
            return None

        df_temp = df[target_column.capitalize()].copy()
        if isinstance(df_temp, pd.DataFrame): df_temp = df_temp.iloc[:, 0]
        df_temp = df_temp.reset_index()

        df = pd.DataFrame({
            'original_idx': df_temp.index,
            'datetime': df_temp.iloc[:, 0].dt.tz_localize(None),
            'target': df_temp.iloc[:, 1].values
        })

        print(f"  [load_yf] Pobrano {len(df)} wierszy")

        file_name = f"{instrument}_{interval}_yf.csv"
        save_df(df, 'raw', file_name)
        print(f"  [save_df] Dane zapisano w pliku data/raw/{file_name}")

        return df

    except Exception as e:
        print(f"  [load_yf] Błąd: {e}")
        return None

def load_data(instrument, interval, target_column, samples_limit=None):
    if any(condition is None for condition in [instrument, interval, target_column]):
        print("  [load_data]: Nieprawidłowe parametry")
        return None
    
    if interval not in SUPPORTED_INTERVALS:
        print("  [load_data]: Nieobsługiwany interwał")
        return None

    path = get_path('raw', f"{instrument}_{interval}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        df = load_yf(instrument, interval, target_column)
    else:
        df = load_csv(target_column, path)

    if df is None or df.empty: return None

    df['instrument'] = instrument
    df['interval'] = interval

    if (df := check_and_repair_data(df)) is None: return None

    if samples_limit:
        if not isinstance(samples_limit, int) or samples_limit <= 0:
            print("  [load_data]: Nieprawidłowa liczba samples_limit")
            return None

        if samples_limit < len(df):
            df = df.tail(samples_limit)
            print(f"  [load_data] Pozostawiono w df ostatnich {samples_limit} wierszy")
        else:
            print(f"  [load_data] Liczba samples_limit jest większa od ilości wierszy df, nie zmieniono liczby wierszy")

    ordered_columns = ['original_idx', 'datetime', 'instrument', 'interval', 'workdays_per_month', 'target']
    df = df[ordered_columns].copy()

    save_df(df, 'raw', f"{instrument}_{interval}_ld.csv")
    print(f"  [save_df] Wiersze df zapisano w pliku data/raw/{instrument}_{interval}_ld.csv")

    return df
