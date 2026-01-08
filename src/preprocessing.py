import pandas as pd
from src.instrument import Instrument

def create_features(instrument: Instrument, f):
    try:
        feature_cols = []
        n = 0
        for item in f:
            if item['feature'] == 'sma':
                periods = list(range(
                    int(item['params']['start']),
                    int(item['params']['stop']),
                    int(item['params']['step'])
                ))
                for period in periods:
                    feature_cols.append(pd.Series(
                        data = instrument.df[item['params']['source_column']].rolling(window=period).mean().shift(1),
                        name = 'feature_' + str(n)
                    ))
                    n += 1
                
        instrument.df = pd.concat([instrument.df] + feature_cols, axis=1)

        print(f"  [create_features] Dodano {n} cech(-y)")

        return True

    except KeyError as e:
        raise KeyError(f"Błąd: brak klucza/kolumny {e}")

    except Exception as e:
        print(f"Przerwano: nieoczekiwany błąd: {e}")
        return False

def create_targets(instrument: Instrument, t):
    try:
        target_cols = []
        n = 0
        for item in t:
            series = instrument.df[item].shift(-1) 
            series.name = 'target_' + str(n)
            target_cols.append(series)
            n += 1
                
        instrument.df = pd.concat([instrument.df] + target_cols, axis=1)

        print(f"  [create_targets] Dodano {n} wartość(-i) docelową(-e)")

        return True

    except KeyError as e:
        raise KeyError(f"Błąd: brak klucza/kolumny {e}")

    except Exception as e:
        print(f"Przerwano: nieoczekiwany błąd: {e}")
        return False

def clean_and_split_data(instrument: Instrument, samples_limit, train_ratio=0.875):
    try:
        instrument.df.dropna(inplace=True)
        
        limit = int(samples_limit)
        if limit > 0 and len(instrument.df) > limit:
            instrument.df = instrument.df.tail(limit)

        ratio = float(train_ratio)
        if ratio <= 0 or ratio > 1:
            ratio = 0.875

        cols_to_keep = [col for col in instrument.df.columns 
            if col == 'datetime' 
            or col.startswith('feature_') 
            or col.startswith('target_')]
        
        instrument.df = instrument.df[cols_to_keep]

        split_idx = int(len(instrument.df) * ratio)
        instrument.df_dict['train'] = instrument.df.iloc[:split_idx].copy().reset_index(drop=True)
        instrument.df_dict['test'] = instrument.df.iloc[split_idx:].copy().reset_index(drop=True)
        instrument.df = pd.DataFrame()
        
        print(f"  [clean_and_split_data] Ustworzono df_dict['train'], df_dict['test']")
        print(f"  [clean_and_split_data] len(df_dict['train']) = {len(instrument.df_dict['train'])}")
        print(f"  [clean_and_split_data] len(df_dict['test']) = {len(instrument.df_dict['test'])}")
        return True

    except (ValueError, TypeError) as e:
        print(f"Błąd: samples_limit musi być liczbą: {e}")
        return False
    except Exception as e:
        print(f"Przerwano: błąd podczas czyszczenia danych: {e}")
        return False

def scale_data(df):
    pass
