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

        print(f"  [create_features] {instrument.name}: Dodano {n} cech(-y)")

        return True

    except KeyError as e:
        raise KeyError(f"Błąd: brak klucza/kolumny {e}")

    except Exception as e:
        print(f"Przerwano: nieoczekiwany błąd: {e}")
        return False

def create_targets(instrument: Instrument, target_list):
    try:
        feature_cols = []
        n = 0
        for item in target_list:
            print(item)
        #     for f in item['features']:
        #         if f['feature'] == 'sma':
        #             periods = list(range(
        #                 int(f['params']['start']),
        #                 int(f['params']['stop']),
        #                 int(f['params']['step'])
        #             ))
        #             for period in periods:
        #                 feature_cols.append(pd.Series(
        #                     data = instrument.df[f['params']['source_column']].rolling(window=period).mean().shift(1),
        #                     name = 'feature_' + str(n)
        #                 ))
        #                 n += 1
        # instrument.df = pd.concat([instrument.df] + feature_cols, axis=1)

        # print(f"  [create_features] {instrument.name}: Dodano {n} cech")

        return True

    except KeyError as e:
        raise KeyError(f"Błąd: brak klucza/kolumny {e}")
    
    except Exception as e:
        print(f"Przerwano: nieoczekiwany błąd: {e}")
        return False

def clean_data(df):
    pass

def scale_data(df):
    pass