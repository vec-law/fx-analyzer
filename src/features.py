import pandas as pd
from src.utils import save_df

def prepare_features(df, indicators, train_ratio):
    if indicators is None or not isinstance(train_ratio, float):
        print("  [prepare_features] Podano nieprawidłowe parametry")
        return None

    all_periods = [period for periods in indicators.values() for period in periods]
    max_period = max(all_periods) if all_periods else 0

    if train_ratio > 0 and train_ratio < 1 and int(len(df) * train_ratio) < max_period + 100:
        print("  [prepare_features] Nieprawidłowy parametr train_ratio")
        return None

    if len(df) <= max_period:
        print("  [prepare_features] Zbyt mało wierszy w df")
        return None
   
    features_columns = []
    unsupported_indicators = set()
    i = 1
    for key, periods in indicators.items():
        for period in periods:
            
            if key == 'sma':
                features_columns.append(pd.Series(
                    data = df['target'].rolling(window=period).mean().shift(1),
                    name = 'feature_' + str(i)
                ))
                i += 1

            elif key == 'med':
                features_columns.append(pd.Series(
                    data = df['target'].rolling(window=period).median().shift(1),
                    name = 'feature_' + str(i)
                ))
                i += 1

            else:
                unsupported_indicators.add(key)
    if unsupported_indicators:
        print(f"  [prepare_features] Wskaźniki {unsupported_indicators} nie są dostępne, nie dodano ich jako cech")

    if not features_columns:
        print(f"  [prepare_features] Nie dodano żadnych cech")
        return None
    
    df = pd.concat([df] + features_columns, axis=1)
    df = df.iloc[max_period:].dropna().reset_index(drop=True)

    df_dict = {}

    instrument = df['instrument'].iloc[0]
    interval = df['interval'].iloc[0]

    split_idx = int(len(df) * train_ratio)
    df_dict['train'] = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_dict['test'] = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"  [prepare_features] Utworzono df_dict")
    print(f"  [prepare_features] len(df_dict['train']) = {len(df_dict['train'])})")
    print(f"  [prepare_features] len(df_dict['test']) = {len(df_dict['test'])})")

    train_file = f"{instrument}_{interval}_train.csv"
    test_file = f"{instrument}_{interval}_test.csv"

    save_df(df_dict['train'], 'pre', train_file)
    print(f"  [save_df] Zapisano df_dict['train'] do pliku data/pre/{train_file}")

    save_df(df_dict['test'], 'pre', test_file)
    print(f"  [save_df] Zapisano df_dict['test'] do pliku data/pre/{test_file}")

    return df_dict

def normalize(df_dict):
    all_cols = df_dict['train'].columns
    cols_to_norm = [c for c in all_cols if c == 'target' or c.startswith('feature_')]

    near_zero = 1e-9
    train_mean = df_dict['train'][cols_to_norm].mean()
    train_std = df_dict['train'][cols_to_norm].std()

    df_dict['stats'] = {
        'mean': train_mean,
        'std': train_std
    }

    df_dict['train_norm'] = (df_dict['train'][cols_to_norm] - train_mean) / (train_std + near_zero)
    df_dict['test_norm'] = (df_dict['test'][cols_to_norm] - train_mean) / (train_std + near_zero)

    instrument = df_dict['train']['instrument'].iloc[0]
    interval = df_dict['train']['interval'].iloc[0]

    train_file = f"{instrument}_{interval}_train_norm.csv"
    test_file = f"{instrument}_{interval}_test_norm.csv"
    stats_file = f"{instrument}_{interval}_stats.csv"

    save_df(df_dict['train_norm'], 'norm', train_file)
    save_df(df_dict['test_norm'], 'norm', test_file)

    stats_df = pd.DataFrame({
        'feature': cols_to_norm,
        'mean': train_mean.values,
        'std': train_std.values
    })
    save_df(stats_df, 'stats', stats_file)

    print(f"  [normalize] Znormalizowano {len(cols_to_norm)} kolumn")
    print(f"  [save_df] Zapisano train_norm do data/norm/{train_file}")
    print(f"  [save_df] Zapisano test_norm do data/norm/{test_file}")
    print(f"  [save_df] Zapisano stats do data/stats/{stats_file}")

    return df_dict
