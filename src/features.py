import pandas as pd
from src.utils import save_df

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
