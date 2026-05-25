def clean_data(df, config):
    try:
        if df is None or df.empty:
            return None
        
        df = df.dropna().copy()
        df = df[df['datetime'].dt.dayofweek < 5]
        
        if df.empty:
            return None

        if config['timeframe_check_period'] is not None and config['timeframe_min_count'] is not None:
            periods = df['datetime'].dt.to_period(config['timeframe_check_period'])
            counts = periods.value_counts()
            current_period = periods.iloc[-1]

            invalid_periods = counts[
                (counts < config['timeframe_min_count']) & (counts.index != current_period)
            ].index

            if not invalid_periods.empty:
                is_invalid = periods.isin(invalid_periods)
                last_invalid_idx = df[is_invalid].index[-1]
                
                df = df.loc[last_invalid_idx:].iloc[1:].copy()

        if df.empty:
            return None

        return df

    except Exception as e:
        raise Exception(f"Błąd: {e}")
