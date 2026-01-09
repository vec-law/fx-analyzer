from src.utils import save_df
from src.utils import save_df


class Cleaner:
    @staticmethod
    def clean_data(container, arg_set):
        try:
            params = arg_set['p']['params']
            repair_gaps = params.get('repair_gaps')
            if container.df is None or container.df.empty:
                return False
            
            container.df = container.df.dropna().copy()
            container.df = container.df[container.df['datetime'].dt.dayofweek < 5]
            
            if container.df.empty:
                print("  [clean_data] Brak danych po usunięciu weekendów")
                return False

            if repair_gaps:
                periods = container.df['datetime'].dt.to_period(container.instrument.check_period)
                counts = periods.value_counts()
                current_period = periods.iloc[-1]

                invalid_periods = counts[
                    (counts < container.instrument.min_count) & (counts.index != current_period)
                ].index

                if not invalid_periods.empty:
                    is_invalid = periods.isin(invalid_periods)
                    last_invalid_date = container.df[is_invalid]['datetime'].iloc[-1]
                    last_invalid_idx = container.df[is_invalid].index[-1]
                    
                    container.df = container.df.loc[last_invalid_idx:].iloc[1:].copy()
                    print(f"  [clean_data] Odcięto historię do {last_invalid_date}")

            if container.df.empty:
                print("  [clean_data] Brak danych po usunięciu luk")
                return False

            save_df(container, 'raw', 'crd')
            print(f"  [clean_data] Pozostawiono {container.df.shape[0]} rekordów")
            
            return True

        except Exception as e:
            print(f"  [clean_data] Nieoczekiwany błąd: {e}")
            return False
