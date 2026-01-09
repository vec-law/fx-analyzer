import pandas as pd


class Preprocessor:
    @staticmethod
    def create_features(container, arg_set):
        try:
            f_config = arg_set['f']['features']
            feature_cols = []
            n = 0
            
            for item in f_config:
                if item['feature'] == 'sma':
                    periods = list(range(
                        int(item['params']['start']),
                        int(item['params']['stop']),
                        int(item['params']['step'])
                    ))
                    for period in periods:
                        series = container.df[item['params']['source_column']].rolling(window=period).mean().shift(1)
                        series.name = 'feature_' + str(n)
                        feature_cols.append(series)
                        n += 1
            
            container.df = pd.concat([container.df] + feature_cols, axis=1)

            print(f"  [create_features] Dodano {n} cech(-y)")
            return True

        except KeyError as e:
            print(f"  [create_features] Błąd: brak klucza/kolumny {e}")
            return False
        except Exception as e:
            print(f"  [create_features] Nieoczekiwany błąd: {e}")
            return False

    @staticmethod
    def create_targets(container, arg_set):
        try:
            t_config = arg_set['t']['targets']
            target_cols = []
            n = 0
            
            for item in t_config:
                series = container.df[item].shift(-1) 
                series.name = 'target_' + str(n)
                target_cols.append(series)
                n += 1
                    
            container.df = pd.concat([container.df] + target_cols, axis=1)

            print(f"  [create_targets] Dodano {n} wartość(-i) docelową(-e)")
            return True

        except Exception as e:
            print(f"  [create_targets] Nieoczekiwany błąd: {e}")
            return False

    @staticmethod
    def cut_and_split_data(container, arg_set):
        try:
            params = arg_set['p']['params']
            samples_limit = params.get('samples_limit', 0)
            train_ratio = params.get('train_ratio', 0.875)

            container.df.dropna(inplace=True)
            
            limit = int(samples_limit)
            if limit > 0 and len(container.df) > limit:
                container.df = container.df.tail(limit)

            cols_to_keep = [col for col in container.df.columns 
                            if col == 'datetime' 
                            or col.startswith('feature_') 
                            or col.startswith('target_')]
            
            container.df = container.df[cols_to_keep]

            split_idx = int(len(container.df) * float(train_ratio))
            
            container.df_dict['train'] = container.df.iloc[:split_idx].copy().reset_index(drop=True)
            container.df_dict['test'] = container.df.iloc[split_idx:].copy().reset_index(drop=True)

            container.df = None
            
            print("  [cut_and_split_data] Utworzono zbiory _train i _test")
            print(f"  [cut_and_split_data] len(df_dict['train']) = {len(container.df_dict['train'])}")
            print(f"  [cut_and_split_data] len(df_dict['test']) = {len(container.df_dict['test'])}")
            return True

        except Exception as e:
            print(f"  [cut_and_split_data] Nieoczekiwany błąd: {e}")
            return False

    @staticmethod
    def scale_data(container):
        try:
            cols_to_norm = [
                col for col in container.df_dict['train'].columns
                if col.startswith('feature_') or col.startswith('target_')
            ]

            train_df = container.df_dict['train']
            train_mean = train_df[cols_to_norm].mean()
            train_std = train_df[cols_to_norm].std().replace(0, 1e-9)

            container.df_dict['stats'] = {'mean': train_mean, 'std': train_std}

            for label in ['train', 'test']:
                source_df = container.df_dict[label]
                norm_key = f"{label}_norm"
                
                container.df_dict[norm_key] = source_df.copy()
                container.df_dict[norm_key][cols_to_norm] = (source_df[cols_to_norm] - train_mean) / train_std

            print("  [scale_data] Utworzono zbiory znormalizowane (_norm)")
            return True

        except Exception as e:
            print(f"  [scale_data] Błąd skalowania: {e}")
            return False
