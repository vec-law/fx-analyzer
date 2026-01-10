import pandas as pd
import torch


class Preprocessor:
    @staticmethod
    def create_features(container):
        try:
            f_config = container.params_set['f']['features']
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
            
            if n == 0:
                print("  [create_features] Błąd: Wygenerowano 0 cech. Sprawdź konfigurację")
                return False

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
    def create_targets(container):
        try:
            t_config = container.params_set['t']['targets']
            target_cols = []
            n = 0
            
            for item in t_config:
                series = container.df[item].shift(-1) 
                series.name = 'target_' + str(n)
                target_cols.append(series)
                n += 1

            if n == 0:
                print("  [create_targets] Błąd: Nie zdefiniowano żadnych wartości docelowych")
                return False
                    
            container.df = pd.concat([container.df] + target_cols, axis=1)

            print(f"  [create_targets] Dodano {n} wartość(-i) docelową(-e)")
            return True

        except Exception as e:
            print(f"  [create_targets] Nieoczekiwany błąd: {e}")
            return False

    @staticmethod
    def cut_and_split_data(container):
        try:
            params = container.params_set['p']['params']
            samples_limit = params.get('samples_limit', 0)
            train_ratio = params.get('train_ratio', 0.875)

            if not container.df.empty:
                if not container.save_df_to_parquet():
                    return False
                print(container.df)
            else:
                print("  [cut_and_split_data] Błąd: Brak danych w df")
                return False

            container.df.dropna(inplace=True)
            if container.df.empty:
                print("  [cut_and_split_data] Błąd: Brak danych po usunięciu wartości NaN (za krótka historia?)")
                return False
            
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
            if not Preprocessor.create_stats(container.df_dict):
                return False
            if not Preprocessor.scale_with_stats(container.df_dict):
                return False

            print("  [scale_data] Utworzono zbiory znormalizowane (_norm)")
            return True

        except Exception as e:
            print(f"  [scale_data] Błąd skalowania: {e}")
            return False
        
    @staticmethod
    def create_stats(df_dict):
        try:
            cols_to_norm = [
                col for col in df_dict['train'].columns
                if col.startswith('feature_') or col.startswith('target_')
            ]

            if 'train' not in df_dict:
                print(f"  [create_stats] Błąd: Brak kolumny train df_dict")
                return False

            train_df = df_dict['train']
            train_mean = train_df[cols_to_norm].mean()
            train_std = train_df[cols_to_norm].std().replace(0, 1e-9)

            df_dict['stats'] = {'mean': train_mean, 'std': train_std}

            print("  [create_stats] Utworzono statystyki")
            return True

        except Exception as e:
            print(f"  [create_stats] Błąd skalowania: {e}")
            return False
    
    @staticmethod
    def scale_with_stats(df_dict):
        try:
            if 'train' not in df_dict or 'test' not in df_dict:
                print(f"  [scale_with_stats] Błąd: Brak kolumn train lub test w df_dict")
                return False
            
            cols_in_train = [
                col for col in df_dict['train'].columns
                if col.startswith('feature_') or col.startswith('target_')
            ]

            has_features = any(col.startswith('feature_') for col in cols_in_train)
            has_targets = any(col.startswith('target_') for col in cols_in_train)

            if not has_features or not has_targets:
                print(f"  [scale_with_stats] Błąd: Brak kolumn feature_ lub target_ w df_dict['train]")
                return False
            
            missing_in_test = [
                col for col in cols_in_train 
                if col not in df_dict['test'].columns
            ]

            if missing_in_test:
                print(f"  [scale_with_stats] Błąd: Różne kolumy feature_ lub target_ w df_dict['train] i df_dict['test']")
                return False
            
            if 'stats' not in df_dict:
                print("  [scale_with_stats] Błąd: Brak 'stats' w df_dict")
                return False
                
            if 'mean' not in df_dict['stats'] or 'std' not in df_dict['stats']:
                print("  [scale_with_stats] Błąd: 'stats' musi zawierać mean i std")
                return False
            
            stats_mean_cols = df_dict['stats']['mean'].index
            stats_std_cols = df_dict['stats']['std'].index

            missing_in_stats = [
                col for col in cols_in_train 
                if col not in stats_mean_cols or col not in stats_std_cols
            ]

            if missing_in_stats:
                print(f"  [scale_with_stats] Błąd: Brak kolumn z df_dict['train'] w df_dict['stats']['mean'] lub df_dict['stats']['std']")
                return False

            for label in ['train', 'test']:
                source_df = df_dict[label]
                norm_key = f"{label}_norm"
                
                df_dict[norm_key] = source_df.copy()
                
                diff = source_df[cols_in_train] - df_dict['stats']['mean']
                df_dict[norm_key][cols_in_train] = diff / df_dict['stats']['std']

                df_dict[norm_key][cols_in_train] = df_dict[norm_key][cols_in_train].fillna(0.0)

            print("  [scale_with_stats] Utworzono zbiory znormalizowane (_norm)")
            return True
        
        except Exception as e:
            print(f"  [scale_with_stats] Błąd podczas skalowania danych: {e}")
            return False

    @staticmethod
    def create_tensors(container):
        try:
            if 'train_norm' not in container.df_dict or 'test_norm' not in container.df_dict:
                print("  [create_tensors] Błąd: Brak zbiorów znormalizowanych (_norm)")
                return False

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            container.ten_dict['device'] = device
            
            cols_to_ten_x = [col for col in container.df_dict['train_norm'].columns
                             if col.startswith('feature_')]
            cols_to_ten_y = [col for col in container.df_dict['train_norm'].columns
                             if col.startswith('target_')]
            
            if not cols_to_ten_x or not cols_to_ten_y:
                print("  [create_tensors] Błąd: Brak kolumn feature_ lub target_ w train_norm")
                return False

            container.ten_dict['x_train_norm'] = torch.tensor(
                container.df_dict['train_norm'][cols_to_ten_x].values, 
                dtype=torch.float32
            ).to(device)
            
            container.ten_dict['y_train_norm'] = torch.tensor(
                container.df_dict['train_norm'][cols_to_ten_y].values, 
                dtype=torch.float32
            ).to(device)

            container.ten_dict['x_test_norm'] = torch.tensor(
                container.df_dict['test_norm'][cols_to_ten_x].values, 
                dtype=torch.float32
            ).to(device)
            
            container.ten_dict['y_test_norm'] = torch.tensor(
                container.df_dict['test_norm'][cols_to_ten_y].values, 
                dtype=torch.float32
            ).to(device)

            print(f"  [create_tensors] Utworzono tensory w ten_dict")
            print(f"  [create_tensors] ten_dict['x_train_norm']: {container.ten_dict['x_train_norm'].shape}")
            print(f"  [create_tensors] ten_dict['y_train_norm']: {container.ten_dict['y_train_norm'].shape}")
            print(f"  [create_tensors] ten_dict['x_test_norm']: {container.ten_dict['x_test_norm'].shape}")
            print(f"  [create_tensors] ten_dict['y_test_norm']: {container.ten_dict['y_test_norm'].shape}")
            return True

        except Exception as e:
            print(f"   [create_tensors] Błąd: {e}")
            return False

