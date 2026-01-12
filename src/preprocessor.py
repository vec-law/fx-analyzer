import pandas as pd
import torch
from src.container import Container

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

            container.df = pd.concat([container.df] + feature_cols, axis=1).copy()

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
                series = container.df[str(item['params']['price'])].shift(int(item['params']['day'])) 
                series.name = 'target_' + str(n)
                target_cols.append(series)
                n += 1

            if n == 0:
                print("  [create_targets] Błąd: Nie zdefiniowano żadnych wartości docelowych")
                return False
                    
            container.df = pd.concat([container.df] + target_cols, axis=1).copy()

            print(f"  [create_targets] Dodano {n} wartość(-i) docelową(-e)")
            return True

        except Exception as e:
            print(f"  [create_targets] Nieoczekiwany błąd: {e}")
            return False

    @staticmethod
    def cut_and_split_data(container):
        try:
            if not container.df.empty:
                if not container.save_df_to_parquet():
                    return False
            else:
                print("  [cut_and_split_data] Błąd: Brak danych w df")
                return False
                
            cols_to_keep = [col for col in container.df.columns 
                            if col == 'datetime' 
                            or col.startswith('feature_') 
                            or col.startswith('target_')]
            
            container.df = container.df[cols_to_keep]

            container.df.dropna(inplace=True)
            if container.df.empty:
                print("  [split_data] Błąd: Brak danych po usunięciu wartości NaN")
                return False

            if not Preprocessor.split_data(container):
                return False
            
            container.df = None

            return True

        except Exception as e:
            print(f"  [cut_and_split_data] Nieoczekiwany błąd: {e}")
            return False
        
    @staticmethod
    def split_data(container):
        try:
            params = container.params_set['p']['params']
            samples_limit = params.get('samples_limit', 0)
            train_ratio = params.get('train_ratio', 0.875)
            
            limit = int(samples_limit)
            if limit > 0 and len(container.df) > limit:
                container.df = container.df.tail(limit)

            split_idx = int(len(container.df) * float(train_ratio))
            
            container.df_dict['train'] = container.df.iloc[:split_idx].copy().reset_index(drop=True)
            container.df_dict['test'] = container.df.iloc[split_idx:].copy().reset_index(drop=True)
            
            print("  [split_data] Utworzono zbiory _train i _test")
            return True

        except Exception as e:
            print(f"  [split_data] Nieoczekiwany błąd: {e}")
            return False

    @staticmethod
    def scale_data(container: Container):
        try:
            if not Preprocessor.create_stats(container.df_dict):
                return False
            if not Preprocessor.scale_with_stats(container):
                return False

            print("  [scale_data] Utworzono zbiory znormalizowane (_norm)")
            return True

        except Exception as e:
            print(f"  [scale_data] Błąd skalowania: {e}")
            return False
        
    @staticmethod
    def create_stats(df_dict):
        try:
            if 'train' not in df_dict:
                print(f"  [create_stats] Błąd: Brak klucza 'train' w df_dict")
                return False

            cols_to_norm = [
                col for col in df_dict['train'].columns
                if col.startswith(('feature_', 'target_'))
            ]

            train_df = df_dict['train']
            train_mean = train_df[cols_to_norm].mean()
            train_std = train_df[cols_to_norm].std().replace(0, 1e-9)

            df_dict['stats'] = {'mean': train_mean, 'std': train_std}

            print("  [create_stats] Utworzono statystyki")
            return True

        except Exception as e:
            print(f"  [create_stats] Błąd: {e}")
            return False

    @staticmethod
    def scale_with_stats(container: Container, col_names=('feature_', 'target_')):
        try:
            df_dict = container.df_dict
            
            if 'stats' not in df_dict:
                print("  [scale_with_stats] Błąd: Brak klucza 'stats'")
                return False

            stats = df_dict['stats']
            
            df_dict['norm'] = {}         
            subsets = ['train', 'test']

            for subset in subsets:
                if subset not in df_dict:
                    continue
                df_subset = df_dict[subset]
                cols_to_scale = [col for col in df_subset.columns if col.startswith(col_names)]
                
                if not cols_to_scale:
                    continue
                
                df_norm = df_subset[cols_to_scale].copy()
                current_mean = stats['mean'][cols_to_scale]
                current_std = stats['std'][cols_to_scale]
                df_norm = (df_norm - current_mean) / current_std
                df_norm = df_norm.fillna(0.0)
                
                df_dict['norm'][subset] = df_norm

            print("  [scale_with_stats] Pomyślnie utworzono zbiory znormalizowane")
            return True

        except Exception as e:
            print(f"  [scale_with_stats] Błąd: {e}")
            return False

    @staticmethod
    def create_tensors(container, col_names=('feature_', 'target_')):
        try:
            df_norm = container.df_dict.get('norm', {})
            subsets = ['train', 'test']
            
            if not all(s in df_norm for s in subsets):
                print("  [create_tensors] Błąd: Brak danych w norm")
                return False

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            container.ten_dict['device'] = device

            all_cols = df_norm['train'].columns
            selected_cols = [c for c in all_cols if c.startswith(col_names)]
            
            x_cols = sorted([c for c in selected_cols if c.startswith('feature_')])
            y_cols = sorted([c for c in selected_cols if c.startswith('target_')])

            cols_map = {}
            if x_cols:
                cols_map['x'] = x_cols
            if y_cols:
                cols_map['y'] = y_cols
            
            if not any(cols_map.values()):
                print(f"  [create_tensors] Błąd: Nie znaleziono kolumn dla: {col_names}")
                return False
            
            container.ten_dict['norm'] = {subset: {} for subset in subsets}
            ten_norm = container.ten_dict['norm']

            for subset in subsets:
                for key, columns in cols_map.items():
                    if columns:
                        ten_norm_np = df_norm[subset][columns].to_numpy()
                        ten_norm[subset][key] = torch.as_tensor(ten_norm_np, dtype=torch.float32).to(device)

            print(f"  [create_tensors] Utworzono tensory na {device}")
            return True

        except Exception as e:
            print(f"   [create_tensors] Błąd: {e}")
            return False
        
    @staticmethod
    def descale_preds(container: 'Container'):
        try:
            stats = container.df_dict.get('stats')
            target_cols = sorted([k for k in stats['mean'].index if k.startswith('target_')])
            
            mean_p = stats['mean'][target_cols].values
            std_p = stats['std'][target_cols].values

            for split in ['train', 'test']:
                if 'p' in container.ten_dict['norm'][split] and split in container.df_dict:
                    p_norm = container.ten_dict['norm'][split]['p']
                    p_orig = p_norm.detach().cpu().numpy() * std_p + mean_p
                    
                    pred_names = [f"pred_{i}" for i in range(len(target_cols))]
                    container.df_dict[split][pred_names] = p_orig

            print("  [descale_preds] Dopisano pred_x do df_dict")
            return True
        except Exception as e:
            print(f"  [descale_preds] Błąd: {e}")
            return False
