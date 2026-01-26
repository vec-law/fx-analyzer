import inspect
import torch

class Preprocessor:
    def __init__(self, config: dict, log_signal):
        self.config = config
        self.log_signal = log_signal

    def split_data(self, df, samples_subset_2, selected_cols):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df.empty:
                self.log_signal.emit(f"[{f_name}] Brak danych df")  
                return None, None

            if selected_cols:
                if not all(col in df.columns for col in selected_cols):
                    self.log_signal.emit(f"[{f_name}] Brak wybranych kolumn w df")  
                    return None, None
                
                df_selected = df[selected_cols].copy()
            else:
                self.log_signal.emit(f"[{f_name}] Nie określono kolumn w df")  
                return None, None

            split_idx = len(df_selected) - samples_subset_2
            if split_idx <= 0:
                self.log_signal.emit(f"[{f_name}] Za mało rekordów w df")  
                return None, None

            df_subset_1 = df_selected.iloc[:split_idx].reset_index(drop=True)
            df_subset_2 = df_selected.iloc[split_idx:].reset_index(drop=True) if samples_subset_2 > 0 else None
            
            self.log_signal.emit(f"[{f_name}] Wykonano split: {split_idx} | {samples_subset_2}")            
            return df_subset_1, df_subset_2

        except Exception as e:
            self.log_signal.emit(f"[{f_name}] Error: {e}")
            return None, None
        
    def calculate_stats(self, df, selected_cols):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df.empty:
                self.log_signal.emit(f"[{f_name}] Brak danych df")  
                return None, None

            if selected_cols:
                if not all(col in df.columns for col in selected_cols):
                    self.log_signal.emit(f"[{f_name}] Brak wybranych kolumn w df")  
                    return None, None
                
                df_selected = df[selected_cols]
            else:
                self.log_signal.emit(f"[{f_name}] Nie określono kolumn w df")  
                return None, None
            
            ser_mean = df_selected.mean()
            ser_std = df_selected.std().replace(0, 1e-9)
            
            self.log_signal.emit(f"[{f_name}] Obliczono statystyki")            
            return ser_mean, ser_std

        except Exception as e:
            self.log_signal.emit(f"[{f_name}] Error: {e}")
            return None, None
        
    def scale_data(self, df, ser_mean, ser_std, selected_cols):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df.empty:
                self.log_signal.emit(f"[{f_name}] Brak danych df")  
                return None
            if ser_mean is None or ser_mean.empty:
                self.log_signal.emit(f"[{f_name}] Brak danych df_mean")  
                return None
            if ser_std is None or ser_std.empty:
                self.log_signal.emit(f"[{f_name}] Brak danych df_std")  
                return None
            
            if selected_cols:
                if not all(col in df.columns for col in selected_cols):
                    self.log_signal.emit(f"[{f_name}] Brak wybranych kolumn w df")  
                    return None
                if not all(col in ser_mean.index for col in selected_cols):
                    self.log_signal.emit(f"[{f_name}] Brak wybranych kolumn w df_mean")  
                    return None
                if not all(col in ser_std.index for col in selected_cols):
                    self.log_signal.emit(f"[{f_name}] Brak wybranych kolumn w df_std")  
                    return None

                df_selected = df[selected_cols].copy()
                df_norm = (df_selected - ser_mean) / ser_std
                df_norm = df_norm.fillna(0.0)

                self.log_signal.emit(f"[{f_name}] Obliczono wartości znormalizowane")            
                return df_norm
            else:
                self.log_signal.emit(f"[{f_name}] Nie określono kolumn")  
                return None

        except Exception as e:
            self.log_signal.emit(f"[{f_name}] Error: {e}")
            return None
        
    def create_tensors(self, df, selected_cols, device):
        f_name = inspect.currentframe().f_code.co_name
        try:
            if df is None or df.empty:
                self.log_signal.emit(f"[{f_name}] Brak danych df")  
                return None
            
            if not device:
                self.log_signal.emit(f"[{f_name}] Nie określono urządzenia")  
                return None
            
            if selected_cols:
                if not all(col in df.columns for col in selected_cols):
                    self.log_signal.emit(f"[{f_name}] Brak wybranych kolumn w df")  
                    return None
                
                np_selected = df[selected_cols].to_numpy()
                ten = torch.as_tensor(np_selected, dtype=torch.float32, device=device)

                self.log_signal.emit(f"[{f_name}] Utworzono tensory")
                return ten
            else:
                self.log_signal.emit(f"[{f_name}] Nie określono kolumn")
                return None

        except Exception as e:
            self.log_signal.emit(f"[{f_name}] Error: {e}")
            return None

        
#     @staticmethod
#     def descale_preds(container: 'Container'):
#         try:
#             stats = container.df_dict.get('stats')
#             target_cols = sorted([k for k in stats['mean'].index if k.startswith('target_')])
            
#             mean_p = stats['mean'][target_cols].values
#             std_p = stats['std'][target_cols].values

#             for split in ['train', 'test']:
#                 if 'p' in container.ten_dict['norm'][split] and split in container.df_dict:
#                     p_norm = container.ten_dict['norm'][split]['p']
#                     p_orig = p_norm.detach().cpu().numpy() * std_p + mean_p
                    
#                     pred_names = [f"pred_{i}" for i in range(len(target_cols))]
#                     container.df_dict[split][pred_names] = p_orig

#             print("  [descale_preds] Dopisano pred_x do df_dict")
#             return True
#         except Exception as e:
#             print(f"  [descale_preds] Błąd: {e}")
#             return False
