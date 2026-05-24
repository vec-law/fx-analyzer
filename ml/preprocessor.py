import torch
import pandas as pd

class Preprocessor:
    def __init__(self):
        pass

    def split_data(self, df, samples_subset_2, selected_cols):
        try:
            if df is None or df.empty:
                return None, None

            if selected_cols:
                if not all(col in df.columns for col in selected_cols):
                    return None, None
                df_selected = df[selected_cols].copy()
            else:
                return None, None

            split_idx = len(df_selected) - samples_subset_2
            if split_idx < 0:
                return None, None
            elif split_idx == 0:
                df_subset_1 = None
            else:
                df_subset_1 = df_selected.iloc[:split_idx].reset_index(drop=True)
            df_subset_2 = df_selected.iloc[split_idx:].reset_index(drop=True) if samples_subset_2 > 0 else None
            
            return df_subset_1, df_subset_2

        except Exception as e:
            raise Exception(f"Błąd: {e}")
        
    def calculate_stats(self, df, selected_cols):
        try:
            if df is None or df.empty:
                return None, None

            if selected_cols:
                if not all(col in df.columns for col in selected_cols):
                    return None, None
                df_selected = df[selected_cols]
            else:
                return None, None
            
            ser_mean = df_selected.mean()
            ser_std = df_selected.std().replace(0, 1e-9)
            
            return ser_mean, ser_std

        except Exception as e:
            raise Exception(f"Błąd: {e}")
        
    def scale_data(self, df, ser_mean, ser_std, selected_cols):
        try:
            if df is None or df.empty:
                return None
            if ser_mean is None or ser_mean.empty:
                return None
            if ser_std is None or ser_std.empty:
                return None
            
            if selected_cols:
                if not all(col in df.columns for col in selected_cols):
                    return None
                if not all(col in ser_mean.index for col in selected_cols):
                    return None
                if not all(col in ser_std.index for col in selected_cols):
                    return None

                df_selected = df[selected_cols].copy()
                df_norm = (df_selected - ser_mean) / ser_std
                df_norm = df_norm.fillna(0.0)
                return df_norm
            else:
                return None

        except Exception as e:
            raise Exception(f"Błąd: {e}")
        
    def create_tensors(self, df, selected_cols, device):
        try:
            if df is None or df.empty:
                return None
            if not device:
                return None
            
            if selected_cols:
                if not all(col in df.columns for col in selected_cols):
                    return None
                np_selected = df[selected_cols].to_numpy()
                ten = torch.as_tensor(np_selected, dtype=torch.float32, device=device)
                return ten
            else:
                return None

        except Exception as e:
            raise Exception(f"Błąd: {e}")

    def descale_data(self, ten_norm, ser_mean, ser_std, selected_cols):
        try:
            if ten_norm is None:
                return None
            if ser_mean is None or ser_mean.empty:
                return None
            if ser_std is None or ser_std.empty:
                return None
            if not selected_cols:
                return None

            if not all(col in ser_mean.index for col in selected_cols) or \
               not all(col in ser_std.index for col in selected_cols):
                return None

            device = ten_norm.device
            ten_mean = torch.tensor(ser_mean[selected_cols].values, dtype=torch.float32).to(device)
            ten_std = torch.tensor(ser_std[selected_cols].values, dtype=torch.float32).to(device)
            ten_denorm = ten_norm * ten_std + ten_mean

            df_denorm = pd.DataFrame(
                ten_denorm.detach().cpu().numpy(), 
                columns=selected_cols
            )
            return df_denorm

        except Exception as e:
            raise Exception(f"Błąd: {e}")