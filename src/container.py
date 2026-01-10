import os
import json
import torch
import pandas as pd
from .instrument import Instrument

class Container:
    def __str__(self):
        return str(self.params_set)

    def __init__(self, params_set, config, id):
        self.params_set = params_set
        self.config = config
        self.id = id
        
        try:
            params = params_set['i']['params']
        except KeyError as e:
            raise KeyError(f"Błąd struktury params_set: brak klucza {e}")

        self.instrument = Instrument(
            name=params['name'],
            type_of_instrument=params['type_of_instrument'],
            interval=params['interval'],
            supported_types=config.get('INSTRUMENT_TYPES', []),
            custom_names=config.get('CUSTOM_NAMES', {}),
            interval_settings=config.get('INTERVAL_SETTINGS', {})
        )
        
        self.df = None
        self.df_dict = {}
        self.ten_dict = {}
        self.mod_dict = {}

    def save_params_set(self):
        try:
            if not self.id:
                return False
            
            target_dir = os.path.join("data", "fit", str(self.id))
            os.makedirs(target_dir, exist_ok=True)
            
            path = os.path.join(target_dir, "params_set.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.params_set, f, indent=4)
            return True
        except Exception as e:
            print(f"Błąd zapisu params: {e}")
            return False

    def save_df_to_parquet(self):
        try:
            if not self.id or self.df is None:
                return False

            target_dir = os.path.join("data", "fit", str(self.id))
            os.makedirs(target_dir, exist_ok=True)
            
            path = os.path.join(target_dir, "data.parquet")
            self.df.to_parquet(path, index=True)

            print(f"  [save_df_to_parquet] Zapisano dane w {path}")
            return True
        except Exception as e:
            print(f"Błąd zapisu df: {e}")
            return False
        
    def load_df_from_parquet(self):
        try:
            if not self.id:
                print("  [load_df_from_parquet] Błąd: Brak ID kontenera")
                return False

            path = os.path.join("data", "fit", str(self.id), "data.parquet")

            if not os.path.exists(path):
                print(f"  [load_df_from_parquet] Błąd: Plik nie istnieje: {path}")
                return False

            df = pd.read_parquet(path)

            targets_source = self.params_set.get('t', {}).get('targets', [])
            target_cols = [f"target_{i}" for i in range(len(targets_source))]
            df = df.drop(columns=[c for c in target_cols if c in df.columns], errors='ignore')
            
            df = df.dropna()

            p_params = self.params_set.get('p', {}).get('params', {})
            samples_limit = p_params.get('samples_limit')
            train_ratio = p_params.get('train_ratio')

            if samples_limit and isinstance(samples_limit, int):
                if train_ratio and isinstance(train_ratio, float):
                    if train_ratio > 0 and train_ratio < 1:
                        test_samples = int(samples_limit * (1 - train_ratio))
                        df = df.tail(test_samples)

            self.df = df
            
            print(f"  [load_df_from_parquet] Wczytano {len(self.df)} wierszy z {path}")
            return True

        except Exception as e:
            print(f"  [load_df_from_parquet] Błąd podczas wczytywania: {e}")
            return False

    def save_model_and_stats(self):
        try:
            if not self.id or 'model' not in self.mod_dict:
                return False

            target_dir = os.path.join("data", "fit", str(self.id))
            os.makedirs(target_dir, exist_ok=True)

            torch.save(self.mod_dict['model'].state_dict(), os.path.join(target_dir, "model.pth"))
            torch.save(self.df_dict['stats'], os.path.join(target_dir, "stats.pth"))
            return True
        except Exception as e:
            print(f"Błąd zapisu modelu: {e}")
            return False