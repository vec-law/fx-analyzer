import os
import json
import pandas as pd
from .instrument import Instrument
from safetensors.torch import save_file, load_file

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
                print("  [load_df_from_parquet] Błąd: Brak id kontenera")
                return False

            path = os.path.join("data", "fit", str(self.id), "data.parquet")

            if not os.path.exists(path):
                print(f"  [load_df_from_parquet] Błąd: Plik nie istnieje: {path}")
                return False

            df = pd.read_parquet(path)

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

            model_path = os.path.join(target_dir, "model.safetensors")
            state_dict = self.mod_dict['model'].state_dict()

            save_dict = {k: v.cpu().contiguous() for k, v in state_dict.items()}
            save_file(save_dict, model_path)

            stats_path = os.path.join(target_dir, "stats.json")
            stats_json = {
                'mean': self.df_dict['stats']['mean'].to_dict(),
                'std': self.df_dict['stats']['std'].to_dict()
            }
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats_json, f, indent=4)

            return True
        except Exception as e:
            print(f"  [save_model_and_stats] Błąd: {e}")
            return False

    def load_stats(self):
        try:
            target_dir = os.path.join("data", "fit", str(self.id))
            stats_path = os.path.join(target_dir, "stats.json")
            
            with open(stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.df_dict['stats'] = {
                'mean': pd.Series(data['mean']),
                'std': pd.Series(data['std'])
            }
            return True
        except Exception as e:
            print(f"  [load_stats] Błąd: {e}")
            return False

    def load_model_weights(self):
        try:
            if 'model' not in self.mod_dict:
                print("  [load_model_weights] Błąd: Brak modelu w mod_dict")
                return False

            target_dir = os.path.join("data", "fit", str(self.id))
            model_path = os.path.join(target_dir, "model.safetensors")
            
            state_dict = load_file(model_path)
            self.mod_dict['model'].load_state_dict(state_dict)
            self.mod_dict['model'].eval()
            return True
        except Exception as e:
            print(f"  [load_model_weights] Error: {e}")
            return False