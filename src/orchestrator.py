from .container import Container
from .loader import Loader
from .cleaner import Cleaner
from .preprocessor import Preprocessor
from .model_manager import ModelManager
from .trainer import Trainer
from .strategy import Strategy
import gc
import torch
import os
import json
import matplotlib.pyplot as plt


class Orchestrator:
    def __init__(self, config):
        self.config = config

        self.instrument_list = config.get('INSTRUMENT_LIST', [])
        self.params_list = config.get('PARAMS_LIST', [])
        self.feature_list = config.get('FEATURE_LIST', [])
        self.target_list = config.get('TARGET_LIST', [])

        self.args = []
        self._validate_config()

    def _validate_config(self):
        required_lists = [
            self.instrument_list,
            self.params_list,
            self.feature_list,
            self.target_list
        ]

        if not all(required_lists):
            raise KeyError(
                "Błąd: Jedna z wymaganych list (INSTRUMENT, PARAMS, FEATURE, TARGET) jest pusta."
            )

    def run_fits(self):
        print(f"{'=' * 90}\nRUN FITS")

        self._create_args()

        for fit_id, arg_set in enumerate(self.args, start=1):
            i = arg_set['i']
            p = arg_set['p']
            f = arg_set['f']
            t = arg_set['t']

            print(f"ZESTAW nr {fit_id}: {i['name']} | {p['name']} | {f['name']} | {t['name']}")
            print(90 * '-')

            self.run_fit(arg_set, fit_id)

    def _create_args(self):
        self.args = [
            {'i': i, 'p': p, 'f': f, 't': t}
            for i in self.instrument_list
            for p in self.params_list
            for f in self.feature_list
            for t in self.target_list
        ]
        print(f"   [_create_args] Przygotowano {len(self.args)} zestaw(ów) do przetestowania\n{'=' * 90}")

    def run_fit(self, arg_set, container_id=1):
        container = None
        try:
            container = Container(arg_set, self.config, container_id)

            if not container.save_params_set():
                return
            if not Loader.load_data(container):
                return
            if not Cleaner.clean_data(container):
                return
            if not Preprocessor.create_features(container):
                return
            if not Preprocessor.create_targets(container):
                return
            if not Preprocessor.cut_and_split_data(container):
                return
            if not Preprocessor.scale_data(container):
                return
            if not Preprocessor.create_tensors(container):
                return
            if not ModelManager.create_model_and_params(container):
                return
            if not Trainer.train_model(container):
                return
            if not Trainer.evaluate_model(container):
                return
            if not container.save_model_and_stats():
                return
            
        except Exception as e:
            print(f"   [run_fit] Błąd: {e}")
        
        finally:
            if container:
                self._release_resources(container)
                container = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
    @staticmethod
    def _release_resources(container):
        container.df = None
        container.df_dict.clear()
        container.ten_dict.clear()
        container.mod_dict.clear()

    def run_preds(self):
        print(f"{'=' * 90}\nRUN PREDS\n{'=' * 90}")
        base_path = os.path.join("data", "fit")
        if not os.path.exists(base_path):
            return
        
        folders = [f.name for f in os.scandir(base_path) if f.is_dir()]

        for folder in folders:
            path = os.path.join(base_path, folder)
            json_path = os.path.join(path, 'params_set.json')

            if not os.path.exists(json_path):
                continue

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    arg_set = json.load(f)

                print(f"ZESTAW nr {folder}: {arg_set['i']['name']} | {arg_set['p']['name']} | {arg_set['f']['name']} | {arg_set['t']['name']}")
                print(90 * '-')
                
                self.run_pred(arg_set, folder)

            except Exception as e:
                print(f"Błąd w {folder}: {e}")
                continue
        
    def run_pred(self, arg_set, container_id):
        container = None
        try:
            container = Container(arg_set, self.config, container_id)

            if not container.load_df_from_parquet():
                return
            if not Preprocessor.split_data(container):
                return
            if not container.load_stats():
                return
            if not Preprocessor.scale_with_stats(container):
                return
            if not Preprocessor.create_tensors(container):
                return
            if not ModelManager.create_model_and_params(container):
                return
            if not container.load_model_weights():
                return
            if not Trainer.predict(container):
                return
            if not Preprocessor.descale_preds(container):
                return            
            if not Strategy.add_indicators_and_clean(container):
                return

                
            # print(container.df_dict['test'].columns)
            # print(container.df_dict['test'])

            plt.plot(range(len(container.df_dict['test']['close'])), container.df_dict['test']['close'])
            plt.plot(range(len(container.df_dict['test']['pred_0'])), container.df_dict['test']['pred_0'])
            plt.show()
        
        except Exception as e:
            print(f"   [run_pred] Błąd: {e}")
        
        finally:
            if container:
                self._release_resources(container)
                container = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    