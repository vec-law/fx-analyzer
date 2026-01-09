from .container import Container
from .loader import Loader
from .cleaner import Cleaner
from .preprocessor import Preprocessor
from .model_manager import ModelManager
from .trainer import Trainer


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

    def run(self):
        print(f"{'=' * 90}\nSTART")
        
        self._create_args()

        for idx, arg_set in enumerate(self.args, start=1):
            i = arg_set['i']
            p = arg_set['p']
            f = arg_set['f']
            t = arg_set['t']

            print(f"ZESTAW nr {idx}: {i['name']} | {p['name']} | {f['name']} | {t['name']}")
            print(90 * '-')

            self._run_pipeline(arg_set, idx)
            print(90 * '=')

    def _create_args(self):
        self.args = [
            {'i': i, 'p': p, 'f': f, 't': t}
            for i in self.instrument_list
            for p in self.params_list
            for f in self.feature_list
            for t in self.target_list
        ]
        print(f"   [_create_args] Przygotowano {len(self.args)} zestaw(ów) do przetestowania\n{'=' * 90}")

    def _run_pipeline(self, arg_set, idx):
        try:
            container = Container(arg_set, self.config, idx)

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

        except Exception as e:
            print(f"   [_run_pipeline] Błąd podczas przetwarzania: {e}")