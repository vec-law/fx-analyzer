from .instrument import Instrument
from .container import Container
from .loader import Loader
from .cleaner import Cleaner
from .preprocessor import Preprocessor


class Analysis:
    def __init__(self, config):
        self.config = config

        self.instrument_list = config.get('INSTRUMENT_LIST', [])
        self.params_list = config.get('PARAMS_LIST', [])
        self.feature_list = config.get('FEATURE_LIST', [])
        self.target_list = config.get('TARGET_LIST', [])

        self.supported_types = config.get('INSTRUMENT_TYPES', [])
        self.custom_names = config.get('CUSTOM_NAMES', {})
        self.interval_settings = config.get('INTERVAL_SETTINGS', {})

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

        for arg_set in self.args:
            i = arg_set['i']
            p = arg_set['p']
            f = arg_set['f']
            t = arg_set['t']

            print(f"ZESTAW: {i['name']} | {p['name']} | {f['name']} | {t['name']}")
            print(90 * '-')

            self._run_pipeline(arg_set)
            print(90 * '=')

    def _create_args(self):
        self.args = [
            {'i': i, 'p': p, 'f': f, 't': t}
            for i in self.instrument_list
            for p in self.params_list
            for f in self.feature_list
            for t in self.target_list
        ]
        print(f"  [_create_args] Przygotowano {len(self.args)} zestaw(ów) do przetestowania\n{'=' * 90}")
        
    def _create_instrument(self, arg_set):
        try:
            params = arg_set['i']['params']
            instrument = Instrument(
                params['name'],
                params['type_of_instrument'],
                params['interval'],
                self.supported_types,
                self.custom_names,
                self.interval_settings
            )
            print(f"  [_create_instrument] Utworzono instrument: {arg_set['i']['name']}")
            return instrument

        except Exception as e:
            print(f"  [_create_instrument] Nieoczekiwany błąd: {e}")
            return None

    def _run_pipeline(self, arg_set):
        try:
            instrument = self._create_instrument(arg_set)
            if not instrument:
                return

            container = Container()
            container.instrument = instrument
            
            if not Loader.load_data(container, arg_set, self.config):
                return

            if not Cleaner.clean_data(container, arg_set):
                return

            if not Preprocessor.create_features(container, arg_set):
                return
            
            if not Preprocessor.create_targets(container, arg_set):
                return

            if not Preprocessor.cut_and_split_data(container, arg_set):
                return
            
            if not Preprocessor.scale_data(container):
                return

        except Exception as e:
            print(f"  [_run_pipeline] Błąd podczas przetwarzania: {e}")
