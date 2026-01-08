from .instrument import Instrument
from src.ingestion import load_data
from src.preprocessing import create_features, create_targets
from src.preprocessing import clean_data, scale_data

class Analysis:
    def __init__(self, config):
        self.config = config
        
        self.instrument_list = config.get('INSTRUMENT_LIST', [])
        self.params_list = config.get('PARAMS_LIST', [])
        self.feature_list = config.get('FEATURE_LIST', [])
        self.target_list = config.get('TARGET_LIST', [])

        if not all([self.instrument_list, self.params_list, self.feature_list, self.target_list]):
            raise KeyError("Błąd: Jedna z list (INSTRUMENT, PARAMS, FEATURE, TARGET) jest pusta lub jej brakuje.")
        
    def run(self):
        print("START")
        self._create_args()

        for a in self.args:
            print()
            print(f"ZESTAW: {a['i']['name']} | {a['p']['name']}| {a['f']['name']}")
            
            print("ETAP 1/8: Inicjalizacja...")
            instrument = self._stage_1_create_instrument(a)
            if not instrument:
                print(">>> Przerwano zestaw (Błąd Etapu 1)")
                continue

            print("ETAP 2/8: Ingestion...")
            if not self._stage_2_ingestion(instrument, a):
                print(">>> Przerwano zestaw (Błąd Etapu 2)")
                continue


            print("ETAP 3/8: Preprocessing...")
            if not self._stage_3_preprocessing(instrument, a):
                print(">>> Przerwano zestaw (Błąd Etapu 3)")
                continue

    def _create_args(self):
        self.args = [
            {'i': i, 'p': p, 'f': f, 't': t}
            for i in self.instrument_list
            for p in self.params_list
            for f in self.feature_list
            for t in self.target_list
        ]
        print(f"  [_create_args] Przygotowano {len(self.args)} zestawów do przetestowania")

    def _stage_1_create_instrument(self, a):
        try:
            instrument = Instrument(
                a['i']['params']['name'],
                a['i']['params']['type'],
                a['i']['params']['interval'],
                self.config
                )
            print(f"  [_stage_1_create_instrument] Utworzenie instrumentu {a['i']['name']}")
            return instrument
        except Exception as e:
            print(f"Nieoczekiwany błąd {e}")
            return None

    def _stage_2_ingestion(self, instrument, a):
        try:
            if not load_data(instrument, a['p']['params']['mode'], a['p']['params']['repair_gaps'], self.config):
                return False
            print(f"  [_stage_2_ingestion] Załadowano dane {a['i']['name']}")
            return True

        except KeyError as e:
            print(f"Błąd klucza w konfiguracji cech: {e}")
            return False

        except Exception as e:
            print(f"Nieoczekiwany błąd {e}")
            return False

    def _stage_3_preprocessing(self, instrument, a):
        try:
            if not create_features(instrument, a['f']['features']):
                return False

            print(f"  [_stage_3_preprocessing] Przetworzono zestaw: {a['f']['name']}")
            return True
            
        except KeyError as e:
            print(f"Błąd klucza w konfiguracji cech: {e}")
            return False

        except Exception as e:
            print(f"Nieoczekiwany błąd: {e}")
            return False

    # def _stage_3_preprocessing(self, instrument, p):
    #     for f in self.feature_list:
    #         if not create_features(instrument, f):
    #             print("Przerwano")
    #             return
    #         if not create_targets(instrument, self.target_list):
    #             print("Przerwano")
    #             return
    #     # if not clean_data(instrument, p):
    #     #     print("Przerwano")
    #     #     return
    #     # if not scale_data(instrument, p):
    #     #     print("Przerwano")
    #     #     return
    #     print("OK")
