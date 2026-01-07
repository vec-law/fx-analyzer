from .instrument import Instrument
from src.ingestion import load_data

class Analysis:
    def __init__(self, config):
        if config is None or not isinstance(config, dict):
            raise ValueError("Błąd: brak 'config'")
        self.config = config

        if 'INSTRUMENT_LIST' not in config or not config['INSTRUMENT_LIST']:
            raise KeyError("Błąd: brak 'INSTRUMENT_LIST'")
        self.instrument_list = config['INSTRUMENT_LIST']
        
        if 'ANALYSIS_PARAMS_LIST' not in config or not config['ANALYSIS_PARAMS_LIST']:
            raise KeyError("Błąd: brak 'ANALYSIS_PARAMS_LIST'")
        
        self.analysis_params_list = config['ANALYSIS_PARAMS_LIST']
    
    def run(self):
        for i in self.instrument_list:
            instrument = Instrument(
                i['name'],
                i['type'],
                i['interval'],
                self.config
            )
            print(f"--- START: {i['name']} ({i['interval']}) ---")

            self._stage_1_set_analysis_params(instrument)

    def _stage_1_set_analysis_params(self, instrument):
        print(f"ETAP 1/8: Ustawianie parametrów...")
        for p in self.analysis_params_list:
            if 'repair_gaps' not in p or 'mode' not in p:
                raise KeyError(f"Brak wymaganych parametrów w: {p}")
            print("OK")

            self._stage_2_ingestion(instrument, p)

    def _stage_2_ingestion(self, instrument, p):
        print(f"ETAP 2/8: Ładowanie danych...")
        if not load_data(instrument, p['mode'], p['repair_gaps'], self.config):
            print("Przerwano")
            return
        print("OK")