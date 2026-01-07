import pandas as pd

class Instrument:
    def __init__(self, name, type, interval, config):
        if 'INSTRUMENT_TYPES' not in config:
            raise KeyError("Błąd: Brak klucza 'INSTRUMENT_TYPES' w konfiguracji")
        if 'OWN_NAMES' not in config:
            raise KeyError("Błąd: Brak klucza 'OWN_NAMES' w konfiguracji")
        if 'INTERVAL_SETTINGS' not in config:
            raise KeyError("Błąd: Brak klucza 'INTERVAL_SETTINGS' w konfiguracji")

        INSTRUMENT_TYPES = config['INSTRUMENT_TYPES']
        OWN_NAMES = config['OWN_NAMES']
        INTERVAL_SETTINGS = config['INTERVAL_SETTINGS']

        if name is None or not isinstance(name, str):
            raise ValueError(f"Błędna nazwa: {name}")
        if type is None or not isinstance(type, str) or type not in INSTRUMENT_TYPES:
            raise ValueError(f"Błędny typ: {type}")
        if interval is None or not isinstance(interval, str) or interval not in INTERVAL_SETTINGS:
            raise ValueError(f"Błędny interwał: {interval}")
        
        if name in OWN_NAMES and type in OWN_NAMES[name]:
            self.ticker = OWN_NAMES[name][type]
        elif type == 'currency_pair' and not name.endswith('=X'):
            self.ticker = f"{name}=X"
        else:
            self.ticker = name

        self.name = name
        self.type = type
        self.interval = interval
        settings = INTERVAL_SETTINGS[self.interval]
        self.history_range = settings['range']
        self.check_period = settings['check_period']
        self.min_count = settings['min_count']
        self.df = pd.DataFrame()
        self.source = None

    def __str__(self):
        return (
            f"Name: {self.name} ({self.type}) | Interval: {self.interval} | Ticker: {self.ticker} | "
            f"Settings: [Range: {self.history_range}, Check: {self.check_period}, Min: {self.min_count}] | "
            f"Data: {self.df.shape[0]}x{self.df.shape[1]}"
        )
