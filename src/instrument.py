import pandas as pd

class Instrument:
    def __init__(self, name, type, interval, instrument_types, own_names, interval_settings):
        if name is None or not isinstance(name, str):
            raise ValueError(f"Błędna nazwa: {name}")
        if type is None or not isinstance(type, str) or type not in instrument_types:
            raise ValueError(f"Błędny typ: {type}")
        if interval is None or not isinstance(interval, str) or interval not in interval_settings:
            raise ValueError(f"Błędny interwał: {interval}")
        
        if name in own_names and type in own_names[name]:
            self.ticker = own_names[name][type]
        elif type == 'currency_pair' and not name.endswith('=X'):
            self.ticker = f"{name}=X"
        else:
            self.ticker = name

        self.name = name
        self.type = type
        self.interval = interval
        settings = interval_settings[self.interval]
        self.history_range = settings['range']
        self.check_period = settings['check_period']
        self.min_count = settings['min_count']
        self.df = pd.DataFrame()
        self.df_dict = {
            'train': pd.DataFrame(),
            'test': pd.DataFrame()
        }
        self.source = None

    def __str__(self):
        return (
            f"Name: {self.name} ({self.type}) | Interval: {self.interval} | Ticker: {self.ticker} | "
            f"Settings: [Range: {self.history_range}, Check: {self.check_period}, Min: {self.min_count}] | "
            f"Data: {self.df.shape[0]}x{self.df.shape[1]}"
        )
