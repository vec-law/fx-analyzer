class Instrument:
    def __init__(
        self, 
        name, 
        type_of_instrument, 
        interval, 
        supported_types, 
        custom_names, 
        interval_settings
    ):
        if name is None or not isinstance(name, str):
            raise ValueError(f"Invalid name: {name}")
        
        if (type_of_instrument is None or not isinstance(type_of_instrument, str) 
                or type_of_instrument not in supported_types):
            raise ValueError(f"Invalid type: {type_of_instrument}")
            
        if (interval is None or not isinstance(interval, str) 
                or interval not in interval_settings):
            raise ValueError(f"Invalid interval: {interval}")

        if name in custom_names and type_of_instrument in custom_names[name]:
            self.ticker = custom_names[name][type_of_instrument]
        elif type_of_instrument == 'currency_pair' and not name.endswith('=X'):
            self.ticker = f"{name}=X"
        else:
            self.ticker = name

        self.name = name
        self.type_of_instrument = type_of_instrument
        self.interval = interval
        
        settings = interval_settings[self.interval]
        self.history_range = settings['range']
        self.check_period = settings['check_period']
        self.min_count = settings['min_count']

    def __str__(self):
        return (
            f"Name: {self.name} ({self.type_of_instrument}) | "
            f"Interval: {self.interval} | Ticker: {self.ticker} | "
            f"Settings: [Range: {self.history_range}, "
            f"Check: {self.check_period}, Min: {self.min_count}]"
        )