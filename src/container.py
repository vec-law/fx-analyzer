from .instrument import Instrument


class Container:
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
