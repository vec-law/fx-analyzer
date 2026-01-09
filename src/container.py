from .instrument import Instrument


class Container:
    __slots__ = ['instrument', 'df', 'df_dict']

    def __init__(self):
        self.instrument = None
        self.df = None
        self.df_dict = {}