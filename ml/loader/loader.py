class Loader:
    def __init__(self, config: dict):
        self.config = config
    
    def load_data(self):
        raise NotImplementedError
