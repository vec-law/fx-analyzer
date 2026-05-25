from ml.loader.yf_loader import YFLoader

def get_loader(config):
    if config['data_source_name'] == 'YF':
        return YFLoader(config)
