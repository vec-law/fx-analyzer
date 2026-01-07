import os

def get_path(instrument_name, instrument_interval, folder, sufix=None):
    filename = f"{instrument_name}_{instrument_interval}"
    if sufix and isinstance(sufix, str):
        filename += '_' + sufix
    filename += ".csv"
    base_dir = os.path.dirname(os.path.abspath(__file__))

    return os.path.normpath(os.path.join(base_dir, "..", "data", folder, filename))

def save_df(instrument, folder, sufix=None):
    path = get_path(instrument.name, instrument.interval, folder, sufix)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    instrument.df.to_csv(path, index=False)
    
    print(f"  [save_df] {instrument.name}: Dane zapisano w pliku {path}")

    return True

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    