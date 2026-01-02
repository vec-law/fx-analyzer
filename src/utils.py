import os

def get_path(folder, filename):

    base_dir = os.path.dirname(os.path.abspath(__file__))

    return os.path.normpath(os.path.join(base_dir, "..", "data", folder, filename))

def save_df(df, folder, filename):
    
    if df is None: return False
    
    path = get_path(folder, filename)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df.to_csv(path, index=False)

    return True

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')