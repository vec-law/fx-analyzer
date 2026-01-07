


# from src.ingestion import load_data
# from src.features import prepare_features, normalize
# from src.model.functions import prepare_tensors, prepare_model_params, train_model, evaluate_model
# from src.strategy import simulate_strategy



# target_column ='close'
# strategy = 1
# indicators = {
#     'sma': list(range(10, 41, 1))}

# max_ind_period = max(max(periods) for periods in indicators.values())
# samples_limit = 4000 + max_ind_period
# train_ratio = 0.875
# seed = 42
# epochs = 1000
# model_num = 1
# repair_gaps = False

__version__ = "1.0.3-alpha"

import json
from src.instruments import Instrument
from src.utils import clear_console
from src.ingestion import load_data

def main():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Błąd konfiguracji: {e}")

    clear_console()
    print(f"--- fx-analyzer v{__version__} ---")

    instrument_name = 'EURUSD'
    instrument_type = 'currency_pair'
    instrument_interval = '1d'
    repair_gaps = True
    mode = 'server'

    instrument = Instrument(instrument_name, instrument_type, instrument_interval, config)

    print(f"ETAP 1/8: Ładowanie danych...")
    if not load_data(instrument, mode, repair_gaps, config):
        print("Przerwano")
        return
    print("OK")

if __name__ == "__main__":
    main()

# print()
# print(f"ETAP 2/8: Preprocessing...")
# if(df_dict := prepare_features(df, indicators, train_ratio)) is None:
#     print("Przerwano")
#     exit()
# print("OK")

# print()
# print(f"ETAP 3/8: Normalizacja danych...")
# if (df_dict := normalize(df_dict)) is None:
#     print("Przerwano")
#     exit()
# print("OK")

# print()
# print(f"ETAP 4/8: Przygotowanie tensorów...")
# if (ten_dict := prepare_tensors(df_dict)) is None:
#     print("Przerwano")
#     exit()
# print("OK")

# print()
# print(f"ETAP 5/8: Konfiguracja modelu...")
# if (mod_dict := prepare_model_params(ten_dict, seed, model_num)) is None:
#     print("Przerwano")
#     exit()
# print("OK")

# print()
# print(f"ETAP 6/8: Trening modelu...")
# if (mod_dict := train_model(ten_dict, mod_dict, epochs)) is None:
#     print("Przerwano")
#     exit()
# print("OK")

# print()
# print(f"ETAP 7/8: Ewaluacja modelu...")
# if (result := evaluate_model(mod_dict, df_dict, ten_dict)) is None:
#     print("Przerwano")
#     exit()
# df_dict, ten_dict, mod_dict = result
# print("OK")

# print()
# print(f"ETAP 8/8: Symulacja strategii...")
# if (strategy_results := simulate_strategy(df_dict, strategy)) is None:
#     print("Przerwano")
#     exit()
# print("OK")
