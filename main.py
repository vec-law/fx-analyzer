__version__ = "1.0.1"

from src.utils import clear_console
from src.ingestion import load_data
from src.features import prepare_features, normalize
from src.model import prepare_tensors, prepare_model_params, train_model, evaluate_model
from src.strategy import simulate_strategy

instrument = 'EURUSD'
interval = '1d'
target_column ='close'
strategy = 1
indicators = {
    'sma': list(range(10, 41, 1))}

max_ind_period = max(max(periods) for periods in indicators.values())
samples_limit = 4000 + max_ind_period
train_ratio = 0.875
seed = 42
epochs = 1000

clear_console()

print(f"ETAP 1/8: Ładowanie danych...")
if(df := load_data(instrument, interval, target_column, samples_limit)) is None:
    print("Przerwano")
    exit()
print("OK")

print()
print(f"ETAP 2/8: Preprocessing...")
if(df_dict := prepare_features(df, indicators, train_ratio)) is None:
    print("Przerwano")
    exit()
print("OK")

print()
print(f"ETAP 3/8: Normalizacja danych...")
if (df_dict := normalize(df_dict)) is None:
    print("Przerwano")
    exit()
print("OK")

print()
print(f"ETAP 4/8: Przygotowanie tensorów...")
if (ten_dict := prepare_tensors(df_dict)) is None:
    print("Przerwano")
    exit()
print("OK")

print()
print(f"ETAP 5/8: Konfiguracja modelu...")
if (mod_dict := prepare_model_params(ten_dict, seed)) is None:
    print("Przerwano")
    exit()
print("OK")

print()
print(f"ETAP 6/8: Trening modelu...")
if (mod_dict := train_model(ten_dict, mod_dict, epochs)) is None:
    print("Przerwano")
    exit()
print("OK")

print()
print(f"ETAP 7/8: Ewaluacja modelu...")
if (result := evaluate_model(mod_dict, df_dict, ten_dict)) is None:
    print("Przerwano")
    exit()
df_dict, ten_dict, mod_dict = result
print("OK")

print()
print(f"ETAP 8/8: Symulacja strategii...")
if (strategy_results := simulate_strategy(df_dict, strategy)) is None:
    print("Przerwano")
    exit()
print("OK")
