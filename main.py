# from src.ingestion import load_data
# from src.features import prepare_features, normalize
# from src.model.functions import prepare_tensors, prepare_model_params, train_model, evaluate_model
# from src.strategy import simulate_strategy

# target_column ='close'
# strategy = 1
# seed = 42
# epochs = 1000
# model_num = 1

__version__ = "1.0.3-alpha"

import json
from src.orchestrator import Orchestrator
from src.utils import clear_console

def main():
    clear_console()

    print(f"fx-analyzer v{__version__}")
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Błąd konfiguracji: {e}")

    orchestrator = Orchestrator(config)
    orchestrator.run()

if __name__ == "__main__":
    main()

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
