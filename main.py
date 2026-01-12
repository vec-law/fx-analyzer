__version__ = "2.0.0-alpha"

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
    orchestrator.run_fits()
    orchestrator.run_preds()

if __name__ == "__main__":
    main()

# print()
# print(f"ETAP 8/8: Symulacja strategii...")
# if (strategy_results := simulate_strategy(df_dict, strategy)) is None:
#     print("Przerwano")
#     exit()
# print("OK")
