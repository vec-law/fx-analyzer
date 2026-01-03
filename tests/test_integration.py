import pytest

from src.ingestion import load_data
from src.features import prepare_features, normalize
from src.model import prepare_tensors, prepare_model_params, train_model, evaluate_model
from src.strategy import simulate_strategy

INSTRUMENT = 'EURUSD'
INTERVAL = '1d'
TARGET = 'close'
INDICATORS = {'sma': list(range(10, 21))} 
TRAIN_RATIO = 0.8
SEED = 42
MIN_EPOCHS = 10  
SAMPLES_LIMIT = 1000

@pytest.fixture(scope="module")
def shared_data():
    """Fixture ładujący dane raz dla wszystkich testów w module."""
    df = load_data(INSTRUMENT, INTERVAL, TARGET, SAMPLES_LIMIT)
    if df is None:
        pytest.fail("Etap 1: Nie udało się załadować danych do testów.")
    return df

def test_pipeline_full_flow(shared_data):
    """
    INTEGRATION TEST: End-to-End (Etapy 1-8)
    """
    # Etap 2: Features
    df_dict = prepare_features(shared_data, INDICATORS, TRAIN_RATIO)
    assert df_dict is not None, "Błąd w Etapie 2 (Features)"

    # Etap 3: Normalizacja
    df_dict = normalize(df_dict)
    assert df_dict is not None, "Błąd w Etapie 3 (Normalizacja)"

    # Etap 4: Tensory
    ten_dict = prepare_tensors(df_dict)
    assert ten_dict is not None, "Błąd w Etapie 4 (Tensory)"

    # Etap 5: Parametry modelu
    mod_dict = prepare_model_params(ten_dict, SEED)
    assert mod_dict is not None, "Błąd w Etapie 5 (Konfiguracja)"

    # Etap 6: Trening (Minimum 10 epok)
    mod_dict = train_model(ten_dict, mod_dict, epochs=MIN_EPOCHS)
    assert mod_dict is not None, f"Błąd w Etapie 6: Trening odrzucony dla {MIN_EPOCHS} epok"

    # Etap 7: Ewaluacja
    result = evaluate_model(mod_dict, df_dict, ten_dict)
    assert result is not None, "Błąd w Etapie 7 (Ewaluacja)"
    df_dict, ten_dict, mod_dict = result

    # Etap 8: Strategia
    strategy_results = simulate_strategy(df_dict, strategy=1)
    assert strategy_results is not None, "Błąd w Etapie 8 (Strategia)"

def test_model_guard_clause_epochs():
    """
    FUNCTIONAL TEST: Weryfikacja bezpiecznika liczby epok.
    """
    # Mockujemy minimalny słownik
    ten_dict = {'train_features_norm': [0]*100} 
    mod_dict = {'model': None}
    
    # Próba uruchomienia dla 1 epoki
    result = train_model(ten_dict, mod_dict, epochs=1)
    
    # Sprawdzamy czy bezpiecznik zwrócił None
    assert result is None, "Zabezpieczenie Etapu 6 NIE zadziałało dla epochs=1"
    