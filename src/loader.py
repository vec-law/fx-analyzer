class Loader:
    def __init__(self, config: dict, log_callback=None):
        """
        config: Konfiguracja (np. parametry ścieżki do danych, itp.)
        log_callback: Funkcja do logowania
        """
        self.config = config
        self.log = log_callback or (lambda msg: None)  # Logowanie jeśli nie przekazano sygnału

    def load_data(self):
        """Ładuje dane na podstawie configu"""
        try:
            self.log("Rozpoczynanie ładowania danych...")

            # Przykładowe ładowanie danych na podstawie konfiguracji
            raw_data = ['przykładowe', 'dane']  # Możesz dodać logikę ładowania danych

            self.log(f"Załadowano {len(raw_data)} rekordów danych.")
            return raw_data  # Zwracamy dane, które trafią do pipeline.raw

        except Exception as e:
            self.log(f"Błąd przy ładowaniu danych: {e}")
            return None
