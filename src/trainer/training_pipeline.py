from src.loader import Loader

class TrainingPipeline:
    def __init__(self, config: dict, log_signal=None):
        """
        config: Parametry treningu (np. epoki, dane, model itp.)
        log_signal: Sygnał do logowania (np. worker.log)
        """
        self.config = config
        self.log_signal = log_signal  # Przyjmujemy już sygnał, a nie funkcję
        self.raw = None  # To tutaj trafią dane po załadowaniu

    def run(self):
        """Uruchamia cały pipeline"""
        try:
            self.log_signal.emit("Start pipeline treningowego")

            # Loader - wczytaj dane na podstawie configu
            loader = Loader(self.config, log_callback=self.log_signal.emit)  # Przekazujemy sygnał logowania
            self.raw = loader.load_data()  # Dane zostaną zapisane w raw

            # Jeśli dane zostały wczytane, kontynuujemy
            if self.raw:
                self.log_signal.emit("Dane zostały wczytane poprawnie.")
            else:
                self.log_signal.emit("Błąd przy ładowaniu danych.")
                return

            # Kolejne etapy pipeline: preprocess, model itd. (jeśli będą)
            # np. self.preprocess()
            # np. self.train_model()

            self.log_signal.emit("Pipeline zakończony pomyślnie")

        except Exception as e:
            self.log_signal.emit(f"Błąd w pipeline: {e}")

    def stop(self):
        """Sygnalizuje pipeline, żeby się zatrzymał"""
        self.log_signal.emit("Pipeline zatrzymany.")
