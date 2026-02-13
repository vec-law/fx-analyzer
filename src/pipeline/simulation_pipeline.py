import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from src.strategy import Strategy
import inspect
import io

class SimulationPipeline:
    def __init__(self, sim_config: dict, log_signal, db_manager, sim_uuid):
        self.sim_config = sim_config
        self.pred_config = None
        self.train_config = None
        self.log_signal = log_signal
        self.db_manager = db_manager
        self.sim_uuid = sim_uuid
        self._is_stopped = False

    def run(self):
        f_name = inspect.currentframe().f_code.co_name
        try:
            self.db_manager.update_simulation_status(self.sim_uuid, 'running')
            self.log_signal.emit(f"[{f_name}] Rozpoczynanie symulacji")

            self.pred_config = self.db_manager.get_prediction_config(self.sim_config['pred_uuid'])
            if self.pred_config is None:
                raise ValueError("Nie pobrano konfiguracji predykcji")
            if self._handle_stop(f_name): return

            self.train_config = self.db_manager.get_training_config(self.pred_config['train_uuid'])
            if self.train_config is None:
                raise ValueError("Nie pobrano konfiguracji treningu")
            if self._handle_stop(f_name): return

            for architecture in self.train_config['architectures']:
                data = self.db_manager.load_prediction_result(self.sim_config['pred_uuid'], architecture)
                if not data:
                    raise ValueError(f"Brak danych dla architektury {architecture}")
                if self._handle_stop(f_name): return

                df = pd.read_parquet(io.BytesIO(data))
                if df is None or df.empty:
                    raise ValueError(f"Nie odczytano danych dla architektury {architecture}")
                if self._handle_stop(f_name): return
                
                for target in self.train_config["target_names"]:
                    plt.scatter(df.index, df['close'], color='r', marker='.')
                    plt.plot(df.index, df[target], color='b')
                plt.savefig(r"output_plot.png", dpi=150)
                plt.close()

                for strategy_name in self.sim_config["strategies"]:
                    strategy = Strategy(self.log_signal, strategy_name)

                    df = strategy.add_signals(df, self.train_config["target_names"])
                    if df is None or df.empty:
                        raise ValueError(f"Nie odczytano danych dla architektury {architecture}")
                    if self._handle_stop(f_name): return

                    print(df)

            self.db_manager.update_simulation_status(self.sim_uuid, "completed")
            self.log_signal.emit(f"[{f_name}] Koniec symulacji")

        except Exception as e:
            self.log_signal.emit(f"[{f_name}] Przerwano z powodu błędu: {e}")
            try:
                self.db_manager.update_simulation_status(self.sim_uuid, 'failed')
            except Exception as db_err:
                self.log_signal.emit(f"[{f_name}] Błąd bazy danych: {db_err}")

    def stop(self):
        f_name = inspect.currentframe().f_code.co_name
        self._is_stopped = True
        self.log_signal.emit(f"[{f_name}] Zatrzymywanie...")

    def _handle_stop(self, f_name):
        if self._is_stopped:
            self.db_manager.update_simulation_status(self.sim_uuid, 'failed')
            self.log_signal.emit(f"[{f_name}] Proces przerwany przez użytkownika")
            return True
        return False
