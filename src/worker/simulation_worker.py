import time
from PyQt6.QtCore import QObject, pyqtSignal

class SimulationWorker(QObject):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)

    def __init__(self, db_manager, sim_uuid):
        super().__init__()
        self.db_manager = db_manager
        self.sim_uuid = sim_uuid
        self._is_running = True

    def run(self):
        try:
            # Poprawne: 'running' istnieje w bazie
            self.db_manager.update_simulation_status(self.sim_uuid, 'running')
            self.log_signal.emit(f"Starting simulation: {self.sim_uuid}")
            
            for i in range(10):
                if not self._is_running:
                    # Jeśli nie masz 'cancelled', użyj 'failed' lub dodaj go do bazy
                    self.db_manager.update_simulation_status(self.sim_uuid, 'failed')
                    self.log_signal.emit("Simulation stopped by user.")
                    break
                
                time.sleep(1)
                self.log_signal.emit(f"Simulation {self.sim_uuid}: Step {i+1}/10...")

            if self._is_running:
                # Poprawne: 'completed' istnieje w bazie
                self.db_manager.update_simulation_status(self.sim_uuid, 'completed')
                self.log_signal.emit("Simulation completed successfully.")

        except Exception as e:
            # ZMIANA: 'error' zamieniony na 'failed', bo tak masz w tabeli status
            self.db_manager.update_simulation_status(self.sim_uuid, 'failed')
            self.log_signal.emit(f"Critical error in worker: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._is_running = False