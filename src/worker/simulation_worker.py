from PyQt6.QtCore import QObject, pyqtSignal
from src.pipeline.simulation_pipeline import SimulationPipeline

class SimulationWorker(QObject):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)

    def __init__(self, db_manager, sim_uuid):
        super().__init__()
        self.db_manager = db_manager
        self.sim_uuid = sim_uuid
        self.sim_pipeline = None

    def run(self):
        try:
            sim_config = self.db_manager.get_simulation_config(self.sim_uuid)
            self.sim_pipeline = SimulationPipeline(
                sim_config=sim_config, 
                log_signal=self.log_signal,
                db_manager=self.db_manager, 
                sim_uuid=self.sim_uuid
            )
            self.sim_pipeline.run()

        except Exception as e:
            error_msg = f"Błąd w wątku roboczym: {e}"
            self.log_signal.emit(error_msg)

            try:
                self.db_manager.update_simulation_status(self.sim_uuid, 'failed')
            except:
                pass

        finally:
            self.finished.emit()

    def stop(self):
        if self.sim_pipeline:
            self.sim_pipeline.stop()