from PyQt6.QtCore import QObject, pyqtSignal
from ml.pipeline.prediction_pipeline import PredictionPipeline

class PredictionWorker(QObject):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)

    def __init__(self, db_manager, pred_uuid):
        super().__init__()
        self.db_manager = db_manager
        self.pred_uuid = pred_uuid
        self.pred_pipeline = None

    def run(self):
        try:
            pred_config = self.db_manager.get_prediction_config(self.pred_uuid)
            self.pred_pipeline = PredictionPipeline(
                pred_config=pred_config, 
                log_signal=self.log_signal,
                db_manager=self.db_manager, 
                pred_uuid=self.pred_uuid
            )
            self.pred_pipeline.run()

        except Exception as e:
            error_msg = f"Błąd w wątku roboczym: {e}"
            self.log_signal.emit(error_msg)

            try:
                self.db_manager.update_prediction_status(self.pred_uuid, 'failed')
            except:
                pass

        finally:
            self.finished.emit()

    def stop(self):
        if self.pred_pipeline:
            self.pred_pipeline.stop()