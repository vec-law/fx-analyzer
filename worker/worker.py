import os
import time
from db.manager import DatabaseManager
from dotenv import load_dotenv
from threading import Thread
from ml.pipeline.training_pipeline import TrainingPipeline

load_dotenv()

def run_worker(db_manager: DatabaseManager):
    while True:
        if db_manager.count_running_trainings() < int(os.getenv("MAX_RUNNING_TRAININGS")) and \
            (train_pending_tasks := db_manager.get_pending_trainings()):

            train_uuid = train_pending_tasks[0]
            
            thread = Thread(
                target=TrainingPipeline(
                    config=db_manager.get_training_config(train_uuid),
                    db_manager=db_manager,
                    train_uuid=train_uuid
                ).run
            )
            thread.start()

        time.sleep(5)
