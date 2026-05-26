import os
import time
from dotenv import load_dotenv
from threading import Thread
from db.queries.trainings import count_running_trainings, get_pending_trainings, get_training_config
from ml.pipeline.training_pipeline import TrainingPipeline

load_dotenv()

def run_worker():
    while True:
        if count_running_trainings() < int(os.getenv("MAX_RUNNING_TRAININGS")) and \
            (train_pending_tasks := get_pending_trainings()):

            train_pending_task = train_pending_tasks[0]
            train_uuid = train_pending_task["train_uuid"]
            user_id = train_pending_task["user_id"]
            
            thread = Thread(
                target=TrainingPipeline(
                    user_id=user_id,
                    train_uuid=train_uuid,
                    config=get_training_config(user_id, train_uuid)
                ).run
            )
            thread.start()

        time.sleep(5)
