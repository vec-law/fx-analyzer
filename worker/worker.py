import os
import time
from db.manager import DatabaseManager
from dotenv import load_dotenv
from threading import Thread

load_dotenv()

def run_worker(db_manager: DatabaseManager):
    while True:
        if db_manager.count_running_trainings() < int(os.getenv("MAX_RUNNING_TRAININGS")) and \
            (train_penging_tasks := db_manager.get_pending_trainings()):
            thread = Thread(target=funkcja, args=(argument,))
            thread.start()
            pass
             # run train_penging_tasks[0]
        time.sleep(5)