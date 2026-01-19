class TrainingPipeline:
    def __init__(self, job_uuid, db_manager, log_cb):
        self.job_uuid = job_uuid
        self.db = db_manager
        self.log = log_cb
        self._running = True

    def stop(self):
        self._running = False
        self.log("Trening zatrzymany.")

    def run(self):
        self.log("Start treningu")
        
        for epoch in range(1, 101):
            if not self._running:
                self.log("Trening przerwany.")
                return
            self.log(f"Epoka {epoch}/100")

        self.log("Trening zakończony.")
