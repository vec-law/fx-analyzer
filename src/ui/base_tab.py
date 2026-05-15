from PyQt6.QtWidgets import QWidget, QLabel

class BaseTab(QWidget):
    def __init__(self):
        super().__init__()

        self.user_id = None
        self.session_token = None

        self.message_label = QLabel("")

    def set_session(self, user_id, session_token):
        self.user_id = user_id
        self.session_token = session_token

    def clear_session(self):
        self.user_id = None
        self.session_token = None
