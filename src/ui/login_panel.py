from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton, QCheckBox, QLabel

class LoginPanel(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        
        self.user_id = None
        self.session_token = None
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()

        user_name_label = QLabel("Nazwa użytkownika:")
        layout.addWidget(user_name_label)

        self.user_name_input = QLineEdit()
        layout.addWidget(self.user_name_input)

        password_label = QLabel("Hasło:")
        layout.addWidget(password_label)
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password_input)

        self.admin_checkbox = QCheckBox("Zarejestruj się jako administrator")
        layout.addWidget(self.admin_checkbox)

        self.register_button = QPushButton("Zarejestruj")
        layout.addWidget(self.register_button)

        self.login_button = QPushButton("Zaloguj")
        layout.addWidget(self.login_button)

        self.logout_button = QPushButton("Wyloguj")
        layout.addWidget(self.logout_button)
        self.logout_button.setEnabled(False)

        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

        layout.addStretch()
        self.setFixedWidth(250)
        self.setLayout(layout)

        self.login_button.clicked.connect(self.handle_login)
        self.logout_button.clicked.connect(self.handle_logout)
        self.register_button.clicked.connect(self.handle_register)

    def handle_login(self):
        try:
            self.show_message("")

            user_name = self.user_name_input.text()
            password = self.password_input.text()

            if not user_name or not password:
                raise ValueError("Żadne z pól (login, hasło) nie może być puste")

            self.user_id, self.session_token = self.db_manager.login_user(user_name, password)

            self.user_name_input.setEnabled(False)
            self.password_input.setEnabled(False)
            self.admin_checkbox.setEnabled(False)
            self.register_button.setEnabled(False)
            self.login_button.setEnabled(False)
            self.logout_button.setEnabled(True)
            self.admin_checkbox.setChecked(False)
            
            self.show_message(f"Zalogowano użytkownika {user_name}", True)

        except Exception as e:
            self.show_message(str(e))

    def handle_logout(self):
        try:
            self.show_message("")

            user_name = self.user_name_input.text()

            self.db_manager.logout_user(user_name)

            self.user_id, self.session_token = None, None

            self.user_name_input.setEnabled(True)
            self.password_input.setEnabled(True)
            self.admin_checkbox.setEnabled(True)
            self.register_button.setEnabled(True)
            self.login_button.setEnabled(True)
            self.logout_button.setEnabled(False)

            self.password_input.clear()

            self.show_message(f"Wylogowano użytkownika {user_name}", True)

        except Exception as e:
            self.show_message(str(e))

    def handle_register(self):
        try:
            self.show_message("")
            
            user_name = self.user_name_input.text()
            password = self.password_input.text()

            if not user_name or not password:
                raise ValueError("Żadne z pól (login, hasło) nie może być puste")
            
            self.db_manager.register_user(user_name, password, self.admin_checkbox.isChecked())

            self.admin_checkbox.setChecked(False)
            self.user_name_input.clear()
            self.password_input.clear()

            self.show_message(f"Zarejestrowano użytkownika {user_name}", True)

        except Exception as e:
            self.show_message(str(e))

    def show_message(self, text, success=False):
        color = "green" if success else "red"
        self.message_label.setStyleSheet(f"color: {color};")
        self.message_label.setText(text)
