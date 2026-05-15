from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import Qt
from src.ui.utils import show_message
from src.ui.base_tab import BaseTab
from src.database_manager import DatabaseManager

class UserManagementTab(BaseTab):
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self.db_manager = db_manager

        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()

        left_column_layout = QVBoxLayout()

        self.load_users_button = QPushButton("Wczytaj użytkowników")
        left_column_layout.addWidget(self.load_users_button)

        self.add_user_button = QPushButton("Dodaj użytkownika")
        left_column_layout.addWidget(self.add_user_button)

        self.del_user_button = QPushButton("Usuń użytkownika")
        left_column_layout.addWidget(self.del_user_button)

        self.block_user_button = QPushButton("Zablokuj użytkownika")
        left_column_layout.addWidget(self.block_user_button)
        
        self.unblock_user_button = QPushButton("Odblokuj użytkownika")
        left_column_layout.addWidget(self.unblock_user_button)

        self.change_password_button = QPushButton("Zmień hasło")
        left_column_layout.addWidget(self.change_password_button)

        left_column_layout.addStretch()
        left_widget = QWidget()
        left_widget.setFixedWidth(200)
        left_widget.setLayout(left_column_layout)
        layout.addWidget(left_widget)

        right_column_layout = QVBoxLayout()

        self.user_table = QTableWidget()
        self.user_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.user_table.verticalHeader().setVisible(False)
        self.user_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        right_column_layout.addWidget(self.user_table)
        right_column_layout.addWidget(self.message_label)

        layout.addLayout(right_column_layout)

        self.setLayout(layout)

        self.load_users_button.clicked.connect(self.handle_load_users)

    def handle_load_users(self):
        try:
            self.db_manager.validate_access(self.user_id, self.session_token, "admin")

            users = self.db_manager.get_users()
            self.user_table.setRowCount(len(users))
            self.user_table.setColumnCount(4)
            self.user_table.setHorizontalHeaderLabels(["ID", "Nazwa", "Rola", "Zablokowany"])

            for row_index, user in enumerate(users):
                for col_index, value in enumerate(user):
                    self.user_table.setItem(row_index, col_index, QTableWidgetItem(str(value)))

        except Exception as e:
            show_message(self.message_label, str(e))
