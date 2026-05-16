from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import Qt
from src.ui.utils import show_message
from src.ui.base_tab import BaseTab
from src.database_manager import DatabaseManager
from src.ui.change_password_popup import ChangePasswordPopup
from src.ui.add_user_popup import AddUserPopup
from src.ui.del_user_popup import DelUserPopup

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
        self.user_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.user_table.verticalHeader().setVisible(False)
        self.user_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        self.user_table.setSortingEnabled(True)
        right_column_layout.addWidget(self.user_table)
        right_column_layout.addWidget(self.message_label)

        layout.addLayout(right_column_layout)

        self.setLayout(layout)

        self.load_users_button.clicked.connect(self.handle_load_users)
        self.add_user_button.clicked.connect(self.handle_add_user)
        self.del_user_button.clicked.connect(self.handle_del_user)
        self.change_password_button.clicked.connect(self.handle_change_password)

    def handle_load_users(self):
        try:
            show_message(self.message_label, "")
            
            self.db_manager.validate_access(self.user_id, self.session_token, "admin")

            users = [u for u in self.db_manager.get_users() if u[0] != self.user_id]
            self.user_table.setRowCount(len(users))
            self.user_table.setColumnCount(4)
            self.user_table.setHorizontalHeaderLabels(["ID", "Nazwa", "Rola", "Zablokowany"])

            for row_index, user in enumerate(users):
                for col_index, value in enumerate(user):
                    self.user_table.setItem(row_index, col_index, QTableWidgetItem(str(value)))

        except Exception as e:
            show_message(self.message_label, str(e))

    def handle_add_user(self):
        try:
            show_message(self.message_label, "")
            
            self.db_manager.validate_access(self.user_id, self.session_token, "admin")

            self.add_user_popup = AddUserPopup(self.db_manager)

            self.add_user_popup.user_added.connect(
                lambda: self.on_action_success("Dodano użytkownika")
            )

            self.add_user_popup.show()

        except Exception as e:
            show_message(self.message_label, str(e))

    def handle_del_user(self):
        try:
            show_message(self.message_label, "")

            selected_user_id = self.get_selected_user_id()

            if selected_user_id is None:
                raise ValueError("Nie wybrano żadnego użytkownika")
            
            self.db_manager.validate_access(self.user_id, self.session_token, "admin")
            
            self.del_user_popup = DelUserPopup(
                selected_user_id,
                self.db_manager
            )

            self.del_user_popup.user_deleted.connect(
                lambda: self.on_action_success("Usunięto użytkownika")
            )
            
            self.del_user_popup.show()

        except Exception as e:
            show_message(self.message_label, str(e))

    def handle_change_password(self):
        try:
            show_message(self.message_label, "")

            selected_user_id = self.get_selected_user_id()

            if selected_user_id is None:
                raise ValueError("Nie wybrano żadnego użytkownika")
            
            self.db_manager.validate_access(self.user_id, self.session_token, "admin")
            
            self.change_password_popup = ChangePasswordPopup(selected_user_id, self.db_manager)

            self.change_password_popup.password_changed.connect(
                lambda: self.on_action_success("Zmieniono hasło użytkownika")
            )
            
            self.change_password_popup.show()

        except Exception as e:
            show_message(self.message_label, str(e))

    def get_selected_user_id(self):
        row_index = self.user_table.currentRow()

        if row_index == -1: return None

        return int(self.user_table.item(row_index, 0).text())
    
    def on_action_success(self, message):
        self.handle_load_users()
        show_message(self.message_label, message, True)

    def set_session(self, user_id, session_token):
        super().set_session(user_id, session_token)
        self.handle_load_users()