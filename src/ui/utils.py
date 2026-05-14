from PyQt6.QtWidgets import QLabel

def show_message(label: QLabel, text: str, success=False):
    color = "green" if success else "red"
    label.setStyleSheet(f"color: {color};")
    label.setText(text)
