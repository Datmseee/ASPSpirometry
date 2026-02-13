from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt

class UserPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("UserPage")

        # Set background color to match other pages
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(246, 250, 255, 255))  # #F6FAFF
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        layout.setAlignment(Qt.AlignTop)

        # User info panel
        user_panel = QFrame(self)
        user_panel.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,0.95);
                border-radius: 14px;
                border: 1px solid #D0E6F7;
                box-shadow: 0 2px 12px rgba(0,159,227,0.08);
                padding: 32px 48px;
            }
        """)
        panel_layout = QVBoxLayout(user_panel)
        panel_layout.setSpacing(12)

        name_label = QLabel("Name: Dr. Kundan")
        name_label.setFont(QFont("Segoe UI", 15, QFont.Bold))
        id_label = QLabel("User ID: CGH-ADMIN-001")
        id_label.setFont(QFont("Segoe UI", 13))
        branch_label = QLabel("Branch: Changi General Hospital")
        branch_label.setFont(QFont("Segoe UI", 13))

        panel_layout.addWidget(name_label)
        panel_layout.addWidget(id_label)
        panel_layout.addWidget(branch_label)

        layout.addWidget(user_panel)
        layout.addStretch(1)