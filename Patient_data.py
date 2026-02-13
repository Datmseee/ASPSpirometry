from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt

class PatientDataPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PatientDataPage")

        # Set background color to match UploadPage
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(246, 250, 255, 255))  # #F6FAFF
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        layout.setAlignment(Qt.AlignTop)

        # Patient 1
        patient1_panel = QFrame(self)
        patient1_panel.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,0.95);
                border-radius: 14px;
                border: 1px solid #D0E6F7;
                box-shadow: 0 2px 12px rgba(0,159,227,0.08);
                padding: 24px 32px;
            }
        """)
        p1_layout = QVBoxLayout(patient1_panel)
        p1_layout.setSpacing(8)

        p1_name = QLabel("Name: John Tan")
        p1_name.setFont(QFont("Segoe UI", 14, QFont.Bold))
        p1_id = QLabel("Patient ID: CGH00123")
        p1_id.setFont(QFont("Segoe UI", 12))
        p1_issue = QLabel("Issue: Truncation of inspiratory limb")
        p1_issue.setFont(QFont("Segoe UI", 12))
        p1_issue.setStyleSheet("color: #D60000; font-weight: bold;")

        p1_layout.addWidget(p1_name)
        p1_layout.addWidget(p1_id)
        p1_layout.addWidget(p1_issue)

        # Patient 2
        patient2_panel = QFrame(self)
        patient2_panel.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,0.95);
                border-radius: 14px;
                border: 1px solid #D0E6F7;
                box-shadow: 0 2px 12px rgba(0,159,227,0.08);
                padding: 24px 32px;
            }
        """)
        p2_layout = QVBoxLayout(patient2_panel)
        p2_layout.setSpacing(8)

        p2_name = QLabel("Name: Sarah Lim")
        p2_name.setFont(QFont("Segoe UI", 14, QFont.Bold))
        p2_id = QLabel("Patient ID: CGH00456")
        p2_id.setFont(QFont("Segoe UI", 12))
        p2_issue = QLabel("Issue: Truncation of both limbs")
        p2_issue.setFont(QFont("Segoe UI", 12))
        p2_issue.setStyleSheet("color: #D60000; font-weight: bold;")

        p2_layout.addWidget(p2_name)
        p2_layout.addWidget(p2_id)
        p2_layout.addWidget(p2_issue)

        layout.addWidget(patient1_panel)
        layout.addWidget(patient2_panel)
        layout.addStretch(1)