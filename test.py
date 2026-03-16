try:
    # On Windows, importing torch after Qt can fail with WinError 1114.
    import torch  # noqa: F401
except Exception as exc:
    print(f"Warning: torch preload failed at startup: {exc}")

from qfluentwidgets import (
    FluentWindow, NavigationItemPosition,
    setTheme, Theme, FluentIcon
)
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette
import sys

from upload_page import UploadPage   # ✅ NEW IMPORT
from download_page import ReportDownloadPage
from Patient_data import PatientDataPage
from user import UserPage


class DashboardPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("DashboardPage")

        # Set background color to match UploadPage
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(246, 250, 255, 255))  # #F6FAFF
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.setAlignment(Qt.AlignTop)

        # Hospital image at the very top, stretched full width
        self.img_label = QLabel(self)
        self.img_label.setStyleSheet("background: #E6F2FF;")
        self.img_label.setFixedHeight(220)
        self.img_label.setSizePolicy(self.img_label.sizePolicy().Expanding, self.img_label.sizePolicy().Fixed)
        self.layout.addWidget(self.img_label)

        self._set_hospital_image()

        # Full-width greeting panel, left-aligned text
        greeting_panel = QFrame(self)
        greeting_panel.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,0.95);
                border-radius: 0px;
                border-bottom: 1px solid #D0E6F7;
                border-top: none;
                border-left: none;
                border-right: none;
                box-shadow: 0 2px 12px rgba(0,159,227,0.04);
                padding: 32px 48px;
            }
        """)
        greeting_layout = QVBoxLayout(greeting_panel)
        greeting_layout.setContentsMargins(0, 0, 0, 0)
        greeting_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        greeting_label = QLabel("Hello! Dr. Kundan")
        greeting_font = QFont()
        greeting_font.setPointSize(22)
        greeting_font.setBold(True)
        greeting_label.setFont(greeting_font)
        greeting_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        greeting_layout.addWidget(greeting_label)
        self.layout.addWidget(greeting_panel)

        # Title and subtitle (centered, but below the full-width panels)
        title = QLabel("Welcome to CGH Spirometry AI Dashboard")
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Your AI-powered assistant for spirometry reports")
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #009FE3;")

        self.layout.addSpacing(20)
        self.layout.addWidget(title)
        self.layout.addWidget(subtitle)
        self.layout.addStretch(1)

    def _set_hospital_image(self):
        pixmap = QPixmap("assets/hospital2.webp")
        if not pixmap.isNull():
            # Stretch to full width of parent
            width = self.width() if self.width() > 0 else 1100  # fallback width
            scaled = pixmap.scaled(width, 220, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.img_label.setPixmap(scaled)
            self.img_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        else:
            self.img_label.setText("Hospital image not found.")
            self.img_label.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._set_hospital_image()


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Changi General Hospital - Spirometry AI Assistant")
        self.resize(1100, 700)
        self._fixed_min_width = 800
        self.setMinimumWidth(self._fixed_min_width)

        # Theme
        setTheme(Theme.LIGHT)

        self._reports_page = ReportDownloadPage()

        # Optional: load your medical_theme.qss if you already created it
        try:
            with open("styles/medical_theme.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print("⚠ styles/medical_theme.qss not found – using default Fluent style.")

        # Hospital logo (make sure assets/cgh_logo.png exists)
        logo_label = QLabel(self)
        pixmap = QPixmap("assets/cgh_logo.png")
        if pixmap.isNull():
            print("⚠ ERROR: Logo not found at assets/cgh_logo.png")
        else:
            logo_label.setPixmap(pixmap.scaled(140, 40))
            self.titleBar.layout().insertWidget(1, logo_label)

        # ✅ Add pages to sidebar – NOTE THE ORDER: (page, icon, text, position)
        self.addSubInterface(
            DashboardPage(),
            FluentIcon.HOME,
            "Dashboard",
            NavigationItemPosition.TOP
        )

        self.upload_page = UploadPage(on_reports_generated=self._on_reports_generated)
        self.addSubInterface(
            self.upload_page,
            FluentIcon.ADD,
            "Upload Report",
            NavigationItemPosition.TOP
        )

        self.addSubInterface(
            self._reports_page,
            FluentIcon.DOWNLOAD,
            "Predicted Reports",
            NavigationItemPosition.TOP
        )

        self.addSubInterface(
            PatientDataPage(),
            FluentIcon.ACCEPT,  # You can choose another icon if you prefer
            "Patient Data",
            NavigationItemPosition.TOP
        )

        self.addSubInterface(
            UserPage(),
            FluentIcon.PEOPLE,
            "Account",
            NavigationItemPosition.BOTTOM
        )

        # Fix: Prevent sidebar expansion from permanently resizing window
        

    def _on_reports_generated(self, reports):
        if not reports:
            return

        self._reports_page.add_reports(reports)
        self.switchTo(self._reports_page)

    def _on_sidebar_expanded(self, expanded: bool):
        # When sidebar is expanded/collapsed, reset minimum width to original
        self.setMinimumWidth(800)

    def resizeEvent(self, event):
        # Always enforce minimum width
        if self.width() < self._fixed_min_width:
            self.resize(self._fixed_min_width, self.height())
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
