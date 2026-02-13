import os
import shutil

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from report_models import GeneratedReport


class ReportDownloadPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ReportDownloadPage")

        self._reports = []
        self._report_keys = set()
        self._empty_item = None
        self._checkboxes = []

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(246, 250, 255, 255))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignTop)

        title = QLabel("Predicted Spirometry Reports")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)

        desc = QLabel("Download the generated prediction PDFs for each patient.")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #009FE3; font-size: 13px;")

        layout.addWidget(title)
        layout.addWidget(desc)

        controls = QHBoxLayout()
        controls.setSpacing(10)

        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self._select_all)
        self.select_all_button.setStyleSheet(
            """
            QPushButton {
                background-color: #E6F2FF;
                color: #006EA0;
                border-radius: 8px;
                padding: 6px 16px;
                font-weight: 600;
                border: 1px solid #BBD9EE;
            }
            QPushButton:disabled {
                color: #94A7B5;
                border-color: #D0E6F7;
            }
            """
        )

        self.clear_selection_button = QPushButton("Clear Selection")
        self.clear_selection_button.clicked.connect(self._clear_selection)
        self.clear_selection_button.setStyleSheet(
            """
            QPushButton {
                background-color: #F5F8FC;
                color: #506070;
                border-radius: 8px;
                padding: 6px 16px;
                font-weight: 600;
                border: 1px solid #D0E6F7;
            }
            QPushButton:disabled {
                color: #94A7B5;
                border-color: #D0E6F7;
            }
            """
        )

        self.download_selected_button = QPushButton("Download Selected")
        self.download_selected_button.clicked.connect(self._download_selected)
        self.download_selected_button.setStyleSheet(
            """
            QPushButton {
                background-color: #009FE3;
                color: white;
                border-radius: 8px;
                padding: 6px 16px;
                font-weight: 600;
                border: none;
            }
            QPushButton:disabled {
                background-color: #B2DFDB;
                color: #f7f7f7;
            }
            QPushButton:hover:!disabled {
                background-color: #007BB5;
            }
            """
        )

        controls.addWidget(self.select_all_button)
        controls.addWidget(self.clear_selection_button)
        controls.addStretch(1)
        controls.addWidget(self.download_selected_button)

        layout.addLayout(controls)

        self.report_list = QListWidget()
        self.report_list.setObjectName("reportListPanel")
        self.report_list.setSizePolicy(self.report_list.sizePolicy().Expanding, self.report_list.sizePolicy().Expanding)
        self.report_list.setStyleSheet(
            """
            QListWidget#reportListPanel {
                background: rgba(255,255,255,0.95);
                border-radius: 14px;
                border: 1px solid #D0E6F7;
                font-size: 14px;
                padding: 12px;
                box-shadow: 0 2px 12px rgba(0,159,227,0.06);
            }
            """
        )
        layout.addWidget(self.report_list)
        self._set_empty_state()
        layout.setStretchFactor(self.report_list, 1)

    def add_reports(self, reports):
        if self._empty_item is not None:
            self.report_list.clear()
            self._empty_item = None
            self._checkboxes = []
        for report in reports:
            key = report.predicted_path or report.source_path
            if key in self._report_keys:
                continue
            self._report_keys.add(key)
            self._reports.append(report)
            self._add_report_item(report)
        self._update_controls_state()

    def _set_empty_state(self):
        self.report_list.clear()
        item = QListWidgetItem("No predicted reports yet.")
        item.setFlags(Qt.ItemIsEnabled)
        self.report_list.addItem(item)
        self._empty_item = item
        self._checkboxes = []
        self._update_controls_state()

    def _add_report_item(self, report: GeneratedReport):
        item = QListWidgetItem()

        container = QFrame()
        container.setStyleSheet(
            """
            QFrame {
                background: #FFFFFF;
                border: 1px solid #DDECF8;
                border-radius: 10px;
            }
            """
        )

        row = QHBoxLayout(container)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(10)

        checkbox = QCheckBox()
        checkbox.setEnabled(report.status == "Completed")
        self._checkboxes.append((checkbox, report))

        name = os.path.basename(report.predicted_path or report.source_path)
        name_label = QLabel(name)
        name_label.setFont(QFont("Segoe UI", 11, QFont.Bold))

        prediction_label = QLabel(self._format_prediction(report))
        prediction_label.setStyleSheet("color: #5A6B7A; font-size: 11px;")

        status_label = QLabel(report.status)
        status_color = "#00C48C" if report.status == "Completed" else "#D60000"
        status_label.setStyleSheet(f"color: {status_color}; font-weight: 600;")
        if report.status != "Completed" and report.message:
            status_label.setToolTip(report.message)

        download_button = QPushButton("Download PDF")
        download_button.setEnabled(report.status == "Completed")
        download_button.setStyleSheet(
            """
            QPushButton {
                background-color: #009FE3;
                color: white;
                border-radius: 8px;
                padding: 6px 16px;
                font-weight: 600;
                border: none;
            }
            QPushButton:disabled {
                background-color: #B2DFDB;
                color: #f7f7f7;
            }
            QPushButton:hover:!disabled {
                background-color: #007BB5;
            }
            """
        )
        download_button.clicked.connect(lambda _, r=report: self._download_report(r))

        info_col = QVBoxLayout()
        info_col.setSpacing(2)
        info_col.addWidget(name_label)
        info_col.addWidget(prediction_label)

        row.addWidget(checkbox)
        row.addLayout(info_col, 1)
        row.addWidget(status_label)
        row.addWidget(download_button)

        item.setSizeHint(container.sizeHint())
        self.report_list.addItem(item)
        self.report_list.setItemWidget(item, container)

    def _format_prediction(self, report: GeneratedReport) -> str:
        if report.status != "Completed":
            return "Prediction failed."
        if report.expiratory_truncation is None and report.inspiratory_truncation is None:
            return "Prediction results will appear after generation."

        exp_label = "Yes" if report.expiratory_truncation else "No"
        insp_label = "Yes" if report.inspiratory_truncation else "No"
        exp_conf = f"{report.exp_confidence:.2f}" if report.exp_confidence is not None else "N/A"
        insp_conf = f"{report.insp_confidence:.2f}" if report.insp_confidence is not None else "N/A"
        return (
            f"Expiratory truncation: {exp_label} ({exp_conf}) | "
            f"Inspiratory truncation: {insp_label} ({insp_conf})"
        )

    def _download_report(self, report: GeneratedReport):
        if report.status != "Completed":
            QMessageBox.warning(self, "Download unavailable", "This report failed to generate.")
            return

        default_name = os.path.basename(report.predicted_path or report.source_path)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Predicted Report",
            default_name,
            "PDF Files (*.pdf)",
        )
        if not save_path:
            return

        if not save_path.lower().endswith(".pdf"):
            save_path += ".pdf"

        source_path = report.predicted_path or report.source_path
        if not source_path or not os.path.exists(source_path):
            QMessageBox.warning(self, "File missing", "The predicted report file was not found.")
            return

        try:
            shutil.copy2(source_path, save_path)
        except Exception as exc:
            QMessageBox.warning(self, "Download failed", f"Unable to save file: {exc}")
            return

        QMessageBox.information(self, "Download complete", f"Saved to: {save_path}")

    def _update_controls_state(self):
        has_items = bool(self._reports)
        self.select_all_button.setEnabled(has_items)
        self.clear_selection_button.setEnabled(has_items)
        self.download_selected_button.setEnabled(has_items)

    def _select_all(self):
        for checkbox, report in self._checkboxes:
            if report.status == "Completed":
                checkbox.setChecked(True)

    def _clear_selection(self):
        for checkbox, _ in self._checkboxes:
            checkbox.setChecked(False)

    def _download_selected(self):
        selected = [r for cb, r in self._checkboxes if cb.isChecked() and r.status == "Completed"]
        if not selected:
            QMessageBox.information(self, "No selection", "Select at least one completed report to download.")
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Download Folder",
            "",
            QFileDialog.ShowDirsOnly
        )
        if not folder:
            return

        saved = 0
        skipped = 0
        for report in selected:
            source_path = report.predicted_path or report.source_path
            if not source_path or not os.path.exists(source_path):
                skipped += 1
                continue

            filename = os.path.basename(source_path)
            target_path = self._unique_path(folder, filename)
            try:
                shutil.copy2(source_path, target_path)
                saved += 1
            except Exception:
                skipped += 1

        QMessageBox.information(
            self,
            "Download complete",
            f"Saved {saved} file(s) to {folder}. Skipped {skipped}.",
        )

    def _unique_path(self, folder, filename):
        base, ext = os.path.splitext(filename)
        candidate = os.path.join(folder, filename)
        if not os.path.exists(candidate):
            return candidate
        index = 1
        while True:
            candidate = os.path.join(folder, f"{base}_{index}{ext}")
            if not os.path.exists(candidate):
                return candidate
            index += 1
