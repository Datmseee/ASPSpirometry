import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QPushButton,
    QFileDialog, QMessageBox, QHBoxLayout, QListWidget, QListWidgetItem, QProgressBar,
    QDialog, QTextEdit
)
from typing import Optional, List

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette

from report_models import GeneratedReport


class DropArea(QFrame):
    """
    Drag and drop area that accepts multiple PDF files or folders.
    Calls the provided callback when valid files are dropped.
    """

    def __init__(self, on_files_selected, parent=None):
        super().__init__(parent)
        self.on_files_selected = on_files_selected

        self.setObjectName("uploadDropArea")
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Drag and drop PDF reports or a folder here")
        subtitle = QLabel("or click the buttons below to select files or a folder")

        title.setAlignment(Qt.AlignCenter)
        subtitle.setAlignment(Qt.AlignCenter)

        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title.setFont(title_font)
        subtitle.setStyleSheet("color: #009FE3; font-size: 12px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isdir(path) or path.lower().endswith(".pdf"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            paths.append(url.toLocalFile())
        if paths:
            self.on_files_selected(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class PredictionWorker(QThread):
    progress = pyqtSignal(int, str)
    status = pyqtSignal(str, str)
    report_ready = pyqtSignal(object)
    batch_done = pyqtSignal(list, list)

    def __init__(self, pdf_paths, asp_root, output_dir, parent=None):
        super().__init__(parent)
        self.pdf_paths = list(pdf_paths)
        self.asp_root = Path(asp_root)
        self.output_dir = Path(output_dir)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(self):
        completed = []
        failed = []
        total = len(self.pdf_paths)

        for idx, pdf_path in enumerate(self.pdf_paths):
            self.status.emit(pdf_path, "Queued")
            try:
                report = self._process_pdf(Path(pdf_path))
            except Exception as exc:
                report = GeneratedReport(
                    source_path=str(pdf_path),
                    predicted_path="",
                    status="Failed",
                    message=str(exc),
                    generated_at=datetime.now(),
                )

            self.report_ready.emit(report)
            if report.status == "Completed":
                completed.append(report)
            else:
                failed.append(report)

            self.status.emit(pdf_path, report.status)
            percent = int(((idx + 1) / max(1, total)) * 100)
            self.progress.emit(percent, f"Processed {idx + 1}/{total}")

        self.batch_done.emit(completed, failed)

    def _process_pdf(self, pdf_path: Path) -> GeneratedReport:
        if re.search(r"_predicted(_\\d+)?\\.pdf$", pdf_path.name, re.IGNORECASE):
            raise RuntimeError("Selected PDF looks like a generated prediction. Please use the original report PDF.")

        if not self.asp_root.exists():
            raise RuntimeError("asp_python_app folder not found.")

        table_script = self.asp_root / "extraction" / "v1_12_table_extract_page2_precise.py"
        classify_script = self.asp_root / "extraction" / "v1_14_table_classify.py"
        graph_script = self.asp_root / "extraction" / "v1_10_graph_extract_page2_contours.py"
        if not table_script.exists():
            raise RuntimeError("Table extraction script not found.")
        if not classify_script.exists():
            raise RuntimeError("Table classification script not found.")
        if not graph_script.exists():
            raise RuntimeError("Graph extraction script not found.")

        temp_pdf_dir = self.asp_root / "inputs" / "ui_queue"
        temp_pdf_dir.mkdir(parents=True, exist_ok=True)
        for existing in temp_pdf_dir.glob("*.pdf"):
            try:
                existing.unlink()
            except Exception:
                pass

        shutil.copy2(pdf_path, temp_pdf_dir / pdf_path.name)

        run_root = self.asp_root / "extracted" / "ui_runs" / self.run_id
        case_id = self._sanitize_case_id(pdf_path.stem)
        out_dir = run_root / case_id

        table_error = ""

        def _tables_csv_exists():
            table_dir = out_dir / "tables_csv"
            if not table_dir.exists():
                return False
            return any(table_dir.glob("*_p2_table.csv"))

        def _run_table(engine, cell_ocr, tesseract_cmd=None, cell_level="fast", timeout=240):
            cmd = [
                sys.executable,
                str(table_script),
                "--pdf-dir",
                str(temp_pdf_dir),
                "--out-dir",
                str(out_dir),
                "--dpi",
                "300",
                "--engine",
                engine,
                "--cell-ocr",
                cell_ocr,
                "--cell-ocr-level",
                cell_level,
            ]
            if tesseract_cmd:
                cmd += ["--tesseract-cmd", tesseract_cmd]
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        def _result_message(result, default):
            msg = (result.stderr or result.stdout or default).strip()
            if len(msg) > 800:
                msg = msg[-800:]
            return msg

        def _find_tesseract():
            candidates = [
                os.environ.get("TESSERACT_CMD"),
                r"C:\Users\yiyan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
            for cand in candidates:
                if cand and Path(cand).exists():
                    return cand
            return ""

        self.status.emit(str(pdf_path), "Table extraction (text)")
        try:
            table_result = _run_table("text", "off", timeout=180)
        except subprocess.TimeoutExpired:
            table_result = None
            table_error = "Table extraction timed out (text engine)."

        table_output = ""
        table_warn = False
        if table_result is not None:
            table_output = f"{table_result.stdout}\n{table_result.stderr}".lower()
            table_warn = "no tokens extracted" in table_output

        if (table_result is None) or (table_result.returncode != 0) or table_warn or not _tables_csv_exists():
            tesseract_cmd = _find_tesseract()
            if not tesseract_cmd:
                table_error = "Table extraction failed (no tokens extracted). Install Tesseract OCR or provide TESSERACT_CMD."
                if table_result is not None and (table_result.returncode != 0 or table_warn):
                    table_error += "\n" + _result_message(table_result, "Table extraction failed.")
            else:
                self.status.emit(str(pdf_path), "Table extraction (OCR)")
                try:
                    ocr_result = _run_table("ocr", "auto", tesseract_cmd, cell_level="fast", timeout=420)
                    if ocr_result.returncode != 0 or not _tables_csv_exists():
                        table_error = _result_message(ocr_result, "Table extraction failed with OCR.")
                except subprocess.TimeoutExpired:
                    table_error = "Table extraction timed out (OCR engine)."

        if not table_error and _tables_csv_exists():
            self.status.emit(str(pdf_path), "Table classification")
            classify_cmd = [
                sys.executable,
                str(classify_script),
                "--tables-dir",
                str(out_dir / "tables_csv"),
                "--out-dir",
                str(out_dir / "tables_classification"),
            ]
            try:
                classify_result = subprocess.run(classify_cmd, capture_output=True, text=True, timeout=180)
                if classify_result.returncode != 0:
                    table_error = _result_message(classify_result, "Table classification failed.")
            except subprocess.TimeoutExpired:
                table_error = "Table classification timed out."

        self.status.emit(str(pdf_path), "Graph extraction")
        graph_cmd = [
            sys.executable,
            str(graph_script),
            "--pdf-dir",
            str(temp_pdf_dir),
            "--out-dir",
            str(out_dir),
            "--preset",
            "cgh_v1",
        ]
        try:
            graph_result = subprocess.run(graph_cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Graph extraction timed out.")
        if graph_result.returncode != 0:
            message = (graph_result.stderr or graph_result.stdout or "Graph extraction failed.").strip()
            if len(message) > 600:
                message = message[-600:]
            raise RuntimeError(message)

        graphs_dir = out_dir / "graphs_png"
        if not graphs_dir.exists():
            raise RuntimeError("Graph extraction did not produce output.")

        if str(self.asp_root) not in sys.path:
            sys.path.insert(0, str(self.asp_root))
        try:
            from model_predicting import FVLPredict
        except Exception as exc:
            raise RuntimeError(f"Model import failed: {exc}") from exc

        labels, confidence, _ = FVLPredict.predict_from_directory(graphs_dir)
        if labels is None or len(labels) == 0:
            raise RuntimeError("No predictions generated.")

        exp_label = bool(labels[0][0])
        insp_label = bool(labels[0][1])
        exp_conf = float(confidence[0][0])
        insp_conf = float(confidence[0][1])

        classification = self._load_classification(out_dir)
        table_rows = self._load_table_rows(out_dir)

        predicted_path = self._build_predicted_path(self.output_dir, case_id)

        return GeneratedReport(
            source_path=str(pdf_path),
            predicted_path=str(predicted_path),
            status="Completed",
            message="",
            generated_at=datetime.now(),
            expiratory_truncation=exp_label,
            inspiratory_truncation=insp_label,
            exp_confidence=exp_conf,
            insp_confidence=insp_conf,
            classification=classification,
            table_rows=table_rows,
            table_error=table_error,
        )

    def _sanitize_case_id(self, stem: str) -> str:
        return re.sub(r"\s+", "_", (stem or "").strip())

    def _build_predicted_path(self, output_dir: Path, base_name: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._sanitize_case_id(base_name)
        candidate = output_dir / f"{safe_name}_predicted.pdf"
        if not candidate.exists():
            return candidate

        index = 1
        while True:
            candidate = output_dir / f"{safe_name}_predicted_{index}.pdf"
            if not candidate.exists():
                return candidate
            index += 1

    def _load_classification(self, out_dir: Path):
        class_dir = out_dir / "tables_classification"
        if not class_dir.exists():
            return None
        files = sorted(class_dir.glob("*_classification.json"))
        if not files:
            return None
        try:
            with files[0].open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _load_table_rows(self, out_dir: Path):
        table_dir = out_dir / "tables_csv"
        if not table_dir.exists():
            return []
        files = sorted(table_dir.glob("*_p2_table.csv"))
        if not files:
            return []
        rows = []
        try:
            with files[0].open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    metric = (row.get("metric") or "").strip()
                    if metric in {"FVC", "FEV1", "FEV1/FVC"}:
                        rows.append(
                            {
                                "metric": metric,
                                "pre_z": row.get("Pre_Z-Score", ""),
                                "post_z": row.get("Post_Z-Score", ""),
                                "pct": row.get("%CHG", ""),
                            }
                        )
        except Exception:
            return []
        return rows

    def _write_prediction_pdf(self, *args, **kwargs):
        raise RuntimeError("PDF generation must run on the UI thread.")

class UploadPage(QWidget):
    """
    Upload PDF Page - supports:
    - Drag and drop multiple PDFs or folders
    - Click-to-upload button (multi-select)
    - Folder selection for bulk prediction
    - Shows selected files in a full-width panel
    - Generate Predictions button with progress bar
    """

    def __init__(self, on_reports_generated=None, parent=None):
        super().__init__(parent)
        self.setObjectName("UploadPage")

        self.on_reports_generated = on_reports_generated
        self.pdf_paths = []  # store selected PDF paths
        self._file_status = {}
        self._file_errors = {}
        self._is_processing = False
        self._completed_reports = []
        self._worker = None

        self._init_ui()

    def _init_ui(self):
        # Set a light blue/white shadow background
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(246, 250, 255, 255))  # #F6FAFF
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(20)

        # Header
        title = QLabel("Upload Spirometry Reports")
        desc = QLabel("Upload one or more spirometry PDF reports, or a folder that contains them. "
                      "These will be used for OCR, AI prediction, and report generation.")

        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #009FE3; font-size: 13px;")

        main_layout.addWidget(title)
        main_layout.addWidget(desc)

        # Drag & Drop Area
        self.drop_area = DropArea(self._handle_files_selected, self)
        self.drop_area.setMinimumHeight(160)
        self.drop_area.setStyleSheet("""
            QFrame#uploadDropArea {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #F6FAFF, stop:1 #E6F2FF);
                border: 2px dashed #009FE3;
                border-radius: 18px;
                box-shadow: 0 2px 16px rgba(0,159,227,0.10);
            }
            QFrame#uploadDropArea:hover {
                border: 2px solid #007BB5;
                background: #E6F2FF;
            }
        """)

        main_layout.addWidget(self.drop_area)

        # Button row
        button_row = QHBoxLayout()
        button_row.setSpacing(15)

        # Upload Button
        self.upload_button = QPushButton("Select PDF Reports")
        self.upload_button.clicked.connect(self._open_file_dialog)
        self.upload_button.setObjectName("primaryUploadButton")
        self.upload_button.setMinimumWidth(180)
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #009FE3;
                color: white;
                border-radius: 10px;
                padding: 12px 28px;
                font-weight: 600;
                font-size: 16px;
                box-shadow: 0 1px 4px rgba(0,159,227,0.10);
                border: none;
                transition: background 0.2s;
            }
            QPushButton:hover {
                background-color: #007BB5;
            }
            QPushButton:pressed {
                background-color: #005F8C;
            }
        """)

        # Folder Button
        self.folder_button = QPushButton("Select Folder")
        self.folder_button.clicked.connect(self._open_folder_dialog)
        self.folder_button.setMinimumWidth(160)
        self.folder_button.setStyleSheet("""
            QPushButton {
                background-color: #4F86D6;
                color: white;
                border-radius: 10px;
                padding: 12px 22px;
                font-weight: 600;
                font-size: 16px;
                box-shadow: 0 1px 4px rgba(0,159,227,0.10);
                border: none;
                transition: background 0.2s;
            }
            QPushButton:hover {
                background-color: #3C6FBF;
            }
            QPushButton:pressed {
                background-color: #2F5A9C;
            }
        """)

        # Generate Predictions Button
        self.generate_button = QPushButton("Generate Predictions")
        self.generate_button.setEnabled(False)
        self.generate_button.setMinimumWidth(180)
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #00C48C;
                color: white;
                border-radius: 10px;
                padding: 12px 28px;
                font-weight: 600;
                font-size: 16px;
                box-shadow: 0 1px 4px rgba(0,196,140,0.10);
                border: none;
                transition: background 0.2s;
            }
            QPushButton:disabled {
                background-color: #B2DFDB;
                color: #f7f7f7;
            }
            QPushButton:hover:!disabled {
                background-color: #009E6D;
            }
            QPushButton:pressed:!disabled {
                background-color: #007A53;
            }
        """)
        self.generate_button.clicked.connect(self._start_generation)

        button_row.addWidget(self.upload_button)
        button_row.addWidget(self.folder_button)
        button_row.addWidget(self.generate_button)
        button_row.addStretch(1)

        main_layout.addLayout(button_row)

        # Full-width panel to show uploaded PDFs
        self.pdf_list = QListWidget()
        self.pdf_list.setObjectName("fileListPanel")
        self.pdf_list.setStyleSheet("""
            QListWidget#fileListPanel {
                background: rgba(255,255,255,0.95);
                border-radius: 14px;
                border: 1px solid #D0E6F7;
                font-size: 14px;
                padding: 12px;
                min-height: 120px;
                box-shadow: 0 2px 12px rgba(0,159,227,0.06);
            }
            QListWidget#fileListPanel::item {
                padding: 8px 0;
            }
        """)
        main_layout.addWidget(self.pdf_list)

        # Progress Bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border-radius: 8px;
                background: #E6F2FF;
                height: 24px;
                font-size: 14px;
                padding: 2px;
            }
            QProgressBar::chunk {
                background-color: #009FE3;
                border-radius: 8px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        main_layout.addStretch(1)

    # ---------- Logic ----------
    def _open_file_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Spirometry PDF Reports",
            "",
            "PDF Files (*.pdf)"
        )
        if paths:
            self._handle_files_selected(paths)

    def _open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Spirometry PDFs",
            "",
            QFileDialog.ShowDirsOnly
        )
        if not folder:
            return

        pdfs = self._collect_pdfs_from_folder(folder)
        if not pdfs:
            QMessageBox.information(self, "No PDFs Found", "No PDF files were found in the selected folder.")
            return
        self._handle_files_selected(pdfs)

    def _collect_pdfs_from_folder(self, folder_path):
        pdfs = []
        for root, _, files in os.walk(folder_path):
            for name in files:
                if name.lower().endswith(".pdf"):
                    pdfs.append(os.path.join(root, name))
        return pdfs

    def _handle_files_selected(self, paths):
        collected = []
        for path in paths:
            if os.path.isdir(path):
                collected.extend(self._collect_pdfs_from_folder(path))
            elif path.lower().endswith(".pdf"):
                collected.append(path)

        if not collected:
            return

        valid_paths = []
        ignored = []
        for path in collected:
            if self._is_valid_source_pdf(path):
                valid_paths.append(path)
            else:
                ignored.append(path)

        if ignored:
            QMessageBox.warning(
                self,
                "Ignored Files",
                "Some PDFs were ignored because they look like generated outputs.\n"
                "Please select the original spirometry PDFs.",
            )

        # Only add new PDFs, avoid duplicates
        for path in valid_paths:
            if path not in self.pdf_paths:
                self.pdf_paths.append(path)
                self._file_status[path] = "Pending"
                self._file_errors[path] = ""
        self._refresh_pdf_list()

    def _refresh_pdf_list(self):
        self.pdf_list.clear()
        if not self.pdf_paths:
            self.pdf_list.addItem("No files selected.")
            self.generate_button.setEnabled(False)
            return

        for path in self.pdf_paths:
            filename = path.split("/")[-1].split("\\")[-1]
            status = self._file_status.get(path, "Pending")
            item = QListWidgetItem(f"{filename}  [{status}]")
            error = self._file_errors.get(path, "")
            tooltip = path
            if error:
                tooltip = f"{path}\n\nError: {error}"
            item.setToolTip(tooltip)
            self.pdf_list.addItem(item)

        self.generate_button.setEnabled(not self._is_processing)

    def _start_generation(self):
        if self._is_processing or not self.pdf_paths:
            return

        invalid = [p for p in self.pdf_paths if not self._is_valid_source_pdf(p)]
        if invalid:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Some selected PDFs are generated outputs (predicted reports) and were removed.\n"
                "Please select the original spirometry PDFs.",
            )
            self.pdf_paths = [p for p in self.pdf_paths if self._is_valid_source_pdf(p)]
            for p in invalid:
                self._file_status.pop(p, None)
                self._file_errors.pop(p, None)
            self._refresh_pdf_list()
            if not self.pdf_paths:
                return

        self._is_processing = True
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting... (%p%)")
        self.generate_button.setEnabled(False)
        self.upload_button.setEnabled(False)
        self.folder_button.setEnabled(False)

        self._completed_reports = []

        for path in self.pdf_paths:
            self._file_status[path] = "Queued"
        self._refresh_pdf_list()

        base_dir = Path(__file__).resolve().parent
        asp_root = base_dir / "asp_python_app"
        output_dir = base_dir / "build" / "predicted_reports"

        self._worker = PredictionWorker(self.pdf_paths, asp_root, output_dir, parent=self)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.status.connect(self._on_worker_status)
        self._worker.report_ready.connect(self._on_worker_report)
        self._worker.batch_done.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        if message:
            self.progress_bar.setFormat(f"{message} (%p%)")

    def _on_worker_status(self, path, status):
        self._file_status[path] = status
        if status == "Processing":
            filename = os.path.basename(path)
            self.progress_bar.setFormat(f"Processing {filename} (%p%)")
        self._refresh_pdf_list()

    def _on_worker_report(self, report):
        if report.status == "Completed":
            try:
                self._render_report_pdf(report)
            except Exception as exc:
                report.status = "Failed"
                report.message = f"PDF generation failed: {exc}"
        self._file_status[report.source_path] = report.status
        self._refresh_pdf_list()
        self._completed_reports.append(report)
        if report.message:
            self._file_errors[report.source_path] = report.message

    def _on_worker_done(self, completed, failed):
        self._worker = None
        reports = list(self._completed_reports)
        completed = [r for r in reports if r.status == "Completed"]
        failed = [r for r in reports if r.status != "Completed"]
        all_reports = completed + failed
        if self.on_reports_generated and all_reports:
            self.on_reports_generated(all_reports)

        if failed:
            first_error = ""
            for report in failed:
                if report.message:
                    first_error = report.message
                    break
            self._show_error_dialog(
                "Prediction Completed",
                f"{len(completed)} reports generated, {len(failed)} failed. "
                f"Open Predicted Reports to download.",
                first_error,
            )
        else:
            QMessageBox.information(
                self,
                "Prediction Completed",
                f"{len(completed)} reports generated successfully. Open Predicted Reports to download.",
            )

        self.progress_bar.setVisible(False)
        self._is_processing = False
        self.upload_button.setEnabled(True)
        self.folder_button.setEnabled(True)

        self.pdf_paths.clear()
        self._file_status.clear()
        self._file_errors.clear()
        self._completed_reports = []
        self._refresh_pdf_list()

    # Public helper for later backend integration
    def get_selected_pdfs(self):
        return self.pdf_paths

    def _show_error_dialog(self, title, summary, details):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(700, 420)

        layout = QVBoxLayout(dialog)
        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(details or "No error details available.")
        layout.addWidget(text)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec_()

    def _is_valid_source_pdf(self, path: str) -> bool:
        name = os.path.basename(path)
        lower = path.replace("/", "\\").lower()
        if "\\build\\predicted_reports\\" in lower:
            return False
        if "\\asp_python_app\\extracted\\" in lower:
            return False
        if re.search(r"_predicted(_\\d+)?\\.pdf$", name, re.IGNORECASE):
            return False
        return True

    def _render_report_pdf(self, report: GeneratedReport) -> None:
        output_path = Path(report.predicted_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import fitz  # PyMuPDF
        except Exception as exc:
            raise RuntimeError(f"PyMuPDF not available: {exc}")

        exp_label = bool(report.expiratory_truncation)
        insp_label = bool(report.inspiratory_truncation)
        exp_conf = report.exp_confidence or 0.0
        insp_conf = report.insp_confidence or 0.0

        if exp_label and insp_label:
            overall = "Expiratory + Inspiratory truncation"
        elif exp_label:
            overall = "Expiratory truncation"
        elif insp_label:
            overall = "Inspiratory truncation"
        else:
            overall = "Normal"

        classification = report.classification or {}
        table_warning = report.table_error or ""
        if table_warning:
            table_warning = table_warning.splitlines()[0]

        lines = [
            f"Source PDF: {Path(report.source_path).name}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "GRAPH PREDICTION",
            f"Overall result: {overall}",
            f"Expiratory truncation: {'Yes' if exp_label else 'No'} (confidence {exp_conf:.2f})",
            f"Inspiratory truncation: {'Yes' if insp_label else 'No'} (confidence {insp_conf:.2f})",
            "",
            "TABLE CLASSIFICATION",
        ]

        if classification:
            lines.extend(
                [
                    f"Pre pattern: {classification.get('pre_pattern', 'N/A')} ({classification.get('pre_severity', 'n/a')})",
                    f"Post pattern: {classification.get('post_pattern', 'N/A')} ({classification.get('post_severity', 'n/a')})",
                    f"Report: {classification.get('report_combined', '')}",
                    f"Bronchodilator response: {classification.get('bronchodilator_response', '')}",
                ]
            )
        else:
            lines.append("Table classification not available.")

        if table_warning:
            lines.append(f"Table extraction warning: {table_warning}")

        if report.table_rows:
            lines.append("")
            lines.append("KEY TABLE Z-SCORES")
            for row in report.table_rows:
                lines.append(
                    f"{row.get('metric', '')}: Pre Z {row.get('pre_z') or 'N/A'}, "
                    f"Post Z {row.get('post_z') or 'N/A'}, %CHG {row.get('pct') or 'N/A'}"
                )

        text = "\n".join(lines)

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)

        margin = 40
        logo_w = 0
        logo_h = 0

        logo_path = Path(__file__).resolve().parent / "assets" / "cgh_logo.png"
        if logo_path.exists():
            try:
                pix = fitz.Pixmap(str(logo_path))
                logo_w = 120
                logo_h = max(1, int(logo_w * pix.height / pix.width))
                if logo_h > 50:
                    logo_h = 50
                    logo_w = max(1, int(logo_h * pix.width / pix.height))
                logo_rect = fitz.Rect(margin, 30, margin + logo_w, 30 + logo_h)
                page.insert_image(logo_rect, filename=str(logo_path))
            except Exception:
                logo_w = 0
                logo_h = 0

        header_top = 30 + (logo_h + 16 if logo_h else 0)
        header_height = 60
        header_left = margin
        header_width = 555 - margin
        header_rect = fitz.Rect(header_left, header_top, header_width, header_top + header_height)
        page.draw_rect(header_rect, color=(0.82, 0.90, 0.97), width=1)
        page.insert_text((header_left + 8, header_top + 22), "SPIROMETRY PREDICTION REPORT", fontsize=14, fontname="helv")
        page.insert_text((header_left + 8, header_top + 42), "Doctor in-charge: Dr Kundan", fontsize=11, fontname="helv")

        body_top = header_top + header_height + 20

        def section_height(line_count, title_height=18, line_height=14, padding=12):
            return title_height + (line_count * line_height) + padding

        def draw_section(y_start, title, body_lines):
            height = section_height(len(body_lines))
            rect = fitz.Rect(margin, y_start, 555, y_start + height)
            page.draw_rect(rect, color=(0.82, 0.90, 0.97), width=1)
            page.insert_text((margin + 8, y_start + 16), title, fontsize=12, fontname="helv")
            y_text = y_start + 30
            for line in body_lines:
                page.insert_text((margin + 10, y_text), line, fontsize=11, fontname="helv")
                y_text += 14
            return y_start + height + 14

        graph_lines = [
            f"Overall result: {overall}",
            f"Expiratory truncation: {'Yes' if exp_label else 'No'} (confidence {exp_conf:.2f})",
            f"Inspiratory truncation: {'Yes' if insp_label else 'No'} (confidence {insp_conf:.2f})",
        ]

        classification_lines = []
        if classification:
            classification_lines.extend(
                [
                    f"Pre pattern: {classification.get('pre_pattern', 'N/A')} ({classification.get('pre_severity', 'n/a')})",
                    f"Post pattern: {classification.get('post_pattern', 'N/A')} ({classification.get('post_severity', 'n/a')})",
                    f"Report: {classification.get('report_combined', '')}",
                    f"Bronchodilator response: {classification.get('bronchodilator_response', '')}",
                ]
            )
        else:
            classification_lines.append("Table classification not available.")
        if table_warning:
            classification_lines.append(f"Table extraction warning: {table_warning}")

        table_lines = []
        if report.table_rows:
            for row in report.table_rows:
                table_lines.append(
                    f"{row.get('metric', '')}: Pre Z {row.get('pre_z') or 'N/A'}, "
                    f"Post Z {row.get('post_z') or 'N/A'}, %CHG {row.get('pct') or 'N/A'}"
                )
        else:
            table_lines.append("No table values available.")

        y = body_top
        y = draw_section(y, "Graph Prediction", graph_lines)
        y = draw_section(y, "Table Classification", classification_lines)
        draw_section(y, "Key Table Z-scores", table_lines)

        doc.save(str(output_path))
        doc.close()
