# PDF Extraction Codes (Offline / Local)

Goal: extract, from the PDFs in `All_Cases/`:
- Page-2 flow-volume graph -> JPEG (for CNN input)
- Page-2 spirometry table -> CSV

No internet / no external APIs: this pipeline uses local libraries + optional local Tesseract OCR.

## Output Layout
By default, scripts write into `extracted/v1/` (at the project root):
- `extracted/v1/graphs_jpeg/<CASE_ID>_p2_graph.jpg` (spaces in the PDF stem are converted to `_`)
- `extracted/v1/graphs_debug/<CASE_ID>_p2_graph_dbg.jpg`
- `extracted/v1/tables_img/<CASE_ID>_p2_table.jpg`
- `extracted/v1/tables_debug/<CASE_ID>_p2_table_dbg.jpg`
- `extracted/v1/tables_csv/<CASE_ID>_p2_table.csv` (spaces in the PDF stem are converted to `_`)
- `extracted/v1/tables_qc/<CASE_ID>_p2_table_qc.json`
- `extracted/v1/tables_combined.csv`

## Install (Offline-Friendly)
You need a local Python environment with the packages below (no internet required *at runtime*).

Python packages:
- `pymupdf` (PyMuPDF / `fitz`) - PDF rendering + text extraction
- `opencv-python`
- `numpy`
- `pandas`
- `pyyaml` (optional; overrides files can also be JSON)
- `pytesseract` (optional; only needed for OCR fallback)

Local OCR binary (optional but recommended):
- Tesseract installed locally (Windows default path is supported).

## Scripts (v1.00)
- `PDF_Extraction_Codes/v1_00_graph_crop_page2.py`
  Crops the page-2 graph region (relative rectangle + optional per-file overrides).

- `PDF_Extraction_Codes/v1_00_table_extract_page2.py`
  Extracts the page-2 table. Default engine is `auto`:
  1) Try PDF text extraction first (best accuracy when the PDF has embedded text).
  2) Fall back to local Tesseract OCR if needed.

- `PDF_Extraction_Codes/v1_00_extract_all_cases.py`
  Convenience runner that calls both scripts over a directory of PDFs.

- `PDF_Extraction_Codes/v1_00_qc_report.py`
  Summarizes `tables_qc/*.json` so you can quickly see which cases need crop overrides or OCR tuning.

## Scripts (v1.10 - High Precision)
- `PDF_Extraction_Codes/v1_10_table_extract_page2_precise.py`
  Refactor focused on your drift problems:
  - strict 13-row schema
  - PRE vs POST vertical partition
  - left-label row anchoring + re-alignment fallbacks
  - Grade splitter to keep FET out of Grade rows

- `PDF_Extraction_Codes/v1_10_graph_extract_page2_contours.py`
  Finds the two main graph boxes on page 2 using contour detection and exports:
  `{case_id}_flow_loop.png` and `{case_id}_vol_time.png` (+ debug overlay).
  Use `--preset cgh_v1` for your current dataset template.

- `PDF_Extraction_Codes/v1_10_extract_all_cases.py`
  Runs v1.10 table + graph extraction in one command.

## Scripts (v1.11 - Z-Score Safety + Stronger Cell OCR)
- `PDF_Extraction_Codes/v1_11_table_extract_page2_precise.py`
  Builds on v1.10 with stronger per-cell OCR (more zoom + gridline removal) and better Z-score tie-breaking
  using LLN/Pred/BEST. Optional `--strict` will blank Z-scores that violate strong LLN/Pred/BEST constraints.

- `PDF_Extraction_Codes/v1_11_extract_all_cases.py`
  Runs v1.11 table + v1.10 graph extraction in one command.

## Scripts (v1.12 - Faster Default + Safer Cell OCR)
- `PDF_Extraction_Codes/v1_12_table_extract_page2_precise.py`
  Default behavior is faster: per-cell OCR runs only for missing/suspicious values (`--cell-ocr auto`).
  If cell OCR fails, numeric cells are blanked (instead of keeping junk text like `$`).

- `PDF_Extraction_Codes/v1_12_extract_all_cases.py`
  Runner for v1.12 table + v1.10 graph extraction.

## Quick Run
From the project root (`/mnt/c/Users/weibo/ASP/Project`):

Windows (recommended: always run from your project venv):

```bat
.\.venv\Scripts\python.exe PDF_Extraction_Codes\v1_00_extract_all_cases.py --pdf-dir All_Cases --out-dir extracted\v1
```

WSL/Linux style (only if your environment supports running the Windows venv Python from bash):

```bash
python PDF_Extraction_Codes/v1_00_extract_all_cases.py --pdf-dir All_Cases --out-dir extracted/v1
```

v1.10 (recommended for your drift issues):

```bat
.\.venv\Scripts\python.exe PDF_Extraction_Codes\v1_10_extract_all_cases.py --pdf-dir All_Cases --out-dir extracted\v1_10
```

```bash
python PDF_Extraction_Codes/v1_10_extract_all_cases.py --pdf-dir All_Cases --out-dir extracted/v1_10
```

If Tesseract is not on PATH, pass:

```bat
.\.venv\Scripts\python.exe PDF_Extraction_Codes\v1_10_table_extract_page2_precise.py --pdf-dir All_Cases --out-dir extracted\v1_10 --dpi 300 --engine ocr --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

```bash
python PDF_Extraction_Codes/v1_00_table_extract_page2.py --pdf-dir All_Cases --out-dir extracted/v1 --tesseract-cmd "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

QC summary:

```bash
python PDF_Extraction_Codes/v1_00_qc_report.py --qc-dir extracted/v1/tables_qc
```

For v1.10:

```bash
python PDF_Extraction_Codes/v1_00_qc_report.py --qc-dir extracted/v1_10/tables_qc
```

Handy Windows runners:
- `PDF_Extraction_Codes\\run_v1_00_table_auto.cmd`
- `PDF_Extraction_Codes\\run_v1_10_extract_all_cases.cmd`
- `PDF_Extraction_Codes\\run_v1_11_extract_all_cases.cmd`

## Crop Overrides
If a few PDFs are misaligned, use overrides:
- Graph overrides: `PDF_Extraction_Codes/v1_00_graph_overrides.yaml`
- Table overrides: `PDF_Extraction_Codes/v1_00_table_overrides.yaml`

Both accept either YAML (preferred) or JSON, with this shape:

```yaml
global:
  dx: 0.0
  dy: 0.0
  grow: 0.0
  # optional: rel_left/top/right/bottom overrides
per_file:
  "DEID_Case 12.pdf":
    dx: -0.02
    grow: 0.01
```

## Status
These v1.00 scripts are built from your previous working attempts in `HAT_programs and files/`,
but they are **not runnable inside the Codex sandbox here** (no Python deps installed).
Run them locally, then we can iterate based on the debug images + QC JSON.
