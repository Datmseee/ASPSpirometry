@echo off
setlocal

REM v1.12 - Run table + graph extraction using the local venv Python.
REM Usage examples:
REM   PDF_Extraction_Codes\run_v1_12_extract_all_cases.cmd
REM   PDF_Extraction_Codes\run_v1_12_extract_all_cases.cmd --limit 10
REM   PDF_Extraction_Codes\run_v1_12_extract_all_cases.cmd --engine ocr --cell-ocr auto --cell-ocr-level fast --strict
REM   PDF_Extraction_Codes\run_v1_12_extract_all_cases.cmd --engine ocr --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"

set PYTHON=.venv\Scripts\python.exe
if not exist "%PYTHON%" (
  echo [ERROR] Could not find %PYTHON%
  echo Run this from the project root and ensure the venv exists.
  exit /b 2
)

"%PYTHON%" PDF_Extraction_Codes\v1_12_extract_all_cases.py --pdf-dir All_Cases --out-dir extracted\v1_12 %*
endlocal
