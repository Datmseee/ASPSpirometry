@echo off
setlocal

REM v1.14 - Table + graph + classification pipeline.
REM Usage:
REM   PDF_Extraction_Codes\run_v1_14_pipeline.cmd --cases "DEID_Case 1.pdf"
REM   PDF_Extraction_Codes\run_v1_14_pipeline.cmd --engine auto --cell-ocr auto --cell-ocr-level fast

set PYTHON=.venv\Scripts\python.exe
if not exist "%PYTHON%" (
  echo [ERROR] Could not find %PYTHON%
  echo Run this from the project root and ensure the venv exists.
  exit /b 2
)

"%PYTHON%" PDF_Extraction_Codes\v1_14_extract_all_cases.py --pdf-dir All_Cases --out-dir extracted\v1_12 --dpi 300 %*
endlocal
