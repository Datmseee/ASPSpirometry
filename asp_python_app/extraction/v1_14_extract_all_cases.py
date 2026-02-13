#!/usr/bin/env python3
"""
v1.14 - Integrated pipeline:
- v1_12_table_extract_page2_precise.py (table extraction)
- v1_10_graph_extract_page2_contours.py (graph extraction)
- v1_14_table_classify.py (table classification)

Integration notes (for teammate):
- This is the orchestration entrypoint for the end-to-end workflow.
- It runs in three steps (tables -> graphs -> classification) and can skip any step:
  --skip-tables / --skip-graphs / --skip-classify
- Classification reads the tables produced in the same --out-dir (tables_csv).
- It is safe to call this from a larger Python app via subprocess, or import and call main().
  If integrating directly, pass args that match your dataset path and desired output dir.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v1.14 table + graph + classification pipeline.")
    parser.add_argument("--pdf-dir", type=Path, default=Path("All_Cases"))
    parser.add_argument("--out-dir", type=Path, default=Path("extracted/v1_12"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cases", type=str, default="")

    # Table extraction options (forwarded).
    parser.add_argument("--engine", choices=["auto", "text", "ocr"], default="auto")
    parser.add_argument("--tesseract-cmd", type=str, default=None)
    parser.add_argument("--tessdata", type=str, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--cell-ocr", choices=["auto", "force", "off"], default="auto")
    parser.add_argument("--cell-ocr-level", choices=["fast", "full"], default="fast")

    # Graph extraction options.
    parser.add_argument("--swap-graphs", action="store_true")
    parser.add_argument("--graph-preset", choices=["auto", "cgh_v1"], default="cgh_v1")

    # Classification options.
    parser.add_argument("--bmi", type=float, default=None)
    parser.add_argument("--bmi-map", type=Path, default=None)

    # Skip flags.
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--skip-graphs", action="store_true")
    parser.add_argument("--skip-classify", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd))
    return subprocess.call(cmd)


def main(**kwargs) -> int:
    args = parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)
    print(args)
    here = Path(__file__).resolve().parent

    table_script = here / "v1_12_table_extract_page2_precise.py"
    graph_script = here / "v1_10_graph_extract_page2_contours.py"
    classify_script = here / "v1_14_table_classify.py"

    rc = 0

    if not args.skip_tables:
        table_cmd = [
            sys.executable,
            str(table_script),
            "--pdf-dir",
            str(args.pdf_dir),
            "--out-dir",
            str(args.out_dir),
            "--dpi",
            str(args.dpi),
            "--engine",
            args.engine,
            "--cell-ocr",
            args.cell_ocr,
            "--cell-ocr-level",
            args.cell_ocr_level,
        ]
        if args.limit:
            table_cmd += ["--limit", str(args.limit)]
        if args.cases:
            table_cmd += ["--cases", args.cases]
        if args.tesseract_cmd:
            table_cmd += ["--tesseract-cmd", args.tesseract_cmd]
        if args.tessdata:
            table_cmd += ["--tessdata", args.tessdata]
        if args.strict:
            table_cmd += ["--strict"]
        rc = max(rc, run(table_cmd))

    if not args.skip_graphs:
        graph_cmd = [
            sys.executable,
            str(graph_script),
            "--pdf-dir",
            str(args.pdf_dir),
            "--out-dir",
            str(args.out_dir),
            "--preset",
            args.graph_preset,
        ]
        if args.limit:
            graph_cmd += ["--limit", str(args.limit)]
        if args.cases:
            # Graph script does not support --cases; run full set if needed.
            pass
        if args.swap_graphs:
            graph_cmd += ["--swap"]
        rc = max(rc, run(graph_cmd))

    if not args.skip_classify:
        classify_cmd = [
            sys.executable,
            str(classify_script),
            "--tables-dir",
            str(args.out_dir / "tables_csv"),
            "--out-dir",
            str(args.out_dir / "tables_classification"),
        ]
        if args.limit:
            classify_cmd += ["--limit", str(args.limit)]
        if args.cases:
            classify_cmd += ["--cases", args.cases]
        if args.bmi is not None:
            classify_cmd += ["--bmi", str(args.bmi)]
        if args.bmi_map:
            classify_cmd += ["--bmi-map", str(args.bmi_map)]
        rc = max(rc, run(classify_cmd))

    return 0 if rc == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
