#!/usr/bin/env python3
"""
v1.10 - Extract Page-2 graphs (Flow-Volume loop + Volume-Time) using contour detection.

This avoids hard-coded bottom-right cropping by trying to find the 2 main plot boxes.
It writes PNGs (lossless) for stable CNN input.

Outputs (default out-dir = extracted/v1_10):
- graphs_png/<CASE_ID>_flow_loop.png
- graphs_png/<CASE_ID>_vol_time.png
- graphs_debug/<CASE_ID>_graphs_dbg.jpg

Notes:
- Graph layouts vary; this script is heuristic. Use the debug overlay to tune thresholds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from v1_00_common import ensure_dir, list_pdfs, sanitize_case_id


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v1.10 contour-based graph extractor (page 2).")
    parser.add_argument("--pdf-dir", type=Path, default=Path("All_Cases"))
    parser.add_argument("--out-dir", type=Path, default=Path("extracted/v1_10"))
    parser.add_argument("--page-index", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=250, help="Render DPI (250 is often enough for graphs).")
    parser.add_argument("--limit", type=int, default=0)

    # Detection knobs (start conservative; tune by debug overlay).
    parser.add_argument("--min-area-frac", type=float, default=0.02, help="Min contour bbox area / page area.")
    parser.add_argument("--max-area-frac", type=float, default=0.35, help="Max contour bbox area / page area.")
    parser.add_argument("--min-y-frac", type=float, default=0.30, help="Ignore boxes above this (table/header region).")
    parser.add_argument("--min-x-frac", type=float, default=0.18, help="Ignore boxes too far left (labels).")
    parser.add_argument("--kernel", type=int, default=15, help="Morph kernel size (odd-ish).")
    parser.add_argument("--swap", action="store_true", help="Swap left/right assignment (vol_time vs flow_loop).")
    parser.add_argument(
        "--preset",
        choices=["auto", "cgh_v1"],
        default="auto",
        help="Optional tuned preset for known templates (cgh_v1 matches current dataset well).",
    )
    return parser.parse_args()


def import_deps() -> Tuple[Any, Any, Any]:
    try:
        import fitz  # PyMuPDF
        import cv2
        import numpy as np
    except Exception as exc:
        raise SystemExit(
            f"Dependency import failed: {exc}. Install/repair: pymupdf opencv-python numpy"
        ) from exc
    return fitz, cv2, np


def render_page(fitz: Any, cv2: Any, np: Any, pdf_path: Path, page_index: int, dpi: int) -> Any:
    doc = fitz.open(str(pdf_path))
    if page_index >= len(doc):
        doc.close()
        raise RuntimeError(f"{pdf_path.name}: missing page index {page_index} (PDF has {len(doc)} pages)")
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img


def iou(a: Box, b: Box) -> float:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x2, b.x2)
    y1 = min(a.y2, b.y2)
    iw = max(0, x1 - x0)
    ih = max(0, y1 - y0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / float(max(1, union))


def detect_graph_boxes(cv2: Any, np: Any, page_bgr: Any, args: argparse.Namespace) -> List[Box]:
    h, w = page_bgr.shape[:2]
    page_area = float(h * w)

    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    # Edge emphasis: graphs have axes and dense traces.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 140)

    k = max(3, int(args.kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Box] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = int(bw * bh)
        if area <= 0:
            continue
        area_frac = area / page_area
        if area_frac < args.min_area_frac or area_frac > args.max_area_frac:
            continue
        if y < int(h * args.min_y_frac):
            continue
        if x < int(w * args.min_x_frac):
            continue
        ar = bw / float(max(1, bh))
        if ar < 0.6 or ar > 2.8:
            continue
        boxes.append(Box(x=x, y=y, w=bw, h=bh, area=area))

    # Keep large first and suppress overlaps.
    boxes.sort(key=lambda b: b.area, reverse=True)
    picked: List[Box] = []
    for b in boxes:
        if all(iou(b, p) < 0.15 for p in picked):
            picked.append(b)
        if len(picked) >= 4:
            break

    # Prefer 2 boxes (graphs). If >2, keep top 2 by area after NMS.
    picked.sort(key=lambda b: b.area, reverse=True)
    return picked[:2]


def main() -> int:
    args = parse_args()
    fitz, cv2, np = import_deps()

    base = Path.cwd()
    pdf_dir = (base / args.pdf_dir).resolve() if not args.pdf_dir.is_absolute() else args.pdf_dir
    out_root = (base / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir

    out_graphs = out_root / "graphs_png"
    out_dbg = out_root / "graphs_debug"
    ensure_dir(out_graphs)
    ensure_dir(out_dbg)

    pdfs = list_pdfs(pdf_dir)
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]

    processed = 0
    for pdf_path in pdfs:
        case_id = sanitize_case_id(pdf_path.stem)
        try:
            page = render_page(fitz, cv2, np, pdf_path, args.page_index, args.dpi)
        except Exception as exc:
            print(f"[ERROR] {pdf_path.name}: render failed: {exc}")
            continue

        # Apply tuned defaults if requested.
        if args.preset == "cgh_v1":
            args.min_area_frac = 0.005
            args.min_x_frac = 0.03
            args.min_y_frac = 0.45
            args.kernel = 11

        boxes = detect_graph_boxes(cv2, np, page, args)
        dbg = page.copy()
        for i, b in enumerate(boxes):
            cv2.rectangle(dbg, (b.x, b.y), (b.x2, b.y2), (0, 255, 0), 3)
            cv2.putText(
                dbg,
                f"box{i+1} {b.w}x{b.h}",
                (b.x + 6, max(18, b.y + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        dbg_path = out_dbg / f"{case_id}_graphs_dbg.jpg"
        cv2.imwrite(str(dbg_path), dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        if len(boxes) < 2:
            print(f"[WARN] {pdf_path.name}: detected {len(boxes)} graph boxes. See {dbg_path.name}")
            continue

        # Assign left/right.
        boxes.sort(key=lambda b: b.x)
        left, right = boxes[0], boxes[1]
        if args.swap:
            left, right = right, left

        # Default assignment: left = vol_time, right = flow_loop (common on many templates).
        vol_time = page[left.y : left.y2, left.x : left.x2].copy()
        flow_loop = page[right.y : right.y2, right.x : right.x2].copy()

        vol_path = out_graphs / f"{case_id}_vol_time.png"
        flow_path = out_graphs / f"{case_id}_flow_loop.png"
        cv2.imwrite(str(vol_path), vol_time)
        cv2.imwrite(str(flow_path), flow_loop)

        processed += 1
        print(f"[OK] {pdf_path.name} -> {vol_path.name}, {flow_path.name}")

    print(f"Done. Graphs written to: {out_graphs}")
    return 0 if processed else 2


if __name__ == "__main__":
    raise SystemExit(main())
