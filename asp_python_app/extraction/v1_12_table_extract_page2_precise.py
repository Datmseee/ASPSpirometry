#!/usr/bin/env python3
"""
v1.12 - High-precision Page-2 table extractor (offline, no APIs).

Focus vs v1.11:
- Much faster default: cell OCR runs only when a value is missing/suspicious (auto mode).
- Safer: if cell OCR fails, numeric cells become blank (not junk like '$').

Key changes vs v1.00:
- Uses strict 13-row schema with row re-alignment safeguards.
- Splits PRE vs POST zones (vertical partition) to reduce "drift".
- Anchors row centers from detected metric labels (left side) when possible.
- Includes "Grade splitter" to prevent Grade rows absorbing FET values.

Engines:
- auto (default): try PDF text extraction first (PyMuPDF), validate, then OCR fallback.
- text: PDF text only
- ocr: local Tesseract OCR only

Outputs (default out-dir = extracted/v1_12):
- tables_csv/<CASE_ID>_p2_table.csv
- tables_debug/<CASE_ID>_p2_table_dbg.jpg
- tables_img/<CASE_ID>_p2_table.png
- tables_qc/<CASE_ID>_p2_table_qc.json
- tables_combined.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from v1_00_common import (
    Token,
    apply_rect_overrides,
    clean_cell_text,
    ensure_dir,
    find_tesseract_cmd,
    has_digit,
    kmeans_1d,
    list_pdfs,
    load_overrides,
    normalize_key,
    sanitize_case_id,
)


# ---------------- Schema ----------------
ROW_NAMES: List[str] = [
    "FVC",
    "FEV1",
    "FEV1/FVC",
    "VC",
    "FEV1/VC",
    "FEF25%-75%",
    "PEF",
    "MVV",
    "Grade FVC N",
    "Grade FEV1 N",
    "FET",
    "FIVC",
    "BEV",
]

OUT_COLUMNS: List[str] = [
    "PRE-BEST",
    "LLN",
    "Pred",
    "Pre_%Pred",
    "Pre_Z-Score",
    "POST-BEST",
    "Post_%Pred",
    "Post_Z-Score",
    "%CHG",
]

PRE_COLUMNS: List[str] = ["PRE-BEST", "LLN", "Pred", "Pre_%Pred", "Pre_Z-Score"]
POST_COLUMNS: List[str] = ["POST-BEST", "Post_%Pred", "Post_Z-Score", "%CHG"]

RESTRICTED_ROWS_PRE_POST = {
    "VC",
    "FEV1/VC",
    "Grade FVC N",
    "Grade FEV1 N",
    "FET",
    "FIVC",
    "BEV",
}
RESTRICTED_ALLOWED_COLS = {"PRE-BEST", "POST-BEST"}


# ---------------- Header/Row Aliases ----------------
HEADER_ALIASES: Dict[str, str] = {
    "prebest": "PRE-BEST",
    "pre-best": "PRE-BEST",
    "pre": "PRE-BEST",
    "lln": "LLN",
    "pred": "Pred",
    "prediction": "Pred",
    # %Pred and Z-Score appear in both PRE and POST; disambiguate by x position (see detect_header_positions()).
    "%pred": "Pre_%Pred",
    "pred%": "Pre_%Pred",
    "pre%pred": "Pre_%Pred",
    "prepred": "Pre_%Pred",
    "zscore": "Pre_Z-Score",
    "prezscore": "Pre_Z-Score",
    "prez": "Pre_Z-Score",
    "postbest": "POST-BEST",
    "post-best": "POST-BEST",
    "post": "POST-BEST",
    # Sometimes OCR yields "post%pred"/"postpred" explicitly.
    "post%pred": "Post_%Pred",
    "postpred": "Post_%Pred",
    "postzscore": "Post_Z-Score",
    "postz": "Post_Z-Score",
    "%chg": "%CHG",
    "pctchg": "%CHG",
    "chg": "%CHG",
}

ROW_ALIASES: Dict[str, str] = {
    normalize_key("FEF25-75%"): "FEF25%-75%",
    normalize_key("FEF 25-75%"): "FEF25%-75%",
    normalize_key("FEF25%-75%"): "FEF25%-75%",
    normalize_key("Grade FVC"): "Grade FVC N",
    normalize_key("Grade FVC N"): "Grade FVC N",
    normalize_key("Grade FEV1"): "Grade FEV1 N",
    normalize_key("Grade FEV1 N"): "Grade FEV1 N",
    normalize_key("FEV1FVC"): "FEV1/FVC",
    normalize_key("FEV1VC"): "FEV1/VC",
}
for _name in ROW_NAMES:
    ROW_ALIASES.setdefault(normalize_key(_name), _name)


@dataclass
class QcResult:
    pdf_name: str
    case_id: str
    engine_used: str
    rows_found: int
    warnings: List[str]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_overrides = here / "v1_00_table_overrides.yaml"

    parser = argparse.ArgumentParser(description="v1.12 precise page-2 table extractor (offline).")
    parser.add_argument("--pdf-dir", type=Path, default=Path("All_Cases"))
    parser.add_argument("--out-dir", type=Path, default=Path("extracted/v1_12"))
    parser.add_argument("--overrides", type=Path, default=default_overrides, help="Band crop overrides YAML/JSON.")

    parser.add_argument("--page-index", type=int, default=1, help="0-based page index (default: 1 = page 2).")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI.")

    # Initial band crop (relative to page image).
    # Slightly wider than v1.00 to avoid cutting off left-side metric labels on some cases.
    parser.add_argument("--rel-left", type=float, default=0.02)
    parser.add_argument("--rel-top", type=float, default=0.305)
    parser.add_argument("--rel-right", type=float, default=0.92)
    parser.add_argument("--rel-bottom", type=float, default=0.505)
    parser.add_argument("--extra-bottom", type=float, default=0.02)

    parser.add_argument("--engine", choices=["auto", "text", "ocr"], default="auto")
    parser.add_argument("--tesseract-cmd", type=str, default=None)
    parser.add_argument("--tessdata", type=str, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Safety mode: if a Z-score violates strong LLN/Pred/BEST constraints, leave it blank + emit a QC warning.",
    )
    parser.add_argument(
        "--cell-ocr",
        choices=["auto", "force", "off"],
        default="auto",
        help="Cell-level OCR strategy: auto=only suspicious/missing cells, force=always on critical columns, off=never.",
    )
    parser.add_argument(
        "--cell-ocr-level",
        choices=["fast", "full"],
        default="fast",
        help="Cell OCR effort: fast=fewer OCR passes, full=more passes (slower).",
    )

    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help='Optional comma-separated list of PDF names or stems to process (e.g. "DEID_Case 15.pdf,DEID_Case 18").',
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max PDFs to process (0 = all).")
    return parser.parse_args()


def import_deps() -> Tuple[Any, Any, Any, Any]:
    try:
        import fitz  # PyMuPDF
        import cv2
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit(
            f"Dependency import failed: {exc}. Install/repair: pymupdf opencv-python numpy pandas"
        ) from exc
    return fitz, cv2, np, pd


def _preprocess_for_ocr(img_bgr: Any, *, scale: float = 3.0, border: int = 10) -> Tuple[Any, float, int]:
    """
    Upscale, enhance contrast, binarize, and pad an image region for Tesseract.

    Steps:
      1. Upscale by `scale` using bicubic interpolation (Tesseract accuracy improves
         significantly above ~300 DPI effective resolution).
      2. Convert to greyscale.
      3. CLAHE: normalises local contrast so faint ink and coloured cell backgrounds
         don't confuse the binariser.
      4. Gaussian blur: removes scanner noise / JPEG artefacts before thresholding.
      5. Otsu binarisation: produces clean black-on-white regardless of the original
         background colour (coloured header rows, grey stripes, etc.).
      6. White border: prevents Tesseract from clipping characters at image edges.

    Returns (processed_img, scale, border) so callers can map token coordinates
    back to the original image space:
        orig_x = (tesseract_x - border) / scale
    """
    import cv2

    up = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    padded = cv2.copyMakeBorder(binary, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    return padded, scale, border


def render_page_image(fitz: Any, cv2: Any, np: Any, pdf_path: Path, page_index: int, dpi: int) -> Any:
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


def crop_band(img: Any, *, rel_rect: Tuple[float, float, float, float], extra_bottom: float) -> Tuple[Any, Tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    l, t, r, b = rel_rect
    x0 = max(0, int(w * l))
    y0 = max(0, int(h * t))
    x1 = min(w, int(w * r))
    y1 = min(h, int(h * min(1.0, b + extra_bottom)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid band crop; adjust rel rect / extra-bottom.")
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


def tighten_table_crop(cv2: Any, np: Any, band: Any) -> Tuple[Any, Tuple[int, int, int, int], Any]:
    """
    Tighten crop to colored table block (HSV mask), with safe fallbacks.
    Returns (table_img, bbox_in_band_px, mask).
    """
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = cv2.bitwise_and(cv2.inRange(s, 40, 255), cv2.inRange(v, 40, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if cv2.countNonZero(mask) < 0.002 * mask.size:
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        edges = cv2.Canny(enhanced, 50, 150)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k2, iterations=2)
        mask = cv2.dilate(mask, k2, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = band.shape[:2]
        return band.copy(), (0, 0, w, h), mask

    rects = [cv2.boundingRect(cnt) for cnt in contours]  # (x,y,w,h)
    x_list, y_list, w_list, h_list = zip(*rects)

    # Union bbox is robust when colored regions are sparse, but can accidentally include
    # the Z-score axis band below the table (edge-heavy). If the union is much taller than
    # the largest component, prefer the largest component.
    x_u = min(x_list)
    y_u = min(y_list)
    x_u_end = max(x0 + w0 for x0, w0 in zip(x_list, w_list))
    y_u_end = max(y0 + h0 for y0, h0 in zip(y_list, h_list))
    w_u = x_u_end - x_u
    h_u = y_u_end - y_u

    x_b, y_b, w_b, h_b = max(rects, key=lambda r: r[2] * r[3])

    x, y, w, h = x_u, y_u, w_u, h_u
    if h_b > 0 and h_u > h_b * 1.18 and w_b >= int(0.55 * band.shape[1]):
        x, y, w, h = x_b, y_b, w_b, h_b

    # Asymmetric padding: labels on the left are often low-saturation and can be excluded by the mask.
    pad = 10
    pad_left = max(pad, int(band.shape[1] * 0.04))
    pad_bottom = max(pad, int(band.shape[0] * 0.02))

    # If the detected bbox starts too far right, force-include the left label column.
    if x > int(band.shape[1] * 0.03):
        x0 = 0
    else:
        x0 = max(x - pad_left, 0)

    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, band.shape[1])
    y1 = min(y + h + pad_bottom, band.shape[0])
    crop_w = x1 - x0
    crop_h = y1 - y0

    band_h, band_w = band.shape[:2]
    band_area = float(band_h * band_w)
    crop_area = float(max(crop_w, 1) * max(crop_h, 1))
    if crop_area < band_area * 0.25 or (crop_w / max(band_w, 1)) < 0.55 or (crop_h / max(band_h, 1)) < 0.5:
        return band.copy(), (0, 0, band_w, band_h), mask

    return band[y0:y1, x0:x1].copy(), (x0, y0, crop_w, crop_h), mask


def tokens_from_pdf_text(
    fitz: Any,
    pdf_path: Path,
    *,
    page_index: int,
    clip_rect_page: Any,
    zoom: float,
    band_offset_px: Tuple[int, int],
    table_bbox_in_band_px: Tuple[int, int, int, int],
) -> List[Token]:
    doc = fitz.open(str(pdf_path))
    page = doc[page_index]
    words = page.get_text("words", clip=clip_rect_page) or []
    doc.close()

    bx, by = band_offset_px
    tx, ty, _, _ = table_bbox_in_band_px
    out: List[Token] = []
    for x0, y0, x1, y1, w, *_ in words:
        text = clean_cell_text(str(w))
        if not text:
            continue
        cx = (float(x0) + float(x1)) / 2.0
        cy = (float(y0) + float(y1)) / 2.0
        out.append(
            Token(
                text=text,
                conf=1.0,
                cx=cx * zoom - bx - tx,
                cy=cy * zoom - by - ty,
                x0=float(x0) * zoom - bx - tx,
                y0=float(y0) * zoom - by - ty,
                x1=float(x1) * zoom - bx - tx,
                y1=float(y1) * zoom - by - ty,
            )
        )
    return out


def tokens_from_tesseract(table_img: Any, *, tesseract_cmd: str, tessdata: Optional[str] = None) -> List[Token]:
    try:
        import pytesseract  # type: ignore
    except Exception as exc:
        raise RuntimeError("pytesseract not installed; cannot use OCR engine.") from exc

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    if tessdata:
        os.environ["TESSDATA_PREFIX"] = tessdata

    processed, scale, border = _preprocess_for_ocr(table_img, scale=3.0, border=10)
    inv = 1.0 / scale
    cfg = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, config=cfg)
    n = len(data.get("text", []))
    toks: List[Token] = []
    for i in range(n):
        txt = clean_cell_text(data["text"][i] or "")
        if not txt:
            continue
        try:
            conf = float(data.get("conf", [0.0] * n)[i])
        except Exception:
            conf = 0.0
        if conf < 30:
            continue
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        if w <= 0 or h <= 0:
            continue
        # Map from preprocessed-image coordinates back to original table_img coordinates.
        x0 = (x - border) * inv
        y0 = (y - border) * inv
        x1 = (x + w - border) * inv
        y1 = (y + h - border) * inv
        toks.append(
            Token(
                text=txt,
                conf=float(conf) / 100.0,
                cx=(x0 + x1) / 2.0,
                cy=(y0 + y1) / 2.0,
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
            )
        )
    return toks


def _tokens_from_tesseract_image(
    img_bgr: Any,
    *,
    tesseract_cmd: str,
    tessdata: Optional[str],
    scale: float,
    offset_x: float,
    offset_y: float,
) -> List[Token]:
    """
    OCR a given BGR image region with Tesseract and return tokens in the *parent* image coordinates.
    We upscale before OCR for accuracy, then map boxes back down.
    """
    import pytesseract  # type: ignore

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    if tessdata:
        os.environ["TESSDATA_PREFIX"] = tessdata

    if scale <= 0:
        scale = 1.0

    processed, actual_scale, border = _preprocess_for_ocr(img_bgr, scale=scale, border=10)
    inv = 1.0 / actual_scale
    cfg = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, config=cfg)
    n = len(data.get("text", []))

    toks: List[Token] = []
    for i in range(n):
        txt = clean_cell_text(data["text"][i] or "")
        if not txt:
            continue
        try:
            conf = float(data.get("conf", [0.0] * n)[i])
        except Exception:
            conf = 0.0
        if conf < 30:
            continue
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        if w <= 0 or h <= 0:
            continue

        # Map back to parent coordinates (undo border padding and scale, then add region offset).
        x0 = ((x - border) * inv) + offset_x
        y0 = ((y - border) * inv) + offset_y
        x1 = ((x + w - border) * inv) + offset_x
        y1 = ((y + h - border) * inv) + offset_y
        toks.append(
            Token(
                text=txt,
                conf=float(conf) / 100.0,
                cx=(x0 + x1) / 2.0,
                cy=(y0 + y1) / 2.0,
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
            )
        )
    return toks


def tokens_from_tesseract_split(
    table_img: Any,
    *,
    tesseract_cmd: str,
    tessdata: Optional[str],
    split_x_guess: Optional[int] = None,
    overlap_frac: float = 0.10,
    scale: float = 2.5,
) -> List[Token]:
    """
    Run OCR separately on PRE and POST halves, then merge tokens.

    Rationale: improves accuracy by zooming in on a smaller region and reducing token confusion
    between PRE and POST columns.
    """
    h, w = table_img.shape[:2]
    if split_x_guess is None:
        # Template heuristic: PRE spans roughly ~60% width.
        split_x_guess = int(w * 0.62)

    overlap = int(w * max(0.0, min(0.25, overlap_frac)))
    pre_x1 = min(w, split_x_guess + overlap)
    post_x0 = max(0, split_x_guess - overlap)

    pre = table_img[:, :pre_x1].copy()
    post = table_img[:, post_x0:].copy()

    toks = []
    toks.extend(
        _tokens_from_tesseract_image(
            pre,
            tesseract_cmd=tesseract_cmd,
            tessdata=tessdata,
            scale=scale,
            offset_x=0.0,
            offset_y=0.0,
        )
    )
    toks.extend(
        _tokens_from_tesseract_image(
            post,
            tesseract_cmd=tesseract_cmd,
            tessdata=tessdata,
            scale=scale,
            offset_x=float(post_x0),
            offset_y=0.0,
        )
    )
    return toks


def group_tokens_by_line(tokens: Sequence[Token], *, max_dy: float = 8.0) -> List[List[Token]]:
    if not tokens:
        return []
    toks = sorted(tokens, key=lambda t: (t.cy, t.cx))
    groups: List[List[Token]] = []
    cur: List[Token] = [toks[0]]
    for t in toks[1:]:
        if abs(t.cy - cur[-1].cy) <= max_dy:
            cur.append(t)
        else:
            groups.append(cur)
            cur = [t]
    groups.append(cur)
    return groups


def detect_body_y0(tokens: Sequence[Token], image_h: int) -> float:
    """
    Estimate where the header ends and the first data row begins.

    Important: using a fixed fraction (e.g. 0.22) tends to accidentally exclude the top rows (FVC/FEV1).
    We instead use the bottom of the detected header tokens when possible.
    """
    if not tokens:
        return image_h * 0.16

    # Only look near the top; avoids matching axis labels like "Z-Score (PRE)" near the bottom.
    top = [t for t in tokens if t.cy <= image_h * 0.30 and re.search(r"[A-Za-z%]", t.text or "")]
    if not top:
        return image_h * 0.16

    header_like = []
    for t in top:
        norm = normalize_key(t.text)
        if norm in HEADER_ALIASES:
            header_like.append(t)

    if not header_like:
        return image_h * 0.16

    y_bottom = max(t.y1 for t in header_like)
    # Keep the body start in a reasonable band.
    body_y0 = y_bottom + 8.0
    body_y0 = max(image_h * 0.08, min(image_h * 0.35, body_y0))
    return float(body_y0)


def detect_header_positions(tokens: Sequence[Token], body_y0: float) -> Dict[str, float]:
    header_tokens = [t for t in tokens if t.cy <= body_y0]
    groups = group_tokens_by_line(header_tokens, max_dy=6.0)

    # Merge adjacent tokens in a group to help match "PRE BEST".
    merged: List[Token] = []
    for g in groups:
        g = sorted(g, key=lambda t: t.cx)
        if not g:
            continue
        cur = g[0]
        for t in g[1:]:
            if (t.x0 - cur.x1) <= 18:
                cur = Token(
                    text=clean_cell_text(f"{cur.text} {t.text}"),
                    conf=min(cur.conf, t.conf),
                    cx=(cur.cx + t.cx) / 2.0,
                    cy=(cur.cy + t.cy) / 2.0,
                    x0=min(cur.x0, t.x0),
                    y0=min(cur.y0, t.y0),
                    x1=max(cur.x1, t.x1),
                    y1=max(cur.y1, t.y1),
                )
            else:
                merged.append(cur)
                cur = t
        merged.append(cur)

    # Pass 1: locate POST-BEST x to disambiguate duplicated headers (%Pred, Z-Score).
    post_best_x: Optional[float] = None
    for t in merged:
        norm = normalize_key(t.text)
        if HEADER_ALIASES.get(norm) == "POST-BEST":
            post_best_x = t.cx if post_best_x is None else max(post_best_x, t.cx)

    def classify(norm: str, cx: float) -> Optional[str]:
        # Disambiguate duplicated headers by x position relative to POST-BEST.
        if norm in {"%pred", "pred%", "prepred", "pre%pred"}:
            if post_best_x is not None and cx > post_best_x + 30:
                return "Post_%Pred"
            return "Pre_%Pred"
        if norm in {"zscore", "prezscore", "prez"}:
            if post_best_x is not None and cx > post_best_x + 30:
                return "Post_Z-Score"
            return "Pre_Z-Score"
        return HEADER_ALIASES.get(norm)

    best: Dict[str, Tuple[float, float]] = {}
    for t in merged:
        norm = normalize_key(t.text)
        canonical = classify(norm, t.cx)
        if not canonical:
            continue
        prev = best.get(canonical)
        if prev is None or t.conf > prev[1]:
            best[canonical] = (t.cx, t.conf)
    return {k: v[0] for k, v in best.items()}


def detect_label_anchors(tokens: Sequence[Token], image_w: int, body_y0: float) -> Dict[int, float]:
    """
    Try to find y-centers for each canonical row by reading metric labels on the left.
    Returns a mapping of row_index -> y_center.
    """
    label_x_cutoff = image_w * 0.28
    left = [t for t in tokens if t.cy > body_y0 and t.cx <= label_x_cutoff]
    groups = group_tokens_by_line(left, max_dy=10.0)

    anchors: Dict[int, Tuple[float, float]] = {}  # idx -> (y, score)
    for g in groups:
        g = sorted(g, key=lambda t: t.cx)
        label = clean_cell_text(" ".join(t.text for t in g))
        if not label:
            continue
        norm = normalize_key(label)
        # Quick alias check.
        canonical = ROW_ALIASES.get(norm)
        score = 1.0 if canonical else 0.0
        if not canonical:
            # Fuzzy match to avoid missing "FEF25-75%" vs "FEF25%-75%".
            import difflib

            best_name = None
            best_score = 0.0
            for name in ROW_NAMES:
                s = difflib.SequenceMatcher(None, norm, normalize_key(name)).ratio()
                if s > best_score:
                    best_score = s
                    best_name = name
            if best_name and best_score >= 0.55:
                canonical = best_name
                score = best_score

        if not canonical or canonical not in ROW_NAMES:
            continue
        idx = ROW_NAMES.index(canonical)
        y = float(sum(t.cy for t in g) / max(1, len(g)))
        prev = anchors.get(idx)
        if prev is None or score > prev[1]:
            anchors[idx] = (y, score)

    return {idx: y for idx, (y, _) in anchors.items()}


def fit_row_centers(np: Any, anchors: Dict[int, float], image_h: int, body_y0: float) -> List[float]:
    """
    Build row centers for all 13 rows.
    - If we have enough label anchors: fit y = a*idx + b.
    - Else: evenly space within body.
    """
    n = len(ROW_NAMES)
    if len(anchors) >= 6:
        xs = np.array(list(anchors.keys()), dtype=np.float64)
        ys = np.array([anchors[i] for i in anchors.keys()], dtype=np.float64)
        a, b = np.polyfit(xs, ys, 1)
        centers = [float(a * i + b) for i in range(n)]
        # Clamp to body.
        lo, hi = body_y0 + 5, image_h - 5
        centers = [float(min(hi, max(lo, c))) for c in centers]
        return centers

    # Fallback: if some anchors exist, estimate median spacing from them.
    if len(anchors) >= 2:
        items = sorted(anchors.items(), key=lambda kv: kv[0])
        diffs = []
        for (i0, y0), (i1, y1) in zip(items, items[1:]):
            di = max(1, i1 - i0)
            diffs.append((y1 - y0) / float(di))
        dy = float(sorted(diffs)[len(diffs) // 2]) if diffs else (image_h - header_zone) / float(n)
        # Seed b from the first anchor.
        i0, y0 = items[0]
        b = y0 - dy * i0
        centers = [float(dy * i + b) for i in range(n)]
        lo, hi = body_y0 + 5, image_h - 5
        centers = [float(min(hi, max(lo, c))) for c in centers]
        return centers

    # No anchors: evenly spaced.
    lo, hi = body_y0 + (image_h - body_y0) * 0.06, image_h * 0.98
    return [float(lo + (hi - lo) * (i + 0.5) / n) for i in range(n)]


def build_windows_from_centers(centers: Sequence[float], *, lo: float, hi: float) -> List[Tuple[float, float]]:
    if not centers:
        return []
    centers = list(centers)
    edges = []
    for i, c in enumerate(centers):
        if i == 0:
            top = lo
        else:
            top = (centers[i - 1] + c) / 2.0
        if i == len(centers) - 1:
            bot = hi
        else:
            bot = (c + centers[i + 1]) / 2.0
        edges.append((float(top), float(bot)))
    return edges


def estimate_data_x0(tokens: Sequence[Token], image_w: int, body_y0: float) -> float:
    """
    Estimate the left edge of the numeric data block.

    Important: label strings like "FEV1/FVC" can extend far right; using label x1 alone often
    cuts off the PRE-BEST column. We instead prefer the left-most numeric tokens.
    """
    numeric = [t for t in tokens if t.cy > body_y0 and has_digit(t.text)]
    if numeric:
        xs = sorted(float(t.x0) for t in numeric)
        # Use a low percentile to avoid a single stray number near the labels.
        idx = int(round(0.05 * (len(xs) - 1)))
        x0_p = xs[max(0, min(len(xs) - 1, idx))]
        x0 = float(x0_p - 8.0)  # small margin into the cell, but avoid grabbing label text
        return float(max(image_w * 0.12, min(image_w * 0.45, x0)))

    # Fallback: use label right edge percentile.
    label_region = [t for t in tokens if t.cy > body_y0 and t.cx <= image_w * 0.45 and re.search(r"[A-Za-z]", t.text)]
    if not label_region:
        return float(image_w * 0.22)
    x1s = sorted(t.x1 for t in label_region)
    x0 = float(x1s[int(0.9 * (len(x1s) - 1))] + 10.0)
    return float(max(image_w * 0.10, min(image_w * 0.45, x0)))


def detect_split_x(headers: Dict[str, float], image_w: int, data_x0: float) -> Optional[float]:
    # If we see the POST header, split just left of it.
    if "POST-BEST" in headers:
        return float(max(data_x0 + 20.0, headers["POST-BEST"] - 25.0))
    # Otherwise, assume a mid split (template-dependent).
    return None


def infer_zone_centers(tokens: Sequence[Token], *, x0: float, x1: float, k: int, image_h: int) -> List[float]:
    body_y0 = detect_body_y0(tokens, image_h)
    body = [t for t in tokens if t.cy > body_y0 and x0 <= t.cx <= x1]
    numeric = [t for t in body if has_digit(t.text)]
    xs = [t.cx for t in numeric] if len(numeric) >= 10 else [t.cx for t in body]
    xs = [x for x in xs if x0 <= x <= x1]
    centers = kmeans_1d(xs, k)
    if not centers:
        centers = [x0 + (x1 - x0) * (i + 0.5) / k for i in range(k)]
    return centers


def apply_header_anchors(
    centers: List[float],
    *,
    headers: Dict[str, float],
    cols: Sequence[str],
) -> List[float]:
    """
    Turn a raw list of cluster centers into a per-column list, using detected header x positions
    when available. This reduces column swapping (e.g., %Pred vs Z-Score).
    """
    if not centers:
        return []
    centers_sorted = sorted(float(x) for x in centers)
    out: List[float] = []
    j = 0
    for col in cols:
        if col in headers:
            out.append(float(headers[col]))
            while j < len(centers_sorted) and centers_sorted[j] <= headers[col]:
                j += 1
        else:
            if j < len(centers_sorted):
                out.append(float(centers_sorted[j]))
                j += 1
            else:
                out.append(out[-1] + max(1.0, 8.0))
    # Monotonic enforce.
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1.0
    return out


def parse_float(text: str) -> Optional[float]:
    s = (text or "").strip()
    if not s:
        return None
    s = s.replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_grade(text: str) -> Optional[str]:
    s = (text or "").strip().upper()
    m = re.search(r"\b([A-F])\b", s)
    return m.group(1) if m else None


def _num_from_text(raw: str) -> Optional[float]:
    """
    More forgiving numeric parser than parse_float().

    Handles common OCR patterns:
    - "-0 49" -> -0.49
    - "1 21"  -> 1.21
    - stray symbols around numbers
    """
    s = (raw or "").strip()
    if not s:
        return None
    s = s.replace(",", ".")
    m = re.match(r"^\s*([+-]?\d+)\s+(\d{1,3})\s*$", s)
    if m:
        s = f"{m.group(1)}.{m.group(2)}"
    s = re.sub(r"[^0-9eE\.\-\+]", "", s)
    m2 = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not m2:
        return None
    try:
        return float(m2.group(0))
    except Exception:
        return None


def _pick_number_string_for_spec(raw: str, spec: CellSpec) -> Optional[str]:
    """
    When OCR accidentally merges label text into a numeric cell, the string can contain multiple numbers
    (e.g., "FEF25-75%1.89"). Prefer the numeric substring most compatible with the expected cell spec.
    """
    s = (raw or "").strip().replace(",", ".")
    if not s:
        return None

    matches = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", s))
    if not matches:
        return None

    # Prefer decimals when the spec expects decimals and a decimal is present.
    if spec.decimals is not None:
        dec = [m.group(0) for m in matches if "." in m.group(0)]
        if dec:
            return dec[-1]

    # Prefer an in-range integer for int-like cells.
    if spec.kind == "int":
        for m in reversed(matches):
            if "." in m.group(0):
                continue
            try:
                v = float(m.group(0))
            except Exception:
                continue
            if spec.lo <= v <= spec.hi:
                return m.group(0)

    # Otherwise, prefer the last match (often the actual value appended to a label).
    return matches[-1].group(0)


@dataclass
class CellSpec:
    kind: str  # float|percent|zscore|grade|int
    lo: float
    hi: float
    decimals: Optional[int] = None


def cell_spec(metric: str, col: str) -> Optional[CellSpec]:
    """
    Expected ranges for validation + decimal repair.
    """
    if metric == "MVV":
        return None

    if metric in {"Grade FVC N", "Grade FEV1 N"} and col in {"PRE-BEST", "POST-BEST"}:
        return CellSpec(kind="grade", lo=0.0, hi=0.0, decimals=None)

    if col in {"Pre_%Pred", "Post_%Pred", "%CHG"}:
        return CellSpec(kind="percent", lo=-200.0, hi=400.0, decimals=1)
    if col in {"Pre_Z-Score", "Post_Z-Score"}:
        return CellSpec(kind="zscore", lo=-10.0, hi=10.0, decimals=2)

    if metric in {"FEV1/FVC"} and col in {"PRE-BEST", "POST-BEST", "LLN", "Pred"}:
        return CellSpec(kind="int", lo=20.0, hi=150.0, decimals=None)

    if metric in {"FEV1/VC"} and col in {"PRE-BEST", "POST-BEST"}:
        return CellSpec(kind="percent", lo=20.0, hi=150.0, decimals=2)

    if metric in {"FVC", "FEV1", "VC", "FIVC"} and col in {"PRE-BEST", "POST-BEST", "LLN", "Pred"}:
        return CellSpec(kind="float", lo=0.2, hi=10.0, decimals=2)
    if metric in {"FEF25%-75%"} and col in {"PRE-BEST", "POST-BEST", "LLN", "Pred"}:
        return CellSpec(kind="float", lo=0.05, hi=15.0, decimals=2)
    if metric in {"PEF"} and col in {"PRE-BEST", "POST-BEST", "LLN", "Pred"}:
        return CellSpec(kind="float", lo=0.10, hi=20.0, decimals=2)
    if metric in {"FET"} and col in {"PRE-BEST", "POST-BEST"}:
        return CellSpec(kind="float", lo=0.1, hi=60.0, decimals=2)
    if metric in {"BEV"} and col in {"PRE-BEST", "POST-BEST"}:
        return CellSpec(kind="float", lo=0.0, hi=2.0, decimals=2)

    if col in {"PRE-BEST", "POST-BEST", "LLN", "Pred"}:
        return CellSpec(kind="float", lo=-9999.0, hi=9999.0, decimals=2)

    return None


def normalize_by_spec(raw: str, spec: Optional[CellSpec]) -> Tuple[str, bool]:
    s = (raw or "").strip()
    if not s or spec is None:
        return "", False

    if spec.kind == "grade":
        g = parse_grade(s)
        return (g or ""), bool(g)

    # Try to pick the right numeric substring when multiple numbers exist (label bleed).
    picked = _pick_number_string_for_spec(s, spec)
    picked = picked if picked is not None else s

    # Preserve a digit-only view for decimal-repair heuristics.
    digits_only = re.sub(r"[^0-9]", "", picked)
    has_dot = "." in picked or "," in picked

    val = _num_from_text(picked)
    if val is None:
        return "", False

    # Repair missing decimals: prefer scaling that matches the expected number of decimals.
    # Example: "308" with decimals=2 is far more likely 3.08 than 30.8 or 0.308.
    if spec.kind in {"float", "percent", "zscore"} and not has_dot and digits_only:
        cand_order = []
        if spec.decimals == 2 and len(digits_only) >= 3:
            cand_order = [val / 100.0, val / 10.0, val / 1000.0]
        elif spec.decimals == 2 and len(digits_only) == 2:
            # Common for Z-scores like "49" meaning "0.49" (missing leading 0 and decimal).
            cand_order = [val / 100.0, val / 10.0, val / 1000.0]
        elif spec.decimals == 1 and len(digits_only) >= 3:
            cand_order = [val / 10.0, val / 100.0, val / 1000.0]
        else:
            cand_order = [val / 10.0, val / 100.0, val / 1000.0]

        if not (spec.lo <= val <= spec.hi):
            for cand in cand_order:
                if spec.lo <= cand <= spec.hi:
                    val = cand
                    break

    ok = spec.lo <= val <= spec.hi
    if not ok:
        return "", False

    if spec.kind == "int":
        return str(int(round(val))), True
    if spec.decimals is None:
        return str(val), True
    fmt = f"{{:.{spec.decimals}f}}"
    return fmt.format(val), True


def ocr_cell_candidates(
    cv2: Any,
    np: Any,
    cell_bgr: Any,
    *,
    tesseract_cmd: str,
    tessdata: Optional[str],
    whitelist: str,
    scale: float = 2.2,
    level: str = "fast",
) -> List[str]:
    import pytesseract  # type: ignore

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    if tessdata:
        os.environ["TESSDATA_PREFIX"] = tessdata

    gray0 = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    gray0 = cv2.resize(gray0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Improve local contrast for faint digits (common in Z-score column).
    try:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray0 = clahe.apply(gray0)
    except Exception:
        pass

    try:
        gray0 = cv2.GaussianBlur(gray0, (3, 3), 0)
    except Exception:
        pass
    # Mild unsharp mask to separate 1 vs 7 and keep '-' visible.
    try:
        blur = cv2.GaussianBlur(gray0, (0, 0), 1.0)
        gray0 = cv2.addWeighted(gray0, 1.7, blur, -0.7, 0)
    except Exception:
        pass

    variants: List[Any] = []

    def _remove_grid_lines(bin_img: Any) -> Any:
        """
        Remove long horizontal/vertical grid lines while keeping short strokes (e.g., minus sign).
        Input must be a 0/255 image with black text on white background.
        """
        inv = 255 - bin_img  # text/lines are white on black
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h = cv2.morphologyEx(inv, cv2.MORPH_OPEN, hk)
        v = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vk)
        lines = cv2.bitwise_or(h, v)
        inv2 = cv2.subtract(inv, lines)
        return 255 - inv2

    def _clear_border(bin_img: Any) -> Any:
        # Strip table borders that can look like stray '-' or '1'.
        h, w = bin_img.shape[:2]
        m = max(3, int(0.03 * min(h, w)))
        bin_img[:m, :] = 255
        bin_img[-m:, :] = 255
        bin_img[:, :m] = 255
        bin_img[:, -m:] = 255
        return bin_img

    # Otsu binarization (base).
    _, bw = cv2.threshold(gray0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = _clear_border(_remove_grid_lines(bw))
    variants.append(bw)
    if level == "full":
        variants.append(255 - bw)

    # Adaptive threshold sometimes keeps thin '-' visible.
    ad = cv2.adaptiveThreshold(
        gray0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 6
    )
    ad = _clear_border(_remove_grid_lines(ad))
    variants.append(ad)
    if level == "full":
        variants.append(255 - ad)

    # Thicken strokes a bit (helps 1 vs 7 and the minus sign).
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    variants.append(_clear_border(cv2.dilate(bw, k, iterations=1)))
    variants.append(_clear_border(cv2.dilate(ad, k, iterations=1)))

    # Skip nearly-blank cells to reduce hallucinated numbers in empty POST columns.
    # Use the *minimum* black count among the binarized variants as a conservative blank detector.
    black_counts = [int((v == 0).sum()) for v in variants[:2]]
    # Use a relative threshold so small 2-digit integers don't get treated as blank.
    area = int(variants[0].shape[0] * variants[0].shape[1]) if variants else 0
    min_black = max(12, int(0.0008 * max(1, area)))
    if black_counts and min(black_counts) < min_black:
        return []

    cfg_base = f'--oem 1 -c tessedit_char_whitelist="{whitelist}"'
    out: List[str] = []
    for img in variants:
        psms = (7, 8) if level == "fast" else (7, 8, 13)
        for psm in psms:  # single line / single word (/ raw line in full mode)
            cfg = f"{cfg_base} --psm {psm}"
            txt = pytesseract.image_to_string(img, config=cfg) or ""
            txt = clean_cell_text(txt)
            if txt and txt not in out:
                out.append(txt)
    return out


def repair_records_with_cell_ocr(
    cv2: Any,
    np: Any,
    table_img: Any,
    records: Dict[str, Dict[str, str]],
    *,
    tokens: Sequence[Token],
    body_y0: float,
    data_x0: float,
    split_x: float,
    row_centers: Sequence[float],
    headers: Dict[str, float],
    tesseract_cmd: Optional[str],
    tessdata: Optional[str],
    warnings: List[str],
    cell_ocr: str = "auto",
    cell_ocr_level: str = "fast",
) -> None:
    """
    Cell-level OCR fallback for values that fail validation.
    """
    if not tesseract_cmd or cell_ocr == "off":
        return

    # Recreate the same windows used by the main token assignment.
    row_windows = build_windows_from_centers(row_centers, lo=body_y0, hi=table_img.shape[0] - 2)
    post_presence = any(t.cx >= split_x and t.cy > body_y0 and has_digit(t.text) for t in tokens)
    # Only attempt to read POST cells if we actually see numeric content in the POST zone.
    post_enabled = post_presence

    pre_centers = infer_zone_centers(tokens, x0=data_x0, x1=split_x, k=len(PRE_COLUMNS), image_h=table_img.shape[0])
    post_centers = (
        infer_zone_centers(tokens, x0=split_x, x1=table_img.shape[1] - 1, k=len(POST_COLUMNS), image_h=table_img.shape[0])
        if post_enabled
        else []
    )
    pre_centers = apply_header_anchors(pre_centers, headers=headers, cols=PRE_COLUMNS)
    post_centers = apply_header_anchors(post_centers, headers=headers, cols=POST_COLUMNS) if post_centers else []
    pre_windows = build_windows_from_centers(pre_centers, lo=data_x0, hi=split_x)
    post_windows = build_windows_from_centers(post_centers, lo=split_x, hi=table_img.shape[1] - 2) if post_centers else []

    def _need_cell_ocr(metric: str, col: str, raw_norm_ok: bool, raw_norm: str) -> bool:
        """
        Decide whether to spend expensive per-cell OCR on this value.
        In auto mode, only do it if the current value is missing or suspicious.
        """
        if cell_ocr == "force":
            return True
        if cell_ocr != "auto":
            return False
        if not raw_norm_ok:
            return True
        spec = cell_spec(metric, col)
        if spec is None:
            return False
        v = _num_from_text(raw_norm) if spec.kind != "grade" else None
        best_key = "PRE-BEST" if col.startswith("Pre_") or col in {"LLN", "Pred"} else "POST-BEST"
        best = _num_from_text(records.get(metric, {}).get(best_key, ""))
        pred = _num_from_text(records.get(metric, {}).get("Pred", ""))
        lln = _num_from_text(records.get(metric, {}).get("LLN", ""))

        # Column-specific suspicion checks.
        if col == "Pred" and lln is not None and v is not None and v <= lln + 0.01:
            return True
        if col == "LLN" and pred is not None and v is not None:
            if v > pred + 0.01:
                return True
            if pred > 0 and v / pred < 0.20:
                return True
        if spec.kind == "percent" and pred is not None and best is not None and pred != 0 and v is not None:
            exp = (best / pred) * 100.0
            if abs(v - exp) >= 5.0:
                return True
        if spec.kind == "zscore" and v is not None and pred is not None and best is not None:
            # Wrong sign is common (missing '-').
            if abs(best - pred) >= max(0.02, 0.01 * max(1.0, abs(pred))):
                expected_sign = -1 if best < pred else 1
                if v != 0 and ((v > 0 and expected_sign < 0) or (v < 0 and expected_sign > 0)):
                    return True
            # BEST<LLN but Z not negative enough.
            if lln is not None and best < lln - 0.01 and v > -1.20:
                return True
            # Outliers.
            if abs(v) > 6.0:
                return True
        return False

    def bbox(row_idx: int, col: str) -> Optional[Tuple[int, int, int, int]]:
        if row_idx < 0 or row_idx >= len(row_windows):
            return None
        y0, y1 = row_windows[row_idx]
        if col in PRE_COLUMNS:
            j = PRE_COLUMNS.index(col)
            if j >= len(pre_windows):
                return None
            x0, x1 = pre_windows[j]
        elif col in POST_COLUMNS:
            j = POST_COLUMNS.index(col)
            if j >= len(post_windows):
                return None
            x0, x1 = post_windows[j]
        else:
            return None
        # Expand slightly to avoid clipping leading '-' or the first digit (common OCR failure in Z-Scores).
        w = max(1.0, (x1 - x0))
        h = max(1.0, (y1 - y0))
        pad_x = max(2.0, 0.08 * w)
        pad_y = max(2.0, 0.15 * h)
        pad_left = pad_x
        pad_right = pad_x
        # Asymmetric padding: the '-' (and sometimes the first digit) is closest to the left edge.
        if col in {"Pre_Z-Score", "Post_Z-Score"}:
            pad_left = max(pad_left, 0.14 * w)
            pad_right = max(pad_right, 0.06 * w)
        elif col in {"Pre_%Pred", "Post_%Pred", "%CHG"}:
            pad_left = max(pad_left, 0.10 * w)
            pad_right = max(pad_right, 0.06 * w)
        return (
            int(max(0, x0 - pad_left)),
            int(max(0, y0 - pad_y)),
            int(min(table_img.shape[1] - 1, x1 + pad_right)),
            int(min(table_img.shape[0] - 1, y1 + pad_y)),
        )

    def pick_best_numeric(metric: str, col: str, raw: str, cands: List[str]) -> Tuple[str, bool]:
        """
        Pick the best candidate using light clinical/structural constraints (not per-case hardcoding).
        Returns (normalized_value, ok).
        """
        spec = cell_spec(metric, col)
        if spec is None:
            return raw.strip(), True

        # Include the original raw as a candidate too.
        all_cands = [raw] + cands
        scored: List[Tuple[float, int, str, str]] = []  # (score, -freq, norm, raw_candidate)

        # Context for constraints.
        best = _num_from_text(records.get(metric, {}).get("PRE-BEST" if col.startswith("Pre_") or col in {"Pred", "LLN"} else "POST-BEST", ""))
        pred = _num_from_text(records.get(metric, {}).get("Pred", ""))
        lln = _num_from_text(records.get(metric, {}).get("LLN", ""))
        pct = _num_from_text(records.get(metric, {}).get("Pre_%Pred" if col.startswith("Pre_") else "Post_%Pred", ""))
        exp_pct = None
        if best is not None and pred is not None and pred != 0:
            exp_pct = (best / pred) * 100.0
        pct_trustworthy = pct is not None and exp_pct is not None and abs(pct - exp_pct) <= 2.5

        # Expected sign for Z from BEST vs Pred.
        expected_sign = 0
        if pred is not None and best is not None:
            if abs(best - pred) >= max(0.02, 0.01 * max(1.0, abs(pred))):
                expected_sign = -1 if best < pred else 1

        # Z-score disambiguation:
        # Prefer LLN/Pred/BEST because LLN is typically the 5th percentile (~Z=-1.64).
        # This helps prevent sign errors and missing leading digits (e.g., -0.34 vs -2.34).
        z_target = None
        if spec.kind == "zscore" and pred is not None and lln is not None and best is not None:
            if pred > lln > 0 and (pred - lln) >= 0.05:
                # Basic plausibility; if OCR botched Pred/LLN, avoid trusting this.
                r = lln / pred if pred else 0.0
                rb = best / pred if pred else 0.0
                if 0.15 <= r <= 1.10 and 0.05 <= rb <= 2.50:
                    scale = (pred - lln)
                    if best >= pred:
                        z = (best - pred) / scale * 1.64
                    elif best >= lln:
                        z = (best - pred) / scale * 1.64
                    else:
                        z = -1.64 - (lln - best) / scale * 1.64
                    z_target = max(-10.0, min(10.0, z))

        # Weak fallback when LLN/Pred/BEST is missing.
        if z_target is None and pct is not None and expected_sign != 0 and spec.kind == "zscore":
            z_target = max(-6.0, min(6.0, (pct - 100.0) / 15.0))

        freq: Dict[str, int] = {}
        for cand in all_cands:
            norm, ok = normalize_by_spec(cand, spec)
            if not ok:
                continue
            v = _num_from_text(norm)
            if v is None:
                continue

            score = 0.0
            # Penalize candidates that don't resemble the expected decimal precision.
            cand_s = str(cand).strip()
            if spec.decimals in {1, 2}:
                mdec = re.search(r"[.,](\\d+)", cand_s)
                if not mdec:
                    score += 0.60
                elif spec.decimals == 2 and len(mdec.group(1)) == 1:
                    score += 0.25

            # Prefer OCR candidates that satisfy basic table invariants.
            if col == "Pred" and lln is not None and v <= lln + 0.01:
                score += 5.0  # Pred should be above LLN (at least not equal/less), otherwise likely digit error.
            if spec.kind == "zscore":
                # Strong sign constraint.
                if expected_sign != 0 and v != 0:
                    if (v > 0 and expected_sign < 0) or (v < 0 and expected_sign > 0):
                        score += 6.0

                # If BEST is clearly below/above Pred, Z close to 0 is unlikely.
                if expected_sign != 0 and abs(v) < 0.15 and best is not None and pred is not None:
                    if abs(best - pred) >= max(0.06, 0.03 * max(1.0, abs(pred))):
                        score += 2.5

                # LLN implies Z should be <= -1.64 when BEST < LLN (typical PFT convention).
                if best is not None and lln is not None and best < lln - 0.01 and v > -1.20:
                    score += 6.0

                # Use rough Z target (from LLN/Pred/BEST or weak %Pred hint) to break OCR ties.
                if z_target is not None:
                    score += 0.75 * abs(v - z_target)
                    if abs(z_target) >= 1.0 and abs(v) < 0.35:
                        score += 1.5
            if spec.kind == "zscore" and abs(v) > 6.0:
                score += 3.0
            freq[norm] = freq.get(norm, 0) + 1
            scored.append((score, 0, norm, str(cand)))

        if not scored:
            return raw.strip(), False

        # No extra Z nudging here: Z tie-breaking happens above using z_target.

        # Tie-break by frequency across variants.
        scored2 = []
        for score, _, norm, cand_raw in scored:
            v = _num_from_text(norm)
            penalty2 = 0.0
            if v is not None:
                # LLN sanity: LLN should usually be a substantial fraction of Pred for volume/flow rows.
                if col == "LLN" and pred is not None and pred > 0 and metric not in {"FEV1/FVC"}:
                    if v / pred < 0.30:
                        penalty2 += 4.0
                    if v > pred:
                        penalty2 += 2.0
            scored2.append((score + penalty2, -freq.get(norm, 1), norm, cand_raw))
        scored2.sort(key=lambda x: (x[0], x[1]))
        best_norm = scored2[0][2]
        return best_norm, True

    for row_idx, metric in enumerate(ROW_NAMES):
        for col in OUT_COLUMNS:
            if metric in RESTRICTED_ROWS_PRE_POST and col not in RESTRICTED_ALLOWED_COLS:
                continue
            if not post_enabled and col in POST_COLUMNS:
                continue

            spec = cell_spec(metric, col)
            raw = records.get(metric, {}).get(col, "")
            norm, ok = normalize_by_spec(raw, spec)

            if ok and not _need_cell_ocr(metric, col, ok, norm):
                records[metric][col] = norm
                continue

            bb = bbox(row_idx, col)
            if not bb:
                continue
            x0, y0, x1, y1 = bb
            if x1 <= x0 or y1 <= y0:
                continue
            cell = table_img[y0:y1, x0:x1].copy()
            wl = "ABCDEF" if spec and spec.kind == "grade" else "0123456789.-%"
            # More zoom for high-risk columns (Z-score / %pred) to reduce digit flips (1<->7 etc.).
            scale = 2.2
            if spec and spec.kind == "zscore":
                scale = 2.9
            elif spec and spec.kind == "percent":
                scale = 2.6
            candidates = ocr_cell_candidates(
                cv2,
                np,
                cell,
                tesseract_cmd=tesseract_cmd,
                tessdata=tessdata,
                whitelist=wl,
                scale=scale,
                level=cell_ocr_level,
            )
            chosen, ok2 = pick_best_numeric(metric, col, raw, candidates)
            if ok2:
                records[metric][col] = chosen
            else:
                # Never keep junk characters (e.g. '$') in numeric columns.
                if spec and spec.kind in {"float", "int", "percent", "zscore"}:
                    records[metric][col] = ""
                else:
                    records[metric][col] = raw.strip()


def enforce_row_constraints(records: Dict[str, Dict[str, str]], warnings: List[str]) -> None:
    """
    Cross-column sanity fixes to reduce systematic OCR scaling errors.
    Example: PEF LLN read as 10.90 when Pred is 3.93; scale LLN down to 1.09.
    """
    for metric in ROW_NAMES:
        for pred_col, lln_col in [("Pred", "LLN")]:
            pred = _num_from_text(records.get(metric, {}).get(pred_col, ""))
            lln = _num_from_text(records.get(metric, {}).get(lln_col, ""))
            if pred is None or lln is None:
                continue
            if pred <= 0:
                continue
            if lln <= pred * 1.10:
                continue

            # Scale down until it's consistent, but do not over-correct.
            cand = lln
            applied = False
            for div in (10.0, 100.0, 1000.0):
                c = lln / div
                if c <= pred * 1.10:
                    cand = c
                    applied = True
                    break

            if applied:
                spec = cell_spec(metric, lln_col)
                txt, ok = normalize_by_spec(str(cand), spec)
                if ok:
                    records[metric][lln_col] = txt
                    warnings.append(f"Constraint fix: {metric} {lln_col} scaled from {lln} to {cand}")


def enforce_zscore_sign(records: Dict[str, Dict[str, str]], warnings: List[str]) -> None:
    """
    Fix missed/extra '-' on Z-Scores using the relationship between BEST and Pred:
      if BEST < Pred => Z should be negative
      if BEST > Pred => Z should be positive
    This only flips the sign when BEST and Pred are present and the disagreement is clear.
    """
    for best_col, z_col in [("PRE-BEST", "Pre_Z-Score"), ("POST-BEST", "Post_Z-Score")]:
        for metric in ROW_NAMES:
            if metric in RESTRICTED_ROWS_PRE_POST:
                continue
            best = _num_from_text(records.get(metric, {}).get(best_col, ""))
            pred = _num_from_text(records.get(metric, {}).get("Pred", ""))
            z = _num_from_text(records.get(metric, {}).get(z_col, ""))
            if best is None or pred is None or z is None:
                continue
            # Ignore tiny deltas where rounding can dominate.
            delta = best - pred
            if abs(delta) < max(0.02, 0.01 * max(1.0, abs(pred))):
                continue
            if delta < 0 and z > 0:
                records[metric][z_col] = f"{-abs(z):.2f}"
                warnings.append(f"Z-sign fix: {metric} {z_col} flipped to negative (best<pred)")
            elif delta > 0 and z < 0:
                records[metric][z_col] = f"{abs(z):.2f}"
                warnings.append(f"Z-sign fix: {metric} {z_col} flipped to positive (best>pred)")


def enforce_zscore_bounds(records: Dict[str, Dict[str, str]], warnings: List[str], *, strict: bool) -> None:
    """
    Guardrail for Z-score plausibility using LLN/Pred/BEST (no per-case hardcoding):
    - If BEST < LLN, Z should usually be <= ~-1.64.
    - If BEST > Pred, Z should usually be >= ~0.
    If violated and strict=True, we blank the Z-score (safer than a wrong medical value).
    """
    if not strict:
        return

    for best_col, z_col in [("PRE-BEST", "Pre_Z-Score"), ("POST-BEST", "Post_Z-Score")]:
        for metric in ROW_NAMES:
            if metric in RESTRICTED_ROWS_PRE_POST:
                continue
            best = _num_from_text(records.get(metric, {}).get(best_col, ""))
            pred = _num_from_text(records.get(metric, {}).get("Pred", ""))
            lln = _num_from_text(records.get(metric, {}).get("LLN", ""))
            z = _num_from_text(records.get(metric, {}).get(z_col, ""))
            if best is None or pred is None or lln is None or z is None:
                continue
            if pred <= lln or pred <= 0 or lln <= 0:
                continue
            r = lln / pred if pred else 0.0
            rb = best / pred if pred else 0.0
            if not (0.15 <= r <= 1.10 and 0.05 <= rb <= 2.50):
                continue

            # Strong, one-sided constraints.
            if best < lln - 0.01 and z > -1.20:
                warnings.append(f"STRICT: {metric} {z_col} violates BEST<LLN (best={best:.2f}, lln={lln:.2f}, z={z:.2f}); blanked")
                records[metric][z_col] = ""
            elif best > pred + 0.01 and z < -0.50:
                warnings.append(f"STRICT: {metric} {z_col} violates BEST>Pred (best={best:.2f}, pred={pred:.2f}, z={z:.2f}); blanked")
                records[metric][z_col] = ""


def enforce_percent_pred(records: Dict[str, Dict[str, str]], warnings: List[str]) -> None:
    """
    If %Pred looks wrong due to digit confusion, recompute from BEST and Pred.
    """
    for best_col, pct_col in [("PRE-BEST", "Pre_%Pred"), ("POST-BEST", "Post_%Pred")]:
        for metric in ROW_NAMES:
            if metric in RESTRICTED_ROWS_PRE_POST:
                continue
            best = _num_from_text(records.get(metric, {}).get(best_col, ""))
            pred = _num_from_text(records.get(metric, {}).get("Pred", ""))
            pct = _num_from_text(records.get(metric, {}).get(pct_col, ""))
            if best is None or pred is None or pred == 0:
                continue
            exp = (best / pred) * 100.0
            if pct is None:
                records[metric][pct_col] = f"{exp:.1f}"
                continue
            # Replace only if clearly off.
            if abs(pct - exp) >= 6.0:
                records[metric][pct_col] = f"{exp:.1f}"
                warnings.append(f"%Pred fix: {metric} {pct_col} replaced {pct:.1f}->{exp:.1f}")


def enforce_best_from_percent(records: Dict[str, Dict[str, str]], warnings: List[str]) -> None:
    """
    Conservative correction for digit-confusion in BEST (e.g., 1 -> 7).
    If Pred and %Pred exist and BEST is wildly inconsistent (ratio too large/small),
    replace BEST using: BEST = Pred * %Pred / 100.
    """
    for best_col, pct_col in [("PRE-BEST", "Pre_%Pred"), ("POST-BEST", "Post_%Pred")]:
        for metric in ROW_NAMES:
            if metric in RESTRICTED_ROWS_PRE_POST:
                continue
            best = _num_from_text(records.get(metric, {}).get(best_col, ""))
            pred = _num_from_text(records.get(metric, {}).get("Pred", ""))
            pct = _num_from_text(records.get(metric, {}).get(pct_col, ""))
            if pred is None or pct is None or pred == 0:
                continue
            exp_best = pred * (pct / 100.0)
            if best is None:
                records[metric][best_col] = f"{exp_best:.2f}"
                continue
            ratio = best / pred if pred else 0.0
            if ratio > 2.0 or ratio < 0.30:
                records[metric][best_col] = f"{exp_best:.2f}"
                warnings.append(f"BEST fix: {metric} {best_col} replaced {best:.2f}->{exp_best:.2f} using %Pred")


def enforce_chg_from_bests(records: Dict[str, Dict[str, str]], warnings: List[str]) -> None:
    """
    %CHG is derivable from PRE-BEST and POST-BEST:
      %CHG = ((POST - PRE) / PRE) * 100
    Prefer the derived value whenever both BEST values exist. This avoids OCR decimal drift like 44.0 vs 4.4.
    """
    for metric in ROW_NAMES:
        if metric in RESTRICTED_ROWS_PRE_POST:
            continue
        pre = _num_from_text(records.get(metric, {}).get("PRE-BEST", ""))
        post = _num_from_text(records.get(metric, {}).get("POST-BEST", ""))
        if pre is None or post is None or pre == 0:
            continue

        exp = (post - pre) / pre * 100.0
        cur = _num_from_text(records.get(metric, {}).get("%CHG", ""))
        txt = f"{exp:.1f}"
        # Overwrite if missing or clearly inconsistent.
        if cur is None or abs(cur - exp) >= 1.2:
            if cur is not None:
                warnings.append(f"%CHG fix: {metric} %CHG replaced {cur:.1f}->{exp:.1f} from BEST values")
            records[metric]["%CHG"] = txt


def enforce_post_best_from_chg(records: Dict[str, Dict[str, str]], warnings: List[str]) -> None:
    """
    Fix POST-BEST using PRE-BEST and %CHG when POST-BEST is clearly wrong (e.g., 0.41 instead of 4.41).
    """
    for metric in ROW_NAMES:
        pre = _num_from_text(records.get(metric, {}).get("PRE-BEST", ""))
        post = _num_from_text(records.get(metric, {}).get("POST-BEST", ""))
        chg = _num_from_text(records.get(metric, {}).get("%CHG", ""))
        if pre is None or chg is None or pre == 0:
            continue

        exp_post = pre * (1.0 + chg / 100.0)

        # Metric-aware plausibility.
        spec = cell_spec(metric, "POST-BEST")
        if spec is None:
            continue
        exp_txt, exp_ok = normalize_by_spec(str(exp_post), spec)
        if not exp_ok:
            continue

        # Determine whether to override.
        if post is None:
            records[metric]["POST-BEST"] = exp_txt
            continue
        ratio = post / pre if pre else 1.0
        # If post is extremely off compared to pre (typical decimal/missing-digit error), override.
        chg_from_post = ((post - pre) / pre) * 100.0
        if ratio < 0.30 or ratio > 2.0:
            records[metric]["POST-BEST"] = exp_txt
            warnings.append(f"POST-BEST fix: {metric} replaced {post:.2f}->{float(exp_txt):.2f} using %CHG")


def apply_grade_split(records: Dict[str, Dict[str, str]], warnings: List[str]) -> None:
    """
    If Grade row cell contains "A 10.53", keep "A" and move "10.53" to FET row below.
    Apply for both PRE-BEST and POST-BEST.
    """
    pattern = re.compile(r"^\s*([A-F])\s+([-+]?\d+(?:\.\d+)?)\s*$", re.I)
    for grade_row, fet_row in [("Grade FVC N", "FET"), ("Grade FEV1 N", "FET")]:
        for col in ["PRE-BEST", "POST-BEST"]:
            val = records.get(grade_row, {}).get(col, "")
            m = pattern.match(val or "")
            if not m:
                # If it has both alpha and digits but not matching the strict pattern, try a softer split.
                if re.search(r"[A-F]", (val or "").upper()) and re.search(r"\d", val or ""):
                    g = parse_grade(val)
                    num = parse_float(val)
                    if g:
                        records[grade_row][col] = g
                    if num is not None and not records.get(fet_row, {}).get(col, "").strip():
                        records[fet_row][col] = f"{num:.2f}"
                        warnings.append(f"Grade splitter (soft): moved {num:.2f} from {grade_row}/{col} to {fet_row}/{col}")
                continue

            grade = m.group(1).upper()
            num = m.group(2)
            records[grade_row][col] = grade
            if not records.get(fet_row, {}).get(col, "").strip():
                records[fet_row][col] = num
                warnings.append(f"Grade splitter: moved {num} from {grade_row}/{col} to {fet_row}/{col}")


def compute_derived(records: Dict[str, Dict[str, str]], warnings: List[str]) -> None:
    """
    Compute derived rows when feasible:
    - FEV1/FVC (integer %)
    - FEV1/VC  (integer %)
    Compute separately for PRE-BEST and POST-BEST.
    """
    for col in ["PRE-BEST", "POST-BEST"]:
        fvc = parse_float(records.get("FVC", {}).get(col, ""))
        fev1 = parse_float(records.get("FEV1", {}).get(col, ""))
        vc = parse_float(records.get("VC", {}).get(col, ""))

        if fvc and fev1:
            pct = int(round((fev1 / fvc) * 100.0))
            cur = _num_from_text(records.get("FEV1/FVC", {}).get(col, ""))
            # Override if missing or obviously invalid/out-of-family.
            if cur is None or cur < 20 or cur > 150 or abs(cur - pct) >= 8:
                records["FEV1/FVC"][col] = str(pct)
                warnings.append(f"Derived {col} FEV1/FVC={pct}")

        if vc and fev1:
            pct = float((fev1 / vc) * 100.0)
            cur = _num_from_text(records.get("FEV1/VC", {}).get(col, ""))
            if cur is None or cur < 20 or cur > 150 or abs(cur - pct) >= 8:
                records["FEV1/VC"][col] = f"{pct:.2f}"
                warnings.append(f"Derived {col} FEV1/VC={pct:.2f}")


def row_realign_top(records: Dict[str, Dict[str, str]], warnings: List[str]) -> None:
    """
    If FVC/FEV1 are empty but FEV1/FVC contains multiple numbers, reassign.
    Apply independently for PRE-BEST and POST-BEST.
    """
    for col in ["PRE-BEST", "POST-BEST"]:
        if records["FVC"][col].strip() or records["FEV1"][col].strip():
            continue
        s = records["FEV1/FVC"][col]
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s or "")
        if len(nums) >= 2:
            records["FVC"][col] = nums[0]
            records["FEV1"][col] = nums[1]
            records["FEV1/FVC"][col] = ""  # will be derived below
            warnings.append(f"Row realign: pulled {nums[0]},{nums[1]} from FEV1/FVC/{col} into FVC+FEV1")


def extract_table_records(
    np: Any,
    tokens: Sequence[Token],
    image_w: int,
    image_h: int,
    *,
    headers: Dict[str, float],
) -> Tuple[Dict[str, Dict[str, str]], List[float], List[float], List[str]]:
    warnings: List[str] = []

    # Estimate data area and row anchors.
    body_y0 = detect_body_y0(tokens, image_h)
    data_x0 = estimate_data_x0(tokens, image_w, body_y0)
    anchors = detect_label_anchors(tokens, image_w, body_y0)
    row_centers = fit_row_centers(np, anchors, image_h, body_y0)
    row_windows = build_windows_from_centers(row_centers, lo=body_y0, hi=image_h - 2)

    split_x = detect_split_x(headers, image_w, data_x0)
    if split_x is None:
        split_x = data_x0 + (image_w - data_x0) * 0.58
    split_x = float(max(data_x0 + 30.0, min(image_w * 0.92, split_x)))

    # If POST is actually missing, treat right side as empty.
    has_post_header = "POST-BEST" in headers
    post_presence = any(t.cx >= split_x and t.cy > body_y0 and has_digit(t.text) for t in tokens)
    # Only assign POST cells if we actually see numeric content in the POST zone.
    post_enabled = post_presence

    pre_centers = infer_zone_centers(tokens, x0=data_x0, x1=split_x, k=len(PRE_COLUMNS), image_h=image_h)
    post_centers = infer_zone_centers(tokens, x0=split_x, x1=image_w - 1, k=len(POST_COLUMNS), image_h=image_h) if post_enabled else []

    # Anchor to detected headers when available.
    pre_centers = apply_header_anchors(pre_centers, headers=headers, cols=PRE_COLUMNS)
    post_centers = apply_header_anchors(post_centers, headers=headers, cols=POST_COLUMNS) if post_centers else []

    pre_windows = build_windows_from_centers(pre_centers, lo=data_x0, hi=split_x)
    post_windows = build_windows_from_centers(post_centers, lo=split_x, hi=image_w - 2) if post_centers else []

    # Prepare empty records.
    records: Dict[str, Dict[str, str]] = {rn: {c: "" for c in OUT_COLUMNS} for rn in ROW_NAMES}

    body_tokens = [t for t in tokens if t.cy > body_y0]

    # Assign tokens to PRE/POST cells based on nearest row window and column window.
    for t in body_tokens:
        if t.cx < data_x0:
            continue
        # Row index by y window.
        r_idx = None
        for i, (y0, y1) in enumerate(row_windows):
            if y0 <= t.cy <= y1:
                r_idx = i
                break
        if r_idx is None or r_idx >= len(ROW_NAMES):
            continue
        row_name = ROW_NAMES[r_idx]

        # Column assignment.
        if t.cx < split_x:
            if not pre_windows:
                continue
            c_idx = None
            for j, (x0, x1) in enumerate(pre_windows):
                if x0 <= t.cx <= x1:
                    c_idx = j
                    break
            if c_idx is None or c_idx >= len(PRE_COLUMNS):
                continue
            col_name = PRE_COLUMNS[c_idx]
        else:
            if not post_enabled or not post_windows:
                continue
            c_idx = None
            for j, (x0, x1) in enumerate(post_windows):
                if x0 <= t.cx <= x1:
                    c_idx = j
                    break
            if c_idx is None or c_idx >= len(POST_COLUMNS):
                continue
            col_name = POST_COLUMNS[c_idx]

        # Append token text (we merge later).
        existing = records[row_name][col_name]
        records[row_name][col_name] = clean_cell_text((existing + " " + t.text).strip()) if existing else clean_cell_text(t.text)

    # Post-processing per schema rules.
    # MVV: always blank.
    for c in OUT_COLUMNS:
        records["MVV"][c] = ""

    # Restricted rows: only PRE-BEST and POST-BEST allowed.
    for rn in RESTRICTED_ROWS_PRE_POST:
        keep_pre = records[rn].get("PRE-BEST", "")
        keep_post = records[rn].get("POST-BEST", "")
        for c in OUT_COLUMNS:
            records[rn][c] = ""
        records[rn]["PRE-BEST"] = keep_pre
        records[rn]["POST-BEST"] = keep_post

    # Grade splitter then re-align + derive.
    apply_grade_split(records, warnings)
    row_realign_top(records, warnings)
    compute_derived(records, warnings)

    # Simple QC: count rows with any content.
    rows_found = sum(1 for rn in ROW_NAMES if any(records[rn][c].strip() for c in OUT_COLUMNS))
    if not post_enabled and has_post_header:
        warnings.append("POST section appears missing; POST columns left blank.")

    return records, row_centers, pre_centers + ([split_x] if pre_centers else []), warnings


def records_to_rows(case_id: str, records: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for rn in ROW_NAMES:
        row: Dict[str, str] = {"case_id": case_id, "metric": rn}
        for c in OUT_COLUMNS:
            row[c] = records.get(rn, {}).get(c, "")
        out.append(row)
    return out


def save_debug_overlay(
    cv2: Any,
    np: Any,
    *,
    band_img: Any,
    table_img: Any,
    table_bbox_in_band_px: Tuple[int, int, int, int],
    data_x0: float,
    split_x: float,
    row_centers: Sequence[float],
    pre_centers: Sequence[float],
    post_centers: Sequence[float],
    out_path: Path,
) -> None:
    band_dbg = band_img.copy()
    tx, ty, tw, th = table_bbox_in_band_px
    cv2.rectangle(band_dbg, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)

    table_dbg = table_img.copy()
    h, w = table_dbg.shape[:2]
    # Data region markers.
    cv2.line(table_dbg, (int(round(data_x0)), 0), (int(round(data_x0)), h - 1), (200, 0, 200), 2)
    cv2.line(table_dbg, (int(round(split_x)), 0), (int(round(split_x)), h - 1), (0, 255, 255), 2)

    for y in row_centers:
        yy = int(round(y))
        cv2.line(table_dbg, (0, yy), (w - 1, yy), (255, 0, 0), 1)

    for x in pre_centers:
        xx = int(round(x))
        cv2.line(table_dbg, (xx, 0), (xx, h - 1), (0, 180, 0), 1)
    for x in post_centers:
        xx = int(round(x))
        cv2.line(table_dbg, (xx, 0), (xx, h - 1), (0, 0, 180), 1)

    # Stack band+table for convenience.
    pad = 6
    band_pad = cv2.copyMakeBorder(band_dbg, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(240, 240, 240))
    table_pad = cv2.copyMakeBorder(table_dbg, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(240, 240, 240))
    width = max(band_pad.shape[1], table_pad.shape[1])
    if band_pad.shape[1] != width:
        band_pad = cv2.copyMakeBorder(band_pad, 0, 0, 0, width - band_pad.shape[1], cv2.BORDER_CONSTANT, value=(240, 240, 240))
    if table_pad.shape[1] != width:
        table_pad = cv2.copyMakeBorder(table_pad, 0, 0, 0, width - table_pad.shape[1], cv2.BORDER_CONSTANT, value=(240, 240, 240))
    combined = np.vstack([band_pad, table_pad])
    cv2.imwrite(str(out_path), combined, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def main() -> int:
    args = parse_args()
    fitz, cv2, np, pd = import_deps()

    base = Path.cwd()
    pdf_dir = (base / args.pdf_dir).resolve() if not args.pdf_dir.is_absolute() else args.pdf_dir
    out_root = (base / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir

    out_tables_img = out_root / "tables_img"
    out_tables_dbg = out_root / "tables_debug"
    out_tables_csv = out_root / "tables_csv"
    out_tables_qc = out_root / "tables_qc"
    for p in [out_tables_img, out_tables_dbg, out_tables_csv, out_tables_qc]:
        ensure_dir(p)

    cfg: Dict[str, Any] = load_overrides(args.overrides)
    base_rect = (args.rel_left, args.rel_top, args.rel_right, args.rel_bottom)
    base_extra = float(args.extra_bottom)

    pdfs = list_pdfs(pdf_dir)
    if args.cases.strip():
        wanted_raw = [c.strip() for c in args.cases.split(",") if c.strip()]
        wanted = set(wanted_raw)
        wanted_sanitized = set(sanitize_case_id(Path(x).stem) for x in wanted_raw)
        filtered = []
        for p in pdfs:
            if p.name in wanted or p.stem in wanted or sanitize_case_id(p.stem) in wanted_sanitized:
                filtered.append(p)
        pdfs = filtered
        if not pdfs:
            print(f"[ERROR] --cases matched 0 PDFs in {pdf_dir}")
            return 2
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]

    tesseract_cmd = find_tesseract_cmd(args.tesseract_cmd)
    if args.engine in {"ocr", "auto"} and args.engine != "text" and not tesseract_cmd:
        print("[WARN] Tesseract not found; OCR fallback disabled. (Set --tesseract-cmd or TESSERACT_CMD)")

    combined_rows: List[Dict[str, str]] = []
    processed = 0

    for pdf_path in pdfs:
        case_id = sanitize_case_id(pdf_path.stem)
        rel_rect, extra_bottom = apply_rect_overrides(
            fname=pdf_path.name,
            base_rect=base_rect,
            cfg=cfg,
            allow_extra_bottom=True,
            base_extra_bottom_frac=base_extra,
        )

        try:
            page_img = render_page_image(fitz, cv2, np, pdf_path, args.page_index, args.dpi)
            band_img, band_bbox = crop_band(page_img, rel_rect=rel_rect, extra_bottom=extra_bottom)
            table_img, table_bbox_in_band, _ = tighten_table_crop(cv2, np, band_img)
        except Exception as exc:
            print(f"[ERROR] {pdf_path.name}: render/crop failed: {exc}")
            continue

        # Optional left-expansion retry: if the band crop trimmed labels, row anchoring fails badly.
        # We detect that after a cheap OCR pass; if needed, re-crop with rel_left forced to 0.0.
        # (Keeps per-file overrides for top/right/bottom and extra-bottom.)
        retry_table_img = None
        retry_table_bbox = None
        retry_band_img = None
        retry_band_bbox = None

        zoom = args.dpi / 72.0
        bx, by, _, _ = band_bbox
        tx, ty, tw, th = table_bbox_in_band
        band_x0_pt = bx / zoom
        band_y0_pt = by / zoom
        table_x0_pt = band_x0_pt + (tx / zoom)
        table_y0_pt = band_y0_pt + (ty / zoom)
        table_x1_pt = table_x0_pt + (tw / zoom)
        table_y1_pt = table_y0_pt + (th / zoom)
        clip_rect_page = fitz.Rect(table_x0_pt, table_y0_pt, table_x1_pt, table_y1_pt)

        tokens: List[Token] = []
        engine_used = ""
        warnings: List[str] = []

        if args.engine in {"text", "auto"}:
            try:
                tokens = tokens_from_pdf_text(
                    fitz,
                    pdf_path,
                    page_index=args.page_index,
                    clip_rect_page=clip_rect_page,
                    zoom=zoom,
                    band_offset_px=(bx, by),
                    table_bbox_in_band_px=table_bbox_in_band,
                )
                engine_used = "text"
            except Exception as exc:
                warnings.append(f"text extraction failed: {exc}")
                tokens = []

        # If auto, validate quickly; if weak, fall back.
        if (args.engine == "ocr" or (args.engine == "auto" and not tokens)) and tesseract_cmd:
            try:
                # Note: we keep token OCR on the full table for stable geometry (headers/rows).
                # Precision comes from selective per-cell OCR repair (see repair_records_with_cell_ocr()).
                tokens = tokens_from_tesseract(table_img, tesseract_cmd=tesseract_cmd, tessdata=args.tessdata)
                engine_used = "ocr"
            except Exception as exc:
                print(f"[ERROR] {pdf_path.name}: OCR failed: {exc}")
                continue

        if not tokens:
            print(f"[WARN] {pdf_path.name}: no tokens extracted.")
            continue

        body_y0 = detect_body_y0(tokens, table_img.shape[0])
        anchors = detect_label_anchors(tokens, table_img.shape[1], body_y0)
        if len(anchors) < 8 and rel_rect[0] > 0.0:
            # Retry with a band that starts at the far left.
            try:
                anchors_before = len(anchors)
                rel_rect2 = (0.0, rel_rect[1], rel_rect[2], rel_rect[3])
                band_img2, band_bbox2 = crop_band(page_img, rel_rect=rel_rect2, extra_bottom=extra_bottom)
                table_img2, table_bbox2, _ = tighten_table_crop(cv2, np, band_img2)

                # Re-OCR tokens on the alternative crop to see if we get more anchors.
                tokens2 = tokens_from_tesseract(table_img2, tesseract_cmd=tesseract_cmd, tessdata=args.tessdata) if engine_used == "ocr" and tesseract_cmd else []
                if engine_used == "text":
                    # Text engine uses page clip; keep original crop in that mode.
                    tokens2 = []
                if tokens2:
                    body_y02 = detect_body_y0(tokens2, table_img2.shape[0])
                    anchors2 = detect_label_anchors(tokens2, table_img2.shape[1], body_y02)
                    if len(anchors2) > len(anchors):
                        retry_table_img = table_img2
                        retry_table_bbox = table_bbox2
                        retry_band_img = band_img2
                        retry_band_bbox = band_bbox2
                        tokens = tokens2
                        body_y0 = body_y02
                        anchors = anchors2
                        warnings.append(f"Auto-recrop: expanded band rel_left from {rel_rect[0]:.2f} to 0.00 (anchors {anchors_before}->{len(anchors2)})")
            except Exception:
                pass

        if retry_table_img is not None:
            table_img = retry_table_img
            table_bbox_in_band = retry_table_bbox  # type: ignore[assignment]
            band_img = retry_band_img  # type: ignore[assignment]
            band_bbox = retry_band_bbox  # type: ignore[assignment]

        # Save the table crop losslessly.
        cv2.imwrite(str(out_tables_img / f"{case_id}_p2_table.png"), table_img)

        headers = detect_header_positions(tokens, body_y0)
        records, row_centers, _col_dbg, warn2 = extract_table_records(
            np,
            tokens,
            table_img.shape[1],
            table_img.shape[0],
            headers=headers,
        )
        warnings.extend(warn2)

        # Normalize + repair common OCR failures (missing decimals, split digits) and optionally re-OCR bad cells.
        data_x0 = estimate_data_x0(tokens, table_img.shape[1], body_y0)
        split_x = detect_split_x(headers, table_img.shape[1], data_x0)
        if split_x is None:
            split_x = data_x0 + (table_img.shape[1] - data_x0) * 0.58
        split_x = float(max(data_x0 + 30.0, min(table_img.shape[1] * 0.92, split_x)))

        repair_records_with_cell_ocr(
            cv2,
            np,
            table_img,
            records,
            tokens=tokens,
            body_y0=body_y0,
            data_x0=data_x0,
            split_x=split_x,
            row_centers=row_centers,
            headers=headers,
            tesseract_cmd=tesseract_cmd,
            tessdata=args.tessdata,
            warnings=warnings,
            cell_ocr=args.cell_ocr,
            cell_ocr_level=args.cell_ocr_level,
        )
        # Re-apply schema rules after repair.
        apply_grade_split(records, warnings)
        row_realign_top(records, warnings)
        compute_derived(records, warnings)
        enforce_row_constraints(records, warnings)
        enforce_post_best_from_chg(records, warnings)
        compute_derived(records, warnings)
        enforce_percent_pred(records, warnings)
        enforce_best_from_percent(records, warnings)
        enforce_chg_from_bests(records, warnings)
        enforce_zscore_sign(records, warnings)
        enforce_zscore_bounds(records, warnings, strict=args.strict)

        # Basic QC score.
        rows_found = sum(1 for rn in ROW_NAMES if any(records[rn][c].strip() for c in OUT_COLUMNS))
        qc = QcResult(pdf_name=pdf_path.name, case_id=case_id, engine_used=engine_used, rows_found=rows_found, warnings=warnings)

        # Persist per-case CSV.
        per_rows = records_to_rows(case_id, records)
        df = pd.DataFrame(per_rows, columns=["case_id", "metric"] + OUT_COLUMNS)
        per_csv = out_tables_csv / f"{case_id}_p2_table.csv"
        df.to_csv(per_csv, index=False, encoding="utf-8-sig")
        combined_rows.extend(per_rows)

        # Debug overlay with anchors.
        pre_centers = infer_zone_centers(tokens, x0=data_x0, x1=split_x, k=len(PRE_COLUMNS), image_h=table_img.shape[0])
        post_centers = infer_zone_centers(tokens, x0=split_x, x1=table_img.shape[1] - 1, k=len(POST_COLUMNS), image_h=table_img.shape[0])

        dbg_path = out_tables_dbg / f"{case_id}_p2_table_dbg.jpg"
        save_debug_overlay(
            cv2,
            np,
            band_img=band_img,
            table_img=table_img,
            table_bbox_in_band_px=table_bbox_in_band,
            data_x0=data_x0,
            split_x=split_x,
            row_centers=row_centers,
            pre_centers=pre_centers,
            post_centers=post_centers,
            out_path=dbg_path,
        )

        qc_path = out_tables_qc / f"{case_id}_p2_table_qc.json"
        qc_path.write_text(json.dumps(asdict(qc), indent=2), encoding="utf-8")

        processed += 1
        print(f"[OK] {pdf_path.name} -> {per_csv.name} (engine={engine_used}, rows_found={rows_found})")

    if combined_rows:
        combo = pd.DataFrame(combined_rows, columns=["case_id", "metric"] + OUT_COLUMNS)
        combo_path = out_root / "tables_combined.csv"
        combo.to_csv(combo_path, index=False, encoding="utf-8-sig")
        print(f"Combined CSV -> {combo_path}")

    print(f"Done. Tables written to: {out_tables_csv}")
    return 0 if processed else 2


if __name__ == "__main__":
    raise SystemExit(main())
