from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_pdfs(pdf_dir: Path) -> List[Path]:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    pdfs = sorted([p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {pdf_dir}")
    return pdfs


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def load_overrides(path: Optional[Path]) -> Dict[str, Any]:
    """
    Loads overrides from YAML (preferred) or JSON.

    Supported shape:
      global: { rel_left, rel_top, rel_right, rel_bottom, dx, dy, grow, extra_bottom_frac }
      per_file: { "<filename.pdf>": { ... } }
    """
    if not path:
        return {}
    if not path.exists():
        return {}

    # Try PyYAML if present; otherwise accept JSON.
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        with path.open("r", encoding="utf-8") as fh:
            return json.loads(fh.read())


def apply_rect_overrides(
    *,
    fname: str,
    base_rect: Tuple[float, float, float, float],
    cfg: Dict[str, Any],
    allow_extra_bottom: bool = False,
    base_extra_bottom_frac: float = 0.0,
) -> Tuple[Tuple[float, float, float, float], float]:
    """
    Apply global/per-file overrides to a relative rectangle.

    dx/dy/grow are relative fractions of page size.
    """
    l, t, r, b = base_rect
    extra_bottom = base_extra_bottom_frac

    g = cfg.get("global") or {}
    l = float(g.get("rel_left", l))
    t = float(g.get("rel_top", t))
    r = float(g.get("rel_right", r))
    b = float(g.get("rel_bottom", b))
    dx = float(g.get("dx", 0.0))
    dy = float(g.get("dy", 0.0))
    grow = float(g.get("grow", 0.0))
    if allow_extra_bottom:
        extra_bottom = float(g.get("extra_bottom_frac", extra_bottom))

    pf = (cfg.get("per_file") or {}).get(fname, {}) or {}
    l = float(pf.get("rel_left", l))
    t = float(pf.get("rel_top", t))
    r = float(pf.get("rel_right", r))
    b = float(pf.get("rel_bottom", b))
    dx += float(pf.get("dx", 0.0))
    dy += float(pf.get("dy", 0.0))
    grow += float(pf.get("grow", 0.0))
    if allow_extra_bottom:
        extra_bottom = float(pf.get("extra_bottom_frac", extra_bottom))

    # Shift
    l += dx
    r += dx
    t += dy
    b += dy

    # Grow/Shrink around center.
    cx = (l + r) / 2.0
    cy = (t + b) / 2.0
    half_w = (r - l) / 2.0
    half_h = (b - t) / 2.0
    half_w *= (1.0 + grow)
    half_h *= (1.0 + grow)
    l, r = cx - half_w, cx + half_w
    t, b = cy - half_h, cy + half_h

    l, t, r, b = map(clamp01, (l, t, r, b))
    if r <= l:
        r = min(1.0, l + 0.01)
    if b <= t:
        b = min(1.0, t + 0.01)
    if allow_extra_bottom:
        extra_bottom = max(0.0, min(0.2, float(extra_bottom)))
    else:
        extra_bottom = 0.0

    return (l, t, r, b), extra_bottom


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9%/]+", "", (text or "").lower())


def sanitize_case_id(stem: str) -> str:
    """
    Make output filenames shell/CSV friendly while keeping them recognizable.
    Example: "DEID_Case 12" -> "DEID_Case_12"
    """
    return re.sub(r"\s+", "_", (stem or "").strip())


def clean_cell_text(text: str) -> str:
    if not text:
        return ""
    s = str(text).replace("\n", " ").strip()
    # Normalize common unicode-ish punctuation into ASCII.
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s)
    # Remove spaces around punctuation commonly broken by OCR.
    s = re.sub(r"(?<=\d)\s+(?=[\.,%])", "", s)
    s = re.sub(r"(?<=[\.,%])\s+(?=\d)", "", s)
    s = re.sub(r"(?<=-)\s+(?=\d)", "", s)
    return s.strip()


def has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def kmeans_1d(values: Sequence[float], k: int, *, max_iter: int = 50) -> List[float]:
    """
    Tiny 1D k-means using median updates (robust to outliers) to avoid sklearn dependency.
    Returns sorted cluster centers.
    """
    if k <= 0:
        return []
    arr = [float(v) for v in values if v is not None]
    if not arr:
        return []

    arr.sort()
    if len(arr) < k:
        lo, hi = arr[0], arr[-1]
        if lo == hi:
            return [lo for _ in range(k)]
        step = (hi - lo) / float(k - 1)
        return [lo + step * i for i in range(k)]

    # Init using quantiles.
    qs = [i / float(k - 1) for i in range(k)]
    centers = [arr[int(round(q * (len(arr) - 1)))] for q in qs]

    for _ in range(max_iter):
        buckets: List[List[float]] = [[] for _ in range(k)]
        for v in arr:
            idx = min(range(k), key=lambda i: abs(v - centers[i]))
            buckets[idx].append(v)
        new_centers = []
        for i, b in enumerate(buckets):
            if not b:
                new_centers.append(centers[i])
            else:
                b.sort()
                new_centers.append(b[len(b) // 2])
        if all(abs(new_centers[i] - centers[i]) < 1e-3 for i in range(k)):
            centers = new_centers
            break
        centers = new_centers

    centers.sort()
    # Enforce monotonic spacing to avoid duplicates collapsing.
    for i in range(1, len(centers)):
        if centers[i] <= centers[i - 1]:
            centers[i] = centers[i - 1] + 1e-3
    return centers


@dataclass
class Token:
    text: str
    conf: float
    cx: float
    cy: float
    x0: float
    y0: float
    x1: float
    y1: float


def find_tesseract_cmd(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    env = os.environ.get("TESSERACT_CMD") or os.environ.get("TESSERACT_EXE")
    if env:
        return env
    which = shutil.which("tesseract")
    if which:
        return which
    # Windows default (if caller passes it, great; otherwise try this).
    default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return default_win if Path(default_win).exists() else None
