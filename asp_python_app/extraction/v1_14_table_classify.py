#!/usr/bin/env python3
"""
v1.14 - Table logic classification (spirometry pattern + severity).

Integration notes (for teammate):
- Purpose: consume the *standard* table CSV produced by v1_12 and emit a
  machine-readable classification JSON (per case) + a combined CSV.
- Inputs:
  - tables_csv/<CASE_ID>_p2_table.csv (columns: metric, Pre_Z-Score, Post_Z-Score, %CHG).
  - Optional BMI (single value via --bmi, or per-case map via --bmi-map CSV with case_id,bmi).
- Outputs:
  - tables_classification/<CASE_ID>_classification.json
  - tables_classification/classification_combined.csv
- Output JSON includes the exact Z-scores and %CHG values used for classification
  (pre_fvc_z, pre_fev1_z, pre_ratio_z, post_*, pct_chg_*). This is the audit trail.
- Expected usage in app:
  1) Run table extraction (v1_12) -> CSVs
  2) Run this script -> classification JSON
  3) Consume JSON in your app (no GUI, batch-safe).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class ClassResult:
    case_id: str
    bmi_used: Optional[float]
    pre_fvc_z: Optional[float]
    pre_fev1_z: Optional[float]
    pre_ratio_z: Optional[float]
    post_fvc_z: Optional[float]
    post_fev1_z: Optional[float]
    post_ratio_z: Optional[float]
    pct_chg_fvc: Optional[float]
    pct_chg_fev1: Optional[float]
    pct_chg_ratio: Optional[float]
    pre_pattern: str
    pre_severity: str
    post_pattern: str
    post_severity: str
    report_pre: str
    report_post: str
    report_combined: str
    bronchodilator_response: str
    warnings: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify spirometry patterns from table CSVs.")
    parser.add_argument("--tables-dir", type=Path, default=Path("extracted/v1_12/tables_csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("extracted/v1_12/tables_classification"))
    parser.add_argument("--cases", type=str, default="", help="Comma-separated list of case IDs or CSV names.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bmi", type=float, default=None, help="Optional BMI applied to all cases.")
    parser.add_argument(
        "--bmi-map",
        type=Path,
        default=None,
        help="Optional CSV mapping with columns: case_id,bmi (overrides --bmi per case).",
    )
    return parser.parse_args()


def _num(s: object) -> Optional[float]:
    try:
        if s is None:
            return None
        if isinstance(s, str) and not s.strip():
            return None
        return float(s)
    except Exception:
        return None


def classify_spirometry(fvc_z: float, fev1_z: float, ratio_z: float) -> Tuple[str, str]:
    if ratio_z >= -1.64 and fvc_z >= -1.64 and fev1_z < -1.64:
        pattern = "Isolated"
    elif ratio_z < -1.64 and fvc_z >= -1.64:
        pattern = "Obstructive"
    elif ratio_z >= -1.64 and fvc_z < -1.64:
        pattern = "Restrictive"
    elif ratio_z < -1.64 and fvc_z < -1.64:
        pattern = "Mixed"
    else:
        pattern = "Normal"
        severity = "normal"

    if pattern != "Normal":
        if fev1_z >= -2.50:
            severity = "mild"
        elif fev1_z >= -4.00:
            severity = "moderate"
        else:
            severity = "severe"

    return pattern, severity


def report(pattern: str, severity: str, bmi: Optional[float]) -> str:
    if pattern == "Isolated":
        text = f"{severity} ventilatory limitation (isolated reduced FEV1)."
    elif pattern == "Obstructive":
        text = f"{severity} airflow obstruction."
    elif pattern == "Restrictive":
        text = (
            f"{severity} ventilatory limitation which appears to be restrictive in pattern.\n"
            "Suggest to correlate clinically for possibility of restrictive lung disease and "
            "perform lung volumes & diffusion capacity if clinically indicated."
        )
        if bmi is not None and bmi >= 30:
            text += f" The restrictive picture may be contributed by raised BMI (BMI {int(bmi)})."
    elif pattern == "Mixed":
        text = (
            f"{severity} ventilatory limitation which appears to be both obstructive and restrictive in pattern.\n"
            "Suggest to correlate clinically for possibility of concomitant restrictive lung disease and "
            "perform lung volumes & diffusion capacity if clinically indicated."
        )
        if bmi is not None and bmi >= 30:
            text += f" The restrictive picture may be contributed by raised BMI (BMI {int(bmi)})."
    else:
        text = f"{severity} limits"
    return text


def check10pct(pct_changes: List[Optional[float]]) -> str:
    changed10pct = any(v is not None and v > 10 for v in pct_changes)
    if changed10pct:
        return "There is Significant bronchodilator response to 400mcg of inhaled Salbutamol."
    return "There is no significant bronchodilator response to 400mcg of inhaled Salbutamol."


def load_bmi_map(path: Optional[Path]) -> Dict[str, float]:
    if not path:
        return {}
    df = pd.read_csv(path)
    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        case_id = str(row.get("case_id", "")).strip()
        bmi = _num(row.get("bmi"))
        if case_id and bmi is not None:
            out[case_id] = float(bmi)
    return out


def classify_table(csv_path: Path, bmi: Optional[float]) -> ClassResult:
    df = pd.read_csv(csv_path)
    case_id = df["case_id"].iloc[0] if "case_id" in df.columns else csv_path.stem.replace("_p2_table", "")
    warnings: List[str] = []

    def z_for(metric: str, col: str) -> Optional[float]:
        try:
            row = df.loc[df["metric"] == metric]
            if row.empty:
                return None
            return _num(row.iloc[0][col])
        except Exception:
            return None

    fvc_pre = z_for("FVC", "Pre_Z-Score")
    fev1_pre = z_for("FEV1", "Pre_Z-Score")
    ratio_pre = z_for("FEV1/FVC", "Pre_Z-Score")
    fvc_post = z_for("FVC", "Post_Z-Score")
    fev1_post = z_for("FEV1", "Post_Z-Score")
    ratio_post = z_for("FEV1/FVC", "Post_Z-Score")

    pct_changes = [
        _num(df.loc[df["metric"] == m, "%CHG"].iloc[0]) if not df.loc[df["metric"] == m].empty else None
        for m in ["FVC", "FEV1", "FEV1/FVC"]
    ]

    if None in (fvc_pre, fev1_pre, ratio_pre):
        warnings.append("Missing pre Z-scores for FVC/FEV1/FEV1-FVC.")
        # Fail safe: mark as unknown.
        pre_pattern, pre_severity = "Unknown", "unknown"
        pre_report = "Insufficient data to classify pre-bronchodilator spirometry."
    else:
        pre_pattern, pre_severity = classify_spirometry(fvc_pre, fev1_pre, ratio_pre)
        pre_report = report(pre_pattern, pre_severity, bmi)

    post_present = fvc_post is not None or fev1_post is not None or ratio_post is not None
    if not post_present:
        post_pattern, post_severity = "N/A", "n/a"
        post_report = ""
        combined = f"Spirometry demonstrates {pre_report}"
    else:
        if None in (fvc_post, fev1_post, ratio_post):
            warnings.append("Partial post Z-scores present; classification may be unreliable.")
            post_pattern, post_severity = "Unknown", "unknown"
            post_report = "Insufficient data to classify post-bronchodilator spirometry."
        else:
            post_pattern, post_severity = classify_spirometry(fvc_post, fev1_post, ratio_post)
            post_report = report(post_pattern, post_severity, bmi)

        if pre_pattern == post_pattern and pre_severity == post_severity:
            combined = f"Both pre- and post- bronchodilator spirometry demonstrate {pre_report}"
        else:
            combined = f"Pre- bronchodilator spirometry demonstrates {pre_report}\nPost- bronchodilator spirometry demonstrates {post_report}"

    broncho = check10pct(pct_changes)

    return ClassResult(
        case_id=case_id,
        bmi_used=bmi,
        pre_fvc_z=fvc_pre,
        pre_fev1_z=fev1_pre,
        pre_ratio_z=ratio_pre,
        post_fvc_z=fvc_post,
        post_fev1_z=fev1_post,
        post_ratio_z=ratio_post,
        pct_chg_fvc=pct_changes[0] if len(pct_changes) > 0 else None,
        pct_chg_fev1=pct_changes[1] if len(pct_changes) > 1 else None,
        pct_chg_ratio=pct_changes[2] if len(pct_changes) > 2 else None,
        pre_pattern=pre_pattern,
        pre_severity=pre_severity,
        post_pattern=post_pattern,
        post_severity=post_severity,
        report_pre=pre_report,
        report_post=post_report,
        report_combined=combined,
        bronchodilator_response=broncho,
        warnings=warnings,
    )


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    allow = {c.strip().lower() for c in args.cases.split(",") if c.strip()} if args.cases else set()
    bmi_map = load_bmi_map(args.bmi_map)

    csvs = sorted(p for p in args.tables_dir.glob("*_p2_table.csv"))
    processed = 0
    results: List[ClassResult] = []

    for csv_path in csvs:
        if allow:
            name = csv_path.name.lower()
            stem = csv_path.stem.lower()
            if name not in allow and stem not in allow and stem.replace("_p2_table", "") not in allow:
                continue
        if args.limit and processed >= args.limit:
            break

        case_id = csv_path.stem.replace("_p2_table", "")
        bmi = bmi_map.get(case_id) if bmi_map else args.bmi
        res = classify_table(csv_path, bmi)

        (out_dir / f"{case_id}_classification.json").write_text(
            json.dumps(asdict(res), indent=2), encoding="utf-8"
        )
        results.append(res)
        processed += 1
        print(f"[OK] {csv_path.name} -> {case_id}_classification.json")

    if results:
        df = pd.DataFrame([asdict(r) for r in results])
        df.to_csv(out_dir / "classification_combined.csv", index=False, encoding="utf-8-sig")
        print(f"Combined CSV -> {out_dir / 'classification_combined.csv'}")

    print(f"Done. Classification written to: {out_dir}")
    return 0 if processed else 2


if __name__ == "__main__":
    raise SystemExit(main())
