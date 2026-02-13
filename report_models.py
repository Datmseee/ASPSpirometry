from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class GeneratedReport:
    source_path: str
    predicted_path: str
    status: str
    message: str = ""
    generated_at: Optional[datetime] = None
    expiratory_truncation: Optional[bool] = None
    inspiratory_truncation: Optional[bool] = None
    exp_confidence: Optional[float] = None
    insp_confidence: Optional[float] = None
    classification: Optional[dict] = None
    table_rows: Optional[list] = None
    table_error: str = ""
