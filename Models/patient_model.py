from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class PatientModel:
    id: str
    name: str
    last_test_datetime: datetime
    status: str
    flags: List[str] = field(default_factory=list)
