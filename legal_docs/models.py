from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np

@dataclass
class CaseDocument:
    """Case document with metadata"""
    id: str
    content: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    """Search result container"""
    case: CaseDocument
    score: float
    rank: int = 0