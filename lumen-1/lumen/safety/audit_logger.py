"""Layer 6: Audit logger for safety events."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AuditEvent:
    timestamp: float
    event_type: str
    layer: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None


class AuditLogger:
    """Logs safety pipeline events for compliance and red-teaming analysis."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path) if log_path else None
        self.events: List[AuditEvent] = []

    def log(self, event_type: str, layer: str, passed: bool, details: Optional[Dict] = None, request_id: Optional[str] = None) -> None:
        event = AuditEvent(
            timestamp=time.time(),
            event_type=event_type,
            layer=layer,
            passed=passed,
            details=details or {},
            request_id=request_id,
        )
        self.events.append(event)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event)) + "\n")

    def summary(self) -> Dict[str, int]:
        blocked = sum(1 for e in self.events if not e.passed)
        return {"total": len(self.events), "blocked": blocked, "passed": len(self.events) - blocked}
