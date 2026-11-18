import threading
from dataclasses import dataclass, field


@dataclass
class QueryManagerResult:
    name: str
    success: bool
    exc: Exception = None
    detail: str = None
