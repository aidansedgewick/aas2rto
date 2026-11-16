import threading
from dataclasses import dataclass, field


@dataclass
class QueryManagerResult:
    name: str
    success: bool = True
    error_message: str = None
