from typing import List, Optional

from pydantic import BaseModel


class SafetyModelResponse(BaseModel):
    """Dataclass that all safety model responses should adhere to"""

    is_safe: bool
    safety_categories: List[str]
    is_valid: bool = True
    reasoning: Optional[str] = None
    total_tokens_used: Optional[float] = None
