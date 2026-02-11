from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal

Provider = Literal["openai", "gemini", "anthropic"]


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(self.extra) if self.extra else {}
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            out["max_output_tokens"] = self.max_output_tokens
        return out
