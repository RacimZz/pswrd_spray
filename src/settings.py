from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Default windowing
    window: str = "10min"  # pandas offset alias
    # Model defaults
    contamination: float = 0.01
    n_estimators: int = 300
    random_state: int = 42

    # Feature thresholds (triage)
    min_fail_for_success_after_fail: int = 20
