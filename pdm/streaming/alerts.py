"""Alert state machine with hysteresis for stable, flicker-free industrial alarms."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from pdm.config import AlertsConfig


class AlertState(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    t: int
    joint: int
    state: AlertState
    fault_type: str
    rul_estimate: Optional[float]
    fused_score: float


class AlertStateMachine:
    """State machine with hysteresis transitions to prevent rapid toggling on threshold boundaries.

    Transition rules:
    - NORMAL -> WARNING: R >= warn for warn_steps (default 3) consecutive steps
    - WARNING -> CRITICAL: R >= critical for crit_steps (default 3) consecutive steps
    - WARNING -> NORMAL: R < warn for recovery_warn_steps (default 5) consecutive steps
    - CRITICAL -> WARNING: R < critical for recovery_crit_steps (default 5) consecutive steps
    """

    def __init__(self, config: Optional[AlertsConfig] = None) -> None:
        self.config = config or AlertsConfig()
        self.state: AlertState = AlertState.NORMAL
        self.consecutive_above_warn: int = 0
        self.consecutive_above_crit: int = 0
        self.consecutive_below_warn: int = 0
        self.consecutive_below_crit: int = 0
        self.alert_history: List[Alert] = []

    def update(
        self,
        t: int,
        joint: int,
        fused_score: float,
        fault_type: str = "none",
        rul_estimate: Optional[float] = None,
    ) -> Alert:
        """Update state machine with latest fused risk score and return Alert status."""
        warn_th = self.config.warn_threshold
        crit_th = self.config.critical_threshold

        # Update counter tracking
        if fused_score >= warn_th:
            self.consecutive_above_warn += 1
            self.consecutive_below_warn = 0
        else:
            self.consecutive_below_warn += 1
            self.consecutive_above_warn = 0

        if fused_score >= crit_th:
            self.consecutive_above_crit += 1
            self.consecutive_below_crit = 0
        else:
            self.consecutive_below_crit += 1
            self.consecutive_above_crit = 0

        # State transition evaluation
        prev_state = self.state

        if self.state == AlertState.NORMAL:
            if self.consecutive_above_warn >= self.config.hysteresis_warn_steps:
                self.state = AlertState.WARNING
                self.consecutive_below_warn = 0

        elif self.state == AlertState.WARNING:
            if self.consecutive_above_crit >= self.config.hysteresis_critical_steps:
                self.state = AlertState.CRITICAL
                self.consecutive_below_crit = 0
            elif self.consecutive_below_warn >= self.config.recovery_warn_steps:
                self.state = AlertState.NORMAL
                self.consecutive_above_warn = 0

        elif self.state == AlertState.CRITICAL:
            if self.consecutive_below_crit >= self.config.recovery_critical_steps:
                self.state = AlertState.WARNING
                self.consecutive_above_crit = 0

        alert = Alert(
            t=t,
            joint=joint,
            state=self.state,
            fault_type=fault_type,
            rul_estimate=rul_estimate,
            fused_score=fused_score,
        )

        if self.state != AlertState.NORMAL or prev_state != AlertState.NORMAL:
            self.alert_history.append(alert)

        return alert
