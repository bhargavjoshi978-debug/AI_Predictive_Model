"""Tests for streaming layer and alert state machine with hysteresis."""

import pytest
from pdm.config import AlertsConfig
from pdm.streaming.alerts import AlertState, AlertStateMachine


def test_alert_state_machine_hysteresis_progression() -> None:
    """Verify NORMAL -> WARNING -> CRITICAL transitions require consecutive steps."""
    cfg = AlertsConfig(
        warn_threshold=0.5,
        critical_threshold=0.8,
        hysteresis_warn_steps=3,
        hysteresis_critical_steps=3,
        recovery_warn_steps=5,
        recovery_critical_steps=5,
    )
    sm = AlertStateMachine(cfg)
    assert sm.state == AlertState.NORMAL

    # Step 1: Score 0.6 (above warn, but only 1 step) -> Still NORMAL
    a1 = sm.update(t=1, joint=0, fused_score=0.6)
    assert a1.state == AlertState.NORMAL

    # Step 2: Score 0.6 (2 consecutive steps) -> Still NORMAL
    a2 = sm.update(t=2, joint=0, fused_score=0.6)
    assert a2.state == AlertState.NORMAL

    # Step 3: Score 0.6 (3 consecutive steps) -> Transitions to WARNING
    a3 = sm.update(t=3, joint=0, fused_score=0.6)
    assert a3.state == AlertState.WARNING

    # Step 4-5: Score 0.9 (above critical, 1 and 2 steps) -> Remains WARNING
    sm.update(t=4, joint=0, fused_score=0.9)
    sm.update(t=5, joint=0, fused_score=0.9)
    assert sm.state == AlertState.WARNING

    # Step 6: 3rd consecutive step above 0.8 -> Transitions to CRITICAL
    a6 = sm.update(t=6, joint=0, fused_score=0.9)
    assert a6.state == AlertState.CRITICAL

    # Recovery: below critical for 4 steps -> Still CRITICAL
    for t in range(7, 11):
        sm.update(t=t, joint=0, fused_score=0.7)
        assert sm.state == AlertState.CRITICAL

    # 5th consecutive step below critical -> Recovers to WARNING
    a11 = sm.update(t=11, joint=0, fused_score=0.7)
    assert a11.state == AlertState.WARNING

    # Recovery: below warn for 5 steps -> Recovers to NORMAL
    for t in range(12, 16):
        sm.update(t=t, joint=0, fused_score=0.2)
        assert sm.state == AlertState.WARNING

    a16 = sm.update(t=16, joint=0, fused_score=0.2)
    assert a16.state == AlertState.NORMAL


def test_alert_no_flicker_on_noisy_boundary() -> None:
    """Verify that noisy boundary oscillation does not cause state toggling flicker."""
    cfg = AlertsConfig(
        warn_threshold=0.5,
        critical_threshold=0.8,
        hysteresis_warn_steps=3,
        hysteresis_critical_steps=3,
        recovery_warn_steps=5,
        recovery_critical_steps=5,
    )
    sm = AlertStateMachine(cfg)

    # Oscillating around warn threshold 0.5
    noisy_scores = [0.49, 0.51, 0.48, 0.52, 0.49, 0.51, 0.47, 0.53, 0.49]
    states = []
    for t, score in enumerate(noisy_scores, 1):
        alert = sm.update(t=t, joint=0, fused_score=score)
        states.append(alert.state)

    # Should remain solidly NORMAL despite jitter crossing 0.5
    assert all(s == AlertState.NORMAL for s in states), f"State flickered: {states}"
