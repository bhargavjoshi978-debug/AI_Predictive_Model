import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure workspace root is in sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pdm.config import AlertsConfig, Config, load_config
from pdm.models.xgboost_model import CLASS_NAMES
from pdm.simulator.robot import EpisodeConfig, GroundTruth, Observation, RobotArm
from pdm.streaming.alerts import Alert, AlertState, AlertStateMachine
from pdm.streaming.detector import LiveDetector, Prediction

st.set_page_config(page_title="Live Simulation | AI PdM", page_icon="🔴", layout="wide")

st.markdown("""
<style>
    .status-card {
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }
    .card-normal { background: rgba(46, 125, 50, 0.2); border: 2px solid #2e7d32; color: #4caf50; }
    .card-warning { background: rgba(249, 168, 37, 0.2); border: 2px solid #f9a825; color: #fbc02d; }
    .card-critical { background: rgba(198, 40, 40, 0.2); border: 2px solid #c62828; color: #ef5350; }
</style>
""", unsafe_allow_html=True)

st.title("🔴 Live Robot Arm Digital Twin Simulation")
st.caption("Real-time telemetry generation, live feature extraction, multi-model inference, and hysteresis alerting.")

# Load base configuration
cfg = load_config()

# Sidebar: Controls
st.sidebar.header("Simulation Scenario Controls")

scenario_choice = st.sidebar.selectbox(
    "Scenario Preset",
    ["Bearing wear", "Gear wear", "Motor overheat", "Cable fault", "Healthy", "Random"],
    index=0,
)

target_joint = st.sidebar.slider("Target Fault Joint", min_value=1, max_value=cfg.simulator.n_joints, value=2) - 1

deg_rate = st.sidebar.slider(
    "Degradation Rate (r)",
    min_value=0.00005,
    max_value=0.0006,
    value=0.00025,
    step=0.00005,
    format="%.5f",
    help="Higher values accelerate time to failure for demonstration purposes.",
)

noise_mult = st.sidebar.slider("Sensor Noise Multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
sim_speed = st.sidebar.slider("Simulation Speed (ticks per cycle)", min_value=1, max_value=30, value=5)

st.sidebar.markdown("---")
st.sidebar.subheader("Active Inference Models")
use_xgb = st.sidebar.checkbox("XGBoost (Proposed)", value=True)
use_rf = st.sidebar.checkbox("Random Forest", value=False)
use_ae = st.sidebar.checkbox("Autoencoder (Anomaly)", value=True)
use_lstm = st.sidebar.checkbox("LSTM (Deep Baseline)", value=False)

active_models_list = []
if use_xgb:
    active_models_list.append("xgboost")
if use_rf:
    active_models_list.append("random_forest")
if use_ae:
    active_models_list.append("autoencoder")
if use_lstm:
    active_models_list.append("lstm")

# Session state initialization
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False
if "sim_step" not in st.session_state:
    st.session_state.sim_step = 0
if "robot_arm" not in st.session_state:
    st.session_state.robot_arm = None
if "detector" not in st.session_state:
    st.session_state.detector = None
if "history" not in st.session_state:
    st.session_state.history = []
if "event_log" not in st.session_state:
    st.session_state.event_log = []

# Control buttons
col_b1, col_b2, col_b3, col_b4 = st.sidebar.columns(4)

with col_b1:
    btn_start = st.button("▶️ Start", use_container_width=True)
with col_b2:
    btn_pause = st.button("⏸️ Pause", use_container_width=True)
with col_b3:
    btn_reset = st.button("🔄 Reset", use_container_width=True)
with col_b4:
    btn_inject = st.button("⚡ Inject Fault", use_container_width=True)

if btn_reset or st.session_state.robot_arm is None:
    st.session_state.sim_running = False
    st.session_state.sim_step = 0
    st.session_state.history = []
    st.session_state.event_log = []

    # Map scenario name to fault type
    scenario_map = {
        "Bearing wear": "bearing",
        "Gear wear": "gear_wear",
        "Motor overheat": "motor_overheat",
        "Cable fault": "cable_fault",
        "Healthy": "none",
        "Random": np.random.choice(["bearing", "gear_wear", "motor_overheat", "cable_fault"]),
    }
    f_type = scenario_map[scenario_choice]

    ep_cfg = EpisodeConfig(
        robot_id=1,
        n_joints=cfg.simulator.n_joints,
        fault_type=f_type,  # type: ignore[arg-type]
        fault_joint=target_joint if f_type != "none" else None,
        onset_step=10,  # quick onset for demonstration
        degradation_rate=deg_rate,
        duration_steps=3000,
        seed=int(time.time()) % 10000,
    )

    arm = RobotArm(cfg.simulator, cfg.dataset.degradation, n_joints=cfg.simulator.n_joints, seed=ep_cfg.seed)
    arm.configure_episode(ep_cfg)
    st.session_state.robot_arm = arm
    st.session_state.detector = LiveDetector(config=cfg, active_models=active_models_list)

if btn_start:
    st.session_state.sim_running = True

if btn_pause:
    st.session_state.sim_running = False

if btn_inject:
    if st.session_state.robot_arm is not None:
        j = st.session_state.robot_arm.joints[target_joint]
        inj_fault = "bearing" if scenario_choice == "Healthy" else scenario_map.get(scenario_choice, "bearing")
        j.inject_fault(inj_fault, st.session_state.sim_step, deg_rate)
        st.session_state.robot_arm.fault_type = inj_fault
        st.session_state.robot_arm.fault_joint = target_joint
        st.sidebar.success(f"Injected {inj_fault} on Joint {target_joint + 1} at step {st.session_state.sim_step}!")

# Update active models in detector
if st.session_state.detector is not None:
    st.session_state.detector.active_models = active_models_list

# Simulation execution step
if st.session_state.sim_running and st.session_state.robot_arm is not None:
    for _ in range(sim_speed):
        obs = st.session_state.robot_arm.step()
        gt = st.session_state.robot_arm.ground_truth()
        preds, alerts, highest_risk = st.session_state.detector.process_step(obs)

        st.session_state.sim_step = obs.t

        # Target joint info
        target_pred = preds[target_joint]
        target_alert = alerts[target_joint]
        target_health = float(gt.health[target_joint])

        # Record history for plotting
        st.session_state.history.append({
            "t": obs.t,
            "target_health": target_health,
            "true_rul": gt.rul_steps if gt.rul_steps is not None else np.nan,
            "pred_rul": target_pred.rul_estimate,
            "fused_risk": target_alert.fused_score,
            "alert_state": target_alert.state.value,
            "anomaly_score": target_pred.anomaly_score,
            "pred_fault": max(target_pred.fault_probs.items(), key=lambda x: x[1])[0],
            "vib_last": obs.joints[target_joint].signals["vibration"][:, 0].copy(),
            "temp": obs.joints[target_joint].slow["temperature_c"],
            "curr": np.mean(obs.joints[target_joint].signals["current"]),
            "fault_probs": target_pred.fault_probs.copy(),
            "joint_states": [a.state.value for a in alerts],
            "joint_healths": [float(h) for h in gt.health],
            "joint_risks": [float(a.fused_score) for a in alerts],
        })

        # Append to event log if state changed or warning/critical
        if target_alert.state in [AlertState.WARNING, AlertState.CRITICAL]:
            if not st.session_state.event_log or st.session_state.event_log[-1]["State"] != target_alert.state.value:
                st.session_state.event_log.append({
                    "Step": obs.t,
                    "Joint": f"Joint {target_joint + 1}",
                    "State": target_alert.state.value,
                    "Predicted Fault": target_pred.model + " -> " + max(target_pred.fault_probs.items(), key=lambda x: x[1])[0],
                    "Fused Risk": f"{target_alert.fused_score:.2f}",
                    "Est. RUL": f"{target_pred.rul_estimate:.0f} steps" if target_pred.rul_estimate is not None else "-",
                })

        # Stop if failed
        if target_health <= 0.10:
            st.session_state.sim_running = False
            break

# UI: Top row 6 Joint Cards
st.subheader("Robot Arm Kinematic Joints Status")
card_cols = st.columns(cfg.simulator.n_joints)

latest_record = st.session_state.history[-1] if st.session_state.history else None

for j_idx in range(cfg.simulator.n_joints):
    with card_cols[j_idx]:
        if latest_record is not None:
            j_state = latest_record["joint_states"][j_idx]
            j_health = latest_record["joint_healths"][j_idx]
            j_risk = latest_record["joint_risks"][j_idx]
        else:
            j_state = "NORMAL"
            j_health = 1.0
            j_risk = 0.0

        if j_state == "NORMAL":
            card_class = "card-normal"
            badge_icon = "🟢"
        elif j_state == "WARNING":
            card_class = "card-warning"
            badge_icon = "🟡"
        else:
            card_class = "card-critical"
            badge_icon = "🔴"

        st.markdown(f"""
        <div class="status-card {card_class}">
            <div style="font-size: 1.1rem; margin-bottom: 4px;">Joint {j_idx + 1}</div>
            <div style="font-size: 1.3rem;">{badge_icon} {j_state}</div>
            <div style="font-size: 0.85rem; margin-top: 6px;">Health: {j_health * 100:.1f}% | Risk: {j_risk:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Main Plotly Charts
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)

    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        # 1. Raw Vibration Waveform & FFT Spectrum
        latest_vib = latest_record["vib_last"]
        n_samples = len(latest_vib)
        t_arr = np.linspace(0, 1.0, n_samples)

        # FFT
        rfft = np.abs(np.fft.rfft(latest_vib * np.hanning(n_samples)))
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / cfg.simulator.sample_rate_hz)

        fig_raw = make_subplots(rows=2, cols=1, subplot_titles=("Raw Vibration X-Axis (1 kHz Window)", "Vibration FFT Spectrum (0-500 Hz)"))
        fig_raw.add_trace(go.Scatter(x=t_arr, y=latest_vib, line=dict(color="#1565c0", width=1.5), name="Vib X"), row=1, col=1)
        fig_raw.add_trace(go.Scatter(x=freqs[:200], y=rfft[:200], line=dict(color="#f9a825", width=1.8), name="FFT Amplitude"), row=2, col=1)
        fig_raw.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_raw, use_container_width=True)

    with row1_c2:
        # 2. Time Series: Fused Risk and Alert Thresholds
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(x=hist_df["t"], y=hist_df["fused_risk"], line=dict(color="#c62828", width=2.5), name="Fused Risk Score R"))
        fig_risk.add_hline(y=0.5, line=dict(color="#f9a825", dash="dash"), annotation_text="WARNING (0.5)")
        fig_risk.add_hline(y=0.8, line=dict(color="#c62828", dash="dash"), annotation_text="CRITICAL (0.8)")
        fig_risk.update_layout(
            title=f"Fused Risk Score & Alert Thresholds (Joint {target_joint + 1})",
            xaxis_title="Simulation Step (t)",
            yaxis_title="Risk Score R ∈ [0, 1]",
            yaxis=dict(range=[-0.05, 1.05]),
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        # 3. Anomaly Score vs. Normalized Threshold
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(x=hist_df["t"], y=hist_df["anomaly_score"], line=dict(color="#7b1fa2", width=2), name="Autoencoder Anomaly Score"))
        fig_anom.add_hline(y=1.0, line=dict(color="red", dash="dash"), annotation_text="Threshold (1.0)")
        fig_anom.update_layout(
            title="Autoencoder Reconstruction Anomaly Score",
            xaxis_title="Simulation Step (t)",
            yaxis_title="Normalized Score",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_anom, use_container_width=True)

    with row2_c2:
        # 4. RUL Estimation vs. Digital Twin Ground Truth
        fig_rul = go.Figure()
        valid_true = hist_df.dropna(subset=["true_rul"])
        fig_rul.add_trace(go.Scatter(x=valid_true["t"], y=valid_true["true_rul"], line=dict(color="gray", dash="dash", width=2), name="Digital Twin Ground Truth"))
        fig_rul.add_trace(go.Scatter(x=hist_df["t"], y=hist_df["pred_rul"], line=dict(color="#1565c0", width=2.5), name="Model RUL Estimate"))
        fig_rul.update_layout(
            title="Remaining Useful Life (RUL): Predicted vs. True Ground Truth",
            xaxis_title="Simulation Step (t)",
            yaxis_title="RUL (steps)",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_rul, use_container_width=True)

# Event log table
st.subheader("📋 Alarm Event Log (Hysteresis Filtered)")
if st.session_state.event_log:
    log_df = pd.DataFrame(st.session_state.event_log).iloc[::-1]  # latest first
    st.dataframe(log_df, use_container_width=True)
else:
    st.info("System operating nominally. No WARNING or CRITICAL alert events recorded.")

# Auto-refresh loop when running
if st.session_state.sim_running:
    time.sleep(0.05)
    st.rerun()
