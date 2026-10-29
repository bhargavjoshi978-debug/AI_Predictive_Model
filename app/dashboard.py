import sys
from pathlib import Path

# Ensure workspace root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(
    page_title="AI Predictive Maintenance Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and clean typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-normal { background-color: rgba(46, 125, 50, 0.25); color: #4caf50; border: 1px solid #2e7d32; }
    .badge-warning { background-color: rgba(249, 168, 37, 0.25); color: #fbc02d; border: 1px solid #f9a825; }
    .badge-critical { background-color: rgba(198, 40, 40, 0.25); color: #ef5350; border: 1px solid #c62828; }
    .disclaimer-box {
        background: rgba(33, 150, 243, 0.1);
        border-left: 4px solid #1976d2;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 15px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Based Predictive Maintenance Simulator for Robotic Equipment")
st.caption("A physics-inspired digital-twin simulation framework based on Rana, Joshi, Kataria (SAKEC, Mumbai).")

st.markdown("""
<div class="disclaimer-box">
    <strong>Mandatory Disclosure:</strong> Results are from synthetic data and are not validated on real robots.
    Figures (85% / 94% / 90% / 92%) represent surveyed literature baselines provided for context only.
</div>
""", unsafe_allow_html=True)

# Overview metrics row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(label="Monitored Kinematics", value="6-DOF Arm", delta="Real-time 1 kHz")
with c2:
    st.metric(label="Active Sensors", value="Vib, Curr, Temp", delta="Digital Twin Proxy")
with c3:
    st.metric(label="Proposed Edge Model", value="XGBoost + FFT", delta="Lightweight CPU")
with c4:
    st.metric(label="Edge Feasibility", value="Raspberry Pi 4", delta="< 50ms Latency Budget")

st.markdown("---")

st.subheader("Interactive Navigation")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    ### 🔴 [1. Live Simulation](Live_Simulation)
    - Real-time tick-by-tick physics telemetry generator.
    - Fault injectors: Bearing wear, Gear wear, Motor overheat, Cable intermittent faults.
    - Live multi-joint alert status cards (`NORMAL → WARNING → CRITICAL`).
    - Real-time vibration waveforms, FFT spectrum, and RUL estimation tracking.
    
    ### 📊 [2. Model Comparison](Model_Comparison)
    - Benchmarked performance of Random Forest, XGBoost, Autoencoder, and LSTM.
    - Multi-class confusion matrices, ROC curves, and RUL regression accuracy.
    - Top 15 XGBoost feature importances (gain-based).
    """)

with col_b:
    st.markdown("""
    ### ⚡ [3. Edge Feasibility](Edge_Feasibility)
    - Single-window inference latency, peak RAM consumption, and serialized footprint.
    - Pass/Fail verification against embedded Raspberry Pi 4 edge budget.
    - Interactive hardware slowdown factor slider.

    ### 🔬 [4. Experiments & Ablations](Experiments)
    - E2: Time-domain vs. FFT feature ablation.
    - E3: Noise robustness sweep under elevated sensor degradation.
    - E4: Data scarcity curve (10% to 100% training volume).
    - E5: Zero-day leave-one-fault-out autoencoder anomaly detection.

    ### ℹ️ [5. About & Methodology](About)
    - Research context, problem statement, mathematical formulas, and literature citations.
    """)

st.info("👈 Use the sidebar navigation to launch the **Live Simulation** or explore benchmarked experimental results.")
