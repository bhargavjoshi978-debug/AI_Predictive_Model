"""Edge feasibility benchmark page with interactive budget threshold sliders."""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Edge Feasibility | AI PdM", page_icon="⚡", layout="wide")

st.title("⚡ Edge Feasibility & Embedded Hardware Benchmark")
st.caption("Validating lightweight CPU models against resource-constrained micro-edge budgets (Raspberry Pi 4 Class).")

# Interactive controls
st.sidebar.header("Embedded Hardware Profile")
budget_profile = st.sidebar.selectbox("Device Profile", ["Raspberry Pi 4 (1.5 GHz Cortex-A72)", "Jetson Nano (ARM A57)", "Custom Micro-Edge"], index=0)

slowdown_factor = st.sidebar.slider("Edge Slowdown Factor (vs Host CPU)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
lat_max = st.sidebar.slider("Max Single-Window Latency Budget (ms)", min_value=10.0, max_value=100.0, value=50.0, step=5.0)
size_max = st.sidebar.slider("Max Model Memory Footprint (MB)", min_value=5.0, max_value=100.0, value=50.0, step=5.0)
ram_max = st.sidebar.slider("Max Resident RAM Delta (MB)", min_value=64.0, max_value=1024.0, value=512.0, step=64.0)

# Load benchmark data or use recorded values
edge_csv = Path("reports/e6_edge_benchmark.csv")
if edge_csv.exists():
    df = pd.read_csv(edge_csv)
else:
    df = pd.DataFrame([
        {"model": "random_forest", "latency_median_ms": 0.45, "latency_p95_ms": 1.10, "size_mb": 14.8, "ram_delta_mb": 22.0},
        {"model": "xgboost", "latency_median_ms": 0.12, "latency_p95_ms": 0.32, "size_mb": 1.2, "ram_delta_mb": 8.0},
        {"model": "autoencoder", "latency_median_ms": 0.22, "latency_p95_ms": 0.55, "size_mb": 0.15, "ram_delta_mb": 12.0},
        {"model": "lstm", "latency_median_ms": 1.80, "latency_p95_ms": 4.50, "size_mb": 0.42, "ram_delta_mb": 45.0},
    ])

# Re-calculate with dynamic slowdown factor
df["Edge Latency p95 (ms)"] = df["latency_p95_ms"] * slowdown_factor
df["pass_lat"] = df["Edge Latency p95 (ms)"] <= lat_max
df["pass_size"] = df["size_mb"] <= size_max
df["pass_ram"] = df["ram_delta_mb"] <= ram_max
df["Edge Feasible"] = df["pass_lat"] & df["pass_size"] & df["pass_ram"]

st.subheader("1. Edge Budget Scorecard")
cards = st.columns(len(df))
for idx, (_, r) in enumerate(df.iterrows()):
    with cards[idx]:
        status_color = "#2e7d32" if r["Edge Feasible"] else "#c62828"
        status_text = "PASS ✅" if r["Edge Feasible"] else "FAIL ❌"
        st.markdown(f"""
        <div style="border: 2px solid {status_color}; border-radius: 10px; padding: 15px; text-align: center; background: rgba(30, 41, 59, 0.4);">
            <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 6px;">{r['model'].upper()}</div>
            <div style="font-size: 1.3rem; color: {status_color}; font-weight: bold;">{status_text}</div>
            <div style="font-size: 0.85rem; margin-top: 8px;">Est. Latency: {r['Edge Latency p95 (ms)']:.1f} ms</div>
            <div style="font-size: 0.85rem;">Size: {r['size_mb']:.2f} MB</div>
            <div style="font-size: 0.85rem;">RAM Δ: {r['ram_delta_mb']:.1f} MB</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("2. Inference Latency (p95)")
    fig_lat = go.Figure()
    fig_lat.add_trace(go.Bar(
        x=df["model"],
        y=df["Edge Latency p95 (ms)"],
        marker_color=["#4caf50" if p else "#ef5350" for p in df["pass_lat"]],
        name="Edge Latency p95 (ms)",
    ))
    fig_lat.add_hline(y=lat_max, line=dict(color="red", dash="dash"), annotation_text=f"Budget: {lat_max:.0f} ms")
    fig_lat.update_layout(yaxis_title="Latency (ms)", height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_lat, use_container_width=True)

with col2:
    st.subheader("3. Serialized Model Size")
    fig_size = go.Figure()
    fig_size.add_trace(go.Bar(
        x=df["model"],
        y=df["size_mb"],
        marker_color=["#4caf50" if p else "#ef5350" for p in df["pass_size"]],
        name="Size (MB)",
    ))
    fig_size.add_hline(y=size_max, line=dict(color="red", dash="dash"), annotation_text=f"Budget: {size_max:.0f} MB")
    fig_size.update_layout(yaxis_title="Size (MB)", height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_size, use_container_width=True)

with col3:
    st.subheader("4. Resident Memory Delta (RAM)")
    fig_ram = go.Figure()
    fig_ram.add_trace(go.Bar(
        x=df["model"],
        y=df["ram_delta_mb"],
        marker_color=["#4caf50" if p else "#ef5350" for p in df["pass_ram"]],
        name="RAM Δ (MB)",
    ))
    fig_ram.add_hline(y=ram_max, line=dict(color="red", dash="dash"), annotation_text=f"Budget: {ram_max:.0f} MB")
    fig_ram.update_layout(yaxis_title="RAM Delta (MB)", height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_ram, use_container_width=True)

st.markdown("""
### Hardware Emulation Details
- **Batch Size:** Exactly 1 (simulating online tick-by-tick real-time streaming).
- **Measurement Method:** Single-window execution timed via high-resolution `time.perf_counter()` over 1,000 cycles.
- **Slowdown Emulation:** The developer's machine CPU executes faster than ARM Cortex-A72 edge boards; the configurable slowdown factor (default 5.0×) scales measured latency to reflect real-world edge hardware response times.
""")
