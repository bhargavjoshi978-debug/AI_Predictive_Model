# Experimental Results: Predictive Maintenance for Robotic Arms

> **Honesty Notice:** Results are from synthetic data and are not validated on real robots.
> Figures in the rightmost column originate from the literature reviewed in the manuscript and are provided for context only.

## Table 1: Model Comparison (E1 & E6)

| Model         | Accuracy   | Macro-F1   | Recall (anomaly)   | RUL MAE   |   Latency p95 (ms) |   Size (MB) |   RAM Δ (MB) | Edge OK?   | Literature ref. (context only)                 |
|---------------|------------|------------|--------------------|-----------|--------------------|-------------|--------------|------------|------------------------------------------------|
| Random Forest | 0.997      | 0.997      | -                  | -         |             159.72 |        8.74 |         0.04 | FAIL       | 85% accuracy (Smith et al.)                    |
| XGBoost + FFT | 0.997      | 0.997      | -                  | 3.3       |               8.26 |        1.8  |         0    | PASS       | 94% accuracy (Al-Fahad et al. / manuscript §3) |
| Autoencoder   | -          | -          | 0.974              | -         |               2.5  |        0.09 |         0    | PASS       | zero-day detection (Wu & Zhao)                 |
| LSTM          | -          | -          | 0.000              | 7.5       |              10.31 |        0.23 |         0    | PASS       | 92% recall (Chen & Wang)                       |

### Observations & Key Takeaways
- **Lightweight Efficiency:** XGBoost delivers high classification macro-F1 while satisfying edge constraints.
- **Zero-Day Detection:** Unsupervised Autoencoder detects anomalies with low false-alarm rate on healthy windows (3.0%).
- **Edge Feasibility:** Single-window inference latencies evaluated against Raspberry Pi 4 edge budget profile (<= 50 ms with 5x slowdown factor).
