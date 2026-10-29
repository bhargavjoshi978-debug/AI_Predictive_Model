# DESIGN.md

Detailed design for the **AI-Based Predictive Maintenance Simulator for Robotic Equipment**.
Read `README.md` (scope) and `ARCHITECTURE.md` (structure) first. This file defines
**what the simulation actually computes**, the default parameters, the dashboard, and
the acceptance criteria.

---

## 1. Design goals

1. Show, with reproducible experiments, that a **lightweight CPU model** can detect robot faults and estimate RUL in real time without real hardware or proprietary data.
2. Mirror the manuscript's method families (Random Forest, XGBoost + FFT, Autoencoder, LSTM, Digital Twin, optional Federated and Graph ideas).
3. Be a clear, demonstrable tool: a user can press "Start", inject a fault, and watch detection and RUL update live.

## 2. Simulated system

A **6-joint industrial robot arm** performing a repetitive pick-and-place cycle.
Each joint is monitored independently; the arm-level state is the worst joint.

Time model:
- Simulation step = 1 "operating second" (configurable).
- Within each step the simulator synthesizes a **raw signal window** (default 1,024 samples at 1 kHz) for each sensor channel. Features are computed from this raw window.
- An episode is typically 2,000–6,000 steps (healthy episodes use the same length, with no degradation).

### 2.1 Per-joint hidden state

| Variable | Range | Meaning |
|---|---|---|
| `h` | 1 → 0 | Health index. 1 = new, `h ≤ 0.1` = failure threshold |
| `load` | 0–1 | Normalized load following the pick-and-place cycle (periodic + noise) |
| `speed` | rad/s | Joint speed following the same cycle |
| `temp` | °C | First-order thermal model |

### 2.2 Degradation model

Health decays only after `onset_step`:

```
h(t) = 1                                           for t < onset
h(t) = max(0, 1 − r · (t − onset)^p + ε_t)         for t ≥ onset
```

- `r` = degradation rate **[configurable]**, sampled per episode from `Uniform(r_min, r_max)`.
- `p` = shape exponent, `p ∈ {1.0 (linear), 1.5 (accelerating)}`; default 1.5 so degradation visibly accelerates near failure.
- `ε_t` = small random-walk noise, `σ = 0.002`.
- **Failure time** = first `t` where `h ≤ 0.1`.
- **True RUL** at step `t` = `failure_time − t` (clipped at 0). Episodes that never fail within their length are **censored** (RUL = NaN, excluded from RUL metrics).
- For RUL training, cap RUL at `RUL_cap = 300` steps (piece-wise linear target) — standard practice and stabilizes regression.

## 3. Fault models

Healthy baseline signals (per joint, per step window):

| Channel | Healthy model |
|---|---|
| Vibration (x, y, z) | `A0 · sin(2π f_rot t) + 0.3·A0 · sin(2π·2 f_rot t) + N(0, σ_v)` where `f_rot ∝ speed`; `A0 ∝ 0.5 + 0.5·load` |
| Servo current (A) | `I0 · (0.4 + 0.6·load) + N(0, σ_i)` with a ripple at PWM harmonic |
| Temperature (°C) | `T_amb + k_load · load · (1 − e^(−t/τ))` plus noise; slow channel (one value per step) |
| Acoustic band energy (virtual, optional) | Band-limited noise energy proportional to `speed`, `N(0, σ_a)` |

Default noise (**[configurable]**): `σ_v = 0.05`, `σ_i = 0.03`, `σ_T = 0.3`, `σ_a = 0.02`.

Each fault changes the signals as a function of `d = 1 − h` (damage, 0 → 0.9):

| Fault type | Physical story | Signal effect (as `d` grows) |
|---|---|---|
| `bearing` | Bearing wear / spalling | Adds a **bearing-defect frequency** (`f_bpf = 3.5·f_rot`) with amplitude `∝ d`; adds impulsive spikes with probability `∝ d²` (raises kurtosis); raises broadband vibration noise by `1 + 2d` |
| `gear_wear` | Gear tooth wear | Adds **sidebands** around the gear-mesh frequency (`f_gm = 12·f_rot`, sidebands at `f_gm ± f_rot`) with amplitude `∝ d`; adds small speed ripple |
| `motor_overheat` | Winding / cooling degradation | Raises temperature by `+35·d` °C and thermal time-constant; raises current by `+25% · d`; slight efficiency drop (current up for same load) |
| `cable_fault` | Frayed/intermittent cable | **Intermittent** current dropouts/spikes with probability `∝ d`; intermittent sensor glitch bursts in vibration; no steady change in mean temperature |
| `none` | Healthy | No degradation |

Fault-mix default for the dataset: 20% healthy, 20% each fault type.

Only **one** fault per episode on a single joint (primary scenario). A "multi-fault"
scenario (two simultaneous faults on different joints) is available in the dashboard
only, not required for training.

### 3.1 Sensor imperfections (realism knobs, **[configurable]**)

- Slow calibration drift: `+0.02%` per 1,000 steps on temperature and current.
- Random dropout: probability `0.1%` per step per channel → NaN (imputed downstream).
- Outlier glitch: probability `0.05%` → value × 3 (tests robustness).

## 4. Feature engineering

Window: `window_len = 1024` samples, `hop = 512` (50% overlap) for the high-rate channels.
Since the simulator already emits one window per step, the default pipeline computes one
feature vector per joint per step; `hop` is used if raw continuous streams are replayed.

Per high-rate channel (vibration x/y/z, current, acoustic):

| Group | Features |
|---|---|
| Time-domain | mean, std, RMS, peak, peak-to-peak, skewness, kurtosis, crest factor |
| Frequency-domain (FFT, Hann window) | dominant frequency, dominant amplitude, spectral centroid, spectral entropy, energy in 8 equal bands, energy at 1×, 2×, bearing-defect and gear-mesh-sideband frequencies (relative to `f_rot`) |

Slow channel (temperature): current value, slope over last 5 steps.

Trend features (per joint): slope of RMS over last 5 windows, rolling max of kurtosis.

Total ≈ 120–150 features per window. Features are standardized with a scaler fitted on
**training healthy windows only**.

For sequence models: sequences of the **last 20 windows** × a selected subset of ~30 features
(top features by XGBoost importance — computed on training data only).

## 5. Models

### 5.1 Random Forest (baseline classifier)
- `n_estimators=200`, `max_depth=12`, `class_weight="balanced"`.
- Target: 5 classes (`none, bearing, gear_wear, motor_overheat, cable_fault`).
- Training windows: only windows where damage `d ≥ 0.15` are labelled with their fault class; earlier windows of the same episode are labelled `none` (fault not yet observable).

### 5.2 XGBoost (proposed lightweight model)
- Classification: `n_estimators=300`, `max_depth=6`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`, early stopping on validation (patience 20).
- RUL regression: separate `XGBRegressor`, same hyperparameters, target = capped RUL, trained on fault episodes only.
- Output feature importance (gain) and expose SHAP-free interpretability: top-15 features bar chart in dashboard.
- Must be the **fastest and smallest** accurate model; this is the headline claim.

### 5.3 Autoencoder (zero-day anomaly detection)
- Dense autoencoder: `input → 64 → 32 → 8 → 32 → 64 → input`, ReLU, `Adam(lr=1e-3)`, MSE loss, ≤ 60 epochs, early stopping, batch 256. CPU only.
- **Trained only on healthy windows** (training split).
- Anomaly score = reconstruction MSE, normalized by the 99th percentile of validation-healthy error.
- Threshold (default): 99th percentile of healthy validation error (**[configurable]**). Score ≥ 1.0 ⇒ anomaly.
- **Zero-day experiment (required):** train the autoencoder with all faults hidden; then evaluate on a **held-out fault type** it never saw (leave-one-fault-out across the 4 faults). Report detection rate per held-out fault.

### 5.4 LSTM (heavy deep-learning baseline)
- Input: sequence of 20 windows × ~30 features.
- Architecture: 2 layers, hidden 64, dropout 0.2, linear head.
- Tasks: (a) RUL regression (capped RUL), (b) binary anomaly (fault-active vs. not).
- `Adam(lr=1e-3)`, ≤ 40 epochs, batch 128, early stopping. CPU only.
- Purpose: demonstrate high recall but higher latency/size than XGBoost, as the manuscript states.

### 5.5 Model comparison summary expected

| Model | Task(s) | Expected behaviour (hypothesis, not a promise) |
|---|---|---|
| Random Forest | Fault class | Good accuracy, larger model, moderate latency |
| XGBoost | Fault class + RUL | Highest accuracy among tabular, lowest latency/size |
| Autoencoder | Anomaly | Detects unseen faults; lower precision on subtle faults |
| LSTM | RUL + anomaly | High recall, highest latency, largest memory |

The results must be reported as measured, even if the hypothesis does not hold.

## 6. Real-time detection & alerting

`LiveDetector` runs once per simulated step per joint:

1. Append window features to ring buffer.
2. Run selected models → `Prediction`.
3. Compute a **fused risk score** `R ∈ [0,1]`:
   `R = max( 1 − P(none) from XGBoost , min(1, anomaly_score / 2) , 1 − min(1, RUL_est / RUL_cap) · 𝟙[fault suspected] )`
   (uses only the models enabled in the UI).
4. Alert state machine:

```
NORMAL ──(R ≥ warn for 3 consecutive steps)──► WARNING
WARNING ──(R ≥ critical for 3 consecutive steps)──► CRITICAL
WARNING ──(R < warn for 5 consecutive steps)──► NORMAL
CRITICAL ──(R < critical for 5 consecutive steps)──► WARNING
```
   Defaults: `warn = 0.5`, `critical = 0.8`. Hysteresis counts **[configurable]**.

5. Log each alert with timestamp, joint, predicted fault, RUL, and fused score.

## 7. Evaluation protocol

### 7.1 Splits
Episode-level, stratified by fault type: 70% train / 15% val / 15% test. Repeat the whole
experiment with **5 different seeds** and report mean ± std.

### 7.2 Metrics

| Task | Metrics |
|---|---|
| Fault classification | Accuracy, macro precision / recall / F1, confusion matrix |
| Anomaly detection | ROC-AUC, recall at 1% false-positive rate, **false alarms per 1,000 healthy windows** |
| RUL | MAE, RMSE (steps), and **NASA-style asymmetric score** (late predictions penalized more) |
| Early warning | **Lead time** = failure step − first CRITICAL alert step; **detection delay** = first WARNING step − fault onset step |
| Edge | median/p95 latency (ms), model size (MB), RAM delta (MB), throughput, pass/fail vs. budget |

### 7.3 Experiments to run (`pdm evaluate`)

| ID | Experiment | Output |
|---|---|---|
| E1 | Model comparison (RF vs XGB vs AE vs LSTM) | Table + bar charts |
| E2 | Feature ablation: time-domain only vs FFT only vs both (XGBoost) | Table — supports manuscript's FFT claim |
| E3 | Noise robustness: sweep sensor noise ×0.5, ×1, ×2, ×4 | Line chart accuracy vs noise |
| E4 | Data-scarcity curve: train on 10%, 25%, 50%, 100% of episodes | Line chart — supports "low-data" claim |
| E5 | Zero-day: leave-one-fault-out autoencoder | Table of detection rate per unseen fault |
| E6 | Edge feasibility benchmark | Table with pass/fail |
| E7 *(optional)* | Federated vs centralized vs local | Table |
| E8 *(optional)* | Graph-aware feature on multi-robot cascade scenario | Table |

### 7.4 Results table template (auto-generated to `reports/results.md`)

| Model | Accuracy | Macro-F1 | Recall (anomaly) | RUL MAE | Latency p95 (ms) | Size (MB) | RAM Δ (MB) | Edge OK? | Literature ref. (context only) |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | | | | | | | | | 85% accuracy (Smith et al.) |
| XGBoost + FFT | | | | | | | | | 94% accuracy (Al-Fahad et al. / manuscript §3) |
| Autoencoder | | | | | | | | | zero-day detection (Wu & Zhao) |
| LSTM | | | | | | | | | 92% recall (Chen & Wang) |

The last column is text only, copied from the manuscript, with the disclaimer from README §3.

## 8. Dashboard design (Streamlit)

Multipage app, light/dark friendly, wide layout.

### Page 1 — **Live Simulation**
- **Sidebar controls:** scenario preset (Healthy / Bearing wear / Gear wear / Motor overheat / Cable fault / Random), target joint (1–6), degradation rate slider, sensor-noise multiplier, simulation speed (1×–50×), active models (checkboxes), **Start / Pause / Reset** buttons, "Inject fault now" button.
- **Top row:** 6 joint status cards (green / amber / red from alert state) with fused risk score.
- **Main charts (Plotly, live-updating):**
  1. Raw vibration window (last step) and its FFT spectrum for the selected joint.
  2. Time series of RMS, kurtosis, temperature, current.
  3. Anomaly score vs. threshold line.
  4. Predicted fault probabilities (stacked bar) and **RUL estimate vs. true RUL** (true shown as a dashed "digital twin ground truth" line).
- **Event log table:** alerts with time, joint, type, RUL.

### Page 2 — **Model Comparison**
- Loads `reports/results.csv`; shows the results table, confusion matrices (model selector), ROC curves, RUL scatter (pred vs. true).
- XGBoost feature-importance bar chart (top 15).

### Page 3 — **Edge Feasibility**
- Bar charts: latency, size, RAM per model, with the edge budget as horizontal threshold lines; green/red pass/fail badges. Slider to change the slowdown factor and budget.

### Page 4 — **Experiments**
- Plots for E2–E5 (and E7/E8 if enabled), each with a one-line takeaway text auto-generated from the numbers (template-based, no LLM).

### Page 5 — **About**
- Problem statement, link to manuscript abstract, the honesty note (README §3), list of cited method families.

### Visual style
- Colour semantics: green `#2e7d32` normal, amber `#f9a825` warning, red `#c62828` critical, blue `#1565c0` for predictions, grey dashed for ground truth.
- Font: system sans-serif. Every chart has axis labels with units and a title.
- Must remain responsive at 1280 px width; no horizontal scroll.

## 9. Default parameter summary

| Parameter | Default |
|---|---|
| seed | 42 |
| n_episodes | 400 |
| n_joints | 6 |
| sample_rate_hz | 1000 |
| window_len / hop | 1024 / 512 |
| episode length | 2000–6000 steps |
| failure threshold on `h` | 0.1 |
| degradation rate `r` | Uniform(2e-5, 8e-5) with `p = 1.5` (tune so failures occur within episode length for ≥ 70% of fault episodes) |
| RUL_cap | 300 steps |
| fault observable at damage | `d ≥ 0.15` |
| warn / critical | 0.5 / 0.8 |
| edge budget | 50 ms, 50 MB, 512 MB, slowdown ×5 |

If any default produces degenerate data (e.g. no failures), the generator must **print a
warning and auto-adjust `r`**, and record the adjusted value in `reports/run_info.json`.

## 10. Acceptance criteria

The build is accepted when **all** hold:

1. `python -m pdm.cli generate` creates a dataset in < 5 min on a laptop CPU, with ≥ 15% of every class present in the train split.
2. Same seed run twice ⇒ byte-identical `features.parquet` (or identical metrics to 6 decimals).
3. `python -m pdm.cli train` finishes in < 15 min CPU-only; no CUDA import errors on a machine without a GPU.
4. XGBoost fault-classification macro-F1 on the test split is **reported**, and XGBoost median inference latency per window is **lower** than LSTM's. (If not, the report must state so honestly.)
5. Autoencoder false-alarm rate on healthy test windows ≤ 2% at the default threshold.
6. Leave-one-fault-out experiment (E5) runs for all 4 faults and outputs a table.
7. Edge benchmark (E6) reports latency, size and RAM for all four models, with pass/fail flags.
8. Dashboard launches with `streamlit run app/dashboard.py`; pressing **Start** with a bearing fault shows alert progression `NORMAL → WARNING → CRITICAL` before the simulated failure, and displays RUL estimate and true RUL.
9. All five dashboard pages render without exceptions using the pre-generated artifacts.
10. `pytest -q` passes (tests listed in `ARCHITECTURE.md §9`).
11. The honesty note appears on the About page and in `reports/results.md`.

## 11. Implementation order for the agent

1. Project skeleton, `config.yaml`, `Config` loader, `requirements.txt`.
2. Simulator (`robot`, `faults`, `sensors`, `generator`) + `test_simulator`.
3. Features + `test_features`, `test_split`.
4. Models in this order: XGBoost → Random Forest → Autoencoder → LSTM → RUL wrapper. Run `test_models`.
5. Evaluation metrics and experiments E1–E6, `report.py`.
6. Streaming detector + alert state machine + `test_stream`.
7. Dashboard pages 1 → 5.
8. Optional E7 / E8.
9. End-to-end test, README quick-start verification, final cleanup.

After each step, run the relevant tests before proceeding.
