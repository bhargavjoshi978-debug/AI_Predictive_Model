# Implementation Plan: AI-Based Predictive Maintenance Simulator

Based on `DESIGN.md §11` and `README.md` / `ARCHITECTURE.md` specifications.

## Status Tracker

- [x] **Step 1: Environment & Skeleton**
  - [x] Virtual environment `.venv` with Python 3.11
  - [x] `requirements.txt` with CPU-only dependencies (pinned)
  - [x] `config.yaml` with all configurable values from `DESIGN.md §9`
  - [x] `pdm/config.py` typed `Config` dataclass and loader
  - [x] `pdm/cli.py` skeleton with `generate`, `train`, `evaluate` subcommands
  - [x] Package structure and initial tests setup

- [x] **Step 2: Simulation Layer (`pdm/simulator/`)**
  - [x] `pdm/simulator/robot.py`: `RobotArm`, `Joint`, state updates (load, speed, thermal, health degradation)
  - [x] `pdm/simulator/faults.py`: `FAULT_REGISTRY`, fault injectors (`bearing`, `gear_wear`, `motor_overheat`, `cable_fault`, `none`)
  - [x] `pdm/simulator/sensors.py`: `VirtualSensors`, vibration (x,y,z), current, temperature, acoustic proxy, noise, drift, dropouts, outliers
  - [x] `pdm/simulator/generator.py`: Episode generator, `EpisodeConfig`, ground truth tracking, parquet persistence
  - [x] Tests: `tests/test_simulator.py` (determinism, monotonic health under fault, healthy h≈1)

- [x] **Step 3: Feature Engineering (`pdm/features/`)**
  - [x] `pdm/features/windowing.py`: Slicing continuous signal streams into windows (`window_len`, `hop`)
  - [x] `pdm/features/extract.py`: Time-domain stats, FFT spectrum features, band energies, fault-specific frequencies, slow & trend features, NaN imputation
  - [x] Data splitting by episode (70/15/15) and healthy-only `StandardScaler` fitting
  - [x] Tests: `tests/test_features.py`, `tests/test_split.py` (no episode leakage, fixed feature size)

- [x] **Step 4: Model Layer (`pdm/models/`)**
  - [x] `pdm/models/base.py`: `BaseModel` protocol, `MODEL_REGISTRY`, serialization interfaces
  - [x] `pdm/models/xgboost_model.py`: Classifier + `XGBRegressor` for RUL, feature importance
  - [x] `pdm/models/random_forest.py`: `RandomForestClassifier` baseline
  - [x] `pdm/models/autoencoder.py`: PyTorch CPU dense autoencoder for zero-day anomaly detection
  - [x] `pdm/models/lstm.py`: PyTorch CPU small 2-layer LSTM for RUL regression & anomaly detection
  - [x] `pdm/models/rul.py`: `RULEstimator` wrapper providing unified RUL interface
  - [x] Tests: `tests/test_models.py` (fit, predict, serialize/load roundtrip, byte-identical/reproducible predictions)

- [x] **Step 5: Evaluation & Edge Benchmark (`pdm/evaluation/`, `pdm/edge/`)**
  - [x] `pdm/evaluation/metrics.py`: Classification metrics, anomaly ROC-AUC & false alarms, RUL MAE/RMSE/NASA asymmetric score, early warning lead times
  - [x] `pdm/edge/benchmark.py`: Single-window latency (p50, p95), serialized size, peak RAM delta via `psutil`, throughput, edge budget check (Raspberry Pi 4 class)
  - [x] `pdm/evaluation/report.py`: Experiments runner (E1-E6), `reports/results.csv`, `reports/results.md` (with honesty disclaimer and literature column), PNG figure generation
  - [x] Tests: `tests/test_edge.py`

- [x] **Step 6: Streaming Layer (`pdm/streaming/`)**
  - [x] `pdm/streaming/stream.py`: Real-time tick generator / episode replay
  - [x] `pdm/streaming/detector.py`: `LiveDetector` with ring buffer, feature extraction, multi-model inference, fused risk score calculation
  - [x] `pdm/streaming/alerts.py`: Hysteresis state machine (`NORMAL -> WARNING -> CRITICAL`)
  - [x] Tests: `tests/test_stream.py`

- [x] **Step 7: Streamlit Dashboard (`app/`)**
  - [x] `app/dashboard.py`: Navigation and main setup
  - [x] `app/pages/1_Live_Simulation.py`: Scenario controls, joint status cards, live vibration/FFT, time-series, RUL vs ground-truth, event logs
  - [x] `app/pages/2_Model_Comparison.py`: Results table, confusion matrices, ROC curves, RUL scatter, XGBoost feature importances
  - [x] `app/pages/3_Edge_Feasibility.py`: Latency/size/RAM charts with budget thresholds and pass/fail badges
  - [x] `app/pages/4_Experiments.py`: E2 (ablation), E3 (noise), E4 (data scarcity), E5 (zero-day LOFO)
  - [x] `app/pages/5_About.py`: Problem statement, literature context, mandatory honesty note

- [x] **Step 8: End-to-End Pipeline & Acceptance Verification**
  - [x] `tests/test_e2e.py`: Fast pipeline verification (< 2 min)
  - [x] Full run: `pdm.cli generate` -> `pdm.cli train` -> `pdm.cli evaluate`
  - [x] Verification of all 11 Acceptance Criteria from `DESIGN.md §10`
  - [x] Green `pytest -q` (17/17 passed)

- [x] **Step 9: Optional Enhancements**
  - [x] E7: Federated Learning demo (`pdm/federated/fedavg.py`)

---
## Assumptions & Interpretations Note
- Python 3.11 located at `C:\Users\bharg\.local\bin\python3.11.exe` will be used to initialize `.venv`.
- PyTorch will use CPU-only wheels (`--extra-index-url https://download.pytorch.org/whl/cpu`).
- Data splits are strictly by episode ID.
- Scaler is fitted exclusively on healthy training windows.
