# ARCHITECTURE.md

System architecture for the **AI-Based Predictive Maintenance Simulator for Robotic Equipment**.
Read `README.md` first for scope. Algorithm and parameter details are in `DESIGN.md`.

---

## 1. Architectural overview

The system is a **layered, offline-first pipeline** with a **streaming mode** layered on top.
Each layer depends only on the layers above it in the diagram.

```
┌──────────────────────────────────────────────────────────────────────┐
│  PRESENTATION      Streamlit dashboard (app/)      CLI (pdm/cli.py)  │
└───────────────▲───────────────────────────▲──────────────────────────┘
                │                           │
┌───────────────┴───────────────────────────┴──────────────────────────┐
│  ORCHESTRATION     Experiment runner · Report builder · Config       │
└───────▲─────────────────▲──────────────────▲─────────────────▲───────┘
        │                 │                  │                 │
┌───────┴──────┐  ┌───────┴───────┐  ┌───────┴───────┐  ┌──────┴───────┐
│  STREAMING   │  │  MODEL LAYER  │  │  EDGE         │  │  FEDERATED   │
│  stream.py   │  │  RF · XGB ·   │  │  BENCHMARK    │  │  (optional)  │
│  detector.py │  │  AE · LSTM ·  │  │  latency/RAM/ │  │  FedAvg      │
│  alerts.py   │  │  RUL          │  │  size         │  │              │
└───────▲──────┘  └───────▲───────┘  └───────▲───────┘  └──────▲───────┘
        │                 │                  │                 │
┌───────┴─────────────────┴──────────────────┴─────────────────┴───────┐
│  FEATURE LAYER     windowing → time-domain + FFT features → scaling  │
└───────────────────────────────▲──────────────────────────────────────┘
                                │
┌───────────────────────────────┴──────────────────────────────────────┐
│  SIMULATION LAYER  RobotArm · Joint · Degradation · Faults · Sensors │
│                    Episode generator (digital-twin style)            │
└──────────────────────────────────────────────────────────────────────┘
```

## 2. Core design principles

1. **Single source of truth for configuration** — everything tunable lives in `config.yaml`, loaded once into a typed `Config` dataclass.
2. **Common model interface** — every model implements `BaseModel` (below), so the evaluator, streamer, edge benchmark and dashboard treat models interchangeably.
3. **Seeded determinism** — every random draw uses a `numpy.random.Generator` derived from `config.seed`. Same seed ⇒ identical dataset and metrics.
4. **No leakage** — train / validation / test splits are made **by episode** (by robot run), never by window.
5. **CPU only** — no CUDA calls anywhere.
6. **Separation of simulation truth from observations** — the simulator keeps hidden ground truth (true health, true fault, true RUL). Models only ever see sensor outputs.

## 3. Module specifications

### 3.1 Simulation layer — `pdm/simulator/`

| Module | Responsibility |
|---|---|
| `robot.py` | `RobotArm` (N joints, default 6). Each `Joint` has: health index `h ∈ [0,1]` (1 = new, 0 = failed), load profile, and a `step(dt)` that advances physical state. |
| `faults.py` | Fault injectors that modify joint dynamics as `h` decays (see DESIGN §3). |
| `sensors.py` | Converts joint state to observed signals + noise, drift and dropout. |
| `generator.py` | Runs many **episodes** (run-to-failure or run-to-censor) and writes a dataset with labels. |

Key classes:

```python
@dataclass
class EpisodeConfig:
    robot_id: int
    n_joints: int
    fault_type: Literal["none","bearing","gear_wear","motor_overheat","cable_fault"]
    fault_joint: int | None
    onset_step: int
    degradation_rate: float
    duration_steps: int
    seed: int

class RobotArm:
    def step(self) -> Observation: ...
    def ground_truth(self) -> GroundTruth: ...   # not visible to models

@dataclass
class Observation:
    t: int
    signals: dict[str, np.ndarray]    # high-rate raw window source
    slow: dict[str, float]            # temperature etc.

@dataclass
class GroundTruth:
    fault_type: str
    fault_joint: int | None
    health: np.ndarray                # per joint
    rul_steps: float | None
```

### 3.2 Feature layer — `pdm/features/`

- `windowing.py`: slice a stream into fixed windows (`window_len`, `hop`).
- `extract.py`: per-window features per channel:
  - Time-domain: mean, std, RMS, peak, peak-to-peak, skewness, kurtosis, crest factor.
  - Frequency-domain (FFT): dominant frequency, spectral centroid, spectral entropy, band energies (N bands).
  - Optional rolling trend features: slope of RMS over last K windows.
- Output: `pandas.DataFrame` (tabular models) **and** a 3-D `numpy` tensor `(n_windows, seq_len, n_features)` (sequence models).
- `StandardScaler` is fitted on **training healthy data only** and persisted with `joblib`.

### 3.3 Model layer — `pdm/models/`

```python
class BaseModel(Protocol):
    name: str
    task: Literal["classification","anomaly","rul"]
    def fit(self, train: Dataset, val: Dataset) -> None: ...
    def predict(self, X) -> np.ndarray: ...          # class ids / RUL / anomaly score
    def predict_proba(self, X) -> np.ndarray | None: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "BaseModel": ...
    def size_bytes(self) -> int: ...
```

| Model | Task | Input | Role in the study |
|---|---|---|---|
| `RandomForestModel` | classification (fault type) | tabular features | Supervised baseline (manuscript: Smith et al.) |
| `XGBoostModel` | classification + RUL regression | tabular time + FFT features | **Proposed lightweight model** (manuscript §3, §4) |
| `AutoencoderModel` | anomaly detection | healthy-only windows | Zero-day fault detection (manuscript §3) |
| `LSTMModel` | RUL / sequence anomaly | sequences of window features | Heavy deep-learning baseline (manuscript: Chen & Wang) |
| `RULEstimator` | RUL | feature trajectory | Wrapper that exposes RUL from XGBoost / LSTM uniformly |

### 3.4 Streaming layer — `pdm/streaming/`

- `stream.py`: wraps a `RobotArm` (or replays a stored episode) and yields one `Observation` per tick at a configurable simulated rate.
- `detector.py`: `LiveDetector` maintains a ring buffer, builds a feature vector once a window fills, calls the selected models, and emits `Prediction` objects.
- `alerts.py`: alert state machine with **hysteresis** (prevents flicker) — states `NORMAL → WARNING → CRITICAL`.

```
Observation ─► ring buffer ─► window full? ─► features ─► models ─► Prediction
                                                                   │
                                                  alert state machine ─► Alert
```

### 3.5 Edge benchmark — `pdm/edge/benchmark.py`

Measures, for each model on the **same** 1,000 test windows:

- median and p95 single-window inference latency (ms) — measured with `time.perf_counter`, batch size 1;
- serialized model size (KB/MB);
- peak resident memory delta (MB) via `psutil`;
- throughput (windows/sec).

Then compares against an **edge budget profile** (config), e.g. *Raspberry Pi 4 class*:
latency ≤ 50 ms, model size ≤ 50 MB, RAM ≤ 512 MB. Because the benchmark runs on the
developer's machine, apply a configurable **slowdown factor** (default 5×) to estimated
edge latency and state this clearly in the report.

### 3.6 Federated layer (optional) — `pdm/federated/`

- 3 simulated "factory cells", each with its own episode set (non-identical fault mixes).
- `fedavg.py`: local training rounds for a small model (logistic regression or small MLP), weighted parameter averaging, 10 rounds.
- Reports: centralized vs. federated vs. isolated-local accuracy.

### 3.7 Evaluation & reporting — `pdm/evaluation/`

- `metrics.py`: accuracy, precision, recall, F1 (macro), ROC-AUC (anomaly), confusion matrix, RUL MAE / RMSE, early-warning lead time, false-alarm rate per 1,000 windows.
- `report.py`: writes `reports/results.csv`, `reports/results.md`, and PNG figures.

## 4. Data flow

### 4.1 Offline (training)

```
config.yaml ─► generator ─► raw episodes (parquet) ─► windowing ─► features
                                                                    │
                          split by episode (train / val / test) ◄───┘
                                         │
                     fit scaler (healthy train only) ─► train models ─► models/*.joblib|.pt
                                         │
                                  evaluate on test ─► reports/
```

### 4.2 Online (dashboard simulation)

```
User picks scenario ─► RobotArm.step() per tick ─► LiveDetector ─► Prediction/Alert
                                                         │
                                         Streamlit session state ─► live charts
```

## 5. Data schemas

### 5.1 Raw episode table (`data/episodes.parquet`)

| column | type | note |
|---|---|---|
| episode_id | int | |
| robot_id | int | |
| t | int | step index |
| joint | int | 0..N-1 |
| vibration_x/y/z | float | high-rate summarized per step or stored as window arrays |
| temperature_c | float | |
| current_a | float | servo current |
| acoustic_band_energy | float | optional virtual acoustic proxy |
| label_fault | str | ground truth, **not a model input** |
| label_joint | int | ground truth |
| health | float | ground truth |
| rul_steps | float | ground truth (NaN for censored) |

### 5.2 Window feature table (`data/features.parquet`)

One row per `(episode_id, joint, window_idx)` with all engineered features + labels
(`y_fault`, `y_anomaly` ∈ {0,1}, `y_rul`).

### 5.3 Prediction object

```python
@dataclass
class Prediction:
    t: int
    joint: int
    fault_probs: dict[str, float]
    anomaly_score: float
    rul_estimate: float | None
    latency_ms: float
    model: str
```

## 6. Configuration (`config.yaml`) — top-level keys

```yaml
seed: 42
simulator: {n_joints: 6, sample_rate_hz: 1000, step_seconds: 1.0, ...}
dataset:   {n_episodes: 400, fault_mix: {...}, split: {train: 0.7, val: 0.15, test: 0.15}}
features:  {window_len: 1024, hop: 512, fft_bands: 8, trend_windows: 5}
models:    {random_forest: {...}, xgboost: {...}, autoencoder: {...}, lstm: {...}}
alerts:    {warn_threshold: 0.5, critical_threshold: 0.8, hysteresis: 3}
edge:      {profile: raspberry_pi_4, latency_ms_max: 50, size_mb_max: 50, ram_mb_max: 512, slowdown_factor: 5}
federated: {enabled: false, cells: 3, rounds: 10}
```

(Full defaults in `DESIGN.md`.)

## 7. Technology decisions and rationale

| Decision | Reason |
|---|---|
| Synthetic data generator instead of public datasets | Matches the manuscript's stated gap: no access to proprietary factory data |
| XGBoost as the proposed model | Manuscript §3/§4: best accuracy / latency / interpretability trade-off |
| Autoencoder on healthy data only | Manuscript §3: handles "zero-day" faults with no fault labels |
| LSTM kept small and CPU-only | Serves as the heavy baseline without requiring a GPU |
| Streamlit | Fast to build, runs locally, good for live-updating charts |
| Parquet for storage | Compact and fast; no database required |

## 8. Error handling & robustness

- Sensor dropout: simulator can emit NaN; `extract.py` must impute (forward-fill, then median) and log a warning.
- Missing model file: `LiveDetector` falls back to available models and shows a UI notice.
- All CLI commands validate that prerequisites exist (e.g. `train` requires `generate` output) and print an actionable message.

## 9. Testing strategy (`tests/`)

| Test | Asserts |
|---|---|
| `test_simulator.py` | Same seed ⇒ identical output; health decreases monotonically under a fault; healthy episodes keep `h≈1` |
| `test_features.py` | Window count formula; feature vector length fixed; no NaN after imputation |
| `test_split.py` | No `episode_id` appears in more than one split |
| `test_models.py` | Each model fits on a tiny dataset, `save`/`load` round-trips with identical predictions |
| `test_stream.py` | Alert state machine honors hysteresis; no flicker on noisy boundary input |
| `test_edge.py` | Benchmark returns all fields; budget pass/fail logic correct |
| `test_e2e.py` | `generate → train → evaluate` on a tiny config completes in < 2 minutes |

## 10. Extensibility

- New fault type: add an injector in `faults.py` and register it in `FAULT_REGISTRY`.
- New model: implement `BaseModel`, register in `MODEL_REGISTRY`; it automatically appears in evaluation, edge benchmark and dashboard.
