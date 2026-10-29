# AI-Based Predictive Maintenance Simulator for Robotic Equipment

A software-only simulation of the framework proposed in the manuscript
**"AI-Based Predictive Maintenance System for Robotic Equipment"**
(Rana, Joshi, Kataria — SAKEC, Mumbai).

> **Instructions for the coding agent (Antigravity):** Read `README.md` first, then
> `ARCHITECTURE.md` (what to build and how the pieces connect), then `DESIGN.md`
> (algorithms, parameters, UI, acceptance criteria). Build exactly what is described.
> Do not invent features beyond the "Scope" section. Where a value is marked
> **[configurable]**, expose it in `config.yaml`.

---

## 1. Problem the simulation addresses

The manuscript concludes that deploying real-time predictive maintenance (PdM) for
industrial robots is blocked by three constraints:

| Constraint | How this simulation removes it |
|---|---|
| No access to proprietary / run-to-failure factory data | A physics-inspired **synthetic robot-arm simulator** (digital-twin style) generates healthy and degrading telemetry, with ground-truth fault labels |
| Expensive sensing hardware (acoustic, high-end IoT) | Sensors are **virtual** (vibration, temperature, servo current, optional acoustic-band energy) |
| High GPU / compute cost | Core models are **lightweight CPU models** (XGBoost, Random Forest, small autoencoder), with a **latency and memory budget** check that emulates an edge device (Raspberry Pi class) |

**Research question the simulation answers:** *Can a lightweight model detect robot
faults and estimate Remaining Useful Life (RUL) in real time, on CPU-only hardware, using
only synthetic data, at accuracy comparable to heavier deep-learning approaches?*

## 2. Scope

### In scope (must build)
1. **Synthetic robot-arm telemetry generator** with healthy operation and 4 fault modes.
2. **Feature engineering** (time-domain stats + FFT features).
3. **Model zoo** reflecting the manuscript's methods:
   - Random Forest (supervised fault classification) — literature ref: 85% accuracy
   - XGBoost + FFT features (fault classification) — literature ref: 94% accuracy
   - Autoencoder trained on healthy data only (zero-day anomaly detection)
   - LSTM (small, CPU) as the "heavy deep-learning" baseline for RUL/anomaly — literature ref: 92% recall
4. **RUL estimation** per component (regression on degradation trajectories).
5. **Real-time streaming simulation**: telemetry arrives tick by tick; models infer live; alerts are raised.
6. **Edge-feasibility benchmark**: per-model inference latency, model size, peak RAM, with pass/fail against an edge budget.
7. **Interactive dashboard** (Streamlit) to run scenarios, watch live streams, and compare models.
8. **Reproducible experiment runner** that outputs a metrics table (CSV + Markdown) suitable for pasting into the paper's Results section.

### Optional (build only after all of the above works)
- **Federated learning demo**: 3 simulated robot cells train locally and aggregate with FedAvg; compare against centralized training.
- **Multi-robot cascade (graph) demo**: a fault in robot *i* raises degradation rate of neighbours; simple graph-aware feature (neighbour mean anomaly score) improves detection.

### Out of scope
Real hardware integration, real datasets, cloud deployment, authentication, databases.

## 3. Important honesty note (put in the app's "About" page)

The figures 85% / 92% / 90% / 94% / 20% come from the **literature surveyed in the
manuscript**. The simulation produces its **own** numbers on **synthetic** data. They are
shown side by side for context only and must **never be labelled as reproductions** of
the cited papers. The UI must display: *"Results are from synthetic data and are not
validated on real robots."*

## 4. Tech stack

- Python 3.11
- `numpy`, `scipy`, `pandas`
- `scikit-learn` (Random Forest, preprocessing, metrics)
- `xgboost`
- `torch` (CPU build) for the autoencoder and LSTM
- `streamlit` + `plotly` for the dashboard
- `pyyaml`, `pytest`, `psutil`, `joblib`, `tabulate`

No GPU required. Everything must run on a laptop CPU.

## 5. Quick start (target behaviour)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pdm.cli generate      # build synthetic dataset -> data/
python -m pdm.cli train         # train all models        -> models/
python -m pdm.cli evaluate      # metrics + edge benchmark -> reports/
streamlit run app/dashboard.py  # launch the interactive simulator
pytest -q                       # run tests
```

## 6. Repository layout

```
pdm-sim/
├── README.md
├── ARCHITECTURE.md
├── DESIGN.md
├── config.yaml
├── requirements.txt
├── pdm/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── simulator/
│   │   ├── robot.py          # RobotArm, Joint, degradation models
│   │   ├── faults.py         # fault injectors
│   │   ├── sensors.py        # virtual sensors + noise
│   │   └── generator.py      # dataset builder (run-to-failure episodes)
│   ├── features/
│   │   ├── windowing.py
│   │   └── extract.py        # time + FFT features
│   ├── models/
│   │   ├── base.py           # common interface
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── autoencoder.py
│   │   ├── lstm.py
│   │   └── rul.py
│   ├── streaming/
│   │   ├── stream.py         # tick-by-tick telemetry feed
│   │   ├── detector.py       # live inference + alert logic
│   │   └── alerts.py
│   ├── edge/
│   │   └── benchmark.py      # latency / size / RAM vs. budget
│   ├── federated/            # optional
│   │   └── fedavg.py
│   └── evaluation/
│       ├── metrics.py
│       └── report.py
├── app/
│   ├── dashboard.py
│   └── pages/
├── data/        # generated
├── models/      # generated
├── reports/     # generated
└── tests/
```

## 7. Definition of done

The project is complete when **all** acceptance criteria in `DESIGN.md §10` pass and
`pytest -q` is green.

## 8. Citation

Rana, M., Joshi, B., Kataria, R. *AI-Based Predictive Maintenance System for Robotic
Equipment.* Shah & Anchor Kutchhi Engineering College, Mumbai.
