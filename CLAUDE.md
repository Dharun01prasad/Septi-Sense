# SeptiSense — project handoff for Claude Code

Read this whole file before doing anything. The team presents this project at a **live evaluation tomorrow (31 Oct 2026)**. The owner (Mukunth) has other work and wants you to drive the remaining software work end to end on this **Windows** laptop, checking in only when you need him (hardware, a decision, or something only a human can do).

## 1. The brief (full text in `00_Brief/`)

Build an educational/research **IoT patient-monitoring and sepsis early-warning prototype**:

1. **Embedded node:** sensors → ESP32. It acquires measurements, validates them, timestamps them and sends telemetry over Wi-Fi.
2. **ML model:** trained on a legal clinical dataset, with a clearly defined target and horizon, and a **held-out evaluation** that guards against class imbalance and leakage.
3. **Laptop backend:** receives telemetry, stores history, runs inference, manages predictions and alerts, and offers an API.
4. **Web dashboard:** shows measurements, trends, predictions with their horizon, configurable alerts, alert history, and connectivity and data-quality status. It must include a short description of the components and their limitations.

It must **never be presented as a clinically validated diagnostic system**. A Raspberry Pi is not required; the laptop is the central system.

**Hardware the team has:** ESP32 DevKit V1, AD8232 ECG module with 3 leads and gel stickers, MAX30102 (pulse/SpO₂), MLX90614 IR thermometer (GY-906), a small SSD1306 OLED, and one breadboard. Mukunth builds the hardware himself, following `01_Hardware/`.

## 2. Ground rules (from the owner — follow strictly)

- **Never invent results, numbers, citations or completion status.** Every number in docs or the dashboard must come from code that was actually run. If something is untested, say "untested".
- **Simulation is not experimental validation.** Synthetic or simulator data is always labelled as such.
- Keep facts, assumptions and things needing verification clearly separated in the docs.
- Build a minimum working version first, then add to it. Verify each step before moving on.
- Explain things so Mukunth can **defend them in front of judges**. He must understand every part.

## 3. Architecture (already designed — keep it)

```
AD8232 / MAX30102 / MLX90614 -> ESP32 (03_Firmware) --HTTP POST 1 record/s, acknowledged--> FastAPI backend (04_Backend)
                                                                                                |-- SQLite storage
                                                                                                |-- breathing rate from ECG+PPG
                                                                                                |-- hourly features -> LightGBM ensemble (02_ML)
                                                                                                |-- alert engine
backend --WebSocket /ws + REST /api/v1--> dashboard (05_Dashboard/index.html, served at http://localhost:8000)
```

### Novel points (the pitch — keep them true)

1. **Sensor-matched model.** It uses only what the wearable measures: HR, SpO₂, temperature, and breathing rate derived from ECG and PPG, plus age and sex. The comparison against a BP-cuff model, a full-ICU model (labs included) and NEWS2 shows the honest cost of that choice.
2. **External validation.** Train and validate on hospital A (PhysioNet 2019 set A, patient-level split). Test once on hospital B (set B), which the model never sees.
3. **Leakage controls.** All features are causal. ICULOS and HospAdmTime are excluded. The deployable model has no measurement-frequency or missingness features (they encode clinician behaviour, which a wearable doesn't have).
4. **Data quality gates the AI.** Every second carries signal-quality indices (ECG, PPG, temperature), lead-off and finger detection, and an ECG/PPG heart-rate agreement check. Bad seconds are excluded from the hourly aggregates, and each prediction is tagged good / partial / poor.
5. **Breathing rate with no breathing sensor.** It fuses three surrogates (EDR, RSA, RIIV) with quality scoring.
6. **Personal baseline features** (deviation from the patient's own past median), a 5-model ensemble that gives an uncertainty band, and **per-prediction reasons** (LightGBM `pred_contrib`, i.e. SHAP values for trees).
7. **Store-and-forward telemetry.** A 15-minute ring buffer on the ESP32, sequence numbers, boot_id, acknowledgements, and gap and late-record accounting.
8. **Replay of held-out real patients** through the same pipeline, so judges can watch a recorded sepsis case being flagged before onset.

## 4. What exists and its status

| Part | Files | Status |
|---|---|---|
| Hardware guide | `01_Hardware/` (wiring diagrams, breadboard hub layout, forearm ECG + wrist band + finger clip, `hardware_test/hardware_test.ino`) | Done. Mukunth is building it. |
| Dataset download | `02_ML/download_data.py` | Written, **not yet run on this laptop**. Fetches PhysioNet 2019 sets A and B into `02_ML/data/` (about 40k .psv files) with 24 reused HTTPS connections. Re-runnable. |
| Features | `02_ML/features.py` | Done. A shared superset builder; `FEATURE_SETS` = `wearable` / `wearable_bp` / `full_icu`. Used by both training and the backend. |
| Training + evaluation | `02_ML/train.py` | Written and tested **only on fake data** (`make_test_data.py`). **No real results exist yet.** Run with `python train.py data`. Outputs: `models/` (5 boosters + `meta.json` with calibration and threshold), `results/metrics.json`, `results/roc_pr.png`, `results/calibration_alerts_features.png`, and `replay/*.json` (3 septic + 1 non-septic held-out hospital-B patients). |
| Firmware | `03_Firmware/SeptiSense_Node/` | Written. Passes a host syntax check with real ArduinoJson and stubbed drivers. **Never run on hardware.** Libraries: SparkFun MAX3010x, Adafruit MLX90614, Adafruit SSD1306 + GFX, ArduinoJson v7. Edit `config.h` for Wi-Fi SSID, password and the laptop IP. |
| Backend | `04_Backend/app.py`, `inference.py`, `respiration.py` | Written and tested in a Linux sandbox with `simulator.py` and a fake-data model. Verified: ingest and ack; a 10 s outage buffered and re-sent with 0 records lost; offline alert raised then cleared; predictions with reasons and quality tags; replay; NaN-safe JSON. The breathing-rate estimator recovered 10/15/20/26 br/min within 0.6 on **synthetic** signals only. |
| Simulator | `04_Backend/simulator.py` | Done. A synthetic ESP32 sending exactly the firmware's JSON. Options: `--outage a:b`, `--leadoff a:b`, `--nofinger a:b`, `--deteriorate N`. |
| **Dashboard** | `05_Dashboard/index.html` | **NOT BUILT. This is the main remaining task.** |
| Docs / report / README | — | **Not written.** |

Backend notes:
- Start it with `python app.py` from `04_Backend`. It prints the LAN IP to put in the ESP32 `config.h`.
- With no trained model it still runs, with risk set to None.
- Config can be read and set at `/api/v1/config`.
- `seconds_per_model_hour` is the clearly labelled demo time compression; 3600 is real time.
- Temperature: the MLX90614 reads **wrist skin** temperature. "Core estimate" = skin + `temp_offset_c` (default 3.5). It can be calibrated against a thermometer via `POST /api/v1/temp/calibrate {"reference_c": 36.8}`. Always label it an *estimate*.

API (all in `app.py`):
- `GET /api/v1/time`
- `POST /api/v1/telemetry`
- `GET /api/v1/patients`
- `POST /api/v1/patients/live/admit {name, age, gender, new_session}`
- `GET /api/v1/patients/{pid}/series?minutes=30` (per-second vitals + resp, live only)
- `GET /api/v1/patients/{pid}/hourly` (hourly rows + predictions with reasons/quality, threshold)
- `GET /api/v1/alerts?patient=`, `POST /api/v1/alerts/{id}/ack`
- `GET/PUT /api/v1/config`
- `POST /api/v1/temp/calibrate`
- `GET /api/v1/device` (online, received, gaps, late, duplicates, loss %, outages, rr, 60 s quality)
- `GET /api/v1/model` (meta + metrics)
- `GET /api/v1/replay/cases`, `POST /api/v1/replay/{pid}/start?speed=2`, `POST /api/v1/replay/{pid}/stop`
- `GET /api/v1/export/{pid}.csv`
- `WS /ws` message types:
  - `live`: ecg samples at 250 Hz, ppg ir at 25 Hz, vitals, sqi, rr, temp_core_est
  - `prediction`: patient, risk, risk_lo/hi, raising/lowering reasons, hour, threshold; for replays also vitals, onset_hour, label_start_hour
  - `alert`: event raised/cleared/acked
  - `status`: the device dict

## 5. Remaining tasks, in order (each has a check)

1. **Environment.** Run `pip install -r requirements.txt` (Python 3.10+).
   - Check: `python -c "import lightgbm, fastapi"`.
2. **Dataset.** In `02_ML`, run `python download_data.py` and re-run it until it prints DONE.
   - Check: about 20,336 files in `data/training_setA` and about 20,000 in `data/training_setB`.
3. **Train.** Run `python train.py data`. On a 2-core machine, features alone take about 10 minutes.
   - Check: `models/meta.json`, `results/metrics.json`, both PNGs and `replay/*.json` exist.
   - Read the metrics critically. If the wearable model's AUROC is near 0.5, or the alarm stats look odd, investigate before moving on.
   - Watch for these possible issues:
     - `boot_ci` speed on 760k rows (reduce `n` if slow)
     - the `alarm_eval` loop speed (fine)
     - Windows multiprocessing: `Pool` needs the `if __name__ == '__main__'` guard, which already exists
     - memory (about 2–3 GB)
4. **Backend smoke test.** Run `python 04_Backend/app.py`, then in a second terminal `python 04_Backend/simulator.py --seconds 120 --outage 30:40 --deteriorate 100`, with `seconds_per_model_hour` set to 20 via PUT `/api/v1/config`.
   - Check: `/api/v1/device` shows gaps 0; hourly predictions appear; alerts raise and clear.
   - Then start a replay and confirm its predictions.
5. **Build the dashboard** at `05_Dashboard/index.html` as a single file, vanilla JS, **no CDN dependencies** (the venue may have no internet).
   - The backend already serves it at `/` and `/static`.
   - **Add a static mount for `02_ML/results`** so the evaluation PNGs show.

   Required views:
   - **Monitor:**
     - an ECG sweep strip (250 Hz) and a PPG strip
     - vital tiles: HR with ECG/PPG and agreement, SpO₂, temperature (skin + "est. core"), breathing rate with quality, HRV RMSSD
     - per-sensor quality bars
     - trends over the last 30 min
   - **Risk panel:**
     - calibrated risk %, the ensemble band, the threshold and "next 6 h"
     - top raising and lowering reasons
     - a quality tag
     - a risk-history chart with the threshold line; for replays, also the label-start and sepsis-onset markers
   - **Alerts:** active alerts with acknowledge, plus the history.
   - **Device panel:** online/offline, last seen, RSSI, received, gaps / loss %, late (buffered) records, outages.
   - **Replay:** a case list with start/stop. Show clearly that it is a recorded, held-out hospital-B patient, hourly data with no waveform.
   - **Model & evaluation:** data from `/api/v1/model`:
     - the dataset, target, split and horizon
     - an ablation table (wearable / +BP / full ICU / NEWS2) with AUROC and AUPRC plus 95 % CIs
     - alerting stats: detection before onset, median lead time, false alarms per patient-day
     - the PNGs
   - **Settings:**
     - alarm threshold, persistence and vital limits
     - demo time compression, with a visible **"DEMO TIME"** badge whenever it isn't 3600
     - temperature calibration
     - admit patient (age/sex)
   - A permanent **"Educational research prototype — not a medical device"** label, plus a **"SIMULATED DATA"** badge whenever `device_id == 'simulator'`.

   Design: a clinical-monitor look, dark background, conventional vital colours (ECG/HR green, SpO₂ cyan, resp yellow, temp orange). It must stay readable on a projector.

   - Check: with the simulator running, every panel updates live. Then stop the simulator and the device panel shows offline within 10 s.
6. **Firmware with Mukunth.** Once his hardware passes `hardware_test.ino`:
   - set `config.h` to the Wi-Fi and the laptop IP (the laptop and ESP32 on the same hotspot)
   - install the 4 libraries
   - flash, and watch the Serial Monitor
   - Fix whatever real-hardware issues appear: the R-peak threshold, MAX30102 finger threshold, OLED height 32 vs 64.
   - On Windows, allow Python through the firewall on port 8000.
7. **Docs.**
   - `README.md`: setup and run steps, architecture, folder map.
   - `REPORT.md`: dataset source and licence (PhysioNet/CinC 2019, CC BY 4.0, cite Reyna et al., Crit Care Med 2020), target and horizon, split, leakage and imbalance precautions, **real metrics from `results/metrics.json`**, ablation, alerting trade-off, limitations, substitutions.
   - Limitations to state:
     - wrist skin temperature, not core
     - SpO₂ from the Maxim algorithm, uncalibrated
     - forearm ECG noise
     - breathing rate validated only on synthetic signals
     - the dataset is ICU hourly data while the device is a wearable (domain shift)
     - not clinically validated
8. **Demo script + Q&A notes for Mukunth:** a 5-minute flow (live sensors → pull a sticker → Wi-Fi off/on → replay a septic patient → evaluation page), plus likely judge questions with honest answers.
9. **Package:** a clean folder, no `data/`, no `.db`. Push to GitHub if Mukunth asks.

## 6. Commands cheat-sheet (Windows, from the project root)

```
pip install -r requirements.txt
cd 02_ML && python download_data.py && python train.py data && cd ..
python 04_Backend/app.py                                  # dashboard: http://localhost:8000
python 04_Backend/simulator.py --seconds 300 --outage 60:75
```
