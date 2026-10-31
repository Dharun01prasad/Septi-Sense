# SeptiSense — Project Report

**IoT-enabled predictive patient monitoring with a sepsis early-warning outlook**

> **Educational / research prototype. Not a medical device. Not clinically validated. It does not diagnose sepsis or any other condition.**
> Every number in this report comes from `02_ML/results/metrics.json`, produced by running `02_ML/train.py` on the real dataset.

---

## Contents

1. [Summary](#1-summary)
2. [System architecture](#2-system-architecture)
3. [How a reading travels through the system](#3-how-a-reading-travels-through-the-system)
4. [Hardware: the wearable node](#4-hardware-the-wearable-node)
5. [Firmware](#5-firmware)
6. [Machine-learning model](#6-machine-learning-model)
7. [Results (held-out evaluation)](#7-results-held-out-evaluation)
8. [Backend](#8-backend)
9. [Web dashboard](#9-web-dashboard)
10. [Phone alerts (SMS)](#10-phone-alerts-sms)
11. [What is novel](#11-what-is-novel)
12. [Limitations](#12-limitations)
13. [Substitutions](#13-substitutions)
14. [Facts, assumptions and what still needs verification](#14-facts-assumptions-and-what-still-needs-verification)
15. [How to run it](#15-how-to-run-it)
16. [Project folder map](#16-project-folder-map)
17. [Questions judges are likely to ask](#17-questions-judges-are-likely-to-ask)
18. [References](#18-references)

---

## 1. Summary

SeptiSense is a wearable monitor plus a laptop-based early-warning system:

- **Sensors on the body** (ECG, pulse/SpO₂, skin temperature) feed an **ESP32** that checks signal quality every second, timestamps each record and sends it over Wi-Fi — buffering up to 15 minutes if Wi-Fi drops.
- A **laptop backend** stores each patient's history, estimates breathing rate from the heart signals, builds hourly features and runs **LightGBM models** that estimate the chance of developing sepsis within **6, 12 and 24 hours**.
- A **web dashboard** explains the patient's condition in plain language, shows the outlook, alerts and history, lets users check previously recorded data, and offers a **Technical details** view with waveforms, signal quality, device statistics and model output.
- **SMS alerts** go to the patient's phone through an Android phone acting as an SMS gateway (works without internet).

**Model in one sentence:** tested on 20,000 patients from a hospital it never saw, the wearable-only model ranks sepsis-risk hours better than the NEWS2 score built from the same vitals (AUROC 0.685 vs 0.633) and warns 42 % of septic patients within 6 hours of onset — about twice what random alarms would catch (19 %) — at 0.35 false-alarm episodes per patient-day. A model that also has lab results reaches AUROC 0.772; that gap is the honest cost of using only what a wearable can measure.

---

## 2. System architecture

```
 Sensors on the body                      ESP32 node                          Laptop
 ───────────────────                      ──────────                          ──────
 AD8232 ECG (3 forearm stickers) ─┐
 MAX30102 pulse + SpO₂ (finger) ──┼──►  acquire · filter · R-peaks ·   ──HTTP POST, 1 record/s──►  FastAPI backend
 MLX90614 IR thermometer (wrist) ─┘     quality indices · timestamp ·      (acknowledged, re-sent      │ SQLite storage
                                        15-min ring buffer · OLED/LED       after Wi-Fi drops)        │ breathing rate (ECG+PPG)
                                                                                                      │ hourly features → LightGBM (6/12/24 h)
                                                                                                      │ alert engine → SMS gateway (Android phone)
                                                                                                      ▼
                                                                         Dashboard (browser, http://localhost:8000)
                                                                         WebSocket /ws (live) + REST /api/v1
```

| Layer | Technology | Files |
|---|---|---|
| Node | ESP32 DevKit V1, Arduino (ESP32 core 3.3.12), FreeRTOS tasks | `03_Firmware/SeptiSense_Node/` |
| Backend | Python, FastAPI + Uvicorn, SQLite, NumPy/pandas/SciPy, LightGBM | `04_Backend/` |
| ML | LightGBM, scikit-learn (isotonic calibration, metrics), matplotlib | `02_ML/` |
| Dashboard | One HTML file, plain JavaScript, no external libraries (works offline) | `05_Dashboard/index.html` |
| SMS | "SMS Gateway for Android" app in Local Server mode | `04_Backend/notify.py` |

---

## 3. How a reading travels through the system

Following one heartbeat from the skin to the patient's screen:

1. **Measure (node, every 4 ms).** The AD8232 output is sampled at 250 Hz on its own processor core, filtered, and an R-peak detector marks each heartbeat. The MAX30102 gives the finger pulse wave and SpO₂; the MLX90614 gives wrist skin temperature once a second.
2. **Check quality (node, every second).** Each one-second record carries quality scores for ECG, PPG and temperature, a lead-off flag, a finger-present flag, and whether the ECG and PPG heart rates agree.
3. **Send (node → laptop).** The record gets a sequence number, a boot ID and a timestamp and is POSTed to the backend, which replies with the highest sequence number it stored. Unacknowledged records stay in a 15-minute ring buffer and are re-sent in order after an outage.
4. **Store (backend).** The record is saved (duplicates are ignored), gaps and late records are counted, and the live waveform is pushed to the dashboard over the WebSocket.
5. **Breathing rate (backend, every 2 s).** From the last 60 s of R-peaks and PPG, three breathing surrogates are estimated and fused into one breathing rate with a quality score.
6. **Hourly features (backend, every ~10 s).** Seconds with poor signal quality are dropped. For each "model hour", the median of the good seconds becomes that hour's HR, SpO₂, temperature and breathing rate — the same hourly format the model was trained on.
7. **Predict.** The same feature code used in training turns the hourly rows into 37 features; three model ensembles give calibrated risks for 6, 12 and 24 h, with an uncertainty band and the top reasons.
8. **Alert.** The alert engine checks the risk (with a 2-hour persistence rule) and the last 30 s of vitals against limits. Alarm-level or repeating alerts trigger an SMS.
9. **Explain.** The dashboard turns this into plain language: "Your breathing rate is slightly raised", "Next 6 hours: Low".

---

## 4. Hardware: the wearable node

| Part | Measures | Connection |
|---|---|---|
| ESP32 DevKit V1 | — (microcontroller, Wi-Fi) | USB power from the laptop |
| AD8232 + 3 gel electrodes | ECG (forearm placement: RA right forearm, LA left forearm, RL reference) | OUTPUT → GPIO34 (ADC1), LO+ → GPIO35, LO− → GPIO32, 3.3 V, GND; SDN tied to 3.3 V |
| MAX30102 | PPG (pulse wave), pulse rate, SpO₂ | I²C 0x57 |
| MLX90614 (GY-906) | Wrist skin temperature + ambient temperature | I²C 0x5A |
| SSD1306 OLED | Local display | I²C 0x3C |

All three I²C modules share SDA = GPIO21 and SCL = GPIO22 at 100 kHz and run from the ESP32's 3V3 pin. The ECG uses an ADC1 pin because ADC2 stops working while Wi-Fi is on. Wiring diagram: `01_Hardware/wiring_esp32_devkit_v1.png`; placement: `01_Hardware/wearable_placement.png`. `01_Hardware/hardware_test/hardware_test.ino` scans the I²C bus (expects 0x3C, 0x57, 0x5A) and streams the ECG for a first check.

---

## 5. Firmware

`03_Firmware/SeptiSense_Node/SeptiSense_Node.ino` (+ `dsp.h`, `config.h`). Two FreeRTOS tasks: the ECG task on core 1 (priority 5) and the network task on core 0.

**ECG (250 Hz)**
- Filters: 0.5 Hz high-pass (baseline wander), 50 Hz notch (mains hum, India), 40 Hz low-pass (muscle noise).
- R-peak detection: simplified Pan-Tompkins — 5–15 Hz band-pass → derivative → square → 150 ms moving integration → adaptive threshold with a 250 ms refractory period.
- Heart rate = 60 / median of the last 8 RR intervals; HRV as RMSSD and SDNN.
- Lead-off: the AD8232 LO+/LO− pins; a second counts as "leads off" when more than half its samples are flagged.
- ECG quality = share of physiologically plausible RR intervals × (1 − 4 × fraction of saturated samples), clipped to 0–1.

**PPG (MAX30102, ~25 Hz effective)**
- Finger detected when the IR level is above 50,000 (a fingertip reflects much more IR light).
- SpO₂ and pulse rate from the sensor maker's (Maxim) algorithm on a 100-sample buffer; values outside 30–220 bpm or 70–100 % are discarded.
- PPG quality: 1.0 if valid and agreeing with the ECG heart rate (or no ECG), 0.6 if valid but disagreeing, 0.3 if not valid, 0 without a finger.
- "HR agree" = ECG and PPG heart rates within 10 bpm.

**Temperature (MLX90614, 1 Hz)**
- Quality 1.0 if the last 10 readings vary less than 0.5 °C, 0.5 if unstable or still settling, 0 outside 25–42 °C. (The backend applies a stricter on-skin check; see §8.)

**Telemetry**
- One record per second: sequence number, boot ID (random per power-up, so the backend can tell a reboot from a gap), uptime and epoch time (synchronised from the backend's `/api/v1/time`, corrected by half the round-trip time), vitals and quality indices. The newest record also carries the one-second ECG and PPG waveforms.
- Every record goes into a 900-record (15 min) ring buffer and stays there until the backend acknowledges it (`ack_seq`). After an outage the backlog is sent oldest-first in batches of 30.

**Local feedback.** The OLED shows HR, SpO₂, temperature, Wi-Fi state and backlog size, plus the risk and alert level returned by the backend. The on-board LED is steady on a warning and blinks on an alarm.

---

## 6. Machine-learning model

### 6.1 Dataset
**PhysioNet/CinC Challenge 2019** (open access, CC BY 4.0; Reyna et al., *Crit Care Med* 2020). Hourly ICU records from two hospital systems:

| | Patients | Patient-hours |
|---|---|---|
| Hospital A (`training_setA`) | 20,336 | 790,215 |
| Hospital B (`training_setB`) | 20,000 | 761,995 |

### 6.2 Prediction target and horizons
- The dataset's `SepsisLabel` is 1 from **6 hours before** clinical Sepsis-3 onset onward. So the **6 h model** answers: *"Will this patient meet Sepsis-3 within the next 6 hours (or already has)?"*
- **12 h and 24 h models** use the same onset (first label hour + 6) with a longer window: label = 1 from 12 h / 24 h before onset onward.
- Unit of prediction: one patient-hour. Each hour uses only that hour and earlier hours.

### 6.3 Split (no patient appears in two sets)
- **Train:** 16,269 hospital-A patients. **Validation:** 4,067 hospital-A patients (20 %, stratified by "ever septic"). Used for early stopping, calibration and choosing the alarm threshold.
- **Test:** all 20,000 **hospital-B** patients, used **once**, never for training or tuning — an external validation on a different hospital system.
- Test prevalence: 1.41 % of hours, 5.71 % of patients.

### 6.4 Features (37, wearable model)
For each of **HR, SpO₂, temperature, breathing rate**: current value (last known), 6-hour mean, SD, min, max, change over 1 h, change over 6 h, and **deviation from the patient's own past median** (personal baseline) — 8 × 4 = 32. Plus **age, sex**, and three rule scores: partial **NEWS2**, a qSOFA breathing-rate criterion (≥ 22/min) and a SIRS-type count. The same function (`02_ML/features.py`) is used in training and in the live backend.

### 6.5 Leakage and imbalance precautions
- **Causal features only:** hour *t* uses hours ≤ *t*; the personal baseline is shifted by one hour.
- **Patient-level splits;** the test hospital is never seen before the final evaluation.
- **ICULOS and HospAdmTime excluded** (time-in-hospital proxies).
- **No measurement-frequency features** in the deployed model: in ICU data, "how often was this measured" reflects clinician concern, which a continuous wearable doesn't have.
- **Imbalance:** moderate positive re-weighting (`scale_pos_weight = √(negatives/positives)`); results reported as AUROC **and AUPRC**, with patient-level alarm statistics; the threshold is chosen on validation data, not on the test set.

### 6.6 Training
- LightGBM (binary; learning rate 0.05, 63 leaves, ≥ 300 samples per leaf, 80 % feature/bagging fractions, L2 = 1), early stopping on validation AUPRC (100 rounds).
- **Ensembles:** 5 models (seeds) for 6 h, 3 each for 12 h and 24 h. The mean is the risk; ±1 SD across models is the uncertainty band.
- **Calibration:** isotonic regression on hospital-A validation, so a risk of 5 % means "about 5 in 100 such hours".
- **Alarm threshold:** the risk at **90 % hour-level specificity on hospital-A validation** — 3.72 % (6 h), 6.41 % (12 h), 9.77 % (24 h).
- **Reasons:** per-prediction feature contributions (LightGBM `pred_contrib`, i.e. tree SHAP), averaged over the ensemble: the top 3 raising and top 2 lowering factors.
- Full run (load, features, 5 model sets, evaluation, plots): 523 s on the team's laptop.

### 6.7 Comparison models (ablation, 6 h, same patients)
- **Wearable + BP cuff:** adds systolic, mean and diastolic BP, shock index and BP-based scores (64 features).
- **Full ICU:** all vitals + 26 lab tests + measurement-recency features (140 features). Comparison only — our device can't supply this.
- **NEWS2 baselines:** the partial NEWS2 score (HR, RR, SpO₂, temperature) and NEWS2 + SBP, used directly as risk scores (no training).

---

## 7. Results (held-out evaluation)

All on **hospital B (20,000 patients), never used for training or tuning.** 95 % CIs: patient-level bootstrap (200 resamples).

### 7.1 Outlook models (wearable inputs only)

| Horizon | AUROC (95 % CI) | AUPRC (95 % CI) | Base rate (hours) | Sensitivity | Specificity | PPV | Brier (vs base rate only) |
|---|---|---|---|---|---|---|---|
| **6 h** | **0.685** (0.672–0.698) | 0.043 (0.038–0.049) | 1.41 % | 32.3 % | 89.6 % | 4.3 % | 0.01383 (0.01395) |
| **12 h** | 0.673 (0.659–0.688) | 0.053 (0.047–0.060) | 2.10 % | 25.5 % | 92.2 % | 6.6 % | 0.02038 (0.02051) |
| **24 h** | 0.663 (0.650–0.679) | 0.073 (0.066–0.081) | 3.25 % | 22.9 % | 92.6 % | 9.4 % | 0.03128 (0.03141) |

Sensitivity, specificity and PPV are at each model's alarm threshold, per hour.

### 7.2 Ablation (6 h): what the wearable gives up

| Model | Features | AUROC (95 % CI) | AUPRC (95 % CI) |
|---|---|---|---|
| **Wearable only (deployed)** | 37 | **0.685** (0.672–0.698) | **0.043** (0.038–0.049) |
| Wearable + BP cuff | 64 | 0.662 (0.649–0.673) | 0.036 (0.033–0.041) |
| Full ICU (vitals + labs) | 140 | 0.772 (0.759–0.784) | 0.054 (0.049–0.060) |
| NEWS2 (HR, RR, SpO₂, T) — no training | — | 0.633 | 0.025 |
| NEWS2 + SBP — no training | — | 0.643 | 0.027 |

### 7.3 Alerting (patient level)
Alarm = risk ≥ threshold for **2 consecutive hours**. "Warned in time" = an alarm hour inside the horizon window before onset. "Chance" = the same rule applied to the alarm sequences of randomly drawn non-septic patients over the same hours (keeps their real alarm rate and timing). 919 septic patients were eligible; 223 already met the label at their first hour (no early warning possible) and are excluded.

| Model | Warned in time | Chance | Ratio | Median lead (IQR) | False-alarm episodes / patient-day |
|---|---|---|---|---|---|
| **Wearable 6 h** | **42.4 %** (390/919) | 18.6 % | 2.3× | 6 h (3.25–6) | 0.35 |
| Wearable 12 h | 42.1 % (387/919) | 20.8 % | 2.0× | 9 h (4–12) | 0.28 |
| Wearable 24 h | 45.9 % (422/919) | 27.3 % | 1.7× | 16 h (7–22.75) | 0.26 |
| Wearable + BP (6 h) | 37.2 % | 17.7 % | 2.1× | 5 h (3–6) | 0.38 |
| Full ICU (6 h) | 39.5 % | 27.6 % | 1.4× | 6 h (4–6) | 0.11 |

Lead time is measured inside the horizon window, so it can't exceed the horizon. For comparison, 64.6 % of septic patients had *some* alarm at any time before onset, but that number includes alarms days earlier and is **not** evidence of predicting the episode.

**Trade-off at other 6 h thresholds** (from `operating_curve`):

| Threshold | Warned in time | Chance | False alarms / patient-day |
|---|---|---|---|
| 2.57 % | 71.5 % | 47.2 % | 0.88 |
| 3.14 % | 53.2 % | 26.8 % | 0.52 |
| **3.72 % (chosen)** | **42.4 %** | **19.0 %** | **0.35** |
| 4.26 % | 35.0 % | 13.3 % | 0.25 |
| 5.43 % | 22.3 % | 6.6 % | 0.12 |
| 8.60 % | 10.4 % | 2.8 % | 0.05 |

(The chance column here uses 5 random draws per patient instead of 20, so the chosen-threshold row differs slightly from §7.3.)

### 7.4 Calibration (6 h, hospital B)
| Predicted risk (bin mean) | Observed frequency | Hours |
|---|---|---|
| 2.0 % | 1.4 % | 757,780 |
| 11.9 % | 10.2 % | 3,482 |
| 21.3 % | 18.7 % | 733 |

The model is slightly over-confident on the new hospital, and almost all hours sit in the lowest bin.

### 7.5 Most used features (6 h, LightGBM gain)
Age, temperature, 6 h mean breathing rate, 6 h max temperature, 6 h min breathing rate, temperature vs. personal baseline, 6 h max HR, 6 h min temperature, HR vs. personal baseline, 6 h min HR, breathing-rate variability, 6 h max breathing rate, 6 h min SpO₂, SIRS-type count, 6 h mean SpO₂.

### 7.6 What the results mean
- **It works, modestly.** The wearable model beats NEWS2 on the same vitals and warns about twice as many septic patients in time as chance, at every horizon.
- **It is weak on its own.** AUROC 0.66–0.69; most alarms are false (PPV 4–9 % per hour). The percentages carry little information beyond the base rate (Brier only ~0.5–1 % better), which is why patients see **Low / Raised / High**, not raw percentages.
- **Further ahead is harder:** AUROC and the gain over chance fall from 6 h to 24 h.
- **Labs matter:** the full-ICU model reaches 0.772. Adding BP did not help on the external hospital (0.662) — likely because BP behaves differently between the two hospitals; note also that the BP comparison model is a single model while the deployed one is a 5-model ensemble.
- **Accuracy is not a useful metric here:** only 1.41 % of hours are sepsis hours, so always answering "no sepsis" would be 98.6 % "accurate". The 6 h model's plain accuracy is 88.8 % (balanced accuracy 60.9 %).
- **Evaluation plots:** `02_ML/results/roc_pr.png` (ROC and precision-recall for all models and NEWS2) and `02_ML/results/calibration_alerts_features.png` (calibration, alert trade-off with the chance line, feature importance).

---

## 8. Backend

`04_Backend/app.py` (FastAPI). Start with `python 04_Backend/app.py`; it prints the LAN IP to put into the firmware's `config.h`.

### 8.1 Accounts and stored history
- **Sign up** with name, phone number, password, age and sex (age and sex are model inputs). Passwords are salted and hashed (PBKDF2-SHA256, 200,000 iterations). Sign-in sets an HTTP-only session cookie valid for 7 days.
- **Start monitoring** links the wearable to the signed-in patient. Everything the device sends from then on belongs to that patient's **monitoring session**; stopping ends it. History keeps every session and every checked recording, with predictions, hourly values and alerts.
- Storage: SQLite (`04_Backend/septisense.db`) — telemetry, breathing-rate estimates, hourly rows, predictions, alerts, users, sessions, uploads, SMS log and settings.

### 8.2 Telemetry handling
- Each record is stored once (`device_id + boot_id + seq` is the key; repeats count as duplicates). The reply carries `ack_seq`, the current risk and alert level (shown on the OLED).
- Device status: online (a record within 10 s), last seen, Wi-Fi RSSI, records received, **gaps** (missing sequence numbers), **loss %**, **late records** (arrived > 5 s after measurement, i.e. re-sent from the buffer), duplicates, backlog and outage periods.

### 8.3 Breathing rate without a breathing sensor (`respiration.py`)
Three surrogates from the last 60 s: **EDR** (R-peak amplitude changes as the chest moves), **RSA** (heart rate speeds up on inhalation) and **RIIV** (the PPG baseline rises and falls with breathing). Each is band-passed to 0.1–0.6 Hz (6–36 breaths/min); the dominant frequency gives an estimate and the share of power at that peak gives its quality. Estimates with quality ≥ 0.25 are fused by a quality-weighted median; the final quality is lowered when the surrogates disagree by more than 3 breaths/min. Breathing rates with quality < 0.3 are not used.

### 8.4 Data quality gates the AI
- For each **model hour**: HR uses seconds with ECG quality ≥ 0.5 (else PPG quality ≥ 0.5); SpO₂ uses PPG quality ≥ 0.5; temperature uses temperature quality ≥ 0.5 **and** a plausible on-skin value (30–40.5 °C — anything colder means the sensor is reading air, not skin). A vital counts for the hour only if at least 30 % of its seconds are good; otherwise it is "missing", exactly as an unmeasured value in the ICU data.
- Each prediction is tagged **good / partial / poor** by how many of the four vitals were missing.
- **Temperature** shown to the patient is an *estimate*: wrist skin + offset (default 3.5 °C), which can be calibrated against a thermometer.

### 8.5 Predictions
For the active monitoring session, about every 10 s: hourly rows → features → 6/12/24 h calibrated risks, uncertainty bands and reasons. **Demo time compression** (`seconds_per_model_hour`, default 3600 = real time) lets one model hour pass in e.g. 20 s for a live demo; the dashboard shows a **DEMO TIME** badge whenever it isn't 3600.

### 8.6 Alert engine (checked every 2 s)

| Alert | Raised when | Severity |
|---|---|---|
| Sepsis risk | 6 h risk ≥ threshold for 2 consecutive model hours | **Alarm** |
| Sepsis risk rising | 6 h risk ≥ 0.6 × threshold | Warning |
| Heart rate | 30-s median outside 45–120 bpm | Warning; **alarm** if ≤ 40 or ≥ 131 (NEWS2 sub-score 3) |
| Blood oxygen | below 92 % | Warning; **alarm** if ≤ 91 % |
| Breathing rate | above 24/min | Warning; **alarm** if ≥ 25 or ≤ 8 |
| Temperature (estimate) | outside 36.0–38.3 °C | Warning only (a wrist estimate is not reliable enough to alarm) |
| ECG sticker off / No finger | > 80 % of the last 30 s | Warning |
| Sensor offline | no record for 10 s | Warning |

All limits are adjustable in Settings. An alert ends by itself when its condition clears. **Acknowledge** records that someone has seen it and stops the red alarm banner; it doesn't close the alert.

### 8.7 Checking a recording (uploads)
CSV or PhysioNet `.psv`, one row per hour, oldest first; columns HR, O2Sat/SpO2, Temp, Resp (any subset; header names are flexible), optional Age, Gender, SBP, SepsisLabel. The rows go through the same features and models as live data. If `SepsisLabel` is present, the recorded onset is marked. The four held-out hospital-B patients chosen by `train.py` (3 septic patients warned within 6 h of onset — picked at random from the 40 with the longest lead — and 1 non-septic patient) are offered as example files. They illustrate the pipeline; they are not a measure of performance.

### 8.8 API (all under `/api/v1`)

| Endpoint | Purpose |
|---|---|
| `GET /time`, `POST /telemetry` | Device side (no sign-in): time sync, telemetry upload + acknowledgement |
| `POST /auth/signup`, `/auth/login`, `/auth/logout`; `GET/PUT /auth/me` | Accounts and profile |
| `POST /monitor/start`, `/monitor/stop` | Link / unlink the wearable to the signed-in patient |
| `GET /me/now` | Home screen: plain-language status, current vitals, 6/12/24 h outlook |
| `GET /me/history` | The patient's sessions and uploads |
| `GET /patients/{key}/hourly`, `/patients/{key}/series` | Hourly predictions; per-second trends |
| `POST /uploads`, `DELETE /uploads/{id}` | Check a recording |
| `GET /replay/cases`, `GET /replay/{pid}.csv` | Example held-out patients |
| `GET /alerts`, `POST /alerts/{id}/ack` | Alerts (with SMS status) |
| `GET/PUT /config`, `POST /temp/calibrate`, `POST /sms/test` | Settings, temperature calibration, test SMS |
| `GET /device`, `GET /model`, `GET /export/{key}.csv` | Device status, model info + evaluation, CSV export |
| `WS /ws` | Live push: `live` (waveforms + vitals), `prediction`, `alert`, `notification`, `status` |

---

## 9. Web dashboard

`05_Dashboard/index.html`, served at `http://localhost:8000`. A single file with no external dependencies, so it works with no internet. Light and dark themes.

| Page | What the patient sees | Extra with **Technical details** on |
|---|---|---|
| **Sign in / Create account** | Phone + password; sign-up asks name, phone, age, sex | — |
| **Home** | Greeting; Start/Stop monitoring; a status card ("Mostly steady. One reading to watch.") with tips; four vital cards (heart rate, blood oxygen, temperature, breathing rate) labelled Normal / Slightly raised or low / Very high or low, with a one-line explanation and the usual range; the **6 / 12 / 24 h outlook** as Low / Raised / High; a "how are you feeling" check-in (not saved) | Live ECG and PPG waveforms; raw numbers; exact risk %, band, threshold, reasons; signal-quality bars; device and connectivity panel; 30-minute trend charts |
| **History** | Every session and recording; summary (overall result, data quality, average HR); outlook at the end; hour-by-hour Low/Raised/High grid with "first raised reading" and recorded-onset markers; CSV download | 6 h risk chart with threshold line |
| **Check a recording** | Upload a file (or pick an example), enter age/sex, get the same result view | Same as History |
| **Alerts** | Active alerts (alarm / warning) with Acknowledge and SMS status; alert history | — |
| **Settings** | Profile; "Text me when there is an alarm"; test SMS; temperature/breathing limits, demo time, temperature offset | All alarm thresholds and limits; calibration against a thermometer |

Always visible: "Research prototype — not a medical device", emergency number 112, the sensor-connection status, a **SIMULATED DATA** badge whenever the device is the simulator, and a **DEMO TIME** badge whenever time is compressed. "How it works" / "About this prototype" opens a short description of the components and limitations.

Normal ranges on the Home page are the **NEWS2** bands (Royal College of Physicians, 2017): heart rate 51–90, breathing 12–20, SpO₂ ≥ 96 %, temperature 36.1–38.0 °C.

---

## 10. Phone alerts (SMS)

- **How it's sent:** an Android phone with a SIM runs "SMS Gateway for Android" in **Local Server** mode on the same Wi-Fi/hotspot as the laptop. The backend asks it over HTTP to send a normal SMS from its SIM. **No internet needed.** Gateway address and credentials are stored once for the whole system.
- **When it's sent** (to the phone number given at sign-up, if the patient has "Text me" on):
  1. an **alarm**-level alert is raised (or a warning escalates to an alarm), or
  2. the **same alert comes back 3 times within 30 minutes**, even if it's only a warning.
- **Limits:** at most one SMS per patient **per alert type** every 15 minutes. All values are adjustable.
- Every attempt (sent, failed, or held back by the rate limit) is logged and shown next to the alert.
- Message wording always says it is a research prototype, not a diagnosis, and gives the emergency number.

---

## 11. What is novel

1. **Sensor-matched model.** The model uses only what the wearable measures, and the ablation shows exactly what that costs (vs BP cuff, full ICU, NEWS2).
2. **External validation.** Trained on one hospital system, tested once on another it never saw.
3. **Honest alert evaluation.** "Warned in time" is measured inside the horizon window and compared with a **random-alarm baseline** that keeps the real alarm rate and timing.
4. **Leakage controls.** Causal features, patient-level splits, no time-in-hospital or measurement-frequency features.
5. **Data quality gates the AI.** Per-second quality indices, lead-off and finger detection, ECG/PPG agreement and an on-skin temperature check decide which seconds feed the model; every prediction carries a quality tag.
6. **Breathing rate with no breathing sensor.** Three surrogates (EDR, RSA, RIIV) fused with quality scoring.
7. **Personal baseline features, ensemble uncertainty band and per-prediction reasons.**
8. **Multi-horizon outlook (6 / 12 / 24 h)** in plain language for non-technical users, with technical details one switch away.
9. **Store-and-forward telemetry.** 15-minute buffer, sequence numbers, boot ID, acknowledgements, gap and late-record accounting.
10. **Offline SMS alerts** through a phone on the same hotspot, with repeat detection and rate limiting.
11. **Same pipeline for live and recorded data.** Uploaded recordings get the identical analysis.

---

## 12. Limitations

- **Not clinically validated; not a diagnostic system.** Retrospective evaluation on ICU data only.
- **Domain shift:** the model learned from hourly ICU charting (ICU patients, nurse-measured vitals). The device is a continuous wearable, possibly on people at home. Its performance on wearable data is unknown.
- **Weak predictor on its own:** AUROC 0.66–0.69; most alarms are false (hour-level PPV 4–9 %).
- **Temperature** is wrist skin temperature plus an offset — an estimate, not core temperature. The IR sensor must touch the skin; readings below 30 °C are rejected as "not on skin".
- **SpO₂** comes from the sensor maker's algorithm and is not calibrated against a reference oximeter.
- **Forearm ECG** is noisier than chest leads and sensitive to movement.
- **Breathing rate** has been checked only on synthetic signals (the earlier sandbox test recovered 10/15/20/26 breaths/min within 0.6), not against a reference on real people.
- **Not measured:** blood pressure, lab tests, consciousness level — so the model sees much less than a clinician.
- **Age** is the single most-used feature: part of the risk reflects "older ICU patients develop sepsis more often".
- **Prototype security:** the device endpoint has no authentication; any signed-in user can change system settings; HTTP (not HTTPS) on the local network.
- **SMS** depends on the gateway phone being on, charged and on the same network.
- Simulated data ("SIMULATED DATA") and time compression ("DEMO TIME") are for demonstration only and are not experimental validation.

---

## 13. Substitutions

| Brief suggests | We used | Why |
|---|---|---|
| ESP8266 NodeMCU or ESP32-S3 | **ESP32 DevKit V1** | Available hardware; dual-core (dedicated ECG core), ADC1 pins usable with Wi-Fi on, enough RAM for the 15-min buffer. A pin map for the ESP32-S3 is in `01_Hardware/wiring_esp32_s3.png` |
| Core body temperature | **Wrist skin temperature (MLX90614) + calibratable offset** | Non-invasive sensor available; always labelled an estimate |
| Breathing sensor | **Breathing rate derived from ECG + PPG** | No breathing sensor available |
| Clinical dataset | **PhysioNet/CinC 2019** (open access) | Legal, public, hourly vitals that match our sensors, two hospital systems for external validation |

---

## 14. Facts, assumptions and what still needs verification

**Verified (code was run)**
- Dataset download and full training/evaluation on the real data (all §7 numbers).
- Firmware and hardware test sketch compile with the real libraries (arduino-cli, ESP32 core 3.3.12): firmware uses 83 % of flash, 34 % of RAM.
- Backend + dashboard used on Windows by the team: sign-up/sign-in, monitoring sessions, alerts and test SMS through the Android gateway (all visible in the database log).

**Assumptions**
- Wrist skin + 3.5 °C approximates core temperature until calibrated with a thermometer.
- Hourly medians of clean wearable seconds are comparable to hourly ICU charted values.
- ICU sepsis patterns transfer to wearable users well enough for a demonstration.

**Needs verification**
- End-to-end run on the real hardware (after the ESP32 repair): R-peak threshold, MAX30102 finger threshold, OLED size, Wi-Fi buffering in practice.
- Breathing-rate accuracy on real people against a counted reference.
- SpO₂ against a reference pulse oximeter; temperature against a thermometer.
- The most recent changes (dark theme, on-skin temperature check, repeated-alert SMS) after their final test run.

---

## 15. How to run it

**Setup (once)**
```
pip install -r requirements.txt
cd 02_ML
python download_data.py      # re-run until it prints DONE (~20,336 + 20,000 files)
python train.py data         # ~10 min; writes models/, results/, replay/
cd ..
```

**Run**
```
python 04_Backend/app.py                 # open http://localhost:8000, create an account, press Start monitoring
python 04_Backend/simulator.py --deteriorate 120      # optional: synthetic device instead of hardware
```
Simulator options: `--outage 60:75` (Wi-Fi drop), `--leadoff a:b`, `--nofinger a:b`, `--deteriorate N` (drifts toward a septic picture over N seconds), `--seconds N`.

**Hardware**
1. Wire as in `01_Hardware/wiring_esp32_devkit_v1.png`; flash `01_Hardware/hardware_test/hardware_test.ino` and check the I²C scan shows 0x3C, 0x57, 0x5A.
2. In `03_Firmware/SeptiSense_Node/config.h` set `WIFI_SSID`, `WIFI_PASS` and `SERVER_HOST` (the IP the backend prints). Laptop and ESP32 must be on the same network; allow Python through the Windows firewall on port 8000.
3. Flash `SeptiSense_Node.ino` (FQBN `esp32:esp32:esp32`) and watch the Serial Monitor at 115200 baud.

**SMS:** install "SMS Gateway for Android", turn on Local Server, start the service, and save its address/username/password once in the backend settings; then use "Send test SMS" in Settings.

---

## 16. Project folder map

```
SeptiSense/
├── 00_Brief/                     participant brief (.docx + .txt)
├── 01_Hardware/                  wiring diagrams, placement pictures, hardware_test.ino, diagram scripts
├── 02_ML/
│   ├── download_data.py          fetches PhysioNet 2019 sets A and B (not stored in git)
│   ├── features.py               shared feature builder (training + backend)
│   ├── train.py                  training, calibration, held-out evaluation, plots, example patients
│   ├── make_test_data.py         fake data for pipeline tests only (never used for results)
│   ├── models/                   trained ensembles (6/12/24 h) + meta.json
│   ├── results/                  metrics.json, roc_pr.png, calibration_alerts_features.png
│   └── replay/                   4 held-out hospital-B example patients
├── 03_Firmware/SeptiSense_Node/  ESP32 firmware (config.h, dsp.h, .ino)
├── 04_Backend/
│   ├── app.py                    FastAPI service: accounts, telemetry, inference, alerts, API
│   ├── inference.py              loads the models, returns risks + reasons
│   ├── respiration.py            breathing rate from ECG + PPG
│   ├── explain.py                plain-language status and outlook (NEWS2 ranges)
│   ├── notify.py                 SMS through the Android gateway
│   └── simulator.py              synthetic device (sends exactly the firmware's JSON)
├── 05_Dashboard/index.html       the web dashboard
├── requirements.txt
└── REPORT.md                     this report
```

---

## 17. Questions judges are likely to ask

**"What's your accuracy?"** — 88.8 %, but that's the wrong number: always saying "no sepsis" scores 98.6 % because only 1.4 % of hours are sepsis hours. The meaningful ones: AUROC 0.685 on an unseen hospital, better than NEWS2 (0.633); 42 % of septic patients warned within 6 h vs 19 % by chance.

**"How do you know it isn't overfitting or leaking?"** — Test hospital never seen until the single final evaluation; patient-level splits; causal features; no time-in-hospital or measurement-frequency features; threshold and calibration chosen on hospital-A validation only.

**"Why does adding blood pressure not help?"** — On the external hospital it didn't transfer (0.662). Likely BP is recorded and behaves differently between the two hospital systems; also that comparison model is a single model vs a 5-model ensemble.

**"Is the 6/12/24-hour prediction real?"** — Each horizon is a separately trained and separately evaluated model (§7.1). It estimates the *chance* of meeting Sepsis-3 within that window; it doesn't predict what will happen. Longer horizons are measurably weaker.

**"Your device measures skin temperature, the dataset has core temperature. Isn't that a mismatch?"** — Yes: we add a calibratable offset, label it an estimate, reject off-skin readings, and list the domain shift as the main limitation.

**"How do you measure breathing rate without a sensor?"** — Three surrogates from the ECG and PPG, fused with quality scores (§8.3). Validated only on synthetic signals so far — stated openly.

**"What happens when Wi-Fi drops?"** — The node keeps up to 15 minutes in a ring buffer and re-sends in order; the backend acknowledges by sequence number and counts gaps, late records and duplicates (shown in the device panel).

**"Why should a patient trust it?"** — They shouldn't treat it as a diagnosis, and the app says so everywhere. It's built for awareness: plain-language readings, Low/Raised/High instead of false precision, and an SMS when something is far off or keeps coming back.

---

## 18. References

- Reyna MA, Josef CS, Jeter R, et al. *Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.* Critical Care Medicine 2020;48(2):210–217. Data: physionet.org/content/challenge-2019 (CC BY 4.0).
- Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 2000;101(23):e215–e220.
- Royal College of Physicians. *National Early Warning Score (NEWS) 2.* London: RCP, 2017.
- Singer M, et al. *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).* JAMA 2016;315(8):801–810.
- Pan J, Tompkins WJ. *A Real-Time QRS Detection Algorithm.* IEEE Trans Biomed Eng 1985;32(3):230–236.
- Ke G, et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- Lundberg SM, et al. *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence 2020;2:56–67.
