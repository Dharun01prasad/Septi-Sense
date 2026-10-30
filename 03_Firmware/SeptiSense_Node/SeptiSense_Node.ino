/*
 * SeptiSense monitoring node - ESP32 DevKit V1 (Arduino-ESP32 core 2.x / 3.x)
 *
 *  AD8232 ECG ......... 250 Hz on core 1 (high-priority task): filtering, R-peaks, HR, HRV, lead-off
 *  MAX30102 ........... 25 Hz effective: PPG waveform, pulse rate, SpO2, finger detection
 *  MLX90614 ........... 1 Hz: skin and ambient temperature
 *  SSD1306 OLED ....... local display of vitals, link state and the backend's risk / alert
 *  Wi-Fi .............. every second one timestamped, sequence-numbered record goes to the laptop
 *                       (HTTP POST, acknowledged). If the link is down, records are kept in a
 *                       15-minute ring buffer and re-sent in order when it comes back.
 *
 * Libraries (Arduino Library Manager):
 *   "SparkFun MAX3010x Pulse and Proximity Sensor Library", "Adafruit MLX90614 Library",
 *   "Adafruit SSD1306" (+ "Adafruit GFX Library"), "ArduinoJson" (v7)
 *
 * Educational / research prototype - not a medical device.
 */
#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_MLX90614.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"
#include "config.h"
#include "dsp.h"

// ======================================================== shared state
static const int ECG_FS = 250;
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

// ECG: last second of filtered samples for streaming, R-peaks, RR history
static int16_t ecgChunk[2][ECG_FS];            // double buffer
static volatile int ecgWrite = 0, ecgIdx = 0;
static volatile bool ecgChunkReady = false;
static volatile int ecgReadyBuf = 1;
struct RPeak { uint16_t idx; int16_t amp; };
static RPeak peaksChunk[2][8];
static volatile uint8_t peaksN[2] = {0, 0};
static float rrHist[64];                       // RR intervals, seconds
static uint32_t rrTime[64];                    // millis() of each beat
static volatile int rrCount = 0, rrHead = 0;
static volatile uint32_t ecgLeadOffSamples = 0, ecgSatSamples = 0, ecgSamples = 0;

// PPG
static uint32_t irBuf[100], redBuf[100];
static int ppgFill = 0, ppgNew = 0;
static uint32_t ppgIrSec[32], ppgRedSec[32];
static int ppgSecN = 0;
static bool maxOk = false, mlxOk = false, oledOk = false;
static int32_t spo2 = -1, hrPpg = -1;
static int8_t spo2Valid = 0, hrPpgValid = 0;
static bool finger = false;

// Temperature
static float skinT = NAN, ambT = NAN;
static float tHist[10]; static int tN = 0;

// Backend feedback (shown on the OLED)
static float lastRisk = NAN;
static String lastAlert = "none";
static String lastMsg = "";

// Time sync: epoch_ms = millis() + timeOffset
static int64_t timeOffset = 0;
static bool timeSynced = false;
static uint32_t seq = 0;
static uint32_t bootId = 0;                    // lets the backend tell a reboot (seq restarts) from a gap

MAX30105 max3010x;
Adafruit_MLX90614 mlx;
Adafruit_SSD1306 oled(128, OLED_HEIGHT, &Wire, -1);

// ======================================================== records + offline buffer
struct Record {
  uint32_t seq; int64_t tMs; uint32_t uptimeMs;
  float hrEcg, hrPpg, spo2, skinT, ambT, rmssd, sdnn;
  float sqiEcg, sqiPpg, sqiTemp;
  bool leadsOff, finger, hrAgree;
};
static const int BACKLOG_N = 900;              // 15 min at 1 record/s
static Record backlog[BACKLOG_N];
static volatile int blHead = 0, blCount = 0;   // ring buffer of not-yet-acknowledged records
static QueueHandle_t liveQ;                    // newest record + its waveforms -> network task
struct LivePacket { Record r; int16_t ecg[ECG_FS]; RPeak peaks[8]; uint8_t nPeaks;
                    uint32_t ir[32], red[32]; uint8_t nPpg; };

static void backlogPush(const Record &r) {
  portENTER_CRITICAL(&mux);
  backlog[(blHead + blCount) % BACKLOG_N] = r;
  if (blCount < BACKLOG_N) blCount++; else blHead = (blHead + 1) % BACKLOG_N;   // overwrite oldest
  portEXIT_CRITICAL(&mux);
}

// ======================================================== ECG task (core 1, every 4 ms)
static void ecgTask(void *) {
  Biquad hp, notch1, lp;
  hp.highpass(ECG_FS, 0.5f);                   // removes baseline wander (breathing, movement)
  notch1.notch(ECG_FS, MAINS_HZ, 30.0f);       // mains hum
  lp.lowpass(ECG_FS, 40.0f);                   // muscle noise
  RPeakDetector det; det.begin();
  float recent[40] = {0}; int rIdx = 0;        // last 160 ms of filtered ECG, to find the R amplitude
  uint32_t lastBeatMs = 0;
  TickType_t wake = xTaskGetTickCount();
  const TickType_t period = pdMS_TO_TICKS(4);
  for (;;) {
    vTaskDelayUntil(&wake, period ? period : 1);
    bool off = digitalRead(PIN_LO_PLUS) || digitalRead(PIN_LO_MINUS);
    int raw = analogRead(PIN_ECG);
    ecgSamples++;
    if (off) ecgLeadOffSamples++;
    if (raw < 30 || raw > 4065) ecgSatSamples++;
    float y = off ? 0.0f : lp.step(notch1.step(hp.step((float)raw)));
    recent[rIdx] = y; rIdx = (rIdx + 1) % 40;
    bool beat = !off && det.step(y) && det.n > 2 * ECG_FS;

    int b = ecgWrite;
    if (ecgIdx < ECG_FS) ecgChunk[b][ecgIdx] = (int16_t)constrain(y, -32000, 32000);
    if (beat) {
      float amp = -1e9;
      for (int k = 0; k < 40; k++) if (recent[k] > amp) amp = recent[k];
      if (peaksN[b] < 8) peaksChunk[b][peaksN[b]++] = {(uint16_t)ecgIdx, (int16_t)constrain(amp, -32000, 32000)};
      uint32_t now = millis();
      if (lastBeatMs) {
        float rr = (now - lastBeatMs) / 1000.0f;
        if (rr > 0.30f && rr < 2.0f) {           // 30-200 bpm only
          portENTER_CRITICAL(&mux);
          rrHist[rrHead] = rr; rrTime[rrHead] = now;
          rrHead = (rrHead + 1) % 64; if (rrCount < 64) rrCount++;
          portEXIT_CRITICAL(&mux);
        }
      }
      lastBeatMs = now;
    }
    if (++ecgIdx >= ECG_FS) {                   // one second complete: swap buffers
      ecgReadyBuf = b;
      ecgWrite = 1 - b;
      peaksN[ecgWrite] = 0;
      ecgIdx = 0;
      ecgChunkReady = true;
    }
  }
}

// HR and HRV from the RR history (last 60 s, outliers removed)
static void ecgStats(float &hr, float &rmssd, float &sdnn, float &plausible) {
  float rr[64]; uint32_t ts[64]; int n;
  portENTER_CRITICAL(&mux);
  n = rrCount;
  for (int i = 0; i < n; i++) { int j = (rrHead - n + i + 64) % 64; rr[i] = rrHist[j]; ts[i] = rrTime[j]; }
  portEXIT_CRITICAL(&mux);
  uint32_t now = millis();
  float win[64]; int m = 0;
  for (int i = 0; i < n; i++) if (now - ts[i] < 60000) win[m++] = rr[i];
  hr = rmssd = sdnn = NAN; plausible = 0;
  if (m < 4) return;
  float last8[8]; int k = 0;
  for (int i = max(0, m - 8); i < m; i++) last8[k++] = win[i];
  float med = medianOf(last8, k);
  if (!(med > 0)) return;
  hr = 60.0f / med;
  float clean[64]; int c = 0;
  for (int i = 0; i < m; i++) if (fabsf(win[i] - med) < 0.2f * med) clean[c++] = win[i];
  plausible = (float)c / m;
  if (c >= 8) {
    float mean = 0; for (int i = 0; i < c; i++) mean += clean[i]; mean /= c;
    float ss = 0, sd = 0;
    for (int i = 0; i < c; i++) ss += (clean[i] - mean) * (clean[i] - mean);
    for (int i = 1; i < c; i++) sd += (clean[i] - clean[i - 1]) * (clean[i] - clean[i - 1]);
    sdnn = sqrtf(ss / (c - 1)) * 1000.0f;
    rmssd = sqrtf(sd / (c - 1)) * 1000.0f;
  }
}

// ======================================================== sensors on the I2C bus (loop, core 1)
static void initSensors() {
  Wire.begin(PIN_SDA, PIN_SCL, 100000);        // 100 kHz: the MLX90614 maximum
  oledOk = oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  if (oledOk) { oled.clearDisplay(); oled.setTextColor(SSD1306_WHITE); oled.setTextSize(1);
                oled.setCursor(0, 0); oled.println("SeptiSense"); oled.println("starting..."); oled.display(); }
  maxOk = max3010x.begin(Wire, I2C_SPEED_STANDARD);
  if (maxOk) max3010x.setup(60, 4, 2, 100, 411, 4096);   // 100 Hz / avg 4 = 25 Hz, red + IR
  mlxOk = mlx.begin(0x5A, &Wire);
  Serial.printf("OLED %s | MAX30102 %s | MLX90614 %s\n", oledOk ? "ok" : "MISSING",
                maxOk ? "ok" : "MISSING", mlxOk ? "ok" : "MISSING");
}

static void pollPpg() {
  if (!maxOk) return;
  max3010x.check();
  while (max3010x.available()) {
    uint32_t r = max3010x.getFIFORed(), ir = max3010x.getFIFOIR();
    max3010x.nextSample();
    if (ppgFill < 100) { redBuf[ppgFill] = r; irBuf[ppgFill] = ir; ppgFill++; }
    else {                                      // slide the 4 s window by one sample
      memmove(redBuf, redBuf + 1, 99 * sizeof(uint32_t)); memmove(irBuf, irBuf + 1, 99 * sizeof(uint32_t));
      redBuf[99] = r; irBuf[99] = ir;
    }
    if (ppgSecN < 32) { ppgIrSec[ppgSecN] = ir; ppgRedSec[ppgSecN] = r; ppgSecN++; }
    finger = ir > 50000;                         // a fingertip reflects much more IR light
    if (++ppgNew >= 25 && ppgFill == 100) {      // recompute once per second on the last 4 s
      ppgNew = 0;
      if (finger) maxim_heart_rate_and_oxygen_saturation(irBuf, 100, redBuf, &spo2, &spo2Valid, &hrPpg, &hrPpgValid);
      else { spo2Valid = hrPpgValid = 0; }
    }
  }
}

static void pollTemp() {
  if (!mlxOk) return;
  skinT = mlx.readObjectTempC();
  ambT = mlx.readAmbientTempC();
  tHist[tN % 10] = skinT; tN++;
}

static float tempSqi() {
  if (!mlxOk || isnan(skinT) || skinT < 25 || skinT > 42) return 0;
  int n = min(tN, 10); if (n < 3) return 0.5f;
  float mn = 99, mx = -99; for (int i = 0; i < n; i++) { mn = min(mn, tHist[i]); mx = max(mx, tHist[i]); }
  return (mx - mn) < 0.5f ? 1.0f : 0.5f;         // unstable reading = not settled / sensor moved
}

// ======================================================== OLED
static void drawOled(const Record &r, bool wifiUp) {
  if (!oledOk) return;
  bool alarm = lastAlert == "alarm";
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.printf("%s %s", wifiUp ? "WiFi" : "OFFLINE", wifiUp ? "" : String(blCount).c_str());
  if (!isnan(lastRisk)) { oled.setCursor(80, 0); oled.printf("R %2.0f%%", lastRisk * 100); }
  oled.setTextSize(OLED_HEIGHT == 64 ? 2 : 1);
  oled.setCursor(0, OLED_HEIGHT == 64 ? 12 : 9);
  if (r.leadsOff) oled.print("ECG off ");
  else if (!isnan(r.hrEcg)) oled.printf("HR %3.0f ", r.hrEcg);
  else oled.print("HR --- ");
  if (OLED_HEIGHT == 64) oled.setCursor(0, 30);
  if (r.finger && r.spo2 > 0) oled.printf("O2 %2.0f%%", r.spo2); else oled.print("O2 --");
  oled.setTextSize(1);
  oled.setCursor(0, OLED_HEIGHT == 64 ? 50 : 24);
  if (!isnan(r.skinT)) oled.printf("Skin %.1fC ", r.skinT);
  if (alarm || lastAlert == "warn") oled.print(alarm ? "ALERT" : "warn");
  if (alarm && (millis() / 500) % 2) oled.invertDisplay(true); else oled.invertDisplay(false);
  oled.display();
  digitalWrite(PIN_LED, alarm ? (millis() / 250) % 2 : (lastAlert == "warn"));
}

// ======================================================== network task (core 0)
static String url(const char *path) { return String("http://") + SERVER_HOST + ":" + SERVER_PORT + path; }

static void syncTime() {
  HTTPClient h;
  uint32_t t0 = millis();
  h.begin(url("/api/v1/time")); h.setTimeout(2000);
  if (h.GET() == 200) {
    JsonDocument d;
    if (!deserializeJson(d, h.getString())) {
      uint32_t rtt = millis() - t0;
      timeOffset = (int64_t)d["epoch_ms"].as<double>() + rtt / 2 - (int64_t)millis();
      timeSynced = true;
      Serial.printf("time synced with backend (rtt %u ms)\n", rtt);
    }
  }
  h.end();
}

static void addRecord(JsonObject o, const Record &r) {
  o["seq"] = r.seq;
  if (timeSynced) o["t_ms"] = (double)r.tMs; else o["t_ms"] = nullptr;
  o["uptime_ms"] = r.uptimeMs;
  JsonObject v = o["vitals"].to<JsonObject>();
  auto put = [&](const char *k, float x) { if (isnan(x) || x < 0) v[k] = nullptr; else v[k] = roundf(x * 10) / 10; };
  put("hr_ecg", r.hrEcg); put("hr_ppg", r.hrPpg); put("spo2", r.spo2);
  put("skin_temp", r.skinT); put("amb_temp", r.ambT); put("rmssd", r.rmssd); put("sdnn", r.sdnn);
  JsonObject q = o["sqi"].to<JsonObject>();
  q["ecg"] = roundf(r.sqiEcg * 100) / 100; q["ppg"] = roundf(r.sqiPpg * 100) / 100; q["temp"] = r.sqiTemp;
  q["leads_off"] = r.leadsOff; q["finger"] = r.finger; q["hr_agree"] = r.hrAgree;
}

static bool postJson(const String &body, JsonDocument &resp) {
  HTTPClient h;
  h.begin(url("/api/v1/telemetry"));
  h.addHeader("Content-Type", "application/json");
  h.setTimeout(3000);
  int code = h.POST(body);
  bool ok = code == 200 && !deserializeJson(resp, h.getString());
  h.end();
  return ok;
}

static void dropAcked(uint32_t ackSeq) {
  portENTER_CRITICAL(&mux);
  while (blCount && backlog[blHead].seq <= ackSeq) { blHead = (blHead + 1) % BACKLOG_N; blCount--; }
  portEXIT_CRITICAL(&mux);
}

static void readFeedback(JsonDocument &resp) {
  if (!resp["risk"].isNull()) lastRisk = resp["risk"].as<float>();
  if (!resp["alert"].isNull()) lastAlert = resp["alert"].as<String>();
  if (!resp["ack_seq"].isNull()) dropAcked(resp["ack_seq"].as<uint32_t>());
}

static void netTask(void *) {
  static LivePacket pkt;
  uint32_t lastSync = 0;
  for (;;) {
    if (WiFi.status() != WL_CONNECTED) {
      WiFi.begin(WIFI_SSID, WIFI_PASS);
      for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) vTaskDelay(pdMS_TO_TICKS(250));
      if (WiFi.status() == WL_CONNECTED) Serial.printf("Wi-Fi up, IP %s\n", WiFi.localIP().toString().c_str());
      else { vTaskDelay(pdMS_TO_TICKS(2000)); continue; }
    }
    if (!timeSynced || millis() - lastSync > 600000) { syncTime(); lastSync = millis(); }

    // 1) backlog first, oldest records in batches of 30 (vitals only)
    while (blCount > 1 && WiFi.status() == WL_CONNECTED) {
      JsonDocument d;
      d["device_id"] = DEVICE_ID; d["boot_id"] = bootId; d["fw"] = FW_VERSION; d["rssi"] = WiFi.RSSI(); d["backlog"] = (int)blCount;
      JsonArray arr = d["records"].to<JsonArray>();
      int n = min((int)blCount - 1, 30);
      for (int i = 0; i < n; i++) addRecord(arr.add<JsonObject>(), backlog[(blHead + i) % BACKLOG_N]);
      String body; serializeJson(d, body);
      JsonDocument resp;
      if (!postJson(body, resp)) break;
      readFeedback(resp);
    }
    // 2) the newest record with its ECG and PPG waveforms
    if (xQueueReceive(liveQ, &pkt, pdMS_TO_TICKS(1500)) == pdTRUE) {
      JsonDocument d;
      d["device_id"] = DEVICE_ID; d["boot_id"] = bootId; d["fw"] = FW_VERSION; d["rssi"] = WiFi.RSSI(); d["backlog"] = (int)blCount;
      JsonArray arr = d["records"].to<JsonArray>();
      JsonObject o = arr.add<JsonObject>();
      addRecord(o, pkt.r);
      JsonObject e = o["ecg"].to<JsonObject>();
      e["fs"] = ECG_FS;
      JsonArray es = e["samples"].to<JsonArray>();
      for (int i = 0; i < ECG_FS; i++) es.add(pkt.ecg[i]);
      JsonArray pk = e["rpeaks"].to<JsonArray>();
      for (int i = 0; i < pkt.nPeaks; i++) { JsonArray p = pk.add<JsonArray>(); p.add(pkt.peaks[i].idx); p.add(pkt.peaks[i].amp); }
      JsonObject p = o["ppg"].to<JsonObject>();
      p["fs"] = 25;
      JsonArray ir = p["ir"].to<JsonArray>(), red = p["red"].to<JsonArray>();
      for (int i = 0; i < pkt.nPpg; i++) { ir.add(pkt.ir[i]); red.add(pkt.red[i]); }
      String body; serializeJson(d, body);
      JsonDocument resp;
      if (postJson(body, resp)) readFeedback(resp);
      else lastMsg = "send failed";
    }
  }
}

// ======================================================== setup / loop
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\nSeptiSense node " FW_VERSION " - educational prototype, not a medical device");
  pinMode(PIN_LO_PLUS, INPUT); pinMode(PIN_LO_MINUS, INPUT); pinMode(PIN_LED, OUTPUT);
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_ECG, ADC_11db);
  initSensors();
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);                          // steadier latency for streaming
  liveQ = xQueueCreate(1, sizeof(LivePacket));   // length 1: always the newest second (xQueueOverwrite)
  bootId = esp_random();
  xTaskCreatePinnedToCore(ecgTask, "ecg", 4096, nullptr, 5, nullptr, 1);
  xTaskCreatePinnedToCore(netTask, "net", 12288, nullptr, 2, nullptr, 0);
}

void loop() {
  static uint32_t lastTemp = 0;
  static Record last = {};
  pollPpg();
  if (millis() - lastTemp >= 1000) { lastTemp = millis(); pollTemp(); }

  if (ecgChunkReady) {                           // once per second: assemble one record
    ecgChunkReady = false;
    int b = ecgReadyBuf;
    static LivePacket pkt;
    memcpy(pkt.ecg, ecgChunk[b], sizeof(pkt.ecg));
    pkt.nPeaks = peaksN[b];
    memcpy(pkt.peaks, peaksChunk[b], sizeof(RPeak) * pkt.nPeaks);
    pkt.nPpg = ppgSecN;
    memcpy(pkt.ir, ppgIrSec, sizeof(uint32_t) * ppgSecN);
    memcpy(pkt.red, ppgRedSec, sizeof(uint32_t) * ppgSecN);
    ppgSecN = 0;

    float hr, rmssd, sdnn, plaus;
    ecgStats(hr, rmssd, sdnn, plaus);
    uint32_t nS = ecgSamples, nOff = ecgLeadOffSamples, nSat = ecgSatSamples;
    ecgSamples = ecgLeadOffSamples = ecgSatSamples = 0;
    Record &r = pkt.r;
    r.seq = ++seq;
    r.uptimeMs = millis();
    r.tMs = timeSynced ? (int64_t)r.uptimeMs + timeOffset : 0;
    r.leadsOff = nOff > nS / 2;
    r.hrEcg = r.leadsOff ? NAN : hr;
    r.rmssd = r.leadsOff ? NAN : rmssd;
    r.sdnn = r.leadsOff ? NAN : sdnn;
    r.finger = finger;
    r.hrPpg = (finger && hrPpgValid && hrPpg > 30 && hrPpg < 220) ? (float)hrPpg : NAN;
    r.spo2 = (finger && spo2Valid && spo2 >= 70 && spo2 <= 100) ? (float)spo2 : NAN;
    r.skinT = skinT; r.ambT = ambT;
    // signal quality indices (0 = unusable, 1 = good)
    float sat = nS ? (float)nSat / nS : 1;
    r.sqiEcg = r.leadsOff ? 0 : constrain(plaus * (1 - 4 * sat), 0.0f, 1.0f);
    r.hrAgree = !isnan(r.hrEcg) && !isnan(r.hrPpg) && fabsf(r.hrEcg - r.hrPpg) <= 10;
    r.sqiPpg = !finger ? 0 : (hrPpgValid && spo2Valid ? (r.hrAgree || isnan(r.hrEcg) ? 1.0f : 0.6f) : 0.3f);
    r.sqiTemp = tempSqi();
    backlogPush(r);                              // every record is kept until the backend acknowledges it
    xQueueOverwrite(liveQ, &pkt);                 // the network task always sends the newest live packet
    last = r;
    Serial.printf("#%lu HRecg %.0f HRppg %.0f SpO2 %.0f skin %.1f | SQI e%.2f p%.2f t%.1f %s%s| risk %.2f %s\n",
                  (unsigned long)r.seq, r.hrEcg, r.hrPpg, r.spo2, r.skinT, r.sqiEcg, r.sqiPpg, r.sqiTemp,
                  r.leadsOff ? "LEADS-OFF " : "", r.finger ? "" : "NO-FINGER ", lastRisk, lastAlert.c_str());
    drawOled(last, WiFi.status() == WL_CONNECTED);
  }
  delay(2);
}
