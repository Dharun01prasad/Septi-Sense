// SeptiSense - hardware power-up test (no extra libraries needed)
// 1) Scans the I2C bus: expect 0x3C (OLED), 0x57 (MAX30102), 0x5A (MLX90614)
// 2) Streams the ECG to the Serial Plotter and reports when a sticker is off
// Board: ESP32 DevKit V1. For an ESP32-S3 change the five pin numbers below.
#include <Wire.h>

#define PIN_SDA      21   // S3: 8
#define PIN_SCL      22   // S3: 9
#define PIN_ECG      34   // S3: 4
#define PIN_LO_PLUS  35   // S3: 5
#define PIN_LO_MINUS 32   // S3: 6

void scanI2C() {
  struct { uint8_t addr; const char *name; } expected[] = {
    {0x3C, "OLED SSD1306"}, {0x57, "MAX30102"}, {0x5A, "MLX90614"}};
  Serial.println("\nScanning I2C bus...");
  for (auto &e : expected) {
    Wire.beginTransmission(e.addr);
    bool ok = Wire.endTransmission() == 0;
    Serial.printf("  0x%02X  %-13s %s\n", e.addr, e.name, ok ? "FOUND" : "MISSING - check its 4 wires");
  }
  int others = 0;
  for (uint8_t a = 1; a < 127; a++) {
    if (a == 0x3C || a == 0x57 || a == 0x5A) continue;
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) { Serial.printf("  0x%02X  (unexpected device)\n", a); others++; }
  }
  if (others) Serial.println("  An unexpected address usually means your module is a variant - tell us which.");
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Wire.begin(PIN_SDA, PIN_SCL, 100000);
  pinMode(PIN_LO_PLUS, INPUT);
  pinMode(PIN_LO_MINUS, INPUT);
  analogReadResolution(12);
  scanI2C();
  Serial.println("\nNow open Tools > Serial Plotter to see the ECG.\n");
  delay(3000);
}

void loop() {
  static uint32_t next = 0, lastMsg = 0;
  if (micros() < next) return;
  next = micros() + 4000;                         // 250 samples per second
  bool off = digitalRead(PIN_LO_PLUS) || digitalRead(PIN_LO_MINUS);
  if (off) {
    if (millis() - lastMsg > 1000) { Serial.println("ecg:0 leads_off:1"); lastMsg = millis(); }
    return;
  }
  Serial.printf("ecg:%d\n", analogRead(PIN_ECG));
}
