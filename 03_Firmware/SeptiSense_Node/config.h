// ===================== SeptiSense node - settings you edit =====================
#pragma once

// Wi-Fi: the ESP32 and the laptop must be on the SAME network (a phone hotspot works well)
#define WIFI_SSID      "YOUR_WIFI_NAME"
#define WIFI_PASS      "YOUR_WIFI_PASSWORD"

// Laptop running the SeptiSense backend: its IP is printed when the backend starts
#define SERVER_HOST    "192.168.1.100"
#define SERVER_PORT    8000

#define DEVICE_ID      "septisense-node-01"
#define FW_VERSION     "1.0.0"

// Display: 64 for a 0.96" 128x64 OLED, 32 for a 0.91" 128x32 OLED
#define OLED_HEIGHT    64

// ---- pins: ESP32 DevKit V1 (see the wiring diagram) ----
#define PIN_SDA        21
#define PIN_SCL        22
#define PIN_ECG        34   // AD8232 OUTPUT (ADC1)
#define PIN_LO_PLUS    35   // AD8232 LO+
#define PIN_LO_MINUS   32   // AD8232 LO-
#define PIN_LED        2    // on-board LED: local alert indicator

// Mains frequency for the ECG notch filter: 50 in India / Europe, 60 in the Americas
#define MAINS_HZ       50
