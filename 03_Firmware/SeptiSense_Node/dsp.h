// SeptiSense - signal processing on the node (no dynamic memory, ISR-safe arithmetic)
#pragma once
#include <math.h>
#include <stdint.h>

// ---------------------------------------------------------------- biquad (RBJ cookbook)
struct Biquad {
  float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, z1 = 0, z2 = 0;
  float step(float x) {                       // transposed direct form II
    float y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
  }
  void set(float B0, float B1, float B2, float A0, float A1, float A2) {
    b0 = B0 / A0; b1 = B1 / A0; b2 = B2 / A0; a1 = A1 / A0; a2 = A2 / A0; z1 = z2 = 0;
  }
  void highpass(float fs, float f0, float q = 0.7071f) {
    float w = 2 * M_PI * f0 / fs, c = cosf(w), al = sinf(w) / (2 * q);
    set((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al);
  }
  void lowpass(float fs, float f0, float q = 0.7071f) {
    float w = 2 * M_PI * f0 / fs, c = cosf(w), al = sinf(w) / (2 * q);
    set((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + al, -2 * c, 1 - al);
  }
  void notch(float fs, float f0, float q) {
    float w = 2 * M_PI * f0 / fs, c = cosf(w), al = sinf(w) / (2 * q);
    set(1, -2 * c, 1, 1 + al, -2 * c, 1 - al);
  }
  void bandpass(float fs, float f0, float q) {   // constant 0 dB peak gain
    float w = 2 * M_PI * f0 / fs, c = cosf(w), al = sinf(w) / (2 * q);
    set(al, 0, -al, 1 + al, -2 * c, 1 - al);
  }
};

// ---------------------------------------------------------------- R-peak detector
// Simplified Pan-Tompkins: 5-15 Hz band-pass -> derivative -> square -> 150 ms moving
// integration -> adaptive threshold with a 250 ms refractory period.
struct RPeakDetector {
  static const int FS = 250;
  static const int MWI_N = 38;                 // 150 ms
  Biquad bp1, bp2;
  float mwiBuf[MWI_N] = {0};
  int mwiIdx = 0;
  float mwiSum = 0, prev1 = 0, prev2 = 0;
  float sigLevel = 0, noiseLevel = 0;          // running estimates of peak and noise level
  float localMax = 0;
  uint32_t n = 0, lastPeak = 0;
  bool above = false;

  void begin() {
    bp1.bandpass(FS, 10.0f, 0.8f);
    bp2.bandpass(FS, 10.0f, 0.8f);
  }
  // returns true when an R-peak is confirmed; the peak happened ~ (mwi delay) samples earlier
  bool step(float x) {
    n++;
    float b = bp2.step(bp1.step(x));
    float d = b - prev2;                       // derivative over 2 samples
    prev2 = prev1; prev1 = b;
    float sq = d * d;
    mwiSum += sq - mwiBuf[mwiIdx];
    mwiBuf[mwiIdx] = sq;
    mwiIdx = (mwiIdx + 1) % MWI_N;
    float m = mwiSum / MWI_N;
    float thr = noiseLevel + 0.30f * (sigLevel - noiseLevel);
    bool hit = false;
    if (!above && m > thr && (n - lastPeak) > FS / 4) {   // rising edge, outside refractory
      above = true;
      localMax = m;
    } else if (above) {
      if (m > localMax) localMax = m;
      if (m < 0.5f * localMax) {                 // falling edge: the beat is complete
        above = false;
        sigLevel = 0.875f * sigLevel + 0.125f * localMax;
        lastPeak = n;
        hit = true;
      }
    } else {
      noiseLevel = 0.995f * noiseLevel + 0.005f * m;
    }
    if (n < FS * 2 && m > sigLevel) sigLevel = m;   // learning phase: first 2 s
    if ((n - lastPeak) > FS * 3) sigLevel *= 0.98f;   // no beat for 3 s: lower the bar slowly
    return hit;
  }
};

// ---------------------------------------------------------------- small helpers
static inline float medianOf(float *v, int n) {
  float t[32];
  if (n > 32) n = 32;
  for (int i = 0; i < n; i++) t[i] = v[i];
  for (int i = 1; i < n; i++) {
    float k = t[i]; int j = i - 1;
    while (j >= 0 && t[j] > k) { t[j + 1] = t[j]; j--; }
    t[j + 1] = k;
  }
  return n ? (n % 2 ? t[n / 2] : 0.5f * (t[n / 2 - 1] + t[n / 2])) : NAN;
}
