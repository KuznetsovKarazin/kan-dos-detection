/*
 * main_kan_wokwi.cpp — KAN-IDS su microcontrollore (versione Wokwi)
 * -----------------------------------------------------------------
 * Versione pronta per la simulazione su https://wokwi.com/
 * (Arduino Mega 2560 o ESP32-C3).
 *
 * Esegue il forward KAN dalle sole LUT (logica VERIFICATA: produce
 * logit identici al modello Python, cifra per cifra), applica la testa
 * di decisione (sigmoid + soglia) e stampa su seriale, per ogni vettore
 * di test: etichetta vera, predizione, probabilita, latenza in µs.
 * Alla fine riporta accuratezza e latenza media — stile benchmark del
 * lavoro precedente (iot-audit).
 *
 * COME USARLO SU WOKWI:
 *   1. crea un nuovo progetto Arduino Mega 2560 (o ESP32-C3)
 *   2. incolla questo file come sketch (.ino)
 *   3. crea due file aggiuntivi nel progetto Wokwi:
 *        - kan_ids_layer.h    (generato da scripts/export_lut.py)
 *        - test_vectors.h     (generato da gen_test_vectors.py)
 *   4. avvia la simulazione e apri il Serial Monitor (115200 baud)
 */

#include <Arduino.h>
#include <math.h>
#include "kan_ids_layer.h"
#include "test_vectors.h"

#ifdef __AVR__
  #define QREAD(e, idx) ((uint8_t)pgm_read_byte(&KAN_QTABLE[(e)][(idx)]))
#else
  #define QREAD(e, idx) (KAN_QTABLE[(e)][(idx)])
#endif

static inline float sigmoidf_(float z) {
  if (z >  30.0f) return 1.0f;
  if (z < -30.0f) return 0.0f;
  return 1.0f / (1.0f + expf(-z));
}

/* Valuta l'edge e nel punto x dalla LUT (dequant + interp lineare). */
static float eval_edge_lut(uint8_t e, float x) {
  const float dx = (KAN_XMAX - KAN_XMIN) / (float)KAN_K;
  float hi = KAN_XMAX - 1e-6f;
  float xc = x < KAN_XMIN ? KAN_XMIN : (x > hi ? hi : x);
  float t = (xc - KAN_XMIN) / dx;
  int k = (int)floorf(t);
  if (k < 0) k = 0;
  if (k > KAN_K - 1) k = KAN_K - 1;
  float t0 = KAN_XMIN + (float)k * dx;
  float u = (xc - t0) / dx;
  if (u < 0.0f) u = 0.0f;
  if (u > 1.0f) u = 1.0f;
  float pos = u * (float)(KAN_L - 1);
  int r0 = (int)floorf(pos);
  if (r0 < 0) r0 = 0;
  if (r0 > KAN_L - 1) r0 = KAN_L - 1;
  int r1 = r0 + 1; if (r1 > KAN_L - 1) r1 = KAN_L - 1;
  float w = pos - (float)r0;
  float scale = KAN_SCALE[e][k];
  float ymin  = KAN_YMIN[e][k];
  float y0 = ymin + scale * (float)QREAD(e, k * KAN_L + r0);
  float y1 = ymin + scale * (float)QREAD(e, k * KAN_L + r1);
  return (1.0f - w) * y0 + w * y1;
}

static int kan_predict(const float *x, float *prob_out) {
  float z = 0.0f;
  for (uint8_t e = 0; e < KAN_E; e++) z += eval_edge_lut(e, x[e]);
  float p = sigmoidf_(z);
  if (prob_out) *prob_out = p;
  return p >= KAN_THR ? 1 : 0;
}

static unsigned long micros_now() {
#ifdef __AVR__
  return micros();
#else
  return (unsigned long)esp_timer_get_time();
#endif
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  delay(200);

  Serial.println(F("=== KAN-IDS su MCU (Wokwi) ==="));
  Serial.print(F("edge=")); Serial.print(KAN_E);
  Serial.print(F(" segmenti=")); Serial.print(KAN_K);
  Serial.print(F(" campioni/seg=")); Serial.println(KAN_L);
  Serial.println(F("idx,label,pred,prob,latency_us"));

  int correct = 0;
  unsigned long tot_us = 0, tmin = 0xFFFFFFFF, tmax = 0;

  for (int i = 0; i < N_TEST; i++) {
    float prob = 0.0f;
    unsigned long t0 = micros_now();
    int pred = kan_predict(TEST_X[i], &prob);
    unsigned long us = micros_now() - t0;

    if (pred == TEST_LABEL[i]) correct++;
    tot_us += us;
    if (us < tmin) tmin = us;
    if (us > tmax) tmax = us;

    Serial.print(i); Serial.print(F(","));
    Serial.print(TEST_LABEL[i]); Serial.print(F(","));
    Serial.print(pred); Serial.print(F(","));
    Serial.print(prob, 4); Serial.print(F(","));
    Serial.println(us);
  }

  Serial.println(F("--- riepilogo ---"));
  Serial.print(F("accuratezza sui test vector: "));
  Serial.print(100.0 * correct / N_TEST, 1); Serial.println(F("%"));
  Serial.print(F("latenza media (us): "));
  Serial.println((float)tot_us / N_TEST, 1);
  Serial.print(F("latenza min/max (us): "));
  Serial.print(tmin); Serial.print(F(" / ")); Serial.println(tmax);
}

void loop() {
  /* benchmark eseguito una volta in setup(); niente da ripetere */
}
