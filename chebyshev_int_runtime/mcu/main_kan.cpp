/*
 * main_kan.cpp — firmware di base KAN-IDS per microcontrollore
 * -----------------------------------------------------------
 * Legge l'header generato (kan_ids_layer.h), esegue il forward KAN
 * dalle sole LUT, applica la testa di decisione (sigmoid + soglia) e
 * misura la latenza di inferenza. Stesso stile dei benchmark del lavoro
 * precedente (iot-audit): output CSV su seriale.
 *
 * Il forward replica ESATTAMENTE il runtime Python verificato in
 * scripts/export_lut.py (clip dominio, segmento uniforme, interp lineare,
 * dequant y = y_min + scale*q), quindi le decisioni coincidono.
 *
 * Target: Arduino Mega 2560 / ESP32-C3 (PlatformIO o Arduino IDE).
 * Compila per __AVR__ con LUT in PROGMEM; su ESP32 le LUT stanno in flash.
 *
 * NOTA: file di base, da integrare con la lettura dei vettori di test
 * reali. Qui include un singolo vettore d'esempio per dimostrare il forward.
 */

#include <Arduino.h>
#include <math.h>
#include "kan_ids_layer.h"   // genera con: python scripts/export_lut.py

#ifdef __AVR__
  #define QREAD(e, idx) ((uint8_t)pgm_read_byte(&KAN_QTABLE[(e)][(idx)]))
#else
  #define QREAD(e, idx) (KAN_QTABLE[(e)][(idx)])
#endif

static inline float sigmoidf(float z) {
  if (z >  30.0f) return 1.0f;
  if (z < -30.0f) return 0.0f;
  return 1.0f / (1.0f + expf(-z));
}

/* Valuta l'edge e nel punto x dalla LUT (dequant + interp lineare). */
static float eval_edge_lut(uint8_t e, float x) {
  const float dx = (KAN_XMAX - KAN_XMIN) / (float)KAN_K;
  /* clip per indicizzazione (boundary half-open) */
  float hi = KAN_XMAX - 1e-6f;
  float xc = x < KAN_XMIN ? KAN_XMIN : (x > hi ? hi : x);
  /* segmento uniforme k e posizione u in [0,1) */
  float t = (xc - KAN_XMIN) / dx;
  int k = (int)floorf(t);
  if (k < 0) k = 0;
  if (k > KAN_K - 1) k = KAN_K - 1;
  float t0 = KAN_XMIN + (float)k * dx;
  float u = (xc - t0) / dx;
  if (u < 0.0f) u = 0.0f;
  if (u > 1.0f) u = 1.0f;
  /* indici LUT r0,r1 e peso w (L campioni, endpoint inclusi) */
  float pos = u * (float)(KAN_L - 1);
  int r0 = (int)floorf(pos);
  if (r0 < 0) r0 = 0;
  if (r0 > KAN_L - 1) r0 = KAN_L - 1;
  int r1 = r0 + 1; if (r1 > KAN_L - 1) r1 = KAN_L - 1;
  float w = pos - (float)r0;
  /* dequant dei due campioni nel segmento k */
  float scale = KAN_SCALE[e][k];
  float ymin  = KAN_YMIN[e][k];
  float y0 = ymin + scale * (float)QREAD(e, k * KAN_L + r0);
  float y1 = ymin + scale * (float)QREAD(e, k * KAN_L + r1);
  return (1.0f - w) * y0 + w * y1;
}

/* Forward completo: logit = somma degli edge; decisione = sigmoid >= soglia. */
static int kan_predict(const float *x, float *prob_out) {
  float z = 0.0f;
  for (uint8_t e = 0; e < KAN_E; e++) z += eval_edge_lut(e, x[e]);
  float p = sigmoidf(z);
  if (prob_out) *prob_out = p;
  return p >= KAN_THR ? 1 : 0;
}

/* Vettore d'esempio (10 feature z-scored). Sostituire con vettori reali. */
static const float demo_x[KAN_E] = {
  0.5f, -0.3f, 1.2f, -0.8f, 0.1f, 0.0f, -1.1f, 0.7f, 0.4f, -0.2f
};

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  Serial.println(F("label,pred,prob,latency_us"));
}

void loop() {
  float prob = 0.0f;

#ifdef __AVR__
  unsigned long t0 = micros();
  int pred = kan_predict(demo_x, &prob);
  unsigned long t1 = micros();
  unsigned long us = t1 - t0;
#else
  uint64_t t0 = esp_timer_get_time();
  int pred = kan_predict(demo_x, &prob);
  uint64_t t1 = esp_timer_get_time();
  unsigned long us = (unsigned long)(t1 - t0);
#endif

  Serial.print(F("demo,"));
  Serial.print(pred);
  Serial.print(F(","));
  Serial.print(prob, 4);
  Serial.print(F(","));
  Serial.println(us);

  delay(1000);
}
