/*
 * main_kan_ml_wokwi.cpp — KAN-IDS MULTICLASS MULTI-LAYER fully-integer
 * --------------------------------------------------------------------
 * Multi-layer (l1 [in->hidden] -> tanh -> l2 [hidden->C]) in aritmetica
 * interamente intera. Catena (zero float):
 *   input Q16.16 (rappr. x in [-1,1]) -> LUT l1 -> accumulo int (scala S)
 *   -> tanh-LUT -> hidden in scala S2 -> LUT l2 (indicizzata in scala S2)
 *   -> accumulo int per classe -> argmax.
 *
 * Headers: kan_ml_layer1.h (con tutte le #define), kan_ml_layer2.h,
 *          kan_ml_tanh.h, test_vectors_ml_q16.h
 *
 * Solo ESP32-C3 (320 KB di LUT). Verificato Python->C.
 */

#include <Arduino.h>
#include "kan_ml_layer1.h"
#include "kan_ml_layer2.h"
#include "kan_ml_tanh.h"
#include "test_vectors_ml_q16.h"

#ifdef __AVR__
  #define L1_RD(e, idx) ((int16_t)pgm_read_word(&KANML_L1[(e)][(idx)]))
  #define L2_RD(e, idx) ((int16_t)pgm_read_word(&KANML_L2[(e)][(idx)]))
  #define TANH_RD(idx)  ((int16_t)pgm_read_word(&KANML_TANH[(idx)]))
#else
  #define L1_RD(e, idx) (KANML_L1[(e)][(idx)])
  #define L2_RD(e, idx) (KANML_L2[(e)][(idx)])
  #define TANH_RD(idx)  (KANML_TANH[(idx)])
#endif

#define FP_SHIFT 16
#define FP_ONE   (1L << FP_SHIFT)

/* --- indicizzazione LUT su dominio [-1,1] da input Q16.16 = (x+1)*2^16 ---
 * off = x_q16 sta in [0, 2*2^16]. Mappa su KSEG segmenti x L campioni. */
static inline int32_t eval_l1(int e, int32_t x_q16) {
  int32_t full = 2 * FP_ONE;                 /* ampiezza dominio [-1,1] in Q16.16 */
  int32_t off = x_q16;
  if (off < 0) off = 0;
  if (off > full - 1) off = full - 1;
  int seg = (int)(((int64_t)off * KANML_KSEG) / full);
  if (seg < 0) seg = 0; if (seg >= KANML_KSEG) seg = KANML_KSEG - 1;
  int64_t num = (int64_t)off * KANML_KSEG - (int64_t)seg * full;   /* [0, full) */
  int64_t pos256 = (num * (KANML_L - 1) * 256) / full;
  int r0 = (int)(pos256 >> 8);
  if (r0 < 0) r0 = 0; if (r0 >= KANML_L - 1) r0 = KANML_L - 2;
  int32_t fr = (int32_t)(pos256 - ((int64_t)r0 << 8));
  if (fr < 0) fr = 0; if (fr > 256) fr = 256;
  int base = seg * KANML_L + r0;
  int16_t v0 = L1_RD(e, base);
  int16_t v1 = L1_RD(e, base + 1);
  return (int32_t)v0 + (((int32_t)(v1 - v0) * fr) >> 8);
}

/* layer2: input gia' intero in scala S2 (= valore in [-1,1]). */
static inline int32_t eval_l2(int e, int32_t x_s2) {
  int32_t full = 2 * KANML_S2;
  int32_t off = x_s2 + KANML_S2;
  if (off < 0) off = 0;
  if (off > full - 1) off = full - 1;
  int seg = (int)(((int64_t)off * KANML_KSEG) / full);
  if (seg < 0) seg = 0; if (seg >= KANML_KSEG) seg = KANML_KSEG - 1;
  int64_t num = (int64_t)off * KANML_KSEG - (int64_t)seg * full;
  int64_t pos256 = (num * (KANML_L - 1) * 256) / full;
  int r0 = (int)(pos256 >> 8);
  if (r0 < 0) r0 = 0; if (r0 >= KANML_L - 1) r0 = KANML_L - 2;
  int32_t fr = (int32_t)(pos256 - ((int64_t)r0 << 8));
  if (fr < 0) fr = 0; if (fr > 256) fr = 256;
  int base = seg * KANML_L + r0;
  int16_t v0 = L2_RD(e, base);
  int16_t v1 = L2_RD(e, base + 1);
  return (int32_t)v0 + (((int32_t)(v1 - v0) * fr) >> 8);
}

/* tanh-LUT: pre-attivazione (scala S) -> valore in scala S2 ([-1,1]) */
static inline int32_t tanh_lut(int32_t h_pre) {
  int32_t v = h_pre + KANML_HMAX;
  int32_t span = 2 * KANML_HMAX;
  if (v < 0) v = 0;
  if (v > span) v = span;
  int idx = (int)(((int64_t)v * (KANML_TL - 1)) / span);
  if (idx < 0) idx = 0; if (idx >= KANML_TL) idx = KANML_TL - 1;
  return TANH_RD(idx);
}

static int kan_ml_predict(const int32_t *x_q16, int32_t *logits) {
  int32_t hpre[KANML_HIDDEN];
  for (int j = 0; j < KANML_HIDDEN; j++) hpre[j] = 0;

  /* layer1: per ogni input i, per ogni hidden j, edge e = i*HIDDEN + j */
  for (int i = 0; i < KANML_INDIM; i++) {
    int32_t xi = x_q16[i];
    for (int j = 0; j < KANML_HIDDEN; j++) {
      int e = i * KANML_HIDDEN + j;
      hpre[j] += eval_l1(e, xi);
    }
  }
  /* tanh -> hidden in scala S2 */
  int32_t hpost[KANML_HIDDEN];
  for (int j = 0; j < KANML_HIDDEN; j++) hpost[j] = tanh_lut(hpre[j]);

  /* layer2: edge e = j*C + c */
  for (int c = 0; c < KANML_C; c++) logits[c] = 0;
  for (int j = 0; j < KANML_HIDDEN; j++) {
    int32_t hj = hpost[j];
    for (int c = 0; c < KANML_C; c++) {
      int e = j * KANML_C + c;
      logits[c] += eval_l2(e, hj);
    }
  }
  int best = 0; int32_t bv = logits[0];
  for (int c = 1; c < KANML_C; c++)
    if (logits[c] > bv) { bv = logits[c]; best = c; }
  return best;
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
  Serial.println(F("=== KAN-IDS MULTICLASS MULTI-LAYER fully-integer ==="));
  Serial.print(F("in=")); Serial.print(KANML_INDIM);
  Serial.print(F(" hidden=")); Serial.print(KANML_HIDDEN);
  Serial.print(F(" C=")); Serial.print(KANML_C);
  Serial.print(F(" edge=")); Serial.println(KANML_INDIM*KANML_HIDDEN + KANML_HIDDEN*KANML_C);
  Serial.println(F("idx,label,pred,latency_us"));

  int correct = 0;
  unsigned long tot = 0, tmin = 0xFFFFFFFF, tmax = 0;
  int32_t logits[KANML_C];

  for (int i = 0; i < N_TEST; i++) {
    unsigned long t0 = micros_now();
    int pred = kan_ml_predict(TEST_XQ[i], logits);
    unsigned long us = micros_now() - t0;
    if (pred == TEST_LABEL[i]) correct++;
    tot += us; if (us < tmin) tmin = us; if (us > tmax) tmax = us;
    Serial.print(i); Serial.print(F(","));
    Serial.print(TEST_LABEL[i]); Serial.print(F(","));
    Serial.print(pred); Serial.print(F(","));
    Serial.println(us);
  }
  Serial.println(F("--- riepilogo ---"));
  Serial.print(F("accuratezza: ")); Serial.print(100.0*correct/N_TEST,1); Serial.println(F("%"));
  Serial.print(F("latenza media (us): ")); Serial.println((float)tot/N_TEST,1);
  Serial.print(F("min/max (us): ")); Serial.print(tmin); Serial.print(F(" / ")); Serial.println(tmax);
}

void loop() {}
