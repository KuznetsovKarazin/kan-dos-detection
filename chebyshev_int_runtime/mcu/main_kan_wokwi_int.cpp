/*
 * main_kan_wokwi_int.cpp — KAN-IDS INTEGER-ONLY su MCU (versione Wokwi)
 * --------------------------------------------------------------------
 * Versione ad ARITMETICA INTERA PURA, secondo il feedback del
 * Prof. Kuznetsov: eliminare completamente il float dall'inferenza.
 *
 * Differenza con main_kan_wokwi.cpp (float):
 *   float:  y = ymin + scale*q ; somma float ; sigmoid ; soglia 0.5
 *   int  :  lookup int16 gia' pre-scalato ; interpolazione intera ;
 *           accumulo int32 ; confronto con soglia intera (0). Zero float.
 *
 * L'indicizzazione usa fixed-point Q16.16 (come il runtime di lut-kan):
 * una sola conversione int per input, poi tutto intero.
 *
 * Per la decisione binaria la sigmoid non serve: sigmoid(z) >= 0.5  <=>  z >= 0.
 *
 * File header: kan_ids_layer_int.h (da export_lut_int.py) + test_vectors.h
 */

#include <Arduino.h>
#include "kan_ids_layer_int.h"
#include "test_vectors.h"

#ifdef __AVR__
  #define TBL_RD(e, idx) ((int16_t)pgm_read_word(&KANI_TABLE[(e)][(idx)]))
#else
  #define TBL_RD(e, idx) (KANI_TABLE[(e)][(idx)])
#endif

#define FP_SHIFT 16
#define FP_ONE   (1L << FP_SHIFT)

/* inverse segment width in Q16.16: K / (xmax - xmin) * 2^16 */
static const int32_t KANI_INV_SEG_FP =
    (int32_t)(((float)KANI_K / (KANI_XMAX - KANI_XMIN)) * (float)FP_ONE + 0.5f);

/* Forward INTERO: ritorna la somma int32 dei contributi (il "logit intero").
 * UNICA operazione float: la conversione iniziale di ogni input in Q16.16
 * (una per input, condivisa da tutti gli edge — come nel runtime di lut-kan). */
static int32_t kan_logit_int(const float *x) {
  int32_t z = 0;
  for (uint8_t i = 0; i < KANI_E; i++) {
    /* clip nel dominio */
    float xi = x[i];
    if (xi < KANI_XMIN) xi = KANI_XMIN;
    float hi = KANI_XMAX - 1e-6f;
    if (xi > hi) xi = hi;

    /* (x - xmin) in Q16.16 — l'unica conversione float, poi solo interi */
    int32_t x_off = (int32_t)((xi - KANI_XMIN) * (float)FP_ONE);
    int32_t pos = (int32_t)(((int64_t)x_off * KANI_INV_SEG_FP) >> FP_SHIFT);

    int seg = (int)(pos >> FP_SHIFT);
    if (seg < 0) seg = 0;
    if (seg >= KANI_K) seg = KANI_K - 1;

    int32_t t = pos - ((int32_t)seg << FP_SHIFT);
    if (t < 0) t = 0; if (t > FP_ONE) t = FP_ONE;

    int32_t u = (int32_t)((int64_t)t * (KANI_L - 1));   /* Q16.16 */
    int idx = (int)(u >> FP_SHIFT);
    if (idx < 0) idx = 0;
    if (idx >= KANI_L - 1) idx = KANI_L - 2;

    int32_t frac256 = (u >> (FP_SHIFT - 8)) & 0xFF;     /* 0..255 */

    int base = seg * KANI_L;
    int16_t v0 = TBL_RD(i, base + idx);
    int16_t v1 = TBL_RD(i, base + idx + 1);
    int32_t v = (int32_t)v0 + (((int32_t)(v1 - v0) * frac256) >> 8);
    z += v;                                             /* accumulo intero */
  }
  return z;
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

  Serial.println(F("=== KAN-IDS INTEGER-ONLY su MCU (Wokwi) ==="));
  Serial.print(F("edge=")); Serial.print(KANI_E);
  Serial.print(F(" K=")); Serial.print(KANI_K);
  Serial.print(F(" L=")); Serial.print(KANI_L);
  Serial.print(F(" FP_BITS=")); Serial.println(KANI_FP_BITS);
  Serial.println(F("idx,label,pred,logit_int,latency_us"));

  int correct = 0;
  unsigned long tot_us = 0, tmin = 0xFFFFFFFF, tmax = 0;

  for (int i = 0; i < N_TEST; i++) {
    unsigned long t0 = micros_now();
    int32_t z = kan_logit_int(TEST_X[i]);
    int pred = (z >= 0) ? 1 : 0;            /* soglia intera: zero float */
    unsigned long us = micros_now() - t0;

    if (pred == TEST_LABEL[i]) correct++;
    tot_us += us;
    if (us < tmin) tmin = us;
    if (us > tmax) tmax = us;

    Serial.print(i); Serial.print(F(","));
    Serial.print(TEST_LABEL[i]); Serial.print(F(","));
    Serial.print(pred); Serial.print(F(","));
    Serial.print(z); Serial.print(F(","));
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

void loop() {}
