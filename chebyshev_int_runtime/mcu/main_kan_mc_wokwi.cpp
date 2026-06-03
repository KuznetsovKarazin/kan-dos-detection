/*
 * main_kan_mc_wokwi.cpp — KAN-IDS MULTICLASS fully-integer su MCU (Wokwi)
 * -----------------------------------------------------------------------
 * Versione multiclass (C classi) ad aritmetica intera pura. Estende il
 * runtime fully-integer binario: invece di un accumulatore + soglia, usa
 * C accumulatori int32 (uno per classe) e decide con argmax.
 *
 * edge index = i*C + c. logit classe c = sum_i phi_{i,c}(x_i).
 * decisione = argmax_c logit_c. Zero float nel loop (input pre-quantizzati
 * in Q16.16 dal preprocessing host).
 *
 * Solo ESP32-C3: la tabella (in_dim*C edge) supera la SRAM del Mega ma sta
 * in flash/SRAM dell'ESP32.
 *
 * Header: kan_ids_mc_int.h (da export_lut_int_multiclass.py) + test_vectors_mc_q16.h
 */

#include <Arduino.h>
#include "kan_ids_mc_int.h"
#include "test_vectors_mc_q16.h"

#ifdef __AVR__
  #define TBL_RD(e, idx) ((int16_t)pgm_read_word(&KANMC_TABLE[(e)][(idx)]))
#else
  #define TBL_RD(e, idx) (KANMC_TABLE[(e)][(idx)])
#endif

#define FP_SHIFT 16
#define FP_ONE   (1L << FP_SHIFT)

static const int32_t KANMC_INV_SEG_FP =
    (int32_t)(((float)KANMC_K / (KANMC_XMAX - KANMC_XMIN)) * (float)FP_ONE + 0.5f);

/* Forward multiclass intero. x_off_q16: input pre-quantizzato Q16.16.
 * Riempie logits[C] (int32). Decisione = argmax. */
static int kan_mc_predict(const int32_t *x_off_q16, int32_t *logits) {
  for (int c = 0; c < KANMC_C; c++) logits[c] = 0;

  for (uint8_t i = 0; i < KANMC_INDIM; i++) {
    int32_t x_off = x_off_q16[i];
    if (x_off < 0) x_off = 0;

    int32_t pos = (int32_t)(((int64_t)x_off * KANMC_INV_SEG_FP) >> FP_SHIFT);
    int seg = (int)(pos >> FP_SHIFT);
    if (seg < 0) seg = 0;
    if (seg >= KANMC_K) seg = KANMC_K - 1;

    int32_t t = pos - ((int32_t)seg << FP_SHIFT);
    if (t < 0) t = 0; if (t > FP_ONE) t = FP_ONE;

    int32_t u = (int32_t)((int64_t)t * (KANMC_L - 1));
    int idx = (int)(u >> FP_SHIFT);
    if (idx < 0) idx = 0;
    if (idx >= KANMC_L - 1) idx = KANMC_L - 2;

    int32_t frac256 = (u >> (FP_SHIFT - 8)) & 0xFF;
    int base_off = seg * KANMC_L + idx;

    /* per ogni classe, l'edge (i,c) ha indice i*C + c */
    for (int c = 0; c < KANMC_C; c++) {
      int e = i * KANMC_C + c;
      int16_t v0 = TBL_RD(e, base_off);
      int16_t v1 = TBL_RD(e, base_off + 1);
      int32_t v = (int32_t)v0 + (((int32_t)(v1 - v0) * frac256) >> 8);
      logits[c] += v;
    }
  }

  int best = 0; int32_t bestv = logits[0];
  for (int c = 1; c < KANMC_C; c++)
    if (logits[c] > bestv) { bestv = logits[c]; best = c; }
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

  Serial.println(F("=== KAN-IDS MULTICLASS fully-integer (Wokwi) ==="));
  Serial.print(F("in_dim=")); Serial.print(KANMC_INDIM);
  Serial.print(F(" C=")); Serial.print(KANMC_C);
  Serial.print(F(" E=")); Serial.print(KANMC_E);
  Serial.print(F(" K=")); Serial.print(KANMC_K);
  Serial.print(F(" L=")); Serial.println(KANMC_L);
  Serial.println(F("idx,label,pred,latency_us"));

  int correct = 0;
  unsigned long tot_us = 0, tmin = 0xFFFFFFFF, tmax = 0;
  int32_t logits[KANMC_C];

  for (int i = 0; i < N_TEST; i++) {
    unsigned long t0 = micros_now();
    int pred = kan_mc_predict(TEST_XQ[i], logits);
    unsigned long us = micros_now() - t0;

    if (pred == TEST_LABEL[i]) correct++;
    tot_us += us;
    if (us < tmin) tmin = us;
    if (us > tmax) tmax = us;

    Serial.print(i); Serial.print(F(","));
    Serial.print(TEST_LABEL[i]); Serial.print(F(","));
    Serial.print(pred); Serial.print(F(","));
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
