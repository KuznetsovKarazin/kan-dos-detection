/*
 * main_kan_wokwi_fullint.cpp — KAN-IDS FULLY-INTEGER su MCU (Wokwi)
 * -----------------------------------------------------------------
 * Versione SENZA ALCUN float nell'inferenza (feedback Prof. Kuznetsov,
 * richiesta "eliminare COMPLETAMENTE il float").
 *
 * Differenza con main_kan_wokwi_int.cpp:
 *   int  (precedente): 1 conversione float per input  (x-xmin)*2^16
 *   fullint (questa) : input ricevuti GIA' come int32 Q16.16 dal
 *                      preprocessing -> zero float nel loop MCU
 *
 * La pre-quantizzazione  x_q = round((x - xmin) * 2^16)  avviene UNA volta
 * nel preprocessing host (dove i dati grezzi vengono comunque trasformati),
 * non nel ciclo di inferenza. Verificato: decisioni identiche al 100%
 * rispetto alla versione int precedente.
 *
 * File header: kan_ids_layer_int.h (stesso della versione int) +
 *              test_vectors_q16.h (input pre-quantizzati Q16.16)
 */

#include <Arduino.h>
#include "kan_ids_layer_int.h"
#include "test_vectors_q16.h"

#ifdef __AVR__
  #define TBL_RD(e, idx) ((int16_t)pgm_read_word(&KANI_TABLE[(e)][(idx)]))
#else
  #define TBL_RD(e, idx) (KANI_TABLE[(e)][(idx)])
#endif

#define FP_SHIFT 16
#define FP_ONE   (1L << FP_SHIFT)

/* inverse segment width in Q16.16, precalcolata come costante intera.
 * KANI_K / (XMAX - XMIN) * 2^16 — il cast float qui e' COMPILE-TIME,
 * non viene eseguito sull'MCU (e' una costante). */
static const int32_t KANI_INV_SEG_FP =
    (int32_t)(((float)KANI_K / (KANI_XMAX - KANI_XMIN)) * (float)FP_ONE + 0.5f);

/* x_off_q16: input gia' pre-quantizzato = round((x - xmin) * 2^16).
 * Tutto il forward e' INTERO: nessuna operazione float a runtime. */
static int32_t kan_logit_fullint(const int32_t *x_off_q16) {
  int32_t z = 0;
  for (uint8_t i = 0; i < KANI_E; i++) {
    int32_t x_off = x_off_q16[i];
    if (x_off < 0) x_off = 0;                        /* clip a [xmin, xmax) */

    int32_t pos = (int32_t)(((int64_t)x_off * KANI_INV_SEG_FP) >> FP_SHIFT);
    int seg = (int)(pos >> FP_SHIFT);
    if (seg < 0) seg = 0;
    if (seg >= KANI_K) seg = KANI_K - 1;

    int32_t t = pos - ((int32_t)seg << FP_SHIFT);
    if (t < 0) t = 0; if (t > FP_ONE) t = FP_ONE;

    int32_t u = (int32_t)((int64_t)t * (KANI_L - 1));
    int idx = (int)(u >> FP_SHIFT);
    if (idx < 0) idx = 0;
    if (idx >= KANI_L - 1) idx = KANI_L - 2;

    int32_t frac256 = (u >> (FP_SHIFT - 8)) & 0xFF;

    int base = seg * KANI_L;
    int16_t v0 = TBL_RD(i, base + idx);
    int16_t v1 = TBL_RD(i, base + idx + 1);
    int32_t v = (int32_t)v0 + (((int32_t)(v1 - v0) * frac256) >> 8);
    z += v;
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

  Serial.println(F("=== KAN-IDS FULLY-INTEGER su MCU (Wokwi) ==="));
  Serial.print(F("edge=")); Serial.print(KANI_E);
  Serial.print(F(" K=")); Serial.print(KANI_K);
  Serial.print(F(" L=")); Serial.println(KANI_L);
  Serial.println(F("(zero float nel loop di inferenza)"));
  Serial.println(F("idx,label,pred,logit_int,latency_us"));

  int correct = 0;
  unsigned long tot_us = 0, tmin = 0xFFFFFFFF, tmax = 0;

  for (int i = 0; i < N_TEST; i++) {
    unsigned long t0 = micros_now();
    int32_t z = kan_logit_fullint(TEST_XQ[i]);
    int pred = (z >= 0) ? 1 : 0;
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
