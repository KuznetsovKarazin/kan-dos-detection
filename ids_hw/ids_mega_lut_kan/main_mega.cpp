/*
 * LUT-KAN IDS — Arduino Mega 2560
 * Identical Serial protocol to ESP32 version → collect_hw.py works unchanged.
 * IMPORTANT: kan_avr_init() must be called in setup() before kan_infer().
 */
#include <Arduino.h>
#include "lut_kan_model.h"

static void send_meta() {
    Serial.println("READY");
    Serial.println("MODEL=LUT_KAN");
    Serial.print("ARCH=");
    Serial.print(KAN_DIM0); Serial.print("->");
    Serial.print(KAN_DIM1); Serial.print("->");
    Serial.print(KAN_DIM2); Serial.print("->"); Serial.println(KAN_DIM3);
    Serial.print("LUT_L=");    Serial.println(KAN_LUT_L);
    Serial.println("BOARD=Arduino Mega 2560");
    Serial.print("F1=");       Serial.println(KAN_F1);
    Serial.print("ROC_AUC="); Serial.println(KAN_ROC_AUC);
    Serial.print("DATASET="); Serial.println(KAN_DATASET);
    Serial.print("FLASH_KB="); Serial.println(KAN_FLASH_KB);
    // Runtime SRAM: static arrays + stack  (~3.5KB)
    uint16_t sram_est = (uint16_t)(KAN_DIM0 * 16u + KAN_DIM1 * 8u + 512u);
    Serial.print("SRAM_USED="); Serial.println(sram_est);
}

static void classify(const float* sample, const char* label, int expected) {
    uint32_t t0 = micros();
    float prob = kan_infer(sample);
    uint32_t dt = micros() - t0;
    int pred = (prob > 0.5f) ? 1 : 0;
    uint16_t sram_est = (uint16_t)(KAN_DIM0 * 16u + KAN_DIM1 * 8u + 512u);
    Serial.print(label);  Serial.print(",");
    Serial.print(pred);   Serial.print(",");
    Serial.print(prob, 4); Serial.print(",");
    Serial.print(dt);     Serial.print(",");
    Serial.print(sram_est); Serial.print(",");
    Serial.println(pred == expected ? 1 : 0);
}

void setup() {
    Serial.begin(115200);
    delay(500);
    kan_avr_init();   // ← initialise far-PROGMEM addresses (AVR only)
    send_meta();
}

void loop() {
    classify(KAN_SAMPLE_ATTACK, "ATTACK", 1);
    classify(KAN_SAMPLE_NORMAL, "NORMAL", 0);
}
