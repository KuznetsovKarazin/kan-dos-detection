/*
 * IDS Embedded -- LUT-KAN
 * Board  : ESP32-C3 SuperMini  (RISC-V 160 MHz, 400 KB SRAM)
 * Model  : lut_kan_model.h
 * Serial : 115200 baud, CSV: label,pred,prob,latency_us,free_heap,correct
 */
#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "lut_kan_model.h"

static void send_meta() {
    Serial.println("READY");
    Serial.println("MODEL=LUT_KAN");
    Serial.print("ARCH=");
    for (int i = 0; i < KAN_N_LAYERS; i++) {
        Serial.print(KAN_IN_DIMS[i]); Serial.print("->");
    }
    Serial.println(KAN_OUT_DIMS[KAN_N_LAYERS - 1]);
    Serial.print("LUT_L=");   Serial.println(KAN_LUT_L);
    Serial.print("N_SEGS=");  Serial.println(KAN_N_SEGS);
    Serial.print("F1=");      Serial.println(KAN_F1);
    Serial.print("ROC_AUC="); Serial.println(KAN_ROC_AUC);
    Serial.print("DATASET="); Serial.println(KAN_DATASET);
    Serial.print("SIZE_KB="); Serial.println(KAN_FLASH_KB);
    // SRAM: report actual runtime heap usage (model is in KAN_SRAM_MODE)
    uint32_t free_heap = ESP.getFreeHeap();
    uint32_t used_sram = 327680 - free_heap;  // rough estimate
    Serial.print("SRAM_USED=");
    Serial.print(used_sram); Serial.print(" B (");
    Serial.print(used_sram / 1024); Serial.print(" KB, model in ");
    Serial.print(KAN_SRAM_MODE); Serial.println(")");
    Serial.print("FREE_HEAP="); Serial.println(free_heap);
    Serial.println("HEADER:label,pred,prob,latency_us,free_heap,correct");
}

static void classify(const float* features, const char* label, int expected) {
    unsigned long t0 = micros();
    float prob       = kan_infer(features);
    unsigned long dt = micros() - t0;
    int pred = (prob > 0.5f) ? 1 : 0;
    int ok   = (pred == expected) ? 1 : 0;
    Serial.print(label);   Serial.print(",");
    Serial.print(pred);    Serial.print(",");
    Serial.print(prob, 4); Serial.print(",");
    Serial.print(dt);      Serial.print(",");
    Serial.print(ESP.getFreeHeap()); Serial.print(",");
    Serial.println(ok);
}

void setup() {
    Serial.begin(115200);
    delay(1200);
    send_meta();
}

void loop() {
    static uint32_t cycle = 0;
    if (++cycle % 20 == 1) send_meta();
    classify(KAN_SAMPLE_ATTACK, "ATTACK", 1);
    classify(KAN_SAMPLE_NORMAL, "NORMAL", 0);
    delay(1000);
}
