# Simulazione Wokwi — versione INTEGER-ONLY

Versione ad aritmetica intera pura (feedback Prof. Kuznetsov). Da
confrontare con la versione float per misurare il guadagno di latenza.

## File necessari (in mcu/)

- `main_kan_wokwi_int.cpp`   — sketch integer-only
- `kan_ids_layer_int.h`      — tabella int16 pre-scalata (da export_lut_int.py)
- `test_vectors.h`           — 40 vettori (gli stessi della versione float)

## Cosa cambia rispetto alla versione float

| | float (main_kan_wokwi) | int (main_kan_wokwi_int) |
|---|---|---|
| tabella | uint8 + scale + ymin | int16 pre-scalata |
| dequant | ymin + scale*q (float) | nessuno (gia' nel valore) |
| accumulo | somma float | somma int32 |
| decisione | sigmoid >= 0.5 | logit_int >= 0 |
| float a runtime | si', molti | solo 1 conversione/input |

## Cosa devi vedere

Accuratezza identica alla versione float (**97.5%**, 39/40): la
quantizzazione intera non cambia le decisioni (verificato: coincidono al
99.92% col modello float, ΔF1 ≈ 0).

La latenza media deve essere MOLTO piu' bassa della versione float.
Obiettivo del prof: speedup nell'ordine di 10x rispetto al float.
Promemoria versione float: Mega 2851 us, ESP32-C3 1665 us.

## Passi (identici a prima)

1. Crea un progetto Wokwi (Arduino Mega 2560, poi ripeti con ESP32-C3)
2. Incolla `main_kan_wokwi_int.cpp` come sketch
3. Crea i file `kan_ids_layer_int.h` e `test_vectors.h` con i rispettivi contenuti
4. Avvia, apri il Serial Monitor (115200 baud), copia il riepilogo

## Confronto da riportare al prof

Per la mail/repo, la tabella che dimostra il punto:

| Board | float (us) | int (us) | speedup |
|---|---|---|---|
| Arduino Mega 2560 | 2851 | (da Wokwi) | (da calcolare) |
| ESP32-C3 | 1665 | (da Wokwi) | (da calcolare) |

Compila la colonna "int" con i numeri di Wokwi e calcola lo speedup
come float/int.
