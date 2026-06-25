# Simulazione su Wokwi — guida passo-passo

Far girare la KAN-IDS su Arduino Mega 2560 / ESP32-C3 simulati, senza
hardware fisico. La logica del forward e' gia stata verificata: in C
produce logit identici al modello Python, cifra per cifra.

## File necessari (tutti in mcu/)

- `main_kan_wokwi.cpp`  — lo sketch
- `kan_ids_layer.h`     — le LUT del modello (gia generato)
- `test_vectors.h`      — 40 vettori di test (20 attacco, 20 normali)

## Cosa devi vedere alla fine

Nel Serial Monitor, dopo la lista dei 40 vettori, il riepilogo deve dire:

```
accuratezza sui test vector: 97.5%
```

39 su 40 corretti. (Un errore: e' il falso positivo atteso, coerente con
l'F1 = 0.969 del modello. Non e' un bug.)

E una latenza media in microsecondi — quello e' il numero nuovo che la
simulazione ti da e che non avevi prima.

## Passo per passo

### 1. Crea il progetto

Vai su https://wokwi.com/, accedi, e crea un nuovo progetto:
- per il test sul dispositivo piu vincolante: **Arduino Mega 2560**
- (opzionale) ripeti poi con **ESP32-C3** per confrontare le latenze

### 2. Incolla lo sketch

Copia tutto il contenuto di `main_kan_wokwi.cpp` nel file principale
dello sketch (su Wokwi e' `sketch.ino` — incolla il C++ li, funziona).

### 3. Aggiungi i due header

Su Wokwi, nel pannello file (icona con il "+" o tasto destro):
- crea un file chiamato `kan_ids_layer.h` e incolla il contenuto
  dell'omonimo file
- crea un file chiamato `test_vectors.h` e incolla il suo contenuto

I nomi devono essere ESATTAMENTE quelli (lo sketch li include con
`#include "kan_ids_layer.h"` e `#include "test_vectors.h"`).

### 4. Avvia e leggi

Premi play. Apri il Serial Monitor (in basso). A 115200 baud vedrai
scorrere i 40 vettori e poi il riepilogo con accuratezza e latenza.

## Se qualcosa non va

**Errore di compilazione "KAN_E not declared"** → il file
`kan_ids_layer.h` non e' stato creato o ha un nome diverso. Controlla il
nome esatto.

**Accuratezza diversa da 97.5%** → probabilmente `kan_ids_layer.h` e
`test_vectors.h` non sono coerenti (generati da run diversi). Rigenerali
insieme: prima `export_lut.py`, poi `gen_test_vectors.py`, dallo stesso
ambiente.

**La latenza sul Mega e' alta (centinaia di µs)** → e' normale: l'AVR a
16 MHz fa molti calcoli float. L'ESP32-C3 a 160 MHz sara molto piu veloce
(nel paper il fattore era ~9-12x). Questo confronto e' anzi un risultato
interessante da riportare.
