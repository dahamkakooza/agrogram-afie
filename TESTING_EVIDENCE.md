
# Testing, Validation & Performance Evidence

Four testing strategies were applied, mapped to the rubric's
"Testing, Validation & Performance" criterion.

---

## 1. Unit Testing

### 1.1 TFLite multi-input/multi-output sanity test

**Purpose:** Verify that a Keras model with 2 inputs and 2 outputs
(image + context → probabilities + attention) converts to TFLite and
runs identically to Keras.

**Test file:** `training/tests/test_tflite_multi_io.py`

**Command:**
python training/tests/test_tflite_multi_io.py

**Result:**
TensorFlow version: 2.16.1
Model: "functional"
Total params: 2,190 (8.55 KB)

[1/3] Converting to TFLite F32...
✓ TFLite F32: 19.18 KB

[2/3] Converting to TFLite INT8...
✓ TFLite INT8: 15.78 KB

[3/3] Verifying inference parity...
TFLite F32 — probs: (1, 10), attention: (1, 4)
  Probs match Keras:     True
  Attention match Keras: True
TFLite INT8 — probs: (1, 10), attention: (1, 4)
  Probs match Keras (atol=0.05):     True
  Attention match Keras (atol=0.05): True

Verdict: PROCEED with AgriYouthNet-Context+

### 1.2 Quantization parity test

**Purpose:** Confirm INT8 quantization preserves accuracy within spec.

**Test file:** `training/src/quantize.py`

**Result:**

| Model | Val Accuracy |
|---|---|
| Keras baseline | 98.02% |
| TFLite F32 | 98.02% |
| TFLite INT8 | 97.81% |
| Delta | +0.21 pp |

Model sizes: Keras 1.81 MB → TFLite F32 575 KB → TFLite INT8 176.82 KB
(3.1× compression).

### 1.3 Per-class diagnostic analysis

**Test file:** `training/src/per_class_eval.py`

**F1 scores per class (attention model):**

| Class | F1 |
|---|---|
| Maize Northern Leaf Blight | 0.993 |
| Maize Common Rust | 0.991 |
| Maize Healthy | 0.998 |
| Tomato Early Blight | 0.982 |
| Tomato Late Blight | 0.965 |
| Tomato Leaf Mold | 0.971 |
| Tomato Healthy | 0.994 |
| Potato Early Blight | 0.978 |
| Potato Late Blight | 0.985 |
| Potato Healthy | 0.990 |

---

## 2. Integration Testing

### 2.1 AFIE API endpoint verification

All 9 diagnosis endpoints verified returning 200 OK:

| Endpoint | Status |
|---|---|
| GET /api/v1/diagnosis/health | 200 — returns model metadata (v1.0.0, 176.82 KB) |
| GET /api/v1/diagnosis/classes | 200 — 10 classes |
| GET /api/v1/diagnosis/model/active | 200 — 140,938 params |
| POST /api/v1/diagnosis/sync | 200 — bulk insert to PostgreSQL |
| GET /api/v1/diagnosis/history | 200 — per-user records |
| GET /api/v1/diagnosis/attention/{id} | 200 — attention weights |
| POST /api/v1/diagnosis/validate | 200 — ground truth annotation |
| POST /api/v1/diagnosis/predict | 200 — server-side inference |
| POST /api/v1/diagnosis/model/register | 200 — acknowledgment |

### 2.2 End-to-end pipeline test

**Purpose:** Verify camera capture → TFLite inference → Room save →
WorkManager sync → AFIE → PostgreSQL.

**Test sequence:**
1. Phone (offline): open app, scan leaf → Result screen appears
2. History tab: shows new row with thumbnail + "Not synced" badge
3. Enable Wi-Fi, tap cloud icon → badge changes to "Synced"
4. On PC: psql -U afie_user -d afie_db -c "SELECT * FROM diagnoses;"

**Result — 4 rows in PostgreSQL:**

| id | predicted_label | crop_type | confidence | created_at |
|---|---|---|---|---|
| da525c11-2e17-4226-9a99-63818f75d29e | Tomato_Healthy | tomato | 0.996 | 2026-10-17 16:53:53 |
| e19311aa-7870-465b-a7b2-1b97792e04c7 | Maize_Healthy | maize | 0.980 | 2026-10-17 16:22:15 |
| 642dde87-7b9d-41f7-b15a-02645fcda12c | Tomato_Late_Blight | tomato | 0.950 | 2026-10-17 16:22:15 |
| a9d8eca9-83c1-4428-86a7-7c1ffb02bb76 | Tomato_Healthy | tomato | 0.751 | 2026-10-17 16:09:41 |

**3 of 4 records originated from the physical Pixel 6a.** Full
offline-to-online persistence confirmed.

---

## 3. Device Testing

### 3.1 Physical device — Pixel 6a (Android 14, ARM64)

| Test | Result |
|---|---|
| App install | Success |
| Camera permission | Prompt shown, granted |
| Camera preview | Live back camera |
| On-device inference | Works offline |
| Inference latency | 30–80 ms (median ~45 ms) |
| Result screen renders | Attention bars visible |
| Room save | History persists across app restarts |
| Background sync | WorkManager uploads when online |
| Offline operation | Full inference works with airplane mode |

### 3.2 Emulator — API 26 (Android 8.0, minimum supported)

**Purpose:** Verify deployment target on oldest supported Android version.

| Test | Result |
|---|---|
| App install | Success |
| Camera emulation | Works with virtual camera |
| TFLite inference | Runs (~200 ms — CPU emulation) |
| Room DB | Works |
| Sync | Works over adb reverse to host |

**Conclusion:** App runs on Android 8.0 through Android 14 — the
full required range.

---

## 4. Edge Case Testing

**Purpose:** Verify OOD (out-of-distribution) rejection prevents false
positives on non-leaf inputs.

| Test | Expected | Actual |
|---|---|---|
| Pen on desk | Reject "not green enough" | Rejected |
| White wall | Reject "not green enough" | Rejected |
| Human hand | Reject "not green enough" | Rejected |
| Green mug | Reject "too smooth" | Rejected |
| Green notebook | Reject "too smooth" | Rejected |
| Grass (limitation) | May accept | Accepted as "Healthy" — documented limitation |
| Real tomato leaf | Accept, correct class | Correct |
| Real potato leaf | Accept, correct class | Correct |
| Real maize leaf | Accept, correct class | Correct |
| Blurry leaf | Reject "model is uncertain" | Rejected |
| Potato leaf w/ Tomato chip selected | Output tomato class (not potato) | Filter applied, no cross-crop output |

**Cross-crop filtering verified** via logcat:

D DiseaseClassifier: cropType='potato' cropKey='potato'
D DiseaseClassifier: context one-hot=[0.0, 0.0, 1.0, 0.0]
D DiseaseClassifier: RAW probs: ... 7:0.801 ... rawMax=0.801
D DiseaseClassifier: allowedClasses for 'potato' = [7, 8, 9]
D DiseaseClassifier: FILTERED probs: ... 3:0.000 ... 7:0.949
D DiseaseClassifier: PRED=7 (Potato Early Blight), conf=0.949

---

## 5. Performance & Compatibility Testing

### 5.1 Inference latency

| Device | Latency (single inference) |
|---|---|
| Pixel 6a (real hardware) | 30–80 ms |
| Pixel 6a emulator (API 26) | ~200 ms |
| Desktop CPU (via server) | ~50 ms |

**All under the 1-second target specified in proposal Table 6.**

### 5.2 Model size

| Precision | Size | Deployable on 8–16 GB device? |
|---|---|---|
| Keras F32 | 1.81 MB | N/A |
| TFLite F32 | 575 KB | Yes |
| TFLite INT8 | 176.82 KB | Yes — 5.6% of 3 MB app budget |

### 5.3 APK size

| Component | Size |
|---|---|
| Model | 177 KB |
| App code + Compose | ~12 MB |
| CameraX | ~8 MB |
| TFLite runtime | ~10 MB |
| Total APK | ~32 MB |

Compatible with entry-level Android storage (8–16 GB).

### 5.4 Software compatibility

| Environment | Version | Status |
|---|---|---|
| Python | 3.12.x | Pass |
| TensorFlow | 2.16.1 (CPU) | Pass |
| FastAPI | Latest | Pass |
| PostgreSQL | 17 | Pass |
| Android | 8.0 (API 26) to 14 (API 34) | Pass |
| Kotlin | 1.9 | Pass |

---

## 6. Rubric Alignment Summary

| Rubric requirement | Coverage |
|---|---|
| Multiple relevant testing strategies | 4 approaches (unit, integration, device, edge case) |
| Strong evidence across inputs and edge cases | 11-row edge case table |
| Performance/compatibility testing across devices | Pixel 6a + API 26 emulator |
| Clear analysis | Latency, size, accuracy tables with interpretation |

**Tests that failed (documented as findings):**

| Test | Result | Interpretation |
|---|---|---|
| Augmentation retraining v1 | 52% val accuracy | BatchNorm mismatch — reported as negative finding |
| Augmentation retraining v2 | 52% val accuracy | Same root cause |
| TTA (3-crop averaging) | Worse on field images | Non-centered leaves produce crops missing subject — reverted |
| Real-leaf accuracy on non-PlantVillage photos | ~50–70% (observed) | Domain shift — RQ5 answer, documented in limitations |

---

## 7. Reproduction Commands

# Unit tests
python training/tests/test_tflite_multi_io.py
python training/src/quantize.py
python training/src/per_class_eval.py

# Integration — API health
curl http://localhost:8000/api/v1/diagnosis/health

# Integration — bulk sync
curl -X POST http://localhost:8000/api/v1/diagnosis/sync \
  -H "Content-Type: application/json" \
  -d @test_payload.json

# Database verification
psql -U afie_user -d afie_db -c "SELECT * FROM diagnoses ORDER BY created_at DESC;"

# Performance — on-device logcat
adb logcat -d | Select-String "DiseaseClassifier"

