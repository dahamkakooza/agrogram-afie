# Scope Alignment — AgriYouthNet-Context+

This document maps every planned feature from the approved capstone proposal (Section 1.3) to its implementation status.

| Planned Feature | Intended Purpose | Status | Evidence |
|---|---|---|---|
| Benchmark 5 baseline CNN architectures | Establish accuracy-efficiency frontier | ✅ Completed | `training/src/train_ablations.py`; results in `logs/ablation_results.json` |
| Develop novel MHCA cross-attention fusion | Replace fixed-weight concat with input-dependent attention | ✅ Completed | `training/src/agriyouthnet_context_plus.py` (`MultiHeadCrossAttention` class) |
| Systematic ablation studies | Isolate contributions of fusion strategy, head count, quantization | ✅ Completed | 6 variants: attention-4h (98.75%), concat (97.71%), gating (97.81%), attention-1h (97.61%), attention-8h (96.77%), narrow (95.84%) |
| INT8 post-training quantization | Reduce model to <200KB while preserving accuracy | ✅ Completed | `training/src/quantize.py`; output 176.82 KB, 0.21 pp accuracy drop |
| Per-class diagnostic analysis | Analyze early-vs-late blight confusion | ✅ Completed | `training/src/per_class_eval.py`; per-class F1 0.965–0.998 |
| Deploy model on entry-level Android | Enable youth access on their own devices | ✅ Completed | `android-app/` (Kotlin + Compose); verified on Pixel 6a; 30–80 ms inference |
| Interpretability — attention weights UI | Show which context features drove diagnosis | ✅ Completed | `ResultScreen.kt` 4-bar chart; persisted in DB |
| AFIE backend integration | Serve model via existing backend | ✅ Completed | `src/api/routes/diagnosis.py` (9 endpoints); verified 200 OK |
| Offline-first operation | Youth may not have reliable internet | ✅ Completed | Room DB; TFLite on-device |
| Background sync to backend | Persist data without blocking UI | ✅ Completed | `worker/SyncWorker.kt` (WorkManager); verified rows in PostgreSQL |
| OOD rejection | Prevent false positives on non-plant objects | ✅ Completed | `ImagePreprocessor.greenFraction()` + `.edgeDensity()` |
| Crop-type class filtering | Prevent cross-crop confusion | ✅ Completed | `DiseaseClassifier.CROP_TO_CLASSES` mask + renormalization |
| Field validation with 20-30 youth | Assess real-world generalization (RQ5) | ⏳ Partial — 3 scans collected, full protocol Week 10 | Sample records in `diagnoses` table |
| Gamification integration | Incentivize youth engagement | ⏳ Deferred — out of ML scope | AFIE's Game Processing module disabled in `.env` |

## Deviations — Justifications

**1. Augmentation retraining abandoned** — Two attempts produced 52% accuracy (vs 98% baseline). BatchNorm running-statistics mismatch. Reported as a negative finding; field-data fine-tuning proposed for future work.

**2. ResNet-50 baseline not retrained locally** — CPU compute constraint. Reference figures from He et al. (2016) used in comparison table.

**3. RQ5 field validation reduced to 3 preliminary scans** — Time constraints; formal 20-30 youth protocol scheduled for Week 10.

**4. Concat baseline not parameter-matched** — Attention wins with 32% FEWER parameters (140K vs 207K). This makes H1 result stronger, not weaker.
