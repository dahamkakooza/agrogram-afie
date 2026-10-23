---BEGIN---
# AgriYouthNet-Context+ 🍃

**Ultra-lightweight plant disease diagnosis for entry-level Android devices**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16-orange)](https://tensorflow.org)
[![Kotlin](https://img.shields.io/badge/Kotlin-1.9-purple)](https://kotlinlang.org)
[![Android API 26+](https://img.shields.io/badge/Android-API%2026%2B-green)](https://android.com)

---

## Overview

AgriYouthNet-Context+ is an **ultra-lightweight plant disease classifier** designed for deployment on entry-level Android smartphones in rural Uganda. The model diagnoses **10 diseases** across maize, tomato, and potato crops using **Multi-Head Cross-Attention (MHCA) fusion** of visual leaf features and crop-type context.

**Key achievement:** 98.75% validation accuracy at **177 KB** (INT8) — deployable on devices with 1–2 GB RAM.

| Metric | Value |
|---|---|
| Validation accuracy | 98.75% |
| INT8 model size | 176.82 KB |
| Inference latency (Pixel 6a) | 30–80 ms |
| Parameter count | 140,938 |
| INT8 accuracy drop | 0.21 pp |
| H1 (attention vs concat) | +1.04 pp, 32% fewer params |

---

## Walkthrough Videos

**Part 1 — Architecture, ML Pipeline, Backend & Testing**
📹 [https://youtu.be/pT5gFy1fJjQ](https://youtu.be/pT5gFy1fJjQ)

**Part 2 — Android Application Live Demo (supplementary)**
📹 [https://www.youtube.com/watch?v=j980XBs6-U4](https://www.youtube.com/watch?v=j980XBs6-U4)

Both videos are set to Unlisted — accessible via link.

---

## Technical Report

📄 **[Read the 5-page technical report](docs/technical_report.pdf)**

Supplementary documentation (full scope alignment + testing evidence):
📄 **[Supplementary documentation](docs/supplementary_documentation.pdf)**

---

## Features

### Machine Learning
- **Custom MHCA fusion layer** — replaces concatenation with adaptive, input-dependent attention
- **INT8 post-training quantization** — 3.1× size reduction with 0.21 pp accuracy loss
- **Per-class diagnostic analysis** — confusion matrices, early-vs-late blight detection
- **OOD rejection** — green-fraction + edge-density + multi-crop consensus filters

### Backend (FastAPI + PostgreSQL)
- 9 REST endpoints under `/api/v1/diagnosis/*`
- Model versioning, per-class accuracy tracking
- Bulk sync from mobile clients
- Attention-weight persistence for interpretability

### Android Application (Kotlin + Jetpack Compose)
- On-device TFLite inference (offline-first)
- Room DB for local diagnosis history
- Background sync via WorkManager
- Attention weight visualization (4-bar chart)
- Camera capture with crop-type selector

---

## Repository Structure
agrogram-afie/
├── src/ # AFIE backend (FastAPI)
│ ├── main.py # Application entry point
│ ├── config/settings.py # Configuration
│ ├── ml/
│ │ └── disease_model.py # TFLite wrapper + inference
│ ├── api/
│ │ ├── schemas/diagnosis.py # Pydantic schemas
│ │ └── routes/diagnosis.py # REST endpoints (raw SQL)
│ └── data/database/
│ └── models_diagnosis.py # SQLAlchemy tables
│
├── training/ # ML training pipeline
│ ├── src/
│ │ ├── agriyouthnet_context_plus.py # Model factory (MHCA)
│ │ ├── dataset.py # PlantVillage loader
│ │ ├── train_numpy.py # Training loop
│ │ ├── train_ablations.py # 6-variant ablation
│ │ ├── quantize.py # INT8 conversion
│ │ └── per_class_eval.py # Diagnostics
│ ├── tests/test_tflite_multi_io.py # Unit tests
│ ├── models/ # Trained weights (gitignored)
│ └── logs/ # Training logs (gitignored)
│
├── android-app/ # Kotlin mobile app
│ ├── app/src/main/
│ │ ├── java/com/agrogram/afie/
│ │ │ ├── MainActivity.kt
│ │ │ ├── ml/DiseaseClassifier.kt # TFLite + OOD
│ │ │ ├── ml/ImagePreprocessor.kt
│ │ │ ├── data/ # Room DB layer
│ │ │ ├── network/ # Retrofit client
│ │ │ ├── worker/SyncWorker.kt # Background sync
│ │ │ └── ui/ # Compose screens
│ │ └── assets/ # TFLite model bundled
│ └── build.gradle.kts
│
├── models/ # Deployment artifacts
│ └── agriyouthnet_context_plus_int8.tflite
│
├── docs/ # Screenshots + reports
│ ├── technical_report.pdf
│ ├── supplementary_documentation.pdf
│ └── screenshots/
│
├── README.md
├── SCOPE_ALIGNMENT.md
├── TESTING_EVIDENCE.md
├── requirements.txt
└── .gitignore


---

## Installation

### Prerequisites

- Python 3.12+
- PostgreSQL 17
- Android Studio (for app deployment)
- Android SDK 34, JDK 17

### Backend

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your database URL:
#   DATABASE_URL="postgresql://afie_user:afie_password@localhost:5432/afie_db"

# Ensure the model files are in place
# models/agriyouthnet_context_plus_int8.tflite
# models/agriyouthnet_context_plus_int8.json

# Start the server (auto-creates DB tables)
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

Training
# Separate venv (avoids TF/FastAPI conflicts)
python -m venv venv-training
venv-training\Scripts\activate
pip install -r training/requirements.txt

# Download PlantVillage dataset (Kaggle: abdallahalidev/plantvillage-dataset)
# Place class folders in training/data/plantvillage/

# Train the primary model
cd training
python src/train_numpy.py

# Run ablations
python src/train_ablations.py

# Quantize
python src/quantize.py

Android

cd android-app

# Open in Android Studio, or build via CLI:
gradlew.bat assembleDebug

# Install on connected device
adb install app/build/outputs/apk/debug/app-debug.apk

Requirements: Android SDK 34, JDK 17, minSdk 26 (Android 8.0)

Note: The app uses adb reverse for backend sync during development:

adb reverse tcp:8000 tcp:8000

API Endpoints
Endpoint	Method	Purpose
/api/v1/diagnosis/health	GET	Model status + metadata
/api/v1/diagnosis/classes	GET	List 10 class labels
/api/v1/diagnosis/model/active	GET	Model card (version, params)
/api/v1/diagnosis/predict	POST	Server-side inference (base64)
/api/v1/diagnosis/sync	POST	Bulk upload from mobile
/api/v1/diagnosis/history	GET	User's diagnosis history
/api/v1/diagnosis/attention/{id}	GET	Attention weights
/api/v1/diagnosis/validate	POST	Ground truth annotation
/api/v1/diagnosis/model/register	POST	Register new model version
Interactive docs: http://localhost:8000/api/v1/docs

Model Architecture
INPUT: [leaf_image (224,224,3)] + [context_vector (4,)]
            │                              │
   ┌────────▼────────┐              ┌──────▼──────┐
   │ Visual Backbone │              │  Context    │
   │ (wide config)   │              │  Encoder    │
   │                 │              │             │
   │ Conv2D(64)      │              │ Dense(128)  │
   │ DW-Sep(128)     │              │ Dense(128)  │
   │ DW-Sep(256)     │              │ Reshape →   │
   │ GAP → 256-D     │              │ (4,32) tokens
   └────────┬────────┘              └──────┬──────┘
            │                              │
            └──────┬───────────────────────┘
                   ▼
         ┌─────────────────────┐
         │  MHCA (4 heads)     │
         │  32 key dims        │
         │  Visual Q → Ctx KV  │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Add + LayerNorm    │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Dropout(0.1)       │
         │  Dense(10, softmax) │
         └──────────┬──────────┘
                    ▼
              class_probs (10,)
              attention_weights (4,)

Total parameters: 140,938
Deployed INT8 size: 176.82 KB

Results
Ablation comparison
Variant	Params	Val Accuracy
Attention-4h (proposed)	140,938	98.75%
Gating fusion	141,966	97.81%
Concatenation	207,502	97.71%
Attention-1h	85,642	97.61%
Attention-8h	214,666	96.77%
Narrow backbone	72,906	95.84%
Hypothesis validation
Hypothesis	Target	Actual	Status
H1: Attention > Concat	≥ +1.00 pp	+1.04 pp	✅ PASS
H2: INT8 <200KB, <1pp drop	<200KB / <1pp	177KB / 0.21pp	✅ PASS

Quick tests:

# Unit test — multi-IO + INT8 conversion parity
python training/tests/test_tflite_multi_io.py

# Quantization parity
python training/src/quantize.py

# Per-class analysis
python training/src/per_class_eval.py

# Integration test — end-to-end sync
curl -X POST http://localhost:8000/api/v1/diagnosis/sync \
  -H "Content-Type: application/json" \
  -d @test_payload.json


Code Quality & Design Patterns
Object-Oriented Design
Factory pattern — build_agriyouthnet_context_plus() returns model variants

Strategy pattern — fusion strategies interchangeable via parameter

Singleton pattern — disease_model (Python) and AppDatabase (Kotlin)

Repository pattern — DiagnosisRepository.kt abstracts persistence

Custom Keras Layer — MultiHeadCrossAttention with proper lifecycle

Algorithms & Custom Logic
Multi-Head Cross-Attention — custom implementation for TFLite compatibility

Green-fraction filter — pixel-level green dominance for OOD rejection

Sobel edge-density — surface texture measurement

Multi-crop consensus voting — 3/4 agreement required

Crop-type class masking — renormalizes softmax per crop family

Raw confidence gating — pre-filter threshold before amplification

Modularity
Layer separation: ml/, data/, network/, ui/, worker/

Configuration via Pydantic Settings with typed validation

No monolithic scripts: 6 training modules, 12 Android files, 9 backend modules

Scope Alignment
Full mapping of proposal features to implementation in SCOPE_ALIGNMENT.md.

Summary:

✅ 11 of 14 planned features fully completed

⏳ 2 deferred (gamification, full field validation Week 10)

⏳ 1 partial (3 preliminary field scans collected)

Deviations justified:

Augmentation retraining failed (BatchNorm mismatch) — reported as negative finding

ResNet-50 not retrained locally (CPU constraint, reference figures used)

Field validation scaled to 3 preliminary scans (formal protocol Week 10)

Limitations
Domain shift — 98% on PlantVillage (lab) vs ~50–70% on real field photos

Augmentation failure — 52% accuracy under standard augmentation (BatchNorm issue)

OOD residuals — uniformly green textures (grass, foliage) can classify as "Healthy"

Cross-crop confusion — mitigated by crop-type filtering but present in raw model

Author
Kakooza Mahad — BSc Software Engineering
African Leadership University
Supervisor: Mr. Emmanuel Adjei
Date: October 2026

License
This project uses the PlantVillage dataset (CC-BY-NC-SA-4.0).
Code released under MIT.