"""
Disease diagnosis API endpoints.

Uses raw SQL (sqlalchemy.text) instead of ORM to bypass a pre-existing
mapper-initialization bug in FarmerDB (broken .farms relationship).
This keeps diagnosis endpoints independent of AFIE's farm/farmer
ORM relationships.
"""
import base64
import io
import json
import uuid
from datetime import datetime
from typing import List
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from PIL import Image
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.diagnosis import (
    DiagnosisSyncItem, DiagnosisSyncResponse,
    DiagnosisHistoryItem, DiagnosisHistoryResponse,
    AttentionResponse, ValidationRequest, ValidationResponse,
    ModelRegisterRequest, DiagnosisHealthResponse,
)
from src.config.logging_config import get_logger
from src.data.database.base import get_db
from src.ml.disease_model import disease_model

logger = get_logger("diagnosis")
router = APIRouter(prefix="/diagnosis", tags=["Disease Diagnosis"])

CROP_INDEX = {"maize": 0, "tomato": 1, "potato": 2, "other": 3}
CONTEXT_KEYS = ["crop_type", "season", "symptoms", "location"]


def _attention_dict(attn_list) -> dict:
    return {k: float(v) for k, v in zip(CONTEXT_KEYS, attn_list)}


def _label_for(idx: int) -> str:
    if disease_model.class_names and idx < len(disease_model.class_names):
        return disease_model.class_names[idx]
    return f"class_{idx}"


# ─── Health & metadata ───────────────────────────────────────────
@router.get("/health", response_model=DiagnosisHealthResponse)
async def diagnosis_health():
    return {
        "status": "healthy" if disease_model.is_loaded else "model_not_loaded",
        "model": disease_model.get_metadata(),
        "classes": disease_model.num_classes or 10,
        "class_names": disease_model.class_names,
    }


@router.get("/classes")
async def list_classes():
    return {
        "classes": [
            {"id": i, "label": name}
            for i, name in enumerate(disease_model.class_names)
        ]
    }


@router.get("/model/active")
async def get_active_model():
    return disease_model.get_metadata()


@router.post("/model/register")
async def register_model(request: ModelRegisterRequest):
    logger.info(f"Model registration requested: v{request.version}")
    return {
        "status": "acknowledged",
        "version": request.version,
        "size_mb": request.size_mb,
        "accuracy": request.accuracy,
        "quantization": request.quantization,
    }


# ─── Server-side inference ───────────────────────────────────────
@router.post("/predict")
async def predict_from_base64(payload: dict):
    if not disease_model.is_loaded:
        raise HTTPException(503, "Disease model not loaded")

    try:
        b64 = payload["image_base64"]
        crop_type = payload.get("crop_type", "other").lower()
        if crop_type not in CROP_INDEX:
            raise ValueError(f"Unknown crop_type: {crop_type}")

        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0

        ctx = np.zeros(4, dtype=np.float32)
        ctx[CROP_INDEX[crop_type]] = 1.0

        result = disease_model.predict(arr, ctx)
        result["attention_weights_named"] = _attention_dict(
            result["attention_weights"])
        return result
    except KeyError as e:
        raise HTTPException(400, f"Missing field: {e}")
    except Exception as e:
        logger.error(f"predict failed: {type(e).__name__}: {e}")
        raise HTTPException(500, f"Inference failed: {e}")


# ─── Sync on-device diagnoses ────────────────────────────────────
@router.post("/sync", response_model=DiagnosisSyncResponse)
async def sync_diagnoses(
    items: List[DiagnosisSyncItem],
    db: AsyncSession = Depends(get_db),
):
    created_ids = []
    now = datetime.utcnow()

    for item in items:
        new_id = uuid.uuid4()
        label = _label_for(item.predicted_class)

        # Coerce user_id / farm_id to UUID safely
        user_id_val = None
        farm_id_val = None
        try:
            if item.user_id:
                user_id_val = uuid.UUID(str(item.user_id))
        except Exception:
            user_id_val = None
        try:
            if item.farm_id:
                farm_id_val = uuid.UUID(str(item.farm_id))
        except Exception:
            farm_id_val = None

        await db.execute(text("""
            INSERT INTO diagnoses
                (id, user_id, farm_id, image_path, crop_type, context_vector,
                 predicted_class, predicted_label, confidence, attention_weights,
                 model_version, inference_location, inference_latency_ms,
                 device_info, created_at, synced_at)
            VALUES
                (:id, :user_id, :farm_id, :image_path, :crop_type,
                 CAST(:context_vector AS json),
                 :predicted_class, :predicted_label, :confidence,
                 CAST(:attention_weights AS json),
                 :model_version, 'on_device', :latency,
                 CAST(:device_info AS json),
                 :created_at, :synced_at)
        """), {
            "id": str(new_id),
            "user_id": str(user_id_val) if user_id_val else None,
            "farm_id": str(farm_id_val) if farm_id_val else None,
            "image_path": item.image_path,
            "crop_type": item.crop_type,
            "context_vector": json.dumps(item.context_vector),
            "predicted_class": item.predicted_class,
            "predicted_label": label,
            "confidence": item.confidence,
            "attention_weights": json.dumps(item.attention_weights),
            "model_version": item.model_version,
            "latency": item.inference_latency_ms,
            "device_info": json.dumps(item.device_info) if item.device_info else None,
            "created_at": item.created_at or now,
            "synced_at": now,
        })
        created_ids.append(str(new_id))

    await db.commit()
    logger.info(f"Synced {len(created_ids)} diagnoses")
    return {"synced": len(created_ids), "ids": created_ids}


# ─── History ─────────────────────────────────────────────────────
@router.get("/history", response_model=DiagnosisHistoryResponse)
async def get_history(
    user_id: UUID,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(text("""
        SELECT id, predicted_label, predicted_class, confidence,
               crop_type, attention_weights, model_version, created_at
        FROM diagnoses
        WHERE user_id = :user_id
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
    """), {"user_id": str(user_id), "limit": limit, "offset": offset})

    rows = result.mappings().all()
    items = []
    for r in rows:
        attn = r["attention_weights"]
        if isinstance(attn, str):
            try:
                attn = json.loads(attn)
            except Exception:
                attn = {}
        items.append(DiagnosisHistoryItem(
            id=str(r["id"]),
            predicted_label=r["predicted_label"],
            predicted_class=r["predicted_class"],
            confidence=r["confidence"],
            crop_type=r["crop_type"],
            attention_weights=attn or {},
            model_version=r["model_version"],
            created_at=r["created_at"],
        ))
    return {"count": len(items), "items": items}


# ─── Attention weights ───────────────────────────────────────────
@router.get("/attention/{diagnosis_id}", response_model=AttentionResponse)
async def get_attention(
    diagnosis_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(text("""
        SELECT id, predicted_label, confidence, attention_weights
        FROM diagnoses WHERE id = :id
    """), {"id": str(diagnosis_id)})
    row = result.mappings().first()
    if row is None:
        raise HTTPException(404, "Diagnosis not found")

    attn = row["attention_weights"]
    if isinstance(attn, str):
        try:
            attn = json.loads(attn)
        except Exception:
            attn = {}
    return {
        "diagnosis_id": str(row["id"]),
        "predicted_label": row["predicted_label"],
        "confidence": row["confidence"],
        "attention_weights": attn or {},
    }


# ─── Field validation ────────────────────────────────────────────
@router.post("/validate", response_model=ValidationResponse)
async def validate_field_image(
    request: ValidationRequest,
    db: AsyncSession = Depends(get_db),
):
    # Verify diagnosis exists
    result = await db.execute(text(
        "SELECT id FROM diagnoses WHERE id = :id"
    ), {"id": str(request.diagnosis_id)})
    if result.first() is None:
        raise HTTPException(404, "Diagnosis not found")

    label = _label_for(request.ground_truth_class)
    now = datetime.utcnow()

    await db.execute(text("""
        INSERT INTO field_validation_images
            (id, diagnosis_id, ground_truth_class, ground_truth_label,
             verified_by, verified_at, notes, is_field_image,
             collection_location, collection_conditions, created_at)
        VALUES
            (:id, :diagnosis_id, :gt_class, :gt_label,
             :verified_by, :verified_at, :notes, :is_field,
             :location, CAST(:conditions AS json), :created_at)
    """), {
        "id": str(uuid.uuid4()),
        "diagnosis_id": str(request.diagnosis_id),
        "gt_class": request.ground_truth_class,
        "gt_label": label,
        "verified_by": str(request.verified_by) if request.verified_by else None,
        "verified_at": now,
        "notes": request.notes,
        "is_field": request.is_field_image,
        "location": request.collection_location,
        "conditions": json.dumps(request.collection_conditions)
                      if request.collection_conditions else None,
        "created_at": now,
    })

    # Mark diagnosis as field-validated
    await db.execute(text(
        "UPDATE diagnoses SET inference_location = 'field_validated' WHERE id = :id"
    ), {"id": str(request.diagnosis_id)})

    await db.commit()
    return {"status": "validated", "diagnosis_id": str(request.diagnosis_id)}