"""Pydantic schemas for the disease diagnosis API."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DiagnosisSyncItem(BaseModel):
    """One on-device diagnosis, sent from Android on sync."""
    user_id: UUID
    farm_id: Optional[UUID] = None
    image_path: str                      # client-side path or URL
    crop_type: str                       # "maize" | "tomato" | "potato" | "other"
    context_vector: List[float] = Field(..., min_length=4, max_length=4)
    predicted_class: int = Field(..., ge=0, le=9)
    confidence: float = Field(..., ge=0.0, le=1.0)
    attention_weights: Dict[str, float]
    model_version: str
    inference_latency_ms: Optional[float] = None
    device_info: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class DiagnosisSyncResponse(BaseModel):
    synced: int
    ids: List[str]


class DiagnosisHistoryItem(BaseModel):
    id: str
    predicted_label: str
    predicted_class: int
    confidence: float
    crop_type: str
    attention_weights: Dict[str, float]
    model_version: str
    created_at: datetime


class DiagnosisHistoryResponse(BaseModel):
    count: int
    items: List[DiagnosisHistoryItem]


class AttentionResponse(BaseModel):
    diagnosis_id: str
    predicted_label: str
    confidence: float
    attention_weights: Dict[str, float]


class ValidationRequest(BaseModel):
    diagnosis_id: UUID
    ground_truth_class: int = Field(..., ge=0, le=9)
    verified_by: Optional[UUID] = None
    notes: Optional[str] = None
    is_field_image: bool = True
    collection_location: Optional[str] = None
    collection_conditions: Optional[Dict[str, Any]] = None


class ValidationResponse(BaseModel):
    status: str
    diagnosis_id: str


class ModelRegisterRequest(BaseModel):
    version: str
    size_mb: float
    accuracy: float
    quantization: str
    per_class_accuracy: Optional[Dict[str, float]] = None


class DiagnosisHealthResponse(BaseModel):
    status: str
    model: Dict[str, Any]
    classes: int
    class_names: List[str]