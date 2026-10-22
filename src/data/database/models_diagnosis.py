"""
SQLAlchemy models for disease diagnosis.

Two tables:
  - diagnoses: one row per on-device diagnosis synced from Android
  - field_validation_images: ground truth annotations for field validation

Tables use plain UUID columns (no FK constraints) to avoid dependency on
the exact schema of existing User/Farm tables.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text, JSON,
)
from sqlalchemy.dialects.postgresql import UUID

from src.data.database.base import Base


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Ownership / context (nullable — no hard FK)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    farm_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    # Input
    image_path = Column(String(512), nullable=False)
    crop_type = Column(String(32), nullable=False)
    context_vector = Column(JSON, nullable=False)         # [0,1,0,0] etc.

    # Output
    predicted_class = Column(Integer, nullable=False)
    predicted_label = Column(String(128), nullable=False)
    confidence = Column(Float, nullable=False)
    attention_weights = Column(JSON, nullable=False)      # {"maize": 0.6, ...}

    # Metadata
    model_version = Column(String(64), nullable=False)
    inference_location = Column(String(32), default="on_device")
    inference_latency_ms = Column(Float, nullable=True)
    device_info = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    synced_at = Column(DateTime, default=datetime.utcnow)


class FieldValidationImage(Base):
    __tablename__ = "field_validation_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    diagnosis_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    ground_truth_class = Column(Integer, nullable=True)
    ground_truth_label = Column(String(128), nullable=True)
    verified_by = Column(UUID(as_uuid=True), nullable=True)
    verified_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    is_field_image = Column(Boolean, default=True)
    collection_location = Column(String(255), nullable=True)
    collection_conditions = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)