"""
Complete SQLAlchemy database models for AFIE service.
Production-ready with all fixes integrated.
"""
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    JSON, ForeignKey, Text, Enum as SQLEnum, UniqueConstraint,
    Index, BigInteger, Numeric, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, validates

from src.data.database.base import Base


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps."""
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow, 
        nullable=False
    )


class FarmDB(Base, TimestampMixin):
    """Database model for farms."""
    
    __tablename__ = "farms"
    
    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    farmer_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    
    # Basic information
    name = Column(String(100), nullable=False)
    location = Column(String(200), nullable=False)
    latitude = Column(Numeric(9, 6), nullable=False)  # -90 to 90 with 6 decimal places
    longitude = Column(Numeric(10, 6), nullable=False)  # -180 to 180 with 6 decimal places
    size_hectares = Column(Float, nullable=False)
    size_category = Column(String(50), nullable=False)  # 'micro', 'smallholder', etc.
    established_year = Column(Integer, nullable=False)
    registration_number = Column(String(50), nullable=True)
    
    # Typology
    typology_type = Column(String(50), nullable=False)  # 'smallholder', 'commercial', etc.
    sub_type = Column(String(50), nullable=True)
    specialization = Column(ARRAY(String), nullable=False, default=list)
    farming_system = Column(String(100), nullable=False)
    technology_level = Column(String(50), nullable=False)
    market_orientation = Column(String(50), nullable=False)
    sustainability_practices = Column(ARRAY(String), nullable=False, default=list)
    certifications = Column(ARRAY(String), nullable=False, default=list)
    
    # Resource constraints
    capital_access_score = Column(Float, nullable=False)  # 0-100
    labor_availability_score = Column(Float, nullable=False)  # 0-100
    equipment_access_score = Column(Float, nullable=False)  # 0-100
    market_proximity_score = Column(Float, nullable=False)  # 0-100
    water_access_score = Column(Float, nullable=False)  # 0-100
    land_availability_score = Column(Float, nullable=False)  # 0-100
    max_capital_per_season = Column(Float, nullable=True)
    max_labor_hours_per_day = Column(Integer, nullable=True)
    equipment_limitations = Column(ARRAY(String), nullable=False, default=list)
    water_restrictions = Column(ARRAY(String), nullable=False, default=list)
    
    # Environmental context
    soil_type = Column(String(50), nullable=False)
    soil_ph = Column(Float, nullable=False)
    soil_organic_matter = Column(Float, nullable=False)  # percentage
    elevation_meters = Column(Float, nullable=False)
    average_rainfall_mm = Column(Float, nullable=False)
    average_temperature_c = Column(Float, nullable=False)
    growing_season_days = Column(Integer, nullable=False)
    irrigation_type = Column(String(50), nullable=False)
    water_source = Column(String(100), nullable=True)
    microclimate_zones = Column(ARRAY(String), nullable=False, default=list)
    topography = Column(ARRAY(String), nullable=False, default=list)
    
    # Goals and preferences
    goals = Column(ARRAY(String), nullable=False, default=list)
    preferences = Column(JSON, nullable=False, default=dict)
    
    # Performance metrics
    confidence_score = Column(Float, nullable=False, default=0.0)  # 0-1
    dna_version = Column(Integer, nullable=False, default=1)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    last_profile_update = Column(DateTime, nullable=True)
    
    # Relationships
    historical_performance = relationship(
        "HistoricalPerformanceDB",
        back_populates="farm",
        cascade="all, delete-orphan",
        lazy="selectin"  # Eager loading for performance
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_farm_location", "latitude", "longitude"),
        Index("idx_farm_farmer", "farmer_id", "is_active"),
        Index("idx_farm_typology", "typology_type", "size_category"),
        UniqueConstraint("farmer_id", "name", name="uq_farmer_farm_name"),
    )
    
    @validates("latitude")
    def validate_latitude(self, key, value):
        """Validate latitude range."""
        if not -90 <= float(value) <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return value
    
    @validates("longitude")
    def validate_longitude(self, key, value):
        """Validate longitude range."""
        if not -180 <= float(value) <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return value
    
    @validates("capital_access_score", "labor_availability_score", 
               "equipment_access_score", "market_proximity_score",
               "water_access_score", "land_availability_score")
    def validate_scores(self, key, value):
        """Validate score ranges."""
        if not 0 <= float(value) <= 100:
            raise ValueError(f"{key} must be between 0 and 100")
        return value
    
    @validates("soil_ph")
    def validate_soil_ph(self, key, value):
        """Validate soil pH range."""
        if not 3.0 <= float(value) <= 10.0:
            raise ValueError("Soil pH must be between 3.0 and 10.0")
        return value
    
    @validates("soil_organic_matter")
    def validate_organic_matter(self, key, value):
        """Validate soil organic matter percentage."""
        if not 0 <= float(value) <= 100:
            raise ValueError("Soil organic matter must be between 0 and 100%")
        return value
    
    def __repr__(self):
        return f"<FarmDB(id={self.id}, name='{self.name}', typology='{self.typology_type}')>"


class HistoricalPerformanceDB(Base, TimestampMixin):
    """Database model for historical farm performance."""
    
    __tablename__ = "historical_performance"
    
    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    farm_id = Column(PG_UUID(as_uuid=True), ForeignKey("farms.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Season information
    year = Column(Integer, nullable=False)
    season = Column(String(50), nullable=False)  # 'main', 'short_rains', etc.
    
    # Crops
    crops_grown = Column(ARRAY(String), nullable=False, default=list)
    
    # Performance metrics
    total_yield_kg = Column(Float, nullable=False)
    total_revenue = Column(Float, nullable=True)
    total_cost = Column(Float, nullable=True)
    water_used_m3 = Column(Float, nullable=True)
    labor_hours = Column(Float, nullable=True)
    inputs_used = Column(JSON, nullable=False, default=dict)
    
    # Qualitative data
    challenges_faced = Column(ARRAY(String), nullable=False, default=list)
    successes = Column(ARRAY(String), nullable=False, default=list)
    notes = Column(Text, nullable=True)
    
    # Calculated fields
    profit = Column(Float, nullable=True)  # revenue - cost
    yield_per_hectare = Column(Float, nullable=True)  # yield / farm size
    
    # Verification
    is_verified = Column(Boolean, nullable=False, default=False)
    verified_by = Column(PG_UUID(as_uuid=True), nullable=True)
    verification_date = Column(DateTime, nullable=True)
    
    # Relationships
    farm = relationship("FarmDB", back_populates="historical_performance")
    
    # Indexes
    __table_args__ = (
        Index("idx_performance_farm_year", "farm_id", "year", "season"),
        Index("idx_performance_year_season", "year", "season"),
        UniqueConstraint("farm_id", "year", "season", name="uq_farm_year_season"),
    )
    
    @validates("year")
    def validate_year(self, key, value):
        """Validate year is realistic."""
        current_year = datetime.utcnow().year
        if not 1900 <= value <= current_year:
            raise ValueError(f"Year must be between 1900 and {current_year}")
        return value
    
    @validates("total_yield_kg", "total_revenue", "total_cost", 
               "water_used_m3", "labor_hours")
    def validate_positive_values(self, key, value):
        """Validate positive values."""
        if value is not None and float(value) < 0:
            raise ValueError(f"{key} cannot be negative")
        return value
    
    def __repr__(self):
        return f"<HistoricalPerformanceDB(farm_id={self.farm_id}, year={self.year}, season='{self.season}')>"


class FarmerDB(Base, TimestampMixin):
    """Database model for farmers."""
    
    __tablename__ = "farmers"
    
    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    user_id = Column(PG_UUID(as_uuid=True), nullable=False, unique=True, index=True)
    
    # Personal information
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    phone_number = Column(String(50), nullable=True)
    
    # Demographics
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    education_level = Column(String(50), nullable=True)
    farming_experience = Column(String(50), nullable=True)
    
    # Location
    country = Column(String(100), nullable=False)
    region = Column(String(100), nullable=False)
    district = Column(String(100), nullable=True)
    village = Column(String(100), nullable=True)
    
    # Communication preferences
    preferred_language = Column(String(10), nullable=False, default="en")
    communication_channels = Column(ARRAY(String), nullable=False, default=["app"])
    notification_frequency = Column(String(20), nullable=False, default="daily")
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    farms = relationship("FarmDB", backref="farmer", lazy="dynamic")
    behavioral_profile = relationship(
        "BehavioralProfileDB",
        back_populates="farmer",
        uselist=False,
        cascade="all, delete-orphan"
    )
    learning_profile = relationship(
        "LearningProfileDB",
        back_populates="farmer",
        uselist=False,
        cascade="all, delete-orphan"
    )
    goals = relationship(
        "FarmerGoalsDB",
        back_populates="farmer",
        uselist=False,
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_farmer_country_region", "country", "region"),
        Index("idx_farmer_experience", "farming_experience"),
    )
    
    @validates("email")
    def validate_email(self, key, value):
        """Validate email format."""
        if "@" not in value:
            raise ValueError("Invalid email format")
        return value.lower().strip()
    
    @validates("age")
    def validate_age(self, key, value):
        """Validate age range."""
        if value is not None and not 18 <= value <= 120:
            raise ValueError("Age must be between 18 and 120")
        return value
    
    def __repr__(self):
        return f"<FarmerDB(id={self.id}, name='{self.first_name} {self.last_name}', email='{self.email}')>"


class BehavioralProfileDB(Base, TimestampMixin):
    """Database model for farmer behavioral profiles."""
    
    __tablename__ = "behavioral_profiles"
    
    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    farmer_id = Column(
        PG_UUID(as_uuid=True), 
        ForeignKey("farmers.id", ondelete="CASCADE"), 
        nullable=False, 
        unique=True, 
        index=True
    )
    
    # Behavioral dimensions
    risk_tolerance = Column(String(50), nullable=False, default="moderate")
    innovation_adoption_speed = Column(Float, nullable=False, default=0.5)  # 0-1
    planning_horizon_days = Column(Integer, nullable=False, default=30)
    decision_confidence = Column(Float, nullable=False, default=0.7)  # 0-1
    
    # Learning preferences
    learning_style = Column(String(50), nullable=False, default="mixed")
    information_depth_preference = Column(String(50), nullable=False, default="moderate")
    feedback_receptiveness = Column(Float, nullable=False, default=0.8)  # 0-1
    
    # Behavioral patterns
    typical_decision_time_hours = Column(Float, nullable=False, default=24.0)
    error_correction_speed = Column(Float, nullable=False, default=0.6)  # 0-1
    stress_response_pattern = Column(String(50), nullable=False, default="adaptive")
    
    # Tracked metrics
    recommendation_adoption_rate = Column(Float, nullable=False, default=0.0)  # 0-1
    success_rate = Column(Float, nullable=False, default=0.0)  # 0-1
    engagement_score = Column(Float, nullable=False, default=0.0)  # 0-1
    
    # Evolution tracking
    evolution_tracking = Column(JSON, nullable=False, default=list)
    
    # Metadata
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    farmer = relationship("FarmerDB", back_populates="behavioral_profile")
    
    # Indexes
    __table_args__ = (
        Index("idx_behavior_risk", "risk_tolerance"),
        Index("idx_behavior_engagement", "engagement_score"),
    )
    
    @validates("innovation_adoption_speed", "decision_confidence", 
               "feedback_receptiveness", "error_correction_speed",
               "recommendation_adoption_rate", "success_rate", "engagement_score")
    def validate_probability_values(self, key, value):
        """Validate probability values between 0 and 1."""
        if not 0 <= float(value) <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
        return value
    
    @validates("planning_horizon_days")
    def validate_planning_horizon(self, key, value):
        """Validate planning horizon."""
        if not 1 <= value <= 365:
            raise ValueError("Planning horizon must be between 1 and 365 days")
        return value
    
    def __repr__(self):
        return f"<BehavioralProfileDB(farmer_id={self.farmer_id}, risk_tolerance='{self.risk_tolerance}')>"


class LearningProfileDB(Base, TimestampMixin):
    """Database model for farmer learning profiles."""
    
    __tablename__ = "learning_profiles"
    
    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    farmer_id = Column(
        PG_UUID(as_uuid=True), 
        ForeignKey("farmers.id", ondelete="CASCADE"), 
        nullable=False, 
        unique=True, 
        index=True
    )
    
    # Learning progress
    completed_modules = Column(ARRAY(String), nullable=False, default=list)
    current_modules = Column(ARRAY(String), nullable=False, default=list)
    skill_levels = Column(JSON, nullable=False, default=dict)
    
    # Learning preferences
    preferred_content_format = Column(ARRAY(String), nullable=False, default=["video", "text"])
    preferred_content_length = Column(String(50), nullable=False, default="medium")
    difficulty_preference = Column(String(50), nullable=False, default="adaptive")
    
    # Learning metrics
    total_learning_hours = Column(Float, nullable=False, default=0.0)
    average_completion_rate = Column(Float, nullable=False, default=0.0)  # 0-1
    knowledge_retention_score = Column(Float, nullable=False, default=0.0)  # 0-1
    last_learning_session = Column(DateTime, nullable=True)
    
    # Progress tracking
    learning_goals = Column(ARRAY(String), nullable=False, default=list)
    progress_milestones = Column(JSON, nullable=False, default=list)
    
    # Relationships
    farmer = relationship("FarmerDB", back_populates="learning_profile")
    
    # Indexes
    __table_args__ = (
        Index("idx_learning_engagement", "total_learning_hours", "average_completion_rate"),
    )
    
    @validates("average_completion_rate", "knowledge_retention_score")
    def validate_probability_values(self, key, value):
        """Validate probability values between 0 and 1."""
        if not 0 <= float(value) <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
        return value
    
    def __repr__(self):
        return f"<LearningProfileDB(farmer_id={self.farmer_id}, completed_modules={len(self.completed_modules)})>"


class FarmerGoalsDB(Base, TimestampMixin):
    """Database model for farmer goals."""
    
    __tablename__ = "farmer_goals"
    
    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    farmer_id = Column(
        PG_UUID(as_uuid=True), 
        ForeignKey("farmers.id", ondelete="CASCADE"), 
        nullable=False, 
        unique=True, 
        index=True
    )
    
    # Goal categories (stored as JSON for flexibility)
    production_goals = Column(ARRAY(String), nullable=False, default=list)
    income_goals = Column(ARRAY(String), nullable=False, default=list)
    sustainability_goals = Column(ARRAY(String), nullable=False, default=list)
    quality_goals = Column(ARRAY(String), nullable=False, default=list)
    lifestyle_goals = Column(ARRAY(String), nullable=False, default=list)
    
    # Goal details
    goal_details = Column(JSON, nullable=False, default=dict)
    
    # Timeframes
    short_term_goals = Column(ARRAY(String), nullable=False, default=list)
    medium_term_goals = Column(ARRAY(String), nullable=False, default=list)
    long_term_goals = Column(ARRAY(String), nullable=False, default=list)
    
    # Progress tracking
    completed_goals = Column(ARRAY(String), nullable=False, default=list)
    in_progress_goals = Column(ARRAY(String), nullable=False, default=list)
    goal_priorities = Column(JSON, nullable=False, default=dict)
    
    # Metadata
    next_review_date = Column(DateTime, nullable=True)
    
    # Relationships
    farmer = relationship("FarmerDB", back_populates="goals")
    
    # Indexes
    __table_args__ = (
        Index("idx_goals_review", "next_review_date"),
    )
    
    def __repr__(self):
        return f"<FarmerGoalsDB(farmer_id={self.farmer_id}, total_goals={len(self.short_term_goals) + len(self.medium_term_goals) + len(self.long_term_goals)})>"


class SystemEventDB(Base, TimestampMixin):
    """Database model for system events (audit log)."""
    
    __tablename__ = "system_events"
    
    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event information
    event_type = Column(String(100), nullable=False, index=True)
    event_source = Column(String(100), nullable=False)  # 'api', 'worker', 'system'
    event_subtype = Column(String(100), nullable=True)
    
    # Actor information
    user_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)
    farmer_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)
    farm_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)
    
    # Event data
    payload = Column(JSON, nullable=True)
    event_event_metadata = Column(JSON, nullable=True)  # FIXED: Changed from 'metadata' to 'event_metadata'
    
    # Status
    status = Column(String(50), nullable=False, default="success")  # 'success', 'failure', 'warning'
    error_message = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)
    
    # Processing
    processed_at = Column(DateTime, nullable=True)
    processing_duration_ms = Column(Integer, nullable=True)
    
    # Correlation
    correlation_id = Column(String(100), nullable=True, index=True)
    parent_event_id = Column(PG_UUID(as_uuid=True), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_events_type_timestamp", "event_type", "created_at"),
        Index("idx_events_correlation", "correlation_id"),
        Index("idx_events_user_timestamp", "user_id", "created_at"),
    )
    
    def __repr__(self):
        return f"<SystemEventDB(id={self.id}, event_type='{self.event_type}', status='{self.status}')>"
# --- Disease diagnosis models (capstone addition) ----------------
from src.data.database.models_diagnosis import (  # noqa: F401, E402
    Diagnosis,
    FieldValidationImage,
)
