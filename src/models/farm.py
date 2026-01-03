from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict, field_validator


class FarmSizeCategory(str, Enum):
    """Farm size classification."""
    MICRO = "micro"  # < 1 hectare
    SMALLHOLDER = "smallholder"  # 1-5 hectares
    MEDIUM = "medium"  # 5-20 hectares
    COMMERCIAL = "commercial"  # 20-100 hectares
    LARGE_SCALE = "large_scale"  # > 100 hectares


class FarmTypologyType(str, Enum):
    """Farm type classification."""
    SMALLHOLDER = "smallholder"
    COMMERCIAL = "commercial"
    LIVESTOCK = "livestock"
    ORGANIC = "organic"
    MIXED = "mixed"
    AGROFORESTRY = "agroforestry"
    HORTICULTURE = "horticulture"
    AQUACULTURE = "aquaculture"


class SoilType(str, Enum):
    """Soil type classification."""
    SANDY = "sandy"
    CLAY = "clay"
    LOAM = "loam"
    SILT = "silt"
    PEAT = "peat"
    CHALK = "chalk"


class IrrigationType(str, Enum):
    """Irrigation method types."""
    NONE = "none"
    RAINFED = "rainfed"
    SURFACE = "surface"
    SPRINKLER = "sprinkler"
    DRIP = "drip"
    CENTER_PIVOT = "center_pivot"


class FarmIdentity(BaseModel):
    """Basic farm identification information."""
    
    model_config = ConfigDict(from_attributes=True)
    
    farm_id: UUID = Field(default_factory=uuid4, description="Unique farm identifier")
    farmer_id: UUID = Field(description="Owner farmer identifier")
    name: str = Field(..., min_length=1, max_length=100, description="Farm name")
    location: str = Field(..., description="Farm location (address)")
    latitude: float = Field(..., ge=-90, le=90, description="Geographic latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Geographic longitude")
    size_hectares: float = Field(..., gt=0, description="Total farm area in hectares")
    size_category: FarmSizeCategory = Field(..., description="Farm size category")
    established_year: int = Field(..., ge=1900, le=datetime.now().year, description="Year farm was established")
    registration_number: Optional[str] = Field(None, max_length=50, description="Official registration number")
    
    @field_validator("latitude", "longitude")
    @classmethod
    def validate_coordinates(cls, v: float, info) -> float:
        """Validate geographic coordinates."""
        if info.field_name == "latitude" and abs(v) > 90:
            raise ValueError("Latitude must be between -90 and 90")
        if info.field_name == "longitude" and abs(v) > 180:
            raise ValueError("Longitude must be between -180 and 180")
        return round(v, 6)  # Store with 6 decimal places (~10cm precision)


class ResourceConstraints(BaseModel):
    """Resource limitations and constraints for the farm."""
    
    model_config = ConfigDict(from_attributes=True)
    
    capital_access_score: float = Field(..., ge=0, le=100, description="Access to capital (0-100)")
    labor_availability_score: float = Field(..., ge=0, le=100, description="Labor availability (0-100)")
    equipment_access_score: float = Field(..., ge=0, le=100, description="Equipment access (0-100)")
    market_proximity_score: float = Field(..., ge=0, le=100, description="Market proximity (0-100)")
    water_access_score: float = Field(..., ge=0, le=100, description="Water access (0-100)")
    land_availability_score: float = Field(..., ge=0, le=100, description="Land availability (0-100)")
    
    max_capital_per_season: Optional[float] = Field(None, gt=0, description="Maximum capital per season (currency)")
    max_labor_hours_per_day: Optional[int] = Field(None, gt=0, description="Maximum labor hours per day")
    equipment_limitations: List[str] = Field(default_factory=list, description="Equipment limitations")
    water_restrictions: List[str] = Field(default_factory=list, description="Water use restrictions")
    
    @field_validator(
        "capital_access_score",
        "labor_availability_score",
        "equipment_access_score",
        "market_proximity_score",
        "water_access_score",
        "land_availability_score"
    )
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Validate score is within 0-100 range."""
        if not 0 <= v <= 100:
            raise ValueError("Score must be between 0 and 100")
        return round(v, 2)


class EnvironmentalContext(BaseModel):
    """Environmental factors affecting the farm."""
    
    model_config = ConfigDict(from_attributes=True)
    
    soil_type: SoilType = Field(..., description="Primary soil type")
    soil_ph: float = Field(..., ge=0, le=14, description="Soil pH level")
    soil_organic_matter: float = Field(..., ge=0, le=100, description="Soil organic matter percentage")
    elevation_meters: float = Field(..., ge=-400, le=5000, description="Elevation in meters")
    average_rainfall_mm: float = Field(..., ge=0, description="Average annual rainfall in mm")
    average_temperature_c: float = Field(..., ge=-50, le=60, description="Average annual temperature in Celsius")
    growing_season_days: int = Field(..., ge=0, le=365, description="Length of growing season in days")
    irrigation_type: IrrigationType = Field(..., description="Primary irrigation method")
    water_source: Optional[str] = Field(None, description="Primary water source")
    microclimate_zones: List[str] = Field(default_factory=list, description="Microclimate zones within farm")
    topography: List[str] = Field(default_factory=list, description="Topographic features")
    
    @field_validator("soil_ph")
    @classmethod
    def validate_soil_ph(cls, v: float) -> float:
        """Validate soil pH is realistic."""
        if not 3.0 <= v <= 10.0:
            raise ValueError("Soil pH must be between 3.0 and 10.0")
        return round(v, 1)
    
    @field_validator("soil_organic_matter")
    @classmethod
    def validate_organic_matter(cls, v: float) -> float:
        """Validate soil organic matter percentage."""
        if not 0 <= v <= 100:
            raise ValueError("Soil organic matter must be between 0 and 100%")
        return round(v, 2)


class HistoricalPerformance(BaseModel):
    """Historical performance data for the farm."""
    
    model_config = ConfigDict(from_attributes=True)
    
    year: int = Field(..., ge=1900, le=datetime.now().year, description="Year of performance data")
    season: str = Field(..., description="Growing season (e.g., 'main', 'short rains')")
    crops_grown: List[str] = Field(..., description="Crops grown in this season")
    total_yield_kg: float = Field(..., ge=0, description="Total yield in kilograms")
    total_revenue: Optional[float] = Field(None, ge=0, description="Total revenue generated")
    total_cost: Optional[float] = Field(None, ge=0, description="Total costs incurred")
    water_used_m3: Optional[float] = Field(None, ge=0, description="Water used in cubic meters")
    labor_hours: Optional[float] = Field(None, ge=0, description="Labor hours used")
    inputs_used: Dict[str, float] = Field(default_factory=dict, description="Inputs used (type: quantity)")
    challenges_faced: List[str] = Field(default_factory=list, description="Challenges faced during season")
    successes: List[str] = Field(default_factory=list, description="Successes achieved")
    
    @property
    def profit(self) -> Optional[float]:
        """Calculate profit if revenue and cost are available."""
        if self.total_revenue is not None and self.total_cost is not None:
            return self.total_revenue - self.total_cost
        return None
    
    @property
    def yield_per_hectare(self, farm_size: float) -> Optional[float]:
        """Calculate yield per hectare."""
        if farm_size > 0:
            return self.total_yield_kg / farm_size
        return None


class FarmTypology(BaseModel):
    """Farm classification and typology."""
    
    model_config = ConfigDict(from_attributes=True)
    
    typology_type: FarmTypologyType = Field(..., description="Primary farm typology")
    sub_type: Optional[str] = Field(None, description="Sub-type classification")
    specialization: List[str] = Field(..., description="Farm specializations")
    farming_system: str = Field(..., description="Farming system (e.g., monoculture, intercropping)")
    technology_level: str = Field(..., description="Technology adoption level")
    market_orientation: str = Field(..., description="Market orientation (subsistence, commercial, etc.)")
    sustainability_practices: List[str] = Field(default_factory=list, description="Sustainability practices employed")
    certifications: List[str] = Field(default_factory=list, description="Certifications held")
    
    @field_validator("specialization")
    @classmethod
    def validate_specialization(cls, v: List[str]) -> List[str]:
        """Validate specialization list."""
        if not v:
            raise ValueError("At least one specialization must be specified")
        return [s.lower().strip() for s in v]


class FarmDNA(BaseModel):
    """
    Complete farm DNA profile - the core data structure for AFIE.
    Represents the unique 'personality' of each farm.
    """
    
    model_config = ConfigDict(from_attributes=True)
    
    # Core identity
    identity: FarmIdentity = Field(..., description="Farm identification")
    typology: FarmTypology = Field(..., description="Farm typology classification")
    
    # Constraints and capabilities
    constraints: ResourceConstraints = Field(..., description="Resource constraints")
    environment: EnvironmentalContext = Field(..., description="Environmental context")
    
    # Goals and preferences
    goals: Optional[List[str]] = Field(default_factory=list, description="Farmer goals")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Farmer preferences")
    
    # Performance
    historical_performance: List[HistoricalPerformance] = Field(
        default_factory=list,
        description="Historical performance data"
    )
    
    # Metadata
    version: int = Field(default=1, description="DNA version")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    confidence_score: float = Field(default=0.0, ge=0, le=1, description="Data confidence score")
    
    @property
    def farm_id(self) -> UUID:
        """Get farm ID from identity."""
        return self.identity.farm_id
    
    @property
    def average_yield(self) -> Optional[float]:
        """Calculate average historical yield."""
        if not self.historical_performance:
            return None
        
        total_yield = sum(perf.total_yield_kg for perf in self.historical_performance)
        return total_yield / len(self.historical_performance)
    
    @property
    def constraint_score(self) -> float:
        """Calculate overall constraint score (lower = more constrained)."""
        scores = [
            self.constraints.capital_access_score,
            self.constraints.labor_availability_score,
            self.constraints.equipment_access_score,
            self.constraints.market_proximity_score,
            self.constraints.water_access_score,
            self.constraints.land_availability_score,
        ]
        return sum(scores) / len(scores) / 100.0  # Normalize to 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FarmDNA":
        """Create instance from dictionary."""
        return cls(**data)