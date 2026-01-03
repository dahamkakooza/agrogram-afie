"""
FarmDNA Database Utilities - Convert between FarmDNA and FarmDB models
"""
from typing import Optional
import uuid
from datetime import datetime

from src.models.farm import (
    FarmDNA, FarmIdentity, FarmTypology, ResourceConstraints, EnvironmentalContext,
    FarmSizeCategory, FarmTypologyType, SoilType, IrrigationType
)
from src.data.database.models import FarmDB


def farm_dna_to_farm_db(farm_dna: FarmDNA, user_id: Optional[uuid.UUID] = None) -> FarmDB:
    """
    Convert FarmDNA Pydantic model to FarmDB SQLAlchemy model.
    
    Args:
        farm_dna: FarmDNA object
        user_id: Optional user ID (will use farmer_id if not provided)
        
    Returns:
        FarmDB object ready for database insertion
    """
    # Ensure we have valid UUIDs
    farmer_id = farm_dna.identity.farmer_id
    if isinstance(farmer_id, str):
        farmer_id = uuid.UUID(farmer_id)
    
    farm_id = farm_dna.identity.farm_id
    if isinstance(farm_id, str):
        farm_id = uuid.UUID(farm_id)
    
    # Get user ID - use farmer_id if not provided
    if user_id is None:
        user_id = farmer_id
    
    return FarmDB(
        # Use farm_id as primary key
        id=farm_id,
        
        # Foreign keys
        farmer_id=farmer_id,
        user_id=user_id,
        
        # Basic information from FarmIdentity
        name=farm_dna.identity.name,
        location=farm_dna.identity.location,
        latitude=float(farm_dna.identity.latitude),
        longitude=float(farm_dna.identity.longitude),
        size_hectares=farm_dna.identity.size_hectares,
        size_category=farm_dna.identity.size_category.value,
        established_year=farm_dna.identity.established_year,
        registration_number=farm_dna.identity.registration_number,
        
        # Typology information
        typology_type=farm_dna.typology.typology_type.value,
        sub_type=farm_dna.typology.sub_type,
        specialization=farm_dna.typology.specialization,
        farming_system=farm_dna.typology.farming_system,
        technology_level=farm_dna.typology.technology_level,
        market_orientation=farm_dna.typology.market_orientation,
        sustainability_practices=farm_dna.typology.sustainability_practices,
        certifications=farm_dna.typology.certifications,
        
        # Resource constraints
        capital_access_score=farm_dna.constraints.capital_access_score,
        labor_availability_score=farm_dna.constraints.labor_availability_score,
        equipment_access_score=farm_dna.constraints.equipment_access_score,
        market_proximity_score=farm_dna.constraints.market_proximity_score,
        water_access_score=farm_dna.constraints.water_access_score,
        land_availability_score=farm_dna.constraints.land_availability_score,
        max_capital_per_season=farm_dna.constraints.max_capital_per_season,
        max_labor_hours_per_day=farm_dna.constraints.max_labor_hours_per_day,
        equipment_limitations=farm_dna.constraints.equipment_limitations,
        water_restrictions=farm_dna.constraints.water_restrictions,
        
        # Environmental context
        soil_type=farm_dna.environment.soil_type.value,
        soil_ph=farm_dna.environment.soil_ph,
        soil_organic_matter=farm_dna.environment.soil_organic_matter,
        elevation_meters=farm_dna.environment.elevation_meters,
        average_rainfall_mm=farm_dna.environment.average_rainfall_mm,
        average_temperature_c=farm_dna.environment.average_temperature_c,
        growing_season_days=farm_dna.environment.growing_season_days,
        irrigation_type=farm_dna.environment.irrigation_type.value,
        water_source=farm_dna.environment.water_source,
        microclimate_zones=farm_dna.environment.microclimate_zones,
        topography=farm_dna.environment.topography,
        
        # Goals and preferences
        goals=farm_dna.goals,
        preferences=farm_dna.preferences or {},
        
        # Metadata
        confidence_score=farm_dna.confidence_score,
        dna_version=farm_dna.version,
        is_active=True,
        last_profile_update=datetime.utcnow()
    )


def farm_db_to_farm_dna(farm_db: FarmDB) -> FarmDNA:
    """
    Convert FarmDB SQLAlchemy model to FarmDNA Pydantic model.
    
    Args:
        farm_db: FarmDB object from database
        
    Returns:
        FarmDNA object
    """
    # Create FarmIdentity
    identity = FarmIdentity(
        farm_id=farm_db.id,
        farmer_id=farm_db.farmer_id,
        name=farm_db.name,
        location=farm_db.location,
        latitude=float(farm_db.latitude),
        longitude=float(farm_db.longitude),
        size_hectares=farm_db.size_hectares,
        size_category=FarmSizeCategory(farm_db.size_category),
        established_year=farm_db.established_year,
        registration_number=farm_db.registration_number
    )
    
    # Create FarmTypology
    typology = FarmTypology(
        typology_type=FarmTypologyType(farm_db.typology_type),
        sub_type=farm_db.sub_type,
        specialization=farm_db.specialization,
        farming_system=farm_db.farming_system,
        technology_level=farm_db.technology_level,
        market_orientation=farm_db.market_orientation,
        sustainability_practices=farm_db.sustainability_practices,
        certifications=farm_db.certifications
    )
    
    # Create ResourceConstraints
    constraints = ResourceConstraints(
        capital_access_score=farm_db.capital_access_score,
        labor_availability_score=farm_db.labor_availability_score,
        equipment_access_score=farm_db.equipment_access_score,
        market_proximity_score=farm_db.market_proximity_score,
        water_access_score=farm_db.water_access_score,
        land_availability_score=farm_db.land_availability_score,
        max_capital_per_season=farm_db.max_capital_per_season,
        max_labor_hours_per_day=farm_db.max_labor_hours_per_day,
        equipment_limitations=farm_db.equipment_limitations,
        water_restrictions=farm_db.water_restrictions
    )
    
    # Create EnvironmentalContext
    environment = EnvironmentalContext(
        soil_type=SoilType(farm_db.soil_type),
        soil_ph=farm_db.soil_ph,
        soil_organic_matter=farm_db.soil_organic_matter,
        elevation_meters=farm_db.elevation_meters,
        average_rainfall_mm=farm_db.average_rainfall_mm,
        average_temperature_c=farm_db.average_temperature_c,
        growing_season_days=farm_db.growing_season_days,
        irrigation_type=IrrigationType(farm_db.irrigation_type),
        water_source=farm_db.water_source,
        microclimate_zones=farm_db.microclimate_zones,
        topography=farm_db.topography
    )
    
    return FarmDNA(
        identity=identity,
        typology=typology,
        constraints=constraints,
        environment=environment,
        goals=farm_db.goals,
        preferences=farm_db.preferences or {},
        historical_performance=[],  # Would need separate query for this
        version=farm_db.dna_version,
        created_at=farm_db.created_at,
        updated_at=farm_db.updated_at,
        confidence_score=farm_db.confidence_score
    )


async def save_farm_to_database(farm_dna: FarmDNA, db_session, user_id: Optional[uuid.UUID] = None) -> FarmDB:
    """
    Save FarmDNA to database.
    
    Args:
        farm_dna: FarmDNA object to save
        db_session: SQLAlchemy async session
        user_id: Optional user ID
        
    Returns:
        Saved FarmDB object
    """
    # Convert to database model
    farm_db = farm_dna_to_farm_db(farm_dna, user_id)
    
    # Check if farm already exists
    existing_farm = await db_session.get(FarmDB, farm_db.id)
    
    if existing_farm:
        # Update existing farm
        for key, value in farm_db.__dict__.items():
            if not key.startswith('_') and key != 'id':
                setattr(existing_farm, key, value)
        farm_db = existing_farm
        farm_db.updated_at = datetime.utcnow()
        print(f"Updated existing farm in database: {farm_db.name}")
    else:
        # Add new farm
        db_session.add(farm_db)
        print(f"Saved new farm to database: {farm_db.name}")
    
    await db_session.commit()
    await db_session.refresh(farm_db)
    
    return farm_db


async def get_farm_from_database(farm_id: uuid.UUID, db_session) -> Optional[FarmDNA]:
    """
    Get FarmDNA from database.
    
    Args:
        farm_id: Farm ID to retrieve
        db_session: SQLAlchemy async session
        
    Returns:
        FarmDNA object or None if not found
    """
    try:
        # Use query.get() instead of await session.get() for SQLAlchemy 2.0
        farm_db = db_session.get(FarmDB, farm_id)
        
        if not farm_db:
            print(f"Farm not found in database: {farm_id}")
            return None
        
        print(f"Found farm in database: {farm_db.name}")
        return farm_db_to_farm_dna(farm_db)
    except Exception as e:
        print(f"Error fetching farm from database: {e}")
        return None
