"""
Integration tests for database operations.
"""
import pytest
import asyncio
from datetime import datetime
from uuid import uuid4

from src.data.database.base import Base, engine, SessionLocal
from src.data.database.models import FarmDB, FarmerDB
from src.data.database.repositories.farm_repo import FarmRepository
from src.models.farm import FarmDNA, FarmIdentity, FarmTypology, ResourceConstraints, EnvironmentalContext


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def db_session():
    """Create database session for tests."""
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async with SessionLocal() as session:
        yield session
    
    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def sample_farm_data():
    """Create sample farm data for testing."""
    farmer_id = uuid4()
    
    farm_dna = FarmDNA(
        identity=FarmIdentity(
            farm_id=uuid4(),
            farmer_id=farmer_id,
            name="Integration Test Farm",
            location="Test Location",
            latitude=1.0,
            longitude=1.0,
            size_hectares=15.0,
            size_category="medium",
            established_year=2018
        ),
        typology=FarmTypology(
            typology_type="commercial",
            specialization=["wheat", "barley"],
            farming_system="monoculture",
            technology_level="advanced",
            market_orientation="export"
        ),
        constraints=ResourceConstraints(
            capital_access_score=80.0,
            labor_availability_score=70.0,
            equipment_access_score=90.0,
            market_proximity_score=60.0,
            water_access_score=75.0,
            land_availability_score=85.0
        ),
        environment=EnvironmentalContext(
            soil_type="clay",
            soil_ph=7.0,
            soil_organic_matter=3.0,
            elevation_meters=1200.0,
            average_rainfall_mm=600.0,
            average_temperature_c=18.0,
            growing_season_days=210,
            irrigation_type="sprinkler"
        )
    )
    
    return farm_dna


class TestFarmRepository:
    """Test FarmRepository integration."""
    
    @pytest.mark.asyncio
    async def test_create_farm(self, db_session, sample_farm_data):
        """Test creating a farm in database."""
        repo = FarmRepository(db_session)
        
        # Create farm
        created_farm = await repo.create_farm(sample_farm_data.to_dict())
        
        assert created_farm["identity"]["farm_id"] == sample_farm_data.identity.farm_id
        assert created_farm["identity"]["name"] == "Integration Test Farm"
        assert created_farm["typology"]["typology_type"] == "commercial"
        assert created_farm["constraints"]["capital_access_score"] == 80.0
        assert created_farm["environment"]["soil_type"] == "clay"
    
    @pytest.mark.asyncio
    async def test_get_farm_by_id(self, db_session, sample_farm_data):
        """Test retrieving farm by ID."""
        repo = FarmRepository(db_session)
        farm_id = sample_farm_data.identity.farm_id
        
        # Get farm
        farm = await repo.get_farm_by_id(farm_id)
        
        assert farm is not None
        assert farm["identity"]["farm_id"] == farm_id
        assert farm["identity"]["name"] == "Integration Test Farm"
    
    @pytest.mark.asyncio
    async def test_update_farm(self, db_session, sample_farm_data):
        """Test updating farm data."""
        repo = FarmRepository(db_session)
        farm_id = sample_farm_data.identity.farm_id
        
        # Update farm
        update_data = sample_farm_data.to_dict()
        update_data["goals"] = ["Increase yield by 20%", "Reduce water usage"]
        update_data["confidence_score"] = 0.8
        
        updated_farm = await repo.update_farm(farm_id, update_data)
        
        assert updated_farm["goals"] == ["Increase yield by 20%", "Reduce water usage"]
        assert updated_farm["confidence_score"] == 0.8
    
    @pytest.mark.asyncio
    async def test_list_farms(self, db_session, sample_farm_data):
        """Test listing farms."""
        repo = FarmRepository(db_session)
        farmer_id = sample_farm_data.identity.farmer_id
        
        # List farms for farmer
        farms = await repo.list_farms(farmer_id=farmer_id)
        
        assert len(farms) > 0
        assert farms[0]["identity"]["farmer_id"] == farmer_id
    
    @pytest.mark.asyncio
    async def test_add_performance_record(self, db_session, sample_farm_data):
        """Test adding performance record."""
        repo = FarmRepository(db_session)
        farm_id = sample_farm_data.identity.farm_id
        
        # Add performance record
        performance_data = {
            "year": 2023,
            "season": "main",
            "crops_grown": ["wheat", "barley"],
            "total_yield_kg": 5000.0,
            "total_revenue": 10000.0,
            "total_cost": 6000.0,
            "water_used_m3": 2000.0,
            "labor_hours": 400.0,
            "inputs_used": {"fertilizer": 100.0, "pesticide": 10.0},
            "challenges_faced": ["drought", "pests"],
            "successes": ["good yield", "quality product"]
        }
        
        record = await repo.add_performance_record(farm_id, performance_data)
        
        assert record["year"] == 2023
        assert record["season"] == "main"
        assert record["total_yield_kg"] == 5000.0
        assert record["profit"] == 4000.0  # revenue - cost