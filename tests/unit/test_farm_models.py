"""
Unit tests for farm models.
"""
import pytest
from datetime import datetime
from uuid import uuid4

from src.models.farm import (
    FarmIdentity, FarmTypology, ResourceConstraints, 
    EnvironmentalContext, HistoricalPerformance, FarmDNA
)
from src.models.farmer import FarmerProfile, BehavioralProfile


class TestFarmIdentity:
    """Test FarmIdentity model."""
    
    def test_create_valid_farm_identity(self):
        """Test creating a valid farm identity."""
        identity = FarmIdentity(
            farmer_id=uuid4(),
            name="Green Valley Farm",
            location="Nakuru, Kenya",
            latitude=-0.303099,
            longitude=36.080000,
            size_hectares=5.0,
            size_category="smallholder",
            established_year=2010
        )
        
        assert identity.name == "Green Valley Farm"
        assert identity.location == "Nakuru, Kenya"
        assert identity.latitude == -0.303099
        assert identity.longitude == 36.080000
        assert identity.size_hectares == 5.0
        assert identity.size_category == "smallholder"
        assert identity.established_year == 2010
    
    def test_invalid_coordinates(self):
        """Test validation of invalid coordinates."""
        with pytest.raises(ValueError):
            FarmIdentity(
                farmer_id=uuid4(),
                name="Test Farm",
                location="Test",
                latitude=100,  # Invalid latitude
                longitude=0,
                size_hectares=1.0,
                size_category="smallholder",
                established_year=2020
            )
    
    def test_negative_farm_size(self):
        """Test validation of negative farm size."""
        with pytest.raises(ValueError):
            FarmIdentity(
                farmer_id=uuid4(),
                name="Test Farm",
                location="Test",
                latitude=0,
                longitude=0,
                size_hectares=-1.0,  # Invalid size
                size_category="smallholder",
                established_year=2020
            )


class TestResourceConstraints:
    """Test ResourceConstraints model."""
    
    def test_create_valid_constraints(self):
        """Test creating valid resource constraints."""
        constraints = ResourceConstraints(
            capital_access_score=75.5,
            labor_availability_score=60.0,
            equipment_access_score=40.0,
            market_proximity_score=90.0,
            water_access_score=80.0,
            land_availability_score=50.0,
            max_capital_per_season=50000.0,
            max_labor_hours_per_day=8
        )
        
        assert constraints.capital_access_score == 75.5
        assert constraints.labor_availability_score == 60.0
        assert constraints.equipment_access_score == 40.0
        assert constraints.market_proximity_score == 90.0
        assert constraints.water_access_score == 80.0
        assert constraints.land_availability_score == 50.0
        assert constraints.max_capital_per_season == 50000.0
        assert constraints.max_labor_hours_per_day == 8
    
    def test_invalid_score_range(self):
        """Test validation of score ranges."""
        with pytest.raises(ValueError):
            ResourceConstraints(
                capital_access_score=150.0,  # Invalid score
                labor_availability_score=60.0,
                equipment_access_score=40.0,
                market_proximity_score=90.0,
                water_access_score=80.0,
                land_availability_score=50.0
            )


class TestFarmDNA:
    """Test FarmDNA model."""
    
    @pytest.fixture
    def sample_farm_dna(self):
        """Create a sample FarmDNA for testing."""
        identity = FarmIdentity(
            farmer_id=uuid4(),
            name="Test Farm",
            location="Test Location",
            latitude=0.0,
            longitude=0.0,
            size_hectares=10.0,
            size_category="medium",
            established_year=2015
        )
        
        typology = FarmTypology(
            typology_type="commercial",
            specialization=["maize", "beans"],
            farming_system="monoculture",
            technology_level="intermediate",
            market_orientation="commercial"
        )
        
        constraints = ResourceConstraints(
            capital_access_score=75.0,
            labor_availability_score=60.0,
            equipment_access_score=40.0,
            market_proximity_score=90.0,
            water_access_score=80.0,
            land_availability_score=50.0
        )
        
        environment = EnvironmentalContext(
            soil_type="loam",
            soil_ph=6.5,
            soil_organic_matter=2.5,
            elevation_meters=1500.0,
            average_rainfall_mm=800.0,
            average_temperature_c=22.0,
            growing_season_days=180,
            irrigation_type="rainfed"
        )
        
        return FarmDNA(
            identity=identity,
            typology=typology,
            constraints=constraints,
            environment=environment,
            goals=["Increase yield", "Reduce costs"],
            preferences={"language": "en", "notifications": True}
        )
    
    def test_create_farm_dna(self, sample_farm_dna):
        """Test creating FarmDNA."""
        assert sample_farm_dna.identity.name == "Test Farm"
        assert sample_farm_dna.typology.typology_type == "commercial"
        assert sample_farm_dna.constraints.capital_access_score == 75.0
        assert sample_farm_dna.environment.soil_ph == 6.5
        assert sample_farm_dna.goals == ["Increase yield", "Reduce costs"]
        assert sample_farm_dna.confidence_score == 0.0
    
    def test_constraint_score_calculation(self, sample_farm_dna):
        """Test constraint score calculation."""
        # Calculate expected score
        scores = [
            sample_farm_dna.constraints.capital_access_score,
            sample_farm_dna.constraints.labor_availability_score,
            sample_farm_dna.constraints.equipment_access_score,
            sample_farm_dna.constraints.market_proximity_score,
            sample_farm_dna.constraints.water_access_score,
            sample_farm_dna.constraints.land_availability_score,
        ]
        expected = sum(scores) / len(scores) / 100.0
        
        assert sample_farm_dna.constraint_score == pytest.approx(expected)
    
    def test_to_from_dict(self, sample_farm_dna):
        """Test conversion to and from dictionary."""
        # Convert to dict
        farm_dict = sample_farm_dna.to_dict()
        
        # Convert back from dict
        farm_from_dict = FarmDNA.from_dict(farm_dict)
        
        # Should be equal
        assert farm_from_dict.identity.name == sample_farm_dna.identity.name
        assert farm_from_dict.typology.typology_type == sample_farm_dna.typology.typology_type
        assert farm_from_dict.constraint_score == sample_farm_dna.constraint_score