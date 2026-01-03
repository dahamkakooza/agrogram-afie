"""
Farm Intelligence Routes - Week 1-2 Implementation
Onboarding and Farm Profiler endpoints
"""
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.data.collectors.onboarding_collector import onboarding_collector
from src.core.farm_intelligence.farm_profiler import farm_profiler
from src.data.database.base import get_db
from src.data.utils.farm_dna_db import save_farm_to_database

# Create router
router = APIRouter(prefix="/farm-intelligence", tags=["farm-intelligence"])


# Request/Response models
class OnboardingRequest(BaseModel):
    """Onboarding data request model."""
    primary_language: str
    farm_size_category: str
    main_crop_animal: str
    biggest_challenge: str
    mobile_literacy: str
    farmer_name: str = "Farmer"
    location: str = "Unknown"


class OnboardingResponse(BaseModel):
    """Onboarding response model."""
    status: str
    onboarding_id: str
    farmer_profile: Dict[str, Any]
    gamification_triggers: Dict[str, Any]
    farm_dna: Dict[str, Any]
    next_steps: List[Dict[str, Any]]


class ProfilerTestRequest(BaseModel):
    """Farm profiler test request model."""
    primary_language: str
    farm_size_category: str
    main_crop_animal: str
    biggest_challenge: str
    mobile_literacy: str
    farmer_id: str = None
    farm_name: str = "Test Farm"
    location: str = "Test Location"


@router.post("/onboarding", response_model=OnboardingResponse, status_code=201)
async def process_onboarding(request: OnboardingRequest):
    """
    Process onboarding data (5 points) and create initial farm profile.
    
    Expected data:
    - primary_language: "en", "sw", "fr", "lg"
    - farm_size_category: "micro", "smallholder", "medium", "commercial", "large_scale"
    - main_crop_animal: "maize", "beans", "coffee", "banana", "cattle", "goats", "poultry"
    - biggest_challenge: "water_access", "capital_access", "market_access", "labor_shortage", "pest_disease", "soil_fertility"
    - mobile_literacy: "none", "basic", "intermediate", "advanced"
    """
    try:
        print(f"🎯 Processing onboarding for {request.farmer_name}")
        
        # Convert to dict
        onboarding_data = request.dict()
        
        # 1. Process onboarding data
        processed_data = onboarding_collector.collect_onboarding_data(onboarding_data)
        
        # 2. Extract minimal data for farm profiler
        minimal_data = onboarding_collector.get_minimal_data_for_farm_profiler(processed_data)
        
        # 3. Create FarmDNA from minimal data using ML-based profiler
        farm_dna = farm_profiler.create_farm_dna_from_minimal(minimal_data)
        
        # 4. Save FarmDNA to database
        db = next(get_db())
        await save_farm_to_database(farm_dna, db)
        print(f"✓ Farm saved to database with ID: {farm_dna.identity.farm_id}")
        
        return {
            "status": "success",
            "onboarding_id": processed_data["onboarding_id"],
            "farmer_profile": processed_data["farmer_profile"],
            "gamification_triggers": processed_data["gamification_triggers"],
            "farm_dna": farm_dna.dict(),
            "next_steps": processed_data["next_steps"]
        }
        
    except Exception as e:
        print(f"❌ Onboarding error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Onboarding processing failed: {str(e)}"
        )


@router.post("/profiler-test", response_model=Dict[str, Any])
async def test_farm_profiler(request: ProfilerTestRequest):
    """Test the farm profiler with sample data."""
    try:
        print("🧪 Testing farm profiler...")
        
        # Convert to dict
        test_data = request.dict()
        
        # Create FarmDNA using ML-based profiler
        farm_dna = farm_profiler.create_farm_dna_from_minimal(test_data)
        
        return {
            "status": "success",
            "farm_dna": farm_dna.dict(),
            "profiler_version": "1.0",
            "confidence_score": farm_dna.confidence_score,
            "farm_type": farm_dna.typology.typology_type.value,
            "water_score": farm_dna.constraints.water_access_score,
            "capital_score": farm_dna.constraints.capital_access_score,
            "ml_model_used": "RandomForestClassifier"
        }
        
    except Exception as e:
        print(f"❌ Profiler test error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Farm profiler test failed: {str(e)}"
        )


@router.get("/status")
async def farm_intelligence_status():
    """Get farm intelligence system status."""
    return {
        "system": "farm_intelligence",
        "version": "1.0",
        "status": "operational",
        "components": {
            "farm_profiler": "active (ML-based)",
            "onboarding_collector": "active",
            "typology_classifier": "active (RandomForest)",
            "constraint_analyzer": "active",
            "context_builder": "active",
            "ml_engine": "scikit-learn 1.8.0"
        },
        "phase": "Week 1-2: Farm Profiler System"
    }
