"""
Recommendation API Router - Expose recommendation engine endpoints
Week 3-4 Implementation - Based on AFIE Blueprint
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

# Import our core components
from src.core.recommendation_engine import (
    RecommendationEngine, RecommendationContext, RecommendationConstraints,
    recommendation_engine
)
from src.models.recommendation import (
    Recommendation, RecommendationType, PriorityLevel,
    RecommendationBatch
)
from src.models.farm import FarmDNA, FarmIdentity, FarmTypology, ResourceConstraints, EnvironmentalContext
from src.data.collectors.onboarding_collector import onboarding_collector

# Create router
router = APIRouter(prefix="/recommendations", tags=["recommendations"])


# ===== Request/Response Models =====

class RecommendationContextModel(BaseModel):
    """Context for recommendation generation."""
    weather: Dict[str, Any] = Field(default_factory=dict, description="Current weather conditions")
    market_prices: Dict[str, float] = Field(default_factory=dict, description="Current market prices")
    season: str = Field("growing", description="Current season (planting, growing, harvest, fallow)")
    current_activities: List[str] = Field(default_factory=list, description="Current farm activities")
    upcoming_events: List[Dict[str, Any]] = Field(default_factory=list, description="Upcoming farm events")
    farmer_feedback: Dict[str, Any] = Field(default_factory=dict, description="Recent farmer feedback")


class RecommendationConstraintsModel(BaseModel):
    """Constraints for recommendation generation."""
    max_recommendations: int = Field(5, ge=1, le=20, description="Maximum number of recommendations")
    max_cost: Optional[float] = Field(None, ge=0, description="Maximum cost constraint")
    max_labor_hours: Optional[float] = Field(None, ge=0, description="Maximum labor hours")
    time_horizon_days: int = Field(30, ge=1, le=365, description="Time horizon for recommendations")
    priority_threshold: PriorityLevel = Field(PriorityLevel.LOW, description="Minimum priority level")
    farmer_preferences: Dict[str, Any] = Field(default_factory=dict, description="Farmer preferences")


class GenerateRecommendationsRequest(BaseModel):
    """Request to generate recommendations."""
    farm_dna_id: str = Field(..., description="ID of the farm DNA profile")
    context: RecommendationContextModel = Field(default_factory=RecommendationContextModel)
    constraints: RecommendationConstraintsModel = Field(default_factory=RecommendationConstraintsModel)
    
    class Config:
        json_schema_extra = {
            "example": {
                "farm_dna_id": "farm_123",
                "context": {
                    "weather": {"temperature": 25, "forecast": "clear"},
                    "market_prices": {"maize": 120.0},
                    "season": "growing",
                    "current_activities": ["weeding", "irrigation"]
                },
                "constraints": {
                    "max_recommendations": 3,
                    "max_cost": 500.0,
                    "max_labor_hours": 40.0,
                    "time_horizon_days": 14
                }
            }
        }


class RecommendationResponse(BaseModel):
    """Single recommendation response."""
    id: str
    type: str
    title: str
    description: str
    priority: str
    confidence_score: float
    total_cost: float
    total_labor: float
    estimated_value: float
    roi_estimate: Optional[float]
    explanation: str
    actions: List[Dict[str, Any]]
    valid_until: Optional[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "rec_123",
                "type": "operational",
                "title": "Harvest before heavy rain",
                "description": "Heavy rain forecasted. Harvest crops now to prevent damage.",
                "priority": "critical",
                "confidence_score": 0.85,
                "total_cost": 150.0,
                "total_labor": 8.5,
                "estimated_value": 450.0,
                "roi_estimate": 200.0,
                "explanation": "Recommended because time-sensitive action required",
                "actions": [
                    {
                        "step_number": 1,
                        "description": "Check crop maturity",
                        "estimated_duration_minutes": 15
                    }
                ],
                "valid_until": "2024-01-15T18:00:00Z"
            }
        }


class GenerateRecommendationsResponse(BaseModel):
    """Response containing generated recommendations."""
    batch_id: str
    farm_dna_id: str
    generated_at: str
    themes: List[str]
    seasonal_focus: str
    total_recommendations: int
    recommendations: List[RecommendationResponse]
    summary: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "batch_123",
                "farm_dna_id": "farm_123",
                "generated_at": "2024-01-10T10:30:00Z",
                "themes": ["water_management", "market_strategies"],
                "seasonal_focus": "growing",
                "total_recommendations": 3,
                "recommendations": [],
                "summary": {
                    "by_type": {"operational": 2, "strategic": 1},
                    "priority_distribution": {"critical": 1, "high": 2},
                    "total_estimated_value": 1250.0
                }
            }
        }


class RecommendationFeedbackRequest(BaseModel):
    """Feedback on a recommendation."""
    accepted: bool = Field(..., description="Whether recommendation was accepted")
    implemented: bool = Field(False, description="Whether recommendation was implemented")
    farmer_rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5 stars")
    notes: Optional[str] = Field(None, description="Additional feedback notes")


class OnboardingWithRecommendationsRequest(BaseModel):
    """Onboarding request that also generates initial recommendations."""
    primary_language: str
    farm_size_category: str
    main_crop_animal: str
    biggest_challenge: str
    mobile_literacy: str
    farmer_name: Optional[str] = None
    location: Optional[str] = None


# ===== Helper Functions =====

async def get_farm_dna(farm_dna_id: str) -> Optional[FarmDNA]:
    """Get farm DNA by ID - fetches from database or creates fallback."""
    try:
        # First try database lookup
        try:
            from src.data.database.base import get_db
            from src.data.database.models import FarmDB
            from src.models.farm import FarmSizeCategory, FarmTypologyType, SoilType, IrrigationType
            from sqlalchemy import select
            
            # Convert to UUID
            try:
                farm_id = uuid.UUID(farm_dna_id) if isinstance(farm_dna_id, str) else farm_dna_id
            except ValueError:
                farm_id = uuid.uuid5(uuid.NAMESPACE_DNS, farm_dna_id)
            
            # Get database session
            db_gen = get_db()
            session = await db_gen.__anext__()
            
            # Query database
            stmt = select(FarmDB).where(FarmDB.id == farm_id)
            result = await session.execute(stmt)
            farm_db = result.scalar_one_or_none()
            
            if farm_db:
                print(f"✅ FOUND farm in database: {farm_db.name}")
                
                # Create FarmDNA from database record
                identity = FarmIdentity(
                    farm_id=str(farm_db.id),
                    farmer_id=farm_db.farmer_id,
                    name=farm_db.name,
                    location=farm_db.location,
                    latitude=float(farm_db.latitude) if farm_db.latitude else 0.0,
                    longitude=float(farm_db.longitude) if farm_db.longitude else 0.0,
                    size_hectares=farm_db.size_hectares,
                    size_category=FarmSizeCategory(farm_db.size_category) if farm_db.size_category else FarmSizeCategory.SMALLHOLDER,
                    established_year=farm_db.established_year,
                    registration_number=farm_db.registration_number
                )
                
                typology = FarmTypology(
                    typology_type=FarmTypologyType(farm_db.typology_type) if farm_db.typology_type else FarmTypologyType.SMALLHOLDER,
                    sub_type=farm_db.sub_type,
                    specialization=farm_db.specialization,
                    farming_system=farm_db.farming_system,
                    technology_level=farm_db.technology_level,
                    market_orientation=farm_db.market_orientation,
                    sustainability_practices=farm_db.sustainability_practices,
                    certifications=farm_db.certifications
                )
                
                constraints = ResourceConstraints(
                    capital_access_score=farm_db.capital_access_score or 40.0,
                    labor_availability_score=farm_db.labor_availability_score or 50.0,
                    equipment_access_score=farm_db.equipment_access_score or 30.0,
                    market_proximity_score=farm_db.market_proximity_score or 70.0,
                    water_access_score=farm_db.water_access_score or 30.0,
                    land_availability_score=farm_db.land_availability_score or 60.0,
                    max_capital_per_season=farm_db.max_capital_per_season,
                    max_labor_hours_per_day=farm_db.max_labor_hours_per_day,
                    equipment_limitations=farm_db.equipment_limitations,
                    water_restrictions=farm_db.water_restrictions
                )
                
                environment = EnvironmentalContext(
                    soil_type=SoilType(farm_db.soil_type) if farm_db.soil_type else SoilType.LOAM,
                    soil_ph=farm_db.soil_ph or 6.5,
                    soil_organic_matter=farm_db.soil_organic_matter or 2.5,
                    elevation_meters=farm_db.elevation_meters,
                    average_rainfall_mm=farm_db.average_rainfall_mm or 1000.0,
                    average_temperature_c=farm_db.average_temperature_c or 25.0,
                    growing_season_days=farm_db.growing_season_days or 120,
                    irrigation_type=IrrigationType(farm_db.irrigation_type) if farm_db.irrigation_type else IrrigationType.RAINFED,
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
                    confidence_score=farm_db.confidence_score or 0.7,
                    version=farm_db.dna_version or "1.0"
                )
                
        except Exception as db_error:
            print(f"⚠️ Database error: {type(db_error).__name__}: {str(db_error)[:100]}...")
        
        # Fallback to farm profiler
        from src.core.farm_intelligence.farm_profiler import farm_profiler
        
        minimal_data = {
            'primary_language': 'en',
            'farm_size_category': 'smallholder',
            'main_crop_animal': 'maize',
            'biggest_challenge': 'water_access',
            'mobile_literacy': 'intermediate',
            'farmer_name': 'Unknown',
            'location': 'Unknown',
            'farm_name': f'Farm {farm_dna_id}'
        }
        
        farm_dna = farm_profiler.create_farm_dna_from_minimal(minimal_data)
        farm_dna.identity.farm_id = str(farm_id)
        print(f"🔄 Created fallback FarmDNA for: {farm_dna_id}")
        return farm_dna
        
    except Exception as e:
        print(f"❌ ERROR in get_farm_dna: {type(e).__name__}: {e}")
        # Ultimate fallback
        from src.core.farm_intelligence.farm_profiler import farm_profiler
        
        minimal_data = {
            'primary_language': 'en',
            'farm_size_category': 'smallholder',
            'main_crop_animal': 'maize',
            'biggest_challenge': 'water_access',
            'mobile_literacy': 'intermediate',
            'farmer_name': 'Fallback Farmer',
            'location': 'Fallback Location',
            'farm_name': 'Fallback Farm'
        }
        
        farm_dna = farm_profiler.create_farm_dna_from_minimal(minimal_data)
        farm_dna.identity.farm_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, farm_dna_id))
        return farm_dna


# ===== API Endpoints =====

@router.post("/generate", response_model=GenerateRecommendationsResponse)
async def generate_recommendations(
    request: GenerateRecommendationsRequest
):
    """
    Generate personalized farming recommendations.
    
    This endpoint generates recommendations based on:
    - Farm DNA profile (constraints, typology, environment)
    - Current context (weather, market, season)
    - Generation constraints (cost, labor, preferences)
    
    Returns a batch of ranked recommendations with scores and explanations.
    """
    try:
        print(f"🎯 Generating recommendations for farm: {request.farm_dna_id}")
        
        # Get farm DNA
        farm_dna = await get_farm_dna(request.farm_dna_id)
        if not farm_dna:
            raise HTTPException(status_code=404, detail="Farm DNA not found")
        
        # Convert context to engine format
        context = RecommendationContext()
        context.weather = request.context.weather
        context.market_prices = request.context.market_prices
        context.season = request.context.season
        context.current_activities = request.context.current_activities
        context.upcoming_events = request.context.upcoming_events
        context.farmer_feedback = request.context.farmer_feedback
        
        # Convert constraints to engine format
        constraints = RecommendationConstraints(
            max_recommendations=request.constraints.max_recommendations,
            max_cost=request.constraints.max_cost,
            max_labor_hours=request.constraints.max_labor_hours,
            time_horizon_days=request.constraints.time_horizon_days,
            priority_threshold=request.constraints.priority_threshold,
            farmer_preferences=request.constraints.farmer_preferences
        )
        
        # Generate recommendations
        batch = recommendation_engine.generate_recommendations(
            farm_dna, context, constraints
        )
        
        # Convert recommendations to response format
        recommendation_responses = []
        for rec in batch.get_priority_sorted():
            rec_dict = rec.to_dict()
            
            # Create simplified action list
            actions = []
            for action in rec_dict.get("actions", []):
                actions.append({
                    "step_number": action["step_number"],
                    "description": action["description"],
                    "estimated_duration_minutes": action.get("estimated_duration_minutes")
                })
            
            recommendation_responses.append(RecommendationResponse(
                id=rec_dict["id"],
                type=rec_dict["type"],
                title=rec_dict["title"],
                description=rec_dict["description"],
                priority=rec_dict["scoring"]["priority"],
                confidence_score=rec_dict["scoring"]["confidence_score"],
                total_cost=rec_dict["scoring"]["total_cost"],
                total_labor=rec_dict["scoring"]["total_labor"],
                estimated_value=rec_dict["scoring"]["estimated_value"],
                roi_estimate=rec_dict["scoring"]["roi_estimate"],
                explanation=rec_dict["metadata"]["explanation"],
                actions=actions,
                valid_until=rec_dict["metadata"]["valid_until"]
            ))
        
        # Create response
        batch_dict = batch.to_dict()
        response = GenerateRecommendationsResponse(
            batch_id=batch_dict["id"],
            farm_dna_id=str(batch_dict["farm_dna_id"]),
            generated_at=batch_dict["created_at"],
            themes=batch_dict["themes"],
            seasonal_focus=batch_dict["seasonal_focus"],
            total_recommendations=batch_dict["summary"]["total_recommendations"],
            recommendations=recommendation_responses,
            summary=batch_dict["summary"]
        )
        
        print(f"✅ Successfully generated {len(recommendation_responses)} recommendations")
        return response
        
    except Exception as e:
        print(f"❌ ERROR generating recommendations: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@router.post("/onboarding-with-recommendations")
async def onboarding_with_recommendations(
    request: OnboardingWithRecommendationsRequest
):
    """
    Complete onboarding with immediate recommendations.
    
    This endpoint:
    1. Collects onboarding data (5 data points)
    2. Creates FarmDNA profile
    3. Generates initial recommendations
    4. Returns everything in one response
    
    Perfect for new farmer onboarding flow.
    """
    try:
        print(f"🚀 Starting onboarding for: {request.farmer_name or 'New Farmer'}")
        
        # 1. Collect onboarding data
        onboarding_data = onboarding_collector.collect_onboarding_data({
            "primary_language": request.primary_language,
            "farm_size_category": request.farm_size_category,
            "main_crop_animal": request.main_crop_animal,
            "biggest_challenge": request.biggest_challenge,
            "mobile_literacy": request.mobile_literacy,
            "farmer_name": request.farmer_name,
            "location": request.location
        })
        
        farm_dna_id = onboarding_data.get("farm_dna", {}).get("identity", {}).get("farm_id")
        if not farm_dna_id:
            farm_dna_id = f"farm_{onboarding_data['onboarding_id']}"
        
        # 2. Get FarmDNA
        farm_dna = await get_farm_dna(farm_dna_id)
        
        # 3. Create context for initial recommendations
        context = RecommendationContext()
        context.season = "planting"
        context.weather = {"forecast": "clear"}
        
        # 4. Generate initial recommendations
        constraints = RecommendationConstraints(
            max_recommendations=3,
            max_cost=200.0,
            time_horizon_days=7
        )
        
        batch = recommendation_engine.generate_recommendations(
            farm_dna, context, constraints
        )
        
        # 5. Prepare response
        recommendations = []
        for rec in batch.recommendations[:3]:
            rec_dict = rec.to_dict()
            recommendations.append({
                "id": rec_dict["id"],
                "type": rec_dict["type"],
                "title": rec_dict["title"],
                "description": rec_dict["description"],
                "priority": rec_dict["scoring"]["priority"],
                "confidence_score": rec_dict["scoring"]["confidence_score"],
                "explanation": rec_dict["metadata"]["explanation"]
            })
        
        return {
            "status": "success",
            "onboarding_id": onboarding_data["onboarding_id"],
            "farm_dna_id": farm_dna_id,
            "farmer_profile": onboarding_data.get("farmer_profile", {}),
            "initial_recommendations": recommendations,
            "gamification_triggers": onboarding_data.get("gamification_triggers", {}),
            "next_steps": [
                {
                    "step": "review_recommendations",
                    "action": "review initial recommendations",
                    "priority": "high"
                },
                {
                    "step": "start_first_game",
                    "action": "begin with first recommended game",
                    "game": onboarding_data.get("gamification_triggers", {}).get("initial_games", [])[0] if onboarding_data.get("gamification_triggers", {}).get("initial_games") else None
                }
            ]
        }
        
    except Exception as e:
        print(f"❌ ERROR in onboarding: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Error during onboarding: {str(e)}")


@router.post("/{recommendation_id}/feedback")
async def submit_recommendation_feedback(
    recommendation_id: str,
    feedback: RecommendationFeedbackRequest,
    farm_dna_id: str = Query(..., description="ID of the farm DNA profile")
):
    """
    Submit feedback on a recommendation.
    
    This feedback is used to:
    - Improve future recommendations
    - Track recommendation success rates
    - Personalize the recommendation engine
    """
    try:
        print(f"📝 Recording feedback for recommendation: {recommendation_id}")
        
        recommendation_engine.log_recommendation_feedback(
            recommendation_id=recommendation_id,
            farm_dna_id=farm_dna_id,
            accepted=feedback.accepted,
            implemented=feedback.implemented,
            farmer_rating=feedback.farmer_rating,
            notes=feedback.notes
        )
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "recommendation_id": recommendation_id,
            "farm_dna_id": farm_dna_id,
            "accepted": feedback.accepted,
            "implemented": feedback.implemented
        }
        
    except Exception as e:
        print(f"❌ ERROR recording feedback: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {str(e)}")


@router.get("/types")
async def get_recommendation_types():
    """
    Get all available recommendation types.
    
    Returns the different types of recommendations the system can generate.
    """
    return {
        "recommendation_types": [
            {
                "id": rt.value,
                "name": rt.value.title(),
                "description": _get_recommendation_type_description(rt),
                "typical_time_horizon": _get_recommendation_time_horizon(rt)
            }
            for rt in RecommendationType
        ]
    }


@router.get("/priorities")
async def get_priority_levels():
    """
    Get all priority levels.
    
    Returns the different priority levels used in recommendations.
    """
    return {
        "priority_levels": [
            {
                "id": pl.value,
                "name": pl.value.title(),
                "description": _get_priority_description(pl),
                "typical_response_time": _get_priority_response_time(pl)
            }
            for pl in PriorityLevel
        ]
    }


@router.get("/context-template")
async def get_context_template():
    """
    Get a template for recommendation context.
    
    Useful for understanding what context information can be provided.
    """
    return {
        "context_template": {
            "weather": {
                "temperature": 25.5,
                "forecast": "clear|cloudy|rain|storm",
                "humidity": 65.0,
                "rainfall_last_24h": 0.0,
                "wind_speed": 10.0
            },
            "market_prices": {
                "maize": 120.0,
                "beans": 200.0,
                "coffee": 500.0
            },
            "season": "planting|growing|harvest|fallow",
            "current_activities": ["weeding", "irrigation", "harvesting"],
            "upcoming_events": [
                {
                    "event": "market_day",
                    "date": "2024-01-15",
                    "importance": "high"
                }
            ],
            "farmer_feedback": {
                "recent_issues": ["pest_infestation"],
                "market_conditions": "stable"
            }
        }
    }


# ===== Helper Functions =====

def _get_recommendation_type_description(rec_type: RecommendationType) -> str:
    """Get description for recommendation type."""
    descriptions = {
        RecommendationType.OPERATIONAL: "Immediate actions to take now (e.g., spray fungicide tomorrow)",
        RecommendationType.STRATEGIC: "Long-term decisions (e.g., switch to drought-resistant varieties)",
        RecommendationType.EDUCATIONAL: "Learning content (e.g., watch video on pest management)",
        RecommendationType.MARKET: "Market decisions (e.g., sell now before price drops)",
        RecommendationType.FINANCIAL: "Financial decisions (e.g., apply for microloan)",
        RecommendationType.ENVIRONMENTAL: "Environmental actions (e.g., plant cover crops)"
    }
    return descriptions.get(rec_type, "Farming recommendation")

def _get_recommendation_time_horizon(rec_type: RecommendationType) -> str:
    """Get typical time horizon for recommendation type."""
    horizons = {
        RecommendationType.OPERATIONAL: "hours to days",
        RecommendationType.STRATEGIC: "weeks to seasons",
        RecommendationType.EDUCATIONAL: "days to weeks",
        RecommendationType.MARKET: "days",
        RecommendationType.FINANCIAL: "days to weeks",
        RecommendationType.ENVIRONMENTAL: "weeks to seasons"
    }
    return horizons.get(rec_type, "varies")

def _get_priority_description(priority: PriorityLevel) -> str:
    """Get description for priority level."""
    descriptions = {
        PriorityLevel.CRITICAL: "Must do immediately - time-sensitive or high-risk",
        PriorityLevel.HIGH: "Should do soon - important for farm outcomes",
        PriorityLevel.MEDIUM: "Consider doing - beneficial but not urgent",
        PriorityLevel.LOW: "Optional - consider when resources allow",
        PriorityLevel.INFORMATIONAL: "For information only - no action required"
    }
    return descriptions.get(priority, "Priority level")

def _get_priority_response_time(priority: PriorityLevel) -> str:
    """Get typical response time for priority level."""
    response_times = {
        PriorityLevel.CRITICAL: "within 24 hours",
        PriorityLevel.HIGH: "within 3 days",
        PriorityLevel.MEDIUM: "within 1 week",
        PriorityLevel.LOW: "when convenient",
        PriorityLevel.INFORMATIONAL: "no time limit"
    }
    return response_times.get(priority, "varies")
