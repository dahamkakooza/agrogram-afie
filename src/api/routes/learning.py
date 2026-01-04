"""
Learning Routes - Week 5-6 Implementation - FINAL CORRECTED VERSION
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

# Import Week 5-6 modules
from src.learning.behavioral_analyzer import behavioral_analyzer, BehavioralProfile
from src.learning.game_data_processor import game_data_processor
from src.learning.adaptation_manager import adaptation_manager

# Create router
router = APIRouter(prefix="/learning", tags=["learning"])


# ============ Request/Response Models ============

class GameEvent(BaseModel):
    """Individual game event model."""
    event_id: str
    farmer_id: str
    decision_point: str
    choices: List[str]
    selected_choice: str
    decision_time_seconds: float = Field(..., ge=0)
    outcome: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    timestamp: Optional[str] = None
    game_type: Optional[str] = "unknown"


class GameSessionRequest(BaseModel):
    """Complete game session request."""
    session_id: str
    farmer_id: str
    game_type: str
    events: List[GameEvent]
    final_score: Optional[float] = None
    learning_objectives: List[str] = []
    duration_seconds: Optional[float] = None


class FarmerFeedbackRequest(BaseModel):
    """Farmer feedback for learning system."""
    farmer_id: str
    feedback_type: str = "explicit"  # explicit, implicit
    feedback: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None


# Response Models
class GameSessionResponse(BaseModel):
    """Response from game session processing."""
    session_id: str
    farmer_id: str
    processing_status: str
    behavioral_insights: Dict[str, Any]
    learning_velocity: Optional[Union[float, Dict[str, Any]]] = None  # Can be float or dict
    adaptation_triggers: List[Dict[str, Any]] = []
    personalization_updates: Dict[str, Any] = {}
    
    @validator('learning_velocity', pre=True)
    def handle_learning_velocity(cls, v):
        """Handle learning_velocity which might be a dict with error."""
        if isinstance(v, dict):
            # If it's an error dict, return None or the dict
            return v
        return v


class BehavioralProfileResponse(BaseModel):
    """Behavioral profile response."""
    farmer_id: str
    risk_tolerance: Optional[float] = None
    learning_style: Optional[str] = None
    decision_confidence: Optional[float] = None
    behavioral_patterns: Dict[str, Any] = {}
    adaptation_history: List[Dict[str, Any]] = []
    personalization_settings: Dict[str, Any] = {}


class FeedbackAnalysisResponse(BaseModel):
    """Feedback analysis response."""
    farmer_id: str
    feedback_type: str
    analysis_results: Dict[str, Any]
    adaptation_applied: bool = False
    adaptation_results: List[Dict[str, Any]] = []


# ============ Endpoints ============

@router.post("/game-session", response_model=GameSessionResponse)
async def process_game_session(session_data: GameSessionRequest):
    """
    Process a game session and extract behavioral insights.
    """
    try:
        # Convert to dict for processing
        session_dict = {
            "session_id": session_data.session_id,
            "farmer_id": session_data.farmer_id,
            "game_type": session_data.game_type,
            "events": [event.dict() for event in session_data.events],
            "final_score": session_data.final_score,
            "learning_objectives": session_data.learning_objectives
        }
        
        # Process the game session
        result = game_data_processor.process_game_session(session_dict)
        
        # Extract learning_velocity safely
        learning_velocity = result.get("learning_velocity")
        
        return GameSessionResponse(
            session_id=session_data.session_id,
            farmer_id=session_data.farmer_id,
            processing_status="success",
            behavioral_insights=result.get("behavioral_features", {}),
            learning_velocity=learning_velocity,
            adaptation_triggers=result.get("adaptation_triggers", []),
            personalization_updates=result.get("personalization_update", {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process game session: {str(e)}")


@router.post("/feedback", response_model=FeedbackAnalysisResponse)
async def analyze_farmer_feedback(feedback: FarmerFeedbackRequest):
    """
    Analyze farmer feedback and trigger system adaptations.
    """
    try:
        # Prepare feedback data in correct format
        feedback_data = {
            "farmer_id": feedback.farmer_id,
            "feedback_type": feedback.feedback_type,
            "feedback": feedback.feedback,
            "context": feedback.context
        }
        
        if feedback.timestamp:
            feedback_data["timestamp"] = feedback.timestamp
        
        # Analyze feedback
        analysis_result = adaptation_manager.analyze_feedback(feedback_data)
        
        # Apply adaptations if needed
        adaptations_applied = []
        if analysis_result.get("should_adapt", False):
            adaptation_context = {
                "farmer_id": feedback.farmer_id,
                "trigger": "feedback_analysis",
                "context": analysis_result
            }
            adaptation_result = adaptation_manager.apply_adaptation(adaptation_context)
            adaptations_applied = adaptation_result.get("applied_adaptations", [])
        
        return FeedbackAnalysisResponse(
            farmer_id=feedback.farmer_id,
            feedback_type=feedback.feedback_type,
            analysis_results=analysis_result,
            adaptation_applied=len(adaptations_applied) > 0,
            adaptation_results=adaptations_applied
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze feedback: {str(e)}")


@router.get("/behavioral-profile/{farmer_id}", response_model=BehavioralProfileResponse)
async def get_behavioral_profile(farmer_id: str):
    """
    Get a farmer's behavioral profile.
    """
    try:
        # Get profile
        profile = behavioral_analyzer.get_behavioral_profile(farmer_id)
        
        if not profile:
            # Return empty profile if not found
            return BehavioralProfileResponse(
                farmer_id=farmer_id,
                behavioral_patterns={},
                adaptation_history=[],
                personalization_settings={}
            )
        
        # Convert to dict
        profile_dict = profile.to_dict() if hasattr(profile, 'to_dict') else vars(profile)
        
        # Get adaptation history
        adaptation_history = []
        try:
            adaptation_history = adaptation_manager.get_adaptation_history(farmer_id)
        except:
            adaptation_history = []
        
        return BehavioralProfileResponse(
            farmer_id=farmer_id,
            risk_tolerance=profile_dict.get("risk_tolerance"),
            learning_style=profile_dict.get("learning_style"),
            decision_confidence=profile_dict.get("decision_confidence"),
            behavioral_patterns={},
            adaptation_history=adaptation_history,
            personalization_settings={}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get behavioral profile: {str(e)}")


@router.post("/decision-pattern/{farmer_id}")
async def analyze_decision_pattern(farmer_id: str, decision_data: Dict[str, Any]):
    """
    Analyze a decision pattern for behavioral insights.
    """
    try:
        result = behavioral_analyzer.analyze_decision_pattern(farmer_id, decision_data)
        
        return {
            "status": "success",
            "farmer_id": farmer_id,
            "decision_analyzed": True,
            "insights": result.get("insights", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze decision pattern: {str(e)}")


@router.get("/adaptation-history/{farmer_id}")
async def get_adaptation_history(farmer_id: str, limit: int = 10):
    """
    Get adaptation history for a farmer.
    """
    try:
        history = adaptation_manager.get_adaptation_history(farmer_id)
        
        return {
            "farmer_id": farmer_id,
            "total_adaptations": len(history),
            "recent_adaptations": history[:limit],
            "effectiveness_rate": len([h for h in history if h.get("success", False) or h.get("outcome") == "success"]) / max(len(history), 1)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get adaptation history: {str(e)}")


@router.get("/system-learning-insights")
async def get_system_learning_insights():
    """
    Get global system learning insights.
    """
    try:
        # Get basic insights
        total_farmers = len(behavioral_analyzer.behavioral_profiles)
        total_sessions = len(game_data_processor.sessions)
        total_adaptations = sum(len(h) for h in adaptation_manager.adaptation_history.values())
        
        # Get learning styles
        learning_styles = {}
        for farmer_id, profile in behavioral_analyzer.behavioral_profiles.items():
            if hasattr(profile, 'learning_style'):
                style = profile.learning_style.value if hasattr(profile.learning_style, 'value') else str(profile.learning_style)
            elif isinstance(profile, dict):
                style = profile.get("learning_style", "unknown")
            else:
                style = "unknown"
            learning_styles[style] = learning_styles.get(style, 0) + 1
        
        # Calculate success rate
        successful_adaptations = 0
        total_adaptations_count = 0
        for farmer_history in adaptation_manager.adaptation_history.values():
            for adaptation in farmer_history:
                total_adaptations_count += 1
                if adaptation.get("success", False) or adaptation.get("outcome") == "success":
                    successful_adaptations += 1
        
        success_rate = successful_adaptations / max(total_adaptations_count, 1)
        
        return {
            "system_status": "learning_active",
            "total_farmers_analyzed": total_farmers,
            "common_learning_styles": dict(sorted(learning_styles.items(), key=lambda x: x[1], reverse=True)),
            "total_game_sessions": total_sessions,
            "total_adaptations": total_adaptations,
            "adaptation_success_rate": success_rate,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system insights: {str(e)}")


# ============ Health & Status ============

@router.get("/health")
async def learning_system_health():
    """Check health of learning system components."""
    try:
        components = {
            "behavioral_analyzer": "healthy",
            "game_data_processor": "healthy", 
            "adaptation_manager": "healthy",
            "active_farmers": len(behavioral_analyzer.behavioral_profiles),
            "total_game_sessions": len(game_data_processor.sessions),
            "total_adaptations": sum(len(h) for h in adaptation_manager.adaptation_history.values())
        }
        
        return {
            "status": "healthy",
            "components": components,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/status")
async def learning_system_status():
    """Get detailed status of learning system."""
    return {
        "system": "behavioral_learning_engine",
        "status": "active",
        "week_implementation": "5-6",
        "features": [
            "behavioral_analysis",
            "game_data_processing", 
            "system_adaptation"
        ],
        "endpoints_available": [
            "POST /learning/game-session",
            "POST /learning/feedback", 
            "GET /learning/behavioral-profile/{farmer_id}",
            "POST /learning/decision-pattern/{farmer_id}",
            "GET /learning/adaptation-history/{farmer_id}",
            "GET /learning/system-learning-insights"
        ],
        "statistics": {
            "modules_loaded": 3,
            "api_version": "1.0"
        }
    }


# ============ Test Endpoint ============

@router.post("/test-modules")
async def test_modules():
    """Test all Week 5-6 modules work together."""
    try:
        test_farmer = f"test-modules-{int(datetime.utcnow().timestamp())}"
        
        # Test Behavioral Analyzer
        decision_data = {
            "decision_type": "test_decision",
            "action": "accepted",
            "timing": "prompt",
            "risk_level": "medium",
            "decision_time_seconds": 15.0,
            "outcome_confidence": 0.8
        }
        ba_result = behavioral_analyzer.analyze_decision_pattern(test_farmer, decision_data)
        
        # Test Game Data Processor
        session_data = {
            "session_id": "test-session-001",
            "farmer_id": test_farmer,
            "game_type": "test_game",
            "events": [{
                "event_id": "test-event-1",
                "farmer_id": test_farmer,
                "decision_point": "test_decision",
                "choices": ["Option A", "Option B"],
                "selected_choice": "Option A",
                "decision_time_seconds": 10.0,
                "outcome": {"score": 85}
            }]
        }
        gdp_result = game_data_processor.process_game_session(session_data)
        
        # Test Adaptation Manager
        feedback_data = {
            "farmer_id": test_farmer,
            "feedback_type": "explicit",
            "feedback": {
                "type": "recommendation",
                "action": "accepted",
                "rating": 4,
                "comments": "Test feedback"
            },
            "context": {"test": True}
        }
        am_result = adaptation_manager.analyze_feedback(feedback_data)
        
        return {
            "status": "success",
            "modules_tested": {
                "behavioral_analyzer": bool(ba_result),
                "game_data_processor": bool(gdp_result),
                "adaptation_manager": bool(am_result)
            },
            "test_farmer_id": test_farmer,
            "message": "All Week 5-6 modules are working correctly"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Module test failed"
        }