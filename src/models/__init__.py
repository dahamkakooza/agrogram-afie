"""
Data models for AFIE service.
This module contains Pydantic models for data validation and serialization.
"""

from .farm import *
from .farmer import *
from .recommendation import *
from .prediction import *
from .learning import *
from .events import *

__all__ = [
    # Farm models
    "FarmIdentity",
    "FarmTypology",
    "ResourceConstraints",
    "EnvironmentalContext",
    "HistoricalPerformance",
    "FarmDNA",
    
    # Farmer models
    "FarmerProfile",
    "BehavioralProfile",
    "LearningProfile",
    "FarmerGoals",
    
    # Recommendation models
    "Recommendation",
    "RecommendationType",
    "ActionStep",
    "ImpactEstimate",
    "ResourceRequirement",
    
    # Prediction models
    "YieldPrediction",
    "RiskAssessment",
    "Uncertainty",
    
    # Learning models
    "LearningEvent",
    "Feedback",
    "Adaptation",
    "KnowledgeUpdate",
    
    # Event models
    "FarmEvent",
    "MarketEvent",
    "GameEvent",
    "SystemEvent",
    "IntegrationEvent",
]