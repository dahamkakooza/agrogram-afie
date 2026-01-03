"""
Recommendation Model - Defines recommendation data structures and logic
Week 3-4 Implementation - Based on AFIE Blueprint
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal


class RecommendationType(str, Enum):
    """Types of recommendations."""
    OPERATIONAL = "operational"  # Immediate actions: "Spray fungicide tomorrow"
    STRATEGIC = "strategic"  # Long-term decisions: "Switch to drought-resistant maize"
    EDUCATIONAL = "educational"  # Learning content: "Watch video on IPM"
    MARKET = "market"  # Market decisions: "Sell maize now, prices will drop"
    FINANCIAL = "financial"  # Financial decisions: "Apply for microloan"
    ENVIRONMENTAL = "environmental"  # Environmental actions: "Plant cover crops"


class PriorityLevel(str, Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"  # Must do immediately
    HIGH = "high"  # Should do soon
    MEDIUM = "medium"  # Consider doing
    LOW = "low"  # Optional
    INFORMATIONAL = "informational"  # Just for information


class ImpactType(str, Enum):
    """Types of impact a recommendation can have."""
    YIELD_INCREASE = "yield_increase"
    COST_REDUCTION = "cost_reduction"
    RISK_REDUCTION = "risk_reduction"
    TIME_SAVINGS = "time_savings"
    QUALITY_IMPROVEMENT = "quality_improvement"
    SUSTAINABILITY = "sustainability"
    MARKET_ACCESS = "market_access"


@dataclass
class ImpactEstimate:
    """
    Estimated impact of implementing a recommendation.
    """
    impact_type: ImpactType
    magnitude: float  # Percentage or absolute value
    unit: str  # "percent", "kg", "USD", "hours"
    confidence: float  # 0.0 to 1.0
    time_horizon: str  # "immediate", "short_term", "season", "long_term"
    
    # Range estimates
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    
    # Supporting evidence
    evidence_source: Optional[str] = None
    supporting_data: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        return f"{self.impact_type}: {self.magnitude} {self.unit} ({self.confidence*100:.0f}% confidence)"


@dataclass
class ResourceRequirement:
    """
    Resources needed to implement a recommendation.
    """
    resource_type: str  # "labor", "capital", "equipment", "materials"
    quantity: float
    unit: str  # "hours", "USD", "kg", "liters"
    quality: Optional[str] = None  # "skilled", "unskilled", "specific_type"
    
    # Timing constraints
    timing: Optional[str] = None  # "before_planting", "during_season", "immediately"
    duration: Optional[str] = None  # "single_day", "continuous", "weekly"
    
    # Availability check
    is_critical: bool = True  # Can't proceed without this resource
    alternatives: List[str] = field(default_factory=list)  # Alternative resources
    
    def __str__(self) -> str:
        return f"{self.resource_type}: {self.quantity} {self.unit}"


@dataclass
class ActionStep:
    """
    A single step in implementing a recommendation.
    """
    step_number: int
    description: str
    detailed_instructions: str
    
    # Resources needed for this specific step
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    
    # Timing information
    estimated_duration_minutes: Optional[int] = None
    best_time_of_day: Optional[str] = None  # "morning", "afternoon", "evening"
    time_sensitivity: Optional[str] = None  # "weather_dependent", "time_critical", "flexible"
    
    # Success criteria
    success_indicators: List[str] = field(default_factory=list)
    common_pitfalls: List[str] = field(default_factory=list)
    
    # Visual aids
    icon: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    
    def __str__(self) -> str:
        return f"Step {self.step_number}: {self.description}"


@dataclass
class Alternative:
    """
    Alternative recommendation or variation.
    """
    id: str
    title: str
    description: str
    
    # How this alternative differs from main recommendation
    differences: Dict[str, Any]
    
    # Pros and cons
    advantages: List[str]
    disadvantages: List[str]
    
    # Comparative metrics
    cost_difference: Optional[float] = None  # Positive = more expensive, Negative = cheaper
    time_difference: Optional[float] = None  # Positive = takes longer, Negative = faster
    impact_difference: Optional[float] = None  # Positive = better impact, Negative = worse
    
    def __str__(self) -> str:
        return f"Alternative: {self.title}"


class Recommendation:
    """
    A personalized farming recommendation.
    """
    
    def __init__(
        self,
        recommendation_type: RecommendationType,
        title: str,
        description: str,
        farm_dna_id: str,
        context: Dict[str, Any]
    ):
        self.id = str(uuid.uuid4())
        self.type = recommendation_type
        self.title = title
        self.description = description
        self.farm_dna_id = farm_dna_id
        self.context = context
        
        # Core components
        self.actions: List[ActionStep] = []
        self.expected_impacts: List[ImpactEstimate] = []
        self.resource_requirements: List[ResourceRequirement] = []
        self.alternatives: List[Alternative] = []
        
        # Scoring and metadata
        self.confidence_score: float = 0.0
        self.priority: PriorityLevel = PriorityLevel.MEDIUM
        self.explanation: str = ""
        self.created_at = datetime.utcnow()
        self.valid_until: Optional[datetime] = None
        self.seasonal_relevance: List[str] = []  # ["planting", "growing", "harvest"]
        
        # Implementation tracking
        self.status: str = "pending"  # pending, accepted, rejected, completed
        self.accepted_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.implementation_notes: List[str] = []
        
        # Farmer interaction
        self.farmer_feedback: Optional[str] = None
        self.farmer_rating: Optional[int] = None  # 1-5 stars
        self.actual_impact: Optional[Dict[str, Any]] = None
        
    def add_action_step(
        self,
        description: str,
        detailed_instructions: str,
        **kwargs
    ) -> 'Recommendation':
        """Add an action step to the recommendation."""
        step_number = len(self.actions) + 1
        action = ActionStep(
            step_number=step_number,
            description=description,
            detailed_instructions=detailed_instructions,
            **kwargs
        )
        self.actions.append(action)
        return self
    
    def add_impact_estimate(
        self,
        impact_type: ImpactType,
        magnitude: float,
        unit: str,
        confidence: float,
        time_horizon: str,
        **kwargs
    ) -> 'Recommendation':
        """Add an impact estimate to the recommendation."""
        impact = ImpactEstimate(
            impact_type=impact_type,
            magnitude=magnitude,
            unit=unit,
            confidence=confidence,
            time_horizon=time_horizon,
            **kwargs
        )
        self.expected_impacts.append(impact)
        return self
    
    def add_resource_requirement(
        self,
        resource_type: str,
        quantity: float,
        unit: str,
        **kwargs
    ) -> 'Recommendation':
        """Add a resource requirement to the recommendation."""
        resource = ResourceRequirement(
            resource_type=resource_type,
            quantity=quantity,
            unit=unit,
            **kwargs
        )
        self.resource_requirements.append(resource)
        return self
    
    def add_alternative(
        self,
        title: str,
        description: str,
        differences: Dict[str, Any],
        advantages: List[str],
        disadvantages: List[str],
        **kwargs
    ) -> 'Recommendation':
        """Add an alternative to the recommendation."""
        alternative = Alternative(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            differences=differences,
            advantages=advantages,
            disadvantages=disadvantages,
            **kwargs
        )
        self.alternatives.append(alternative)
        return self
    
    def calculate_total_cost(self) -> float:
        """Calculate total estimated cost of implementation."""
        total = 0.0
        for resource in self.resource_requirements:
            if resource.resource_type == "capital" and resource.unit == "USD":
                total += resource.quantity
        return total
    
    def calculate_total_labor(self) -> float:
        """Calculate total estimated labor hours required."""
        total = 0.0
        for resource in self.resource_requirements:
            if resource.resource_type == "labor" and resource.unit == "hours":
                total += resource.quantity
        return total
    
    def get_estimated_value(self) -> float:
        """Calculate estimated monetary value of the recommendation."""
        total_value = 0.0
        
        for impact in self.expected_impacts:
            if impact.impact_type == ImpactType.YIELD_INCREASE:
                # Convert percentage increase to estimated monetary value
                # This would need farm-specific yield data for accurate calculation
                value = impact.magnitude * impact.confidence * 100  # Simplified
                total_value += value
            elif impact.impact_type == ImpactType.COST_REDUCTION:
                total_value += impact.magnitude * impact.confidence
        
        return total_value
    
    def get_roi_estimate(self) -> Optional[float]:
        """Calculate estimated return on investment."""
        total_cost = self.calculate_total_cost()
        if total_cost == 0:
            return None
        
        total_value = self.get_estimated_value()
        return (total_value - total_cost) / total_cost * 100
    
    def set_urgency_based_on_context(self) -> None:
        """Set urgency level based on context and timing."""
        context = self.context
        
        # Check for time-sensitive conditions
        if context.get("weather_alert") == "heavy_rain_forecast":
            if "planting" in context.get("current_activities", []):
                self.priority = PriorityLevel.CRITICAL
                self.valid_until = datetime.utcnow() + timedelta(hours=24)
                
        elif context.get("pest_infestation_level") == "severe":
            self.priority = PriorityLevel.HIGH
            self.valid_until = datetime.utcnow() + timedelta(days=3)
            
        elif context.get("market_opportunity") == "price_spike":
            self.priority = PriorityLevel.HIGH
            self.valid_until = datetime.utcnow() + timedelta(days=7)
            
        else:
            self.priority = PriorityLevel.MEDIUM
            self.valid_until = datetime.utcnow() + timedelta(days=30)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "farm_dna_id": self.farm_dna_id,
            "context": self.context,
            
            "actions": [
                {
                    "step_number": action.step_number,
                    "description": action.description,
                    "detailed_instructions": action.detailed_instructions,
                    "estimated_duration_minutes": action.estimated_duration_minutes,
                    "resource_requirements": [
                        {
                            "resource_type": req.resource_type,
                            "quantity": req.quantity,
                            "unit": req.unit
                        }
                        for req in action.resource_requirements
                    ]
                }
                for action in self.actions
            ],
            
            "expected_impacts": [
                {
                    "impact_type": impact.impact_type.value,
                    "magnitude": impact.magnitude,
                    "unit": impact.unit,
                    "confidence": impact.confidence,
                    "time_horizon": impact.time_horizon
                }
                for impact in self.expected_impacts
            ],
            
            "resource_requirements": [
                {
                    "resource_type": req.resource_type,
                    "quantity": req.quantity,
                    "unit": req.unit,
                    "is_critical": req.is_critical
                }
                for req in self.resource_requirements
            ],
            
            "alternatives": [
                {
                    "id": alt.id,
                    "title": alt.title,
                    "description": alt.description,
                    "differences": alt.differences,
                    "advantages": alt.advantages,
                    "disadvantages": alt.disadvantages
                }
                for alt in self.alternatives
            ],
            
            "scoring": {
                "confidence_score": self.confidence_score,
                "priority": self.priority.value,
                "total_cost": self.calculate_total_cost(),
                "total_labor": self.calculate_total_labor(),
                "estimated_value": self.get_estimated_value(),
                "roi_estimate": self.get_roi_estimate()
            },
            
            "metadata": {
                "explanation": self.explanation,
                "created_at": self.created_at.isoformat(),
                "valid_until": self.valid_until.isoformat() if self.valid_until else None,
                "status": self.status,
                "seasonal_relevance": self.seasonal_relevance
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recommendation':
        """Create a recommendation from dictionary data."""
        rec = cls(
            recommendation_type=RecommendationType(data["type"]),
            title=data["title"],
            description=data["description"],
            farm_dna_id=data["farm_dna_id"],
            context=data["context"]
        )
        
        rec.id = data["id"]
        rec.confidence_score = data["scoring"]["confidence_score"]
        rec.priority = PriorityLevel(data["scoring"]["priority"])
        rec.explanation = data["metadata"]["explanation"]
        rec.status = data["metadata"]["status"]
        rec.seasonal_relevance = data["metadata"]["seasonal_relevance"]
        
        # Reconstruct actions
        for action_data in data["actions"]:
            action = ActionStep(
                step_number=action_data["step_number"],
                description=action_data["description"],
                detailed_instructions=action_data["detailed_instructions"],
                estimated_duration_minutes=action_data["estimated_duration_minutes"],
                resource_requirements=[
                    ResourceRequirement(
                        resource_type=req_data["resource_type"],
                        quantity=req_data["quantity"],
                        unit=req_data["unit"]
                    )
                    for req_data in action_data["resource_requirements"]
                ]
            )
            rec.actions.append(action)
        
        # Reconstruct impacts
        for impact_data in data["expected_impacts"]:
            impact = ImpactEstimate(
                impact_type=ImpactType(impact_data["impact_type"]),
                magnitude=impact_data["magnitude"],
                unit=impact_data["unit"],
                confidence=impact_data["confidence"],
                time_horizon=impact_data["time_horizon"]
            )
            rec.expected_impacts.append(impact)
        
        return rec
    
    def __str__(self) -> str:
        return f"{self.type.value.title()} Recommendation: {self.title}"


class RecommendationBatch:
    """
    A batch of related recommendations.
    """
    
    def __init__(self, farm_dna_id: str, context: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.farm_dna_id = farm_dna_id
        self.context = context
        self.created_at = datetime.utcnow()
        
        self.recommendations: List[Recommendation] = []
        self.themes: List[str] = []  # ["water_management", "pest_control", "market_timing"]
        self.seasonal_focus: Optional[str] = None  # "planting", "growing", "harvest"
        
    def add_recommendation(self, recommendation: Recommendation) -> None:
        """Add a recommendation to the batch."""
        self.recommendations.append(recommendation)
    
    def get_priority_sorted(self) -> List[Recommendation]:
        """Get recommendations sorted by priority."""
        priority_order = {
            PriorityLevel.CRITICAL: 0,
            PriorityLevel.HIGH: 1,
            PriorityLevel.MEDIUM: 2,
            PriorityLevel.LOW: 3,
            PriorityLevel.INFORMATIONAL: 4
        }
        
        return sorted(
            self.recommendations,
            key=lambda r: (priority_order[r.priority], -r.confidence_score)
        )
    
    def get_by_type(self, rec_type: RecommendationType) -> List[Recommendation]:
        """Get recommendations of a specific type."""
        return [r for r in self.recommendations if r.type == rec_type]
    
    def get_total_estimated_value(self) -> float:
        """Get total estimated value of all recommendations in batch."""
        return sum(r.get_estimated_value() for r in self.recommendations)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary."""
        return {
            "id": self.id,
            "farm_dna_id": self.farm_dna_id,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "themes": self.themes,
            "seasonal_focus": self.seasonal_focus,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "summary": {
                "total_recommendations": len(self.recommendations),
                "by_type": {
                    rt.value: len(self.get_by_type(rt))
                    for rt in RecommendationType
                },
                "total_estimated_value": self.get_total_estimated_value(),
                "priority_distribution": {
                    p.value: len([r for r in self.recommendations if r.priority == p])
                    for p in PriorityLevel
                }
            }
        }
    
    def __str__(self) -> str:
        return f"Recommendation Batch ({len(self.recommendations)} recommendations)"


# Factory function for creating common recommendation types
def create_operational_recommendation(
    title: str,
    description: str,
    farm_dna_id: str,
    context: Dict[str, Any]
) -> Recommendation:
    """Factory function for creating operational recommendations."""
    rec = Recommendation(
        recommendation_type=RecommendationType.OPERATIONAL,
        title=title,
        description=description,
        farm_dna_id=farm_dna_id,
        context=context
    )
    rec.set_urgency_based_on_context()
    return rec


def create_strategic_recommendation(
    title: str,
    description: str,
    farm_dna_id: str,
    context: Dict[str, Any]
) -> Recommendation:
    """Factory function for creating strategic recommendations."""
    rec = Recommendation(
        recommendation_type=RecommendationType.STRATEGIC,
        title=title,
        description=description,
        farm_dna_id=farm_dna_id,
        context=context
    )
    rec.priority = PriorityLevel.MEDIUM
    rec.valid_until = datetime.utcnow() + timedelta(days=90)  # 3 months for strategic decisions
    return rec
