from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict, field_validator


class EducationLevel(str, Enum):
    """Farmer education level."""
    NONE = "none"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    VOCATIONAL = "vocational"
    TERTIARY = "tertiary"
    POSTGRADUATE = "postgraduate"


class FarmingExperience(str, Enum):
    """Farming experience level."""
    BEGINNER = "beginner"  # < 2 years
    INTERMEDIATE = "intermediate"  # 2-10 years
    EXPERIENCED = "experienced"  # 10-20 years
    EXPERT = "expert"  # > 20 years


class LearningStyle(str, Enum):
    """Preferred learning style."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"
    MIXED = "mixed"


class RiskTolerance(str, Enum):
    """Risk tolerance level."""
    VERY_CONSERVATIVE = "very_conservative"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


class FarmerProfile(BaseModel):
    """Farmer personal and demographic information."""
    
    model_config = ConfigDict(from_attributes=True)
    
    farmer_id: UUID = Field(default_factory=uuid4, description="Unique farmer identifier")
    user_id: UUID = Field(..., description="Associated user account ID")
    first_name: str = Field(..., min_length=1, max_length=50, description="First name")
    last_name: str = Field(..., min_length=1, max_length=50, description="Last name")
    email: str = Field(..., description="Email address")
    phone_number: Optional[str] = Field(None, description="Phone number")
    
    # Demographics
    age: Optional[int] = Field(None, ge=18, le=120, description="Age in years")
    gender: Optional[str] = Field(None, description="Gender")
    education_level: Optional[EducationLevel] = Field(None, description="Education level")
    farming_experience: Optional[FarmingExperience] = Field(None, description="Farming experience")
    
    # Location
    country: str = Field(..., description="Country")
    region: str = Field(..., description="Region/State")
    district: Optional[str] = Field(None, description="District")
    village: Optional[str] = Field(None, description="Village")
    
    # Communication preferences
    preferred_language: str = Field(default="en", description="Preferred language code")
    communication_channels: List[str] = Field(
        default_factory=lambda: ["app"],
        description="Preferred communication channels"
    )
    notification_frequency: str = Field(default="daily", description="Notification frequency")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    is_active: bool = Field(default=True, description="Active status")
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation."""
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower().strip()
    
    @property
    def full_name(self) -> str:
        """Get full name."""
        return f"{self.first_name} {self.last_name}".strip()


class BehavioralProfile(BaseModel):
    """Farmer behavioral patterns and tendencies."""
    
    model_config = ConfigDict(from_attributes=True)
    
    farmer_id: UUID = Field(..., description="Farmer identifier")
    
    # Behavioral dimensions
    risk_tolerance: RiskTolerance = Field(default=RiskTolerance.MODERATE, description="Risk tolerance")
    innovation_adoption_speed: float = Field(default=0.5, ge=0, le=1, description="Innovation adoption speed (0-1)")
    planning_horizon_days: int = Field(default=30, ge=1, le=365, description="Typical planning horizon in days")
    decision_confidence: float = Field(default=0.7, ge=0, le=1, description="Decision confidence level (0-1)")
    
    # Learning preferences
    learning_style: LearningStyle = Field(default=LearningStyle.MIXED, description="Preferred learning style")
    information_depth_preference: str = Field(default="moderate", description="Preferred information depth")
    feedback_receptiveness: float = Field(default=0.8, ge=0, le=1, description="Feedback receptiveness (0-1)")
    
    # Behavioral patterns
    typical_decision_time_hours: float = Field(default=24.0, ge=0, description="Typical decision time in hours")
    error_correction_speed: float = Field(default=0.6, ge=0, le=1, description="Error correction speed (0-1)")
    stress_response_pattern: str = Field(default="adaptive", description="Stress response pattern")
    
    # Tracked metrics
    recommendation_adoption_rate: float = Field(default=0.0, ge=0, le=1, description="Recommendation adoption rate")
    success_rate: float = Field(default=0.0, ge=0, le=1, description="Success rate of adopted recommendations")
    engagement_score: float = Field(default=0.0, ge=0, le=1, description="Overall engagement score")
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last profile update")
    evolution_tracking: List[Dict[str, Any]] = Field(default_factory=list, description="Behavior evolution history")
    
    @property
    def is_risk_averse(self) -> bool:
        """Check if farmer is risk averse."""
        return self.risk_tolerance in [RiskTolerance.VERY_CONSERVATIVE, RiskTolerance.CONSERVATIVE]
    
    @property
    def is_early_adopter(self) -> bool:
        """Check if farmer is an early adopter."""
        return self.innovation_adoption_speed > 0.7
    
    def update_from_feedback(
        self,
        feedback_score: float,
        adoption_speed_hours: Optional[float] = None
    ):
        """Update behavioral profile based on feedback."""
        # Update adoption rate
        if adoption_speed_hours:
            # Faster adoption increases innovation adoption score
            if adoption_speed_hours < 24:
                self.innovation_adoption_speed = min(1.0, self.innovation_adoption_speed + 0.05)
            elif adoption_speed_hours > 72:
                self.innovation_adoption_speed = max(0.0, self.innovation_adoption_speed - 0.03)
        
        # Update decision confidence based on feedback
        if feedback_score > 0.7:
            self.decision_confidence = min(1.0, self.decision_confidence + 0.02)
        elif feedback_score < 0.3:
            self.decision_confidence = max(0.0, self.decision_confidence - 0.05)
        
        # Record evolution
        self.evolution_tracking.append({
            "timestamp": datetime.utcnow().isoformat(),
            "feedback_score": feedback_score,
            "adoption_speed_hours": adoption_speed_hours,
            "updated_fields": ["decision_confidence", "innovation_adoption_speed"]
        })
        
        self.last_updated = datetime.utcnow()


class LearningProfile(BaseModel):
    """Farmer learning progress and preferences."""
    
    model_config = ConfigDict(from_attributes=True)
    
    farmer_id: UUID = Field(..., description="Farmer identifier")
    
    # Learning progress
    completed_modules: List[str] = Field(default_factory=list, description="Completed learning modules")
    current_modules: List[str] = Field(default_factory=list, description="Currently active modules")
    skill_levels: Dict[str, float] = Field(
        default_factory=dict,
        description="Skill levels by category (0-1)"
    )
    
    # Learning preferences
    preferred_content_format: List[str] = Field(
        default_factory=lambda: ["video", "text"],
        description="Preferred content formats"
    )
    preferred_content_length: str = Field(default="medium", description="Preferred content length")
    difficulty_preference: str = Field(default="adaptive", description="Preferred difficulty level")
    
    # Learning metrics
    total_learning_hours: float = Field(default=0.0, ge=0, description="Total learning hours")
    average_completion_rate: float = Field(default=0.0, ge=0, le=1, description="Average completion rate")
    knowledge_retention_score: float = Field(default=0.0, ge=0, le=1, description="Knowledge retention score")
    
    # Progress tracking
    learning_goals: List[str] = Field(default_factory=list, description="Learning goals")
    progress_milestones: List[Dict[str, Any]] = Field(default_factory=list, description="Progress milestones")
    
    # Metadata
    last_learning_session: Optional[datetime] = Field(None, description="Last learning session timestamp")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    def add_completed_module(self, module_name: str, score: float):
        """Add a completed learning module."""
        if module_name not in self.completed_modules:
            self.completed_modules.append(module_name)
        
        # Update metrics
        self.total_learning_hours += 1.0  # Assuming 1 hour per module
        
        # Update skill level if applicable
        if "skill_category" in module_name.lower():
            category = module_name.split("_")[0]
            self.skill_levels[category] = max(
                self.skill_levels.get(category, 0.0),
                min(score, 1.0)
            )
        
        self.last_learning_session = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def get_skill_gaps(self, required_skills: Dict[str, float]) -> Dict[str, float]:
        """Identify skill gaps based on required skills."""
        gaps = {}
        for skill, required_level in required_skills.items():
            current_level = self.skill_levels.get(skill, 0.0)
            if current_level < required_level:
                gaps[skill] = required_level - current_level
        
        return gaps


class FarmerGoals(BaseModel):
    """Farmer goals and objectives."""
    
    model_config = ConfigDict(from_attributes=True)
    
    farmer_id: UUID = Field(..., description="Farmer identifier")
    
    # Goal categories
    production_goals: List[str] = Field(default_factory=list, description="Production goals")
    income_goals: List[str] = Field(default_factory=list, description="Income goals")
    sustainability_goals: List[str] = Field(default_factory=list, description="Sustainability goals")
    quality_goals: List[str] = Field(default_factory=list, description="Quality goals")
    lifestyle_goals: List[str] = Field(default_factory=list, description="Lifestyle goals")
    
    # Goal details
    goal_details: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Detailed goal information"
    )
    
    # Timeframes
    short_term_goals: List[str] = Field(default_factory=list, description="Short-term goals (< 1 year)")
    medium_term_goals: List[str] = Field(default_factory=list, description="Medium-term goals (1-3 years)")
    long_term_goals: List[str] = Field(default_factory=list, description="Long-term goals (> 3 years)")
    
    # Progress tracking
    completed_goals: List[str] = Field(default_factory=list, description="Completed goals")
    in_progress_goals: List[str] = Field(default_factory=list, description="Goals in progress")
    goal_priorities: Dict[str, int] = Field(default_factory=dict, description="Goal priorities (1-5)")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    next_review_date: Optional[datetime] = Field(None, description="Next goal review date")
    
    def add_goal(
        self,
        goal: str,
        category: str,
        timeframe: str,
        priority: int = 3,
        details: Optional[Dict[str, Any]] = None
    ):
        """Add a new goal."""
        # Add to appropriate category
        if category == "production":
            self.production_goals.append(goal)
        elif category == "income":
            self.income_goals.append(goal)
        elif category == "sustainability":
            self.sustainability_goals.append(goal)
        elif category == "quality":
            self.quality_goals.append(goal)
        elif category == "lifestyle":
            self.lifestyle_goals.append(goal)
        
        # Add to timeframe
        if timeframe == "short":
            self.short_term_goals.append(goal)
        elif timeframe == "medium":
            self.medium_term_goals.append(goal)
        elif timeframe == "long":
            self.long_term_goals.append(goal)
        
        # Store details
        if details:
            self.goal_details[goal] = details
        
        # Set priority
        self.goal_priorities[goal] = max(1, min(priority, 5))
        
        # Mark as in progress
        self.in_progress_goals.append(goal)
        
        self.updated_at = datetime.utcnow()
    
    def complete_goal(self, goal: str):
        """Mark a goal as completed."""
        if goal in self.in_progress_goals:
            self.in_progress_goals.remove(goal)
            self.completed_goals.append(goal)
            self.updated_at = datetime.utcnow()