"""
Behavioral Learning Engine - Week 5-6 Implementation
Analyze farmer decisions and update behavioral models
"""
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import json


class LearningStyle(Enum):
    """Learning style types based on VARK model."""
    VISUAL = "visual"      # Learns by seeing
    AUDITORY = "auditory"  # Learns by hearing
    READ_WRITE = "read_write"  # Learns by reading/writing
    KINESTHETIC = "kinesthetic"  # Learns by doing
    MULTIMODAL = "multimodal"    # Mixed learning style


class StressResponsePattern(Enum):
    """How farmer responds to stress/risk."""
    RISK_AVERSE = "risk_averse"      # Avoids risk
    RISK_NEUTRAL = "risk_neutral"    # Calculated decisions
    RISK_SEEKING = "risk_seeking"    # Takes risks
    ADAPTIVE = "adaptive"            # Adjusts based on context


@dataclass
class BehavioralProfile:
    """Comprehensive behavioral profile of a farmer."""
    farmer_id: str
    risk_tolerance: float = 50.0  # 0-100 (0=risk averse, 100=risk seeking)
    innovation_adoption: float = 50.0  # 0-100 (speed of adopting new practices)
    planning_horizon: int = 30  # days (typical planning timeframe)
    learning_style: LearningStyle = LearningStyle.MULTIMODAL
    decision_confidence: float = 50.0  # 0-100 (confidence in own decisions)
    stress_response: StressResponsePattern = StressResponsePattern.RISK_NEUTRAL
    
    # Derived metrics
    consistency_score: float = 50.0  # 0-100 (consistency in decisions)
    adaptability_score: float = 50.0  # 0-100 (ability to adapt to changes)
    learning_velocity: float = 1.0  # Rate of learning improvement
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    last_decision_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "farmer_id": self.farmer_id,
            "risk_tolerance": self.risk_tolerance,
            "innovation_adoption": self.innovation_adoption,
            "planning_horizon": self.planning_horizon,
            "learning_style": self.learning_style.value,
            "decision_confidence": self.decision_confidence,
            "stress_response": self.stress_response.value,
            "consistency_score": self.consistency_score,
            "adaptability_score": self.adaptability_score,
            "learning_velocity": self.learning_velocity,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_decision_at": self.last_decision_at.isoformat() if self.last_decision_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BehavioralProfile':
        """Create from dictionary."""
        # Parse dates
        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
        updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        last_decision_at = datetime.fromisoformat(data["last_decision_at"]) if data.get("last_decision_at") else None
        
        return cls(
            farmer_id=data["farmer_id"],
            risk_tolerance=data.get("risk_tolerance", 50.0),
            innovation_adoption=data.get("innovation_adoption", 50.0),
            planning_horizon=data.get("planning_horizon", 30),
            learning_style=LearningStyle(data.get("learning_style", "multimodal")),
            decision_confidence=data.get("decision_confidence", 50.0),
            stress_response=StressResponsePattern(data.get("stress_response", "risk_neutral")),
            consistency_score=data.get("consistency_score", 50.0),
            adaptability_score=data.get("adaptability_score", 50.0),
            learning_velocity=data.get("learning_velocity", 1.0),
            created_at=created_at,
            updated_at=updated_at,
            last_decision_at=last_decision_at
        )


class BehavioralAnalyzer:
    """
    Analyzes farmer decisions and updates behavioral models.
    
    Features:
    1. Decision Pattern Recognition
    2. Behavioral Dimension Scoring
    3. Pattern Evolution Tracking
    4. Learning Progress Detection
    """
    
    def __init__(self):
        self.decision_history = defaultdict(list)
        self.feedback_history = defaultdict(list)
        self.behavioral_profiles = {}
        
        print("✅ BehavioralAnalyzer initialized")
    
    def analyze_decision_pattern(self, farmer_id: str, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single decision event.
        
        Args:
            farmer_id: Unique farmer identifier
            decision_data: {
                "decision_type": "recommendation_adoption",
                "action": "accepted" | "rejected" | "modified",
                "timing": "immediate" | "delayed" | "never",
                "risk_level": "low" | "medium" | "high",
                "decision_time_seconds": float,
                "outcome_confidence": float  # 0-1
            }
        
        Returns:
            Analysis results with behavioral insights
        """
        print(f"🔍 Analyzing decision pattern for farmer: {farmer_id}")
        
        # Store decision in history
        self.decision_history[farmer_id].append({
            **decision_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Get or create behavioral profile
        profile = self.get_behavioral_profile(farmer_id)
        
        # Update profile based on decision
        updated_profile = self._update_profile_from_decision(profile, decision_data)
        
        # Store updated profile
        self.behavioral_profiles[farmer_id] = updated_profile
        
        # Calculate pattern insights
        insights = self._calculate_decision_insights(farmer_id, decision_data)
        
        return {
            "farmer_id": farmer_id,
            "behavioral_profile": updated_profile.to_dict(),
            "insights": insights,
            "recommendations": self._generate_behavioral_recommendations(updated_profile)
        }
    
    def get_behavioral_profile(self, farmer_id: str) -> BehavioralProfile:
        """Get or create behavioral profile for farmer."""
        if farmer_id in self.behavioral_profiles:
            return self.behavioral_profiles[farmer_id]
        
        # Try to load from database
        profile = self._load_profile_from_db(farmer_id)
        if profile:
            self.behavioral_profiles[farmer_id] = profile
            return profile
        
        # Create new profile with default values
        profile = BehavioralProfile(farmer_id=farmer_id)
        self.behavioral_profiles[farmer_id] = profile
        return profile
    
    def _load_profile_from_db(self, farmer_id: str) -> Optional[BehavioralProfile]:
        """Load behavioral profile from database (placeholder)."""
        # TODO: Implement database loading
        # For now, return None to create new profile
        return None
    
    def _update_profile_from_decision(self, profile: BehavioralProfile, 
                                     decision_data: Dict[str, Any]) -> BehavioralProfile:
        """Update behavioral profile based on decision analysis."""
        action = decision_data.get("action", "")
        risk_level = decision_data.get("risk_level", "medium")
        decision_time = decision_data.get("decision_time_seconds", 60.0)
        outcome_confidence = decision_data.get("outcome_confidence", 0.5)
        
        # Update risk tolerance based on action and risk level
        risk_adjustment = self._calculate_risk_adjustment(action, risk_level)
        profile.risk_tolerance = np.clip(profile.risk_tolerance + risk_adjustment, 0, 100)
        
        # Update innovation adoption based on timing
        if action == "accepted":
            # Quick adoption increases innovation score
            if decision_time < 3600:  # Less than 1 hour
                profile.innovation_adoption = np.clip(profile.innovation_adoption + 5, 0, 100)
        
        # Update decision confidence based on outcome confidence
        profile.decision_confidence = np.clip(
            (profile.decision_confidence + outcome_confidence * 100) / 2,
            0, 100
        )
        
        # Update learning style based on decision patterns
        profile.learning_style = self._infer_learning_style(profile.farmer_id)
        
        # Update stress response pattern
        profile.stress_response = self._infer_stress_response(profile.farmer_id)
        
        # Update timestamps
        profile.updated_at = datetime.utcnow()
        profile.last_decision_at = datetime.utcnow()
        
        return profile
    
    def _calculate_risk_adjustment(self, action: str, risk_level: str) -> float:
        """Calculate risk tolerance adjustment."""
        adjustment_map = {
            ("accepted", "low"): 2.0,
            ("accepted", "medium"): 5.0,
            ("accepted", "high"): 8.0,
            ("rejected", "low"): -3.0,
            ("rejected", "medium"): -5.0,
            ("rejected", "high"): -10.0,
            ("modified", "low"): 0.0,
            ("modified", "medium"): 2.0,
            ("modified", "high"): 4.0,
        }
        return adjustment_map.get((action, risk_level), 0.0)
    
    def _infer_learning_style(self, farmer_id: str) -> LearningStyle:
        """Infer learning style from decision patterns."""
        decisions = self.decision_history.get(farmer_id, [])
        
        if not decisions:
            return LearningStyle.MULTIMODAL
        
        # Simple heuristic: count decision types
        visual_count = sum(1 for d in decisions if d.get("decision_type") == "visual_demo")
        auditory_count = sum(1 for d in decisions if d.get("decision_type") == "verbal_advice")
        reading_count = sum(1 for d in decisions if d.get("decision_type") == "written_guide")
        practice_count = sum(1 for d in decisions if d.get("decision_type") == "hands_on")
        
        counts = {
            LearningStyle.VISUAL: visual_count,
            LearningStyle.AUDITORY: auditory_count,
            LearningStyle.READ_WRITE: reading_count,
            LearningStyle.KINESTHETIC: practice_count
        }
        
        # Return style with most decisions, or multimodal if tie
        max_style = max(counts.items(), key=lambda x: x[1])
        if max_style[1] > 0:
            return max_style[0]
        return LearningStyle.MULTIMODAL
    
    def _infer_stress_response(self, farmer_id: str) -> StressResponsePattern:
        """Infer stress response pattern from decisions under pressure."""
        decisions = self.decision_history.get(farmer_id, [])
        high_risk_decisions = [d for d in decisions if d.get("risk_level") == "high"]
        
        if not high_risk_decisions:
            return StressResponsePattern.RISK_NEUTRAL
        
        # Analyze pattern in high-risk decisions
        accept_count = sum(1 for d in high_risk_decisions if d.get("action") == "accepted")
        reject_count = sum(1 for d in high_risk_decisions if d.get("action") == "rejected")
        modify_count = sum(1 for d in high_risk_decisions if d.get("action") == "modified")
        
        total = len(high_risk_decisions)
        accept_ratio = accept_count / total
        reject_ratio = reject_count / total
        
        if accept_ratio > 0.7:
            return StressResponsePattern.RISK_SEEKING
        elif reject_ratio > 0.7:
            return StressResponsePattern.RISK_AVERSE
        elif modify_count > 0:
            return StressResponsePattern.ADAPTIVE
        else:
            return StressResponsePattern.RISK_NEUTRAL
    
    def _calculate_decision_insights(self, farmer_id: str, 
                                   latest_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate insights from decision patterns."""
        decisions = self.decision_history.get(farmer_id, [])
        
        if len(decisions) < 2:
            return {"message": "Insufficient data for pattern analysis"}
        
        # Calculate consistency
        recent_decisions = decisions[-5:]  # Last 5 decisions
        actions = [d.get("action") for d in recent_decisions]
        consistency = len(set(actions)) / len(actions) if actions else 1.0
        
        # Calculate learning velocity (improvement over time)
        if len(decisions) >= 3:
            early_decisions = decisions[:3]
            recent_decisions = decisions[-3:]
            
            early_success = sum(1 for d in early_decisions if d.get("action") == "accepted") / 3
            recent_success = sum(1 for d in recent_decisions if d.get("action") == "accepted") / 3
            
            learning_velocity = recent_success - early_success
        else:
            learning_velocity = 0.0
        
        return {
            "decision_count": len(decisions),
            "consistency_score": round((1 - consistency) * 100, 2),  # Lower consistency = higher score
            "learning_velocity": round(learning_velocity, 3),
            "success_rate": round(sum(1 for d in decisions if d.get("action") == "accepted") / len(decisions), 3),
            "average_decision_time": round(np.mean([d.get("decision_time_seconds", 60) for d in decisions]), 2),
            "risk_exposure": self._calculate_risk_exposure(decisions)
        }
    
    def _calculate_risk_exposure(self, decisions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk exposure metrics."""
        risk_counts = {"low": 0, "medium": 0, "high": 0}
        for decision in decisions:
            risk_level = decision.get("risk_level", "medium")
            if risk_level in risk_counts:
                risk_counts[risk_level] += 1
        
        total = len(decisions)
        if total == 0:
            return {"low": 0, "medium": 0, "high": 0}
        
        return {
            "low": round(risk_counts["low"] / total, 3),
            "medium": round(risk_counts["medium"] / total, 3),
            "high": round(risk_counts["high"] / total, 3)
        }
    
    def _generate_behavioral_recommendations(self, profile: BehavioralProfile) -> List[Dict[str, Any]]:
        """Generate recommendations based on behavioral profile."""
        recommendations = []
        
        # Risk tolerance based recommendations
        if profile.risk_tolerance < 30:
            recommendations.append({
                "type": "risk_management",
                "title": "Consider low-risk investments",
                "description": "Your risk tolerance is low. Focus on proven, low-risk practices.",
                "priority": "medium"
            })
        elif profile.risk_tolerance > 70:
            recommendations.append({
                "type": "risk_management",
                "title": "Diversify risk exposure",
                "description": "Consider spreading risks across multiple strategies.",
                "priority": "high"
            })
        
        # Learning style based recommendations
        if profile.learning_style == LearningStyle.VISUAL:
            recommendations.append({
                "type": "learning_optimization",
                "title": "Use visual learning materials",
                "description": "You learn best visually. Watch demonstration videos.",
                "priority": "low"
            })
        elif profile.learning_style == LearningStyle.KINESTHETIC:
            recommendations.append({
                "type": "learning_optimization",
                "title": "Hands-on practice recommended",
                "description": "Learn by doing. Try practical exercises.",
                "priority": "low"
            })
        
        # Planning horizon recommendations
        if profile.planning_horizon < 7:
            recommendations.append({
                "type": "planning_improvement",
                "title": "Consider longer-term planning",
                "description": "Try extending your planning horizon for better resource management.",
                "priority": "medium"
            })
        
        return recommendations
    
    def track_pattern_evolution(self, farmer_id: str, timeframe_days: int = 30) -> Dict[str, Any]:
        """Track changes in behavioral patterns over time."""
        decisions = self.decision_history.get(farmer_id, [])
        
        if not decisions:
            return {"error": "No decision history available"}
        
        # Filter decisions within timeframe
        cutoff_date = datetime.utcnow() - timedelta(days=timeframe_days)
        recent_decisions = [
            d for d in decisions 
            if datetime.fromisoformat(d["timestamp"]) > cutoff_date
        ]
        
        if len(recent_decisions) < 3:
            return {"error": "Insufficient recent data"}
        
        # Calculate evolution metrics
        early_period = recent_decisions[:len(recent_decisions)//2]
        late_period = recent_decisions[len(recent_decisions)//2:]
        
        evolution = {
            "timeframe_days": timeframe_days,
            "total_decisions": len(recent_decisions),
            "success_rate_change": self._calculate_success_rate_change(early_period, late_period),
            "risk_tolerance_change": self._calculate_risk_tolerance_change(early_period, late_period),
            "decision_speed_change": self._calculate_decision_speed_change(early_period, late_period),
            "learning_progress": self._assess_learning_progress(early_period, late_period),
            "detected_patterns": self._detect_behavioral_patterns(recent_decisions)
        }
        
        return evolution
    
    def _calculate_success_rate_change(self, early: List, late: List) -> float:
        """Calculate change in success rate between periods."""
        early_success = sum(1 for d in early if d.get("action") == "accepted") / len(early) if early else 0
        late_success = sum(1 for d in late if d.get("action") == "accepted") / len(late) if late else 0
        return round(late_success - early_success, 3)
    
    def _calculate_risk_tolerance_change(self, early: List, late: List) -> float:
        """Calculate change in risk tolerance between periods."""
        early_risk = sum(1 for d in early if d.get("risk_level") == "high") / len(early) if early else 0
        late_risk = sum(1 for d in late if d.get("risk_level") == "high") / len(late) if late else 0
        return round(late_risk - early_risk, 3)
    
    def _calculate_decision_speed_change(self, early: List, late: List) -> float:
        """Calculate change in decision speed between periods."""
        early_speed = np.mean([d.get("decision_time_seconds", 60) for d in early]) if early else 60
        late_speed = np.mean([d.get("decision_time_seconds", 60) for d in late]) if late else 60
        return round(early_speed - late_speed, 2)  # Positive = faster decisions
    
    def _assess_learning_progress(self, early: List, late: List) -> Dict[str, Any]:
        """Assess learning progress between periods."""
        return {
            "improvement_rate": self._calculate_success_rate_change(early, late),
            "confidence_growth": 0.0,  # Placeholder
            "skill_acquisition": "intermediate"  # Placeholder
        }
    
    def _detect_behavioral_patterns(self, decisions: List) -> List[str]:
        """Detect specific behavioral patterns."""
        patterns = []
        
        # Check for consistency pattern
        actions = [d.get("action") for d in decisions]
        if len(set(actions)) == 1:
            patterns.append("high_consistency")
        
        # Check for risk pattern
        risk_levels = [d.get("risk_level") for d in decisions]
        if all(r == "high" for r in risk_levels):
            patterns.append("risk_seeking_tendency")
        elif all(r == "low" for r in risk_levels):
            patterns.append("risk_averse_tendency")
        
        # Check for learning pattern (improving success rate)
        if len(decisions) >= 5:
            first_half = decisions[:len(decisions)//2]
            second_half = decisions[len(decisions)//2:]
            first_success = sum(1 for d in first_half if d.get("action") == "accepted") / len(first_half)
            second_success = sum(1 for d in second_half if d.get("action") == "accepted") / len(second_half)
            
            if second_success > first_success + 0.2:  # 20% improvement
                patterns.append("rapid_learner")
        
        return patterns
    
    def get_behavioral_summary(self, farmer_id: str) -> Dict[str, Any]:
        """Get comprehensive behavioral summary for a farmer."""
        profile = self.get_behavioral_profile(farmer_id)
        decisions = self.decision_history.get(farmer_id, [])
        
        summary = {
            "farmer_id": farmer_id,
            "behavioral_profile": profile.to_dict(),
            "decision_statistics": {
                "total_decisions": len(decisions),
                "success_rate": round(sum(1 for d in decisions if d.get("action") == "accepted") / max(len(decisions), 1), 3),
                "average_risk_level": self._calculate_average_risk(decisions),
                "decision_frequency": self._calculate_decision_frequency(decisions)
            },
            "learning_assessment": {
                "learning_style_strength": self._assess_learning_style_strength(profile.learning_style),
                "adaptability_level": self._assess_adaptability_level(decisions),
                "growth_potential": self._assess_growth_potential(profile, decisions)
            },
            "recommendation_fit_score": self._calculate_recommendation_fit_score(profile)
        }
        
        return summary
    
    def _calculate_average_risk(self, decisions: List) -> str:
        """Calculate average risk level."""
        if not decisions:
            return "medium"
        
        risk_scores = {"low": 1, "medium": 2, "high": 3}
        total_score = sum(risk_scores.get(d.get("risk_level", "medium"), 2) for d in decisions)
        avg_score = total_score / len(decisions)
        
        if avg_score < 1.5:
            return "low"
        elif avg_score < 2.5:
            return "medium"
        else:
            return "high"
    
    def _calculate_decision_frequency(self, decisions: List) -> str:
        """Calculate decision frequency."""
        if len(decisions) < 2:
            return "unknown"
        
        timestamps = [datetime.fromisoformat(d["timestamp"]) for d in decisions if "timestamp" in d]
        if len(timestamps) < 2:
            return "unknown"
        
        timestamps.sort()
        intervals = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
        avg_interval = np.mean(intervals)
        
        if avg_interval < 7:
            return "high_frequency"
        elif avg_interval < 30:
            return "medium_frequency"
        else:
            return "low_frequency"
    
    def _assess_learning_style_strength(self, learning_style: LearningStyle) -> str:
        """Assess strength of learning style preference."""
        # Placeholder - in real implementation, this would analyze consistency
        return "moderate"
    
    def _assess_adaptability_level(self, decisions: List) -> str:
        """Assess adaptability level from decision patterns."""
        if len(decisions) < 5:
            return "unknown"
        
        # Count different types of decisions
        action_types = set(d.get("action") for d in decisions)
        decision_types = set(d.get("decision_type", "") for d in decisions)
        
        diversity_score = len(action_types) + len(decision_types)
        
        if diversity_score >= 6:
            return "high"
        elif diversity_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _assess_growth_potential(self, profile: BehavioralProfile, decisions: List) -> str:
        """Assess growth potential based on behavioral profile and history."""
        # Simple heuristic based on multiple factors
        score = 0
        
        # Higher risk tolerance = higher growth potential (to a point)
        if 40 <= profile.risk_tolerance <= 70:
            score += 2
        elif profile.risk_tolerance > 70:
            score += 1
        
        # Higher innovation adoption = higher growth potential
        if profile.innovation_adoption > 60:
            score += 2
        
        # Learning from decisions
        if len(decisions) >= 3:
            success_rate = sum(1 for d in decisions if d.get("action") == "accepted") / len(decisions)
            if success_rate > 0.6:
                score += 2
        
        if score >= 5:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"
    
    def _calculate_recommendation_fit_score(self, profile: BehavioralProfile) -> float:
        """Calculate how well recommendations should fit the farmer's profile."""
        # Base score on behavioral dimensions
        base_score = 50.0
        
        # Adjust based on profile characteristics
        adjustments = {
            "risk_tolerance": (profile.risk_tolerance - 50) * 0.1,  # Centered around 50
            "decision_confidence": (profile.decision_confidence - 50) * 0.05,
            "consistency_score": (profile.consistency_score - 50) * 0.05,
        }
        
        total_adjustment = sum(adjustments.values())
        fit_score = base_score + total_adjustment
        
        return round(np.clip(fit_score, 0, 100), 2)


# Create singleton instance for easy import
behavioral_analyzer = BehavioralAnalyzer()