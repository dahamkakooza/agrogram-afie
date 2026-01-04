"""
Adaptation Manager - Week 5-6 Implementation
Adapt system behavior based on farmer feedback and learning patterns
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Import from existing system
try:
    from src.core.recommendation_engine import RecommendationEngine, recommendation_engine
    from src.models.recommendation import Recommendation, RecommendationType, PriorityLevel
    from src.models.farm import FarmDNA
except ImportError:
    print("⚠️ Recommendation engine imports not available, using mock data")

# Import learning modules
from .behavioral_analyzer import behavioral_analyzer, BehavioralProfile, LearningStyle
from .game_data_processor import game_data_processor


class AdaptationTrigger(Enum):
    """Types of triggers that cause system adaptation."""
    FEEDBACK_EXPLICIT = "feedback_explicit"      # Direct farmer feedback
    FEEDBACK_IMPLICIT = "feedback_implicit"      # Behavioral patterns
    OUTCOME_OBSERVED = "outcome_observed"        # Actual results
    PATTERN_DETECTED = "pattern_detected"        # Behavioral pattern
    PERFORMANCE_CHANGE = "performance_change"    # Game/skill performance
    SYSTEM_LEARNING = "system_learning"          # Internal learning


class AdaptationAction(Enum):
    """Types of adaptation actions."""
    ADJUST_RECOMMENDATION_WEIGHT = "adjust_recommendation_weight"
    MODIFY_COMMUNICATION_STYLE = "modify_communication_style"
    CHANGE_TIMING_FREQUENCY = "change_timing_frequency"
    ADJUST_COMPLEXITY = "adjust_complexity"
    UPDATE_PERSONALIZATION = "update_personalization"
    ADD_NEW_CONTENT = "add_new_content"
    REMOVE_INEFFECTIVE = "remove_ineffective"
    INTRODUCE_VARIATION = "introduce_variation"
    LEARN_NEW_PATTERN = "learn_new_pattern"  # Added missing enum value


@dataclass
class AdaptationRule:
    """Rule for system adaptation."""
    trigger: AdaptationTrigger
    condition: Dict[str, Any]  # Conditions that must be met
    action: AdaptationAction
    parameters: Dict[str, Any]  # Action parameters
    weight: float = 1.0  # Rule importance/confidence
    last_applied: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    
    def should_apply(self, context: Dict[str, Any]) -> bool:
        """Check if rule should apply in given context."""
        # Check trigger
        if context.get("trigger") != self.trigger.value:
            return False
        
        # Check conditions
        for key, expected_value in self.condition.items():
            if context.get(key) != expected_value:
                return False
        
        # Check cooldown period (don't apply same rule too frequently)
        if self.last_applied:
            hours_since = (datetime.utcnow() - self.last_applied).total_seconds() / 3600
            if hours_since < 24:  # 24-hour cooldown
                return False
        
        return True
    
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptation rule."""
        self.last_applied = datetime.utcnow()
        
        result = {
            "rule_applied": True,
            "trigger": self.trigger.value,
            "action": self.action.value,
            "parameters": self.parameters,
            "weight": self.weight,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Record success/failure
        if context.get("success", True):
            self.success_count += 1
            result["outcome"] = "success"
        else:
            self.failure_count += 1
            result["outcome"] = "failure"
        
        # Update weight based on success rate
        total_applications = self.success_count + self.failure_count
        if total_applications > 5:
            success_rate = self.success_count / total_applications
            # Adjust weight based on success (0.5 to 1.5 range)
            self.weight = 0.5 + success_rate
        
        return result


class AdaptationManager:
    """
    Adapt system behavior based on farmer feedback and learning.
    
    Features:
    1. Feedback Analysis (explicit & implicit)
    2. Adaptation Rules Engine
    3. Personalization Evolution
    4. System Learning Tracking
    """
    
    def __init__(self):
        self.adaptation_rules = self._initialize_default_rules()
        self.adaptation_history = defaultdict(list)
        self.personalization_profiles = defaultdict(dict)
        self.feedback_analysis_cache = {}
        
        print("✅ AdaptationManager initialized")
    
    def _initialize_default_rules(self) -> List[AdaptationRule]:
        """Initialize default adaptation rules."""
        rules = [
            # Rule 1: Farmer rejects recommendation
            AdaptationRule(
                trigger=AdaptationTrigger.FEEDBACK_EXPLICIT,
                condition={"action": "rejected", "feedback_type": "recommendation"},
                action=AdaptationAction.ADJUST_RECOMMENDATION_WEIGHT,
                parameters={
                    "adjustment": -0.3,
                    "scope": "similar_recommendations",
                    "decay_rate": 0.1  # Per week
                },
                weight=0.8
            ),
            
            # Rule 2: Farmer succeeds with variation
            AdaptationRule(
                trigger=AdaptationTrigger.OUTCOME_OBSERVED,
                condition={"outcome": "success_with_variation", "confidence": "high"},
                action=AdaptationAction.LEARN_NEW_PATTERN,
                parameters={
                    "pattern_type": "successful_variation",
                    "confidence_increase": 0.2,
                    "application_scope": "similar_contexts"
                },
                weight=0.9
            ),
            
            # Rule 3: Rapid learner detected
            AdaptationRule(
                trigger=AdaptationTrigger.PATTERN_DETECTED,
                condition={"pattern": "rapid_learner", "confidence": "high"},
                action=AdaptationAction.ADJUST_COMPLEXITY,
                parameters={
                    "complexity_increase": 0.25,
                    "pace_increase": 0.3,
                    "depth_increase": 0.2
                },
                weight=0.7
            ),
            
            # Rule 4: Consistent failure pattern
            AdaptationRule(
                trigger=AdaptationTrigger.PERFORMANCE_CHANGE,
                condition={"performance_trend": "declining", "sessions": ">=3"},
                action=AdaptationAction.ADJUST_COMPLEXITY,
                parameters={
                    "complexity_decrease": 0.3,
                    "additional_support": True,
                    "review_fundamentals": True
                },
                weight=0.85
            ),
            
            # Rule 5: Visual learner detected
            AdaptationRule(
                trigger=AdaptationTrigger.PATTERN_DETECTED,
                condition={"learning_style": "visual", "consistency": "high"},
                action=AdaptationAction.MODIFY_COMMUNICATION_STYLE,
                parameters={
                    "visual_content_increase": 0.4,
                    "text_content_decrease": 0.2,
                    "diagram_preference": True
                },
                weight=0.75
            ),
            
            # Rule 6: Risk-averse farmer
            AdaptationRule(
                trigger=AdaptationTrigger.PATTERN_DETECTED,
                condition={"risk_profile": "averse", "consistency": "high"},
                action=AdaptationAction.ADJUST_RECOMMENDATION_WEIGHT,
                parameters={
                    "low_risk_bias": 0.4,
                    "incremental_changes": True,
                    "success_evidence_weight": 0.8
                },
                weight=0.8
            ),
            
            # Rule 7: High engagement detected
            AdaptationRule(
                trigger=AdaptationTrigger.PATTERN_DETECTED,
                condition={"engagement_level": "high", "frequency": ">=daily"},
                action=AdaptationAction.CHANGE_TIMING_FREQUENCY,
                parameters={
                    "frequency_increase": 0.2,
                    "content_volume": "moderate",
                    "push_notifications": True
                },
                weight=0.65
            ),
            
            # Rule 8: Information overload
            AdaptationRule(
                trigger=AdaptationTrigger.FEEDBACK_IMPLICIT,
                condition={"dropoff_rate": "high", "completion_rate": "low"},
                action=AdaptationAction.ADJUST_COMPLEXITY,
                parameters={
                    "simplify_content": True,
                    "chunk_information": True,
                    "reduce_frequency": 0.3
                },
                weight=0.9
            )
        ]
        
        return rules
    
    # ... (rest of the file remains exactly the same)
    # All other methods should stay unchanged
    
    def analyze_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze farmer feedback (explicit and implicit).
        
        Args:
            feedback_data: {
                "farmer_id": "farmer_001",
                "feedback_type": "explicit" | "implicit",
                "feedback": {
                    "type": "recommendation" | "content" | "system",
                    "action": "accepted" | "rejected" | "modified",
                    "rating": 1-5,
                    "comments": "text feedback",
                    "timestamp": "iso_string"
                },
                "context": {
                    "recommendation_id": "rec_123",
                    "content_type": "game" | "advice",
                    "session_id": "session_123"
                }
            }
        
        Returns:
            Analysis results and adaptation triggers
        """
        print(f"📊 Analyzing feedback for farmer: {feedback_data.get('farmer_id')}")
        
        farmer_id = feedback_data["farmer_id"]
        feedback_type = feedback_data["feedback_type"]
        feedback = feedback_data.get("feedback", {})
        context = feedback_data.get("context", {})
        
        analysis = {
            "farmer_id": farmer_id,
            "timestamp": datetime.utcnow().isoformat(),
            "feedback_type": feedback_type,
            "original_feedback": feedback,
            "context": context
        }
        
        # Analyze based on feedback type
        if feedback_type == "explicit":
            analysis.update(self._analyze_explicit_feedback(feedback, context))
        else:  # implicit
            analysis.update(self._analyze_implicit_feedback(feedback, context))
        
        # Store in history
        self.adaptation_history[farmer_id].append(analysis)
        
        # Check for adaptation triggers
        triggers = self._check_adaptation_triggers(analysis)
        if triggers:
            analysis["adaptation_triggers"] = triggers
            # Apply adaptations
            for trigger in triggers:
                self.apply_adaptation(farmer_id, trigger, analysis)
        
        # Update personalization profile
        self._update_personalization_profile(farmer_id, analysis)
        
        # Cache analysis for quick reference
        cache_key = f"{farmer_id}_{datetime.utcnow().strftime('%Y%m%d')}"
        self.feedback_analysis_cache[cache_key] = analysis
        
        return analysis
    
    def _analyze_explicit_feedback(self, feedback: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze explicit (direct) feedback."""
        analysis = {
            "feedback_category": "explicit",
            "sentiment": self._analyze_sentiment(feedback.get("comments", "")),
            "action_taken": feedback.get("action", "unknown"),
            "rating": feedback.get("rating", 0),
            "confidence": self._calculate_feedback_confidence(feedback)
        }
        
        # Categorize feedback
        feedback_type = feedback.get("type", "unknown")
        if feedback_type == "recommendation":
            analysis["recommendation_impact"] = self._assess_recommendation_impact(
                feedback.get("action"), 
                feedback.get("rating", 3)
            )
        elif feedback_type == "content":
            analysis["content_effectiveness"] = self._assess_content_effectiveness(
                feedback.get("rating", 3),
                feedback.get("comments", "")
            )
        
        return analysis
    
    def _analyze_implicit_feedback(self, feedback: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze implicit (behavioral) feedback."""
        analysis = {
            "feedback_category": "implicit",
            "behavioral_indicators": self._extract_behavioral_indicators(feedback, context),
            "engagement_level": self._assess_engagement_level(feedback, context),
            "learning_signals": self._detect_learning_signals(feedback, context)
        }
        
        # Get behavioral profile for context
        farmer_id = context.get("farmer_id")
        if farmer_id:
            try:
                profile = behavioral_analyzer.get_behavioral_profile(farmer_id)
                analysis["behavioral_context"] = {
                    "risk_tolerance": profile.risk_tolerance,
                    "learning_style": profile.learning_style.value,
                    "decision_confidence": profile.decision_confidence
                }
            except:
                pass
        
        return analysis
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment from text feedback."""
        if not text:
            return {"polarity": "neutral", "score": 0, "keywords": []}
        
        # Simple keyword-based sentiment analysis
        positive_words = ["good", "great", "excellent", "helpful", "useful", "thanks", "thank"]
        negative_words = ["bad", "poor", "useless", "wrong", "confusing", "difficult", "hard"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        score = positive_count - negative_count
        
        if score > 0:
            polarity = "positive"
        elif score < 0:
            polarity = "negative"
        else:
            polarity = "neutral"
        
        return {
            "polarity": polarity,
            "score": score,
            "keywords": [w for w in positive_words + negative_words if w in text_lower]
        }
    
    def _calculate_feedback_confidence(self, feedback: Dict[str, Any]) -> float:
        """Calculate confidence level of feedback."""
        confidence = 0.5  # Base confidence
        
        # Rating increases confidence
        rating = feedback.get("rating", 0)
        if rating in [1, 5]:  # Extreme ratings are more confident
            confidence += 0.3
        elif rating in [2, 4]:
            confidence += 0.15
        
        # Detailed comments increase confidence
        if feedback.get("comments") and len(feedback["comments"]) > 20:
            confidence += 0.2
        
        # Action taken increases confidence
        if feedback.get("action") in ["accepted", "rejected"]:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _assess_recommendation_impact(self, action: str, rating: int) -> Dict[str, Any]:
        """Assess impact of recommendation feedback."""
        impact_map = {
            "accepted": {"adoption_rate": "high", "satisfaction": "high" if rating >= 4 else "medium"},
            "rejected": {"adoption_rate": "low", "satisfaction": "low" if rating <= 2 else "medium"},
            "modified": {"adoption_rate": "medium", "satisfaction": "medium"},
            "unknown": {"adoption_rate": "unknown", "satisfaction": "unknown"}
        }
        
        assessment = impact_map.get(action, impact_map["unknown"])
        assessment["rating_impact"] = "positive" if rating >= 4 else "negative" if rating <= 2 else "neutral"
        
        return assessment
    
    def _assess_content_effectiveness(self, rating: int, comments: str) -> Dict[str, Any]:
        """Assess effectiveness of content based on feedback."""
        effectiveness = {
            "rating_based": "high" if rating >= 4 else "low" if rating <= 2 else "medium",
            "engagement_indicators": [],
            "improvement_areas": []
        }
        
        # Analyze comments for specific feedback
        comment_lower = comments.lower()
        
        if "confus" in comment_lower:
            effectiveness["improvement_areas"].append("clarity")
        if "too long" in comment_lower or "length" in comment_lower:
            effectiveness["improvement_areas"].append("brevity")
        if "helpful" in comment_lower or "useful" in comment_lower:
            effectiveness["engagement_indicators"].append("utility")
        if "interesting" in comment_lower or "engaging" in comment_lower:
            effectiveness["engagement_indicators"].append("engagement")
        
        return effectiveness
    
    def _extract_behavioral_indicators(self, feedback: Dict[str, Any], 
                                     context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract behavioral indicators from implicit feedback."""
        indicators = []
        
        # Time-based indicators
        if "time_spent" in feedback:
            time_spent = feedback["time_spent"]
            indicators.append({
                "type": "engagement_duration",
                "value": time_spent,
                "category": "high" if time_spent > 300 else "medium" if time_spent > 60 else "low"
            })
        
        # Interaction indicators
        if "interaction_count" in feedback:
            count = feedback["interaction_count"]
            indicators.append({
                "type": "interaction_frequency",
                "value": count,
                "category": "high" if count > 10 else "medium" if count > 3 else "low"
            })
        
        # Completion indicators
        if "completion_rate" in feedback:
            rate = feedback["completion_rate"]
            indicators.append({
                "type": "completion_behavior",
                "value": rate,
                "category": "high" if rate > 0.8 else "medium" if rate > 0.5 else "low"
            })
        
        # Return patterns
        if "return_frequency" in feedback:
            freq = feedback["return_frequency"]
            indicators.append({
                "type": "loyalty_behavior",
                "value": freq,
                "category": "regular" if freq > 3 else "occasional" if freq > 0 else "new"
            })
        
        return indicators
    
    def _assess_engagement_level(self, feedback: Dict[str, Any], 
                               context: Dict[str, Any]) -> str:
        """Assess overall engagement level."""
        engagement_score = 0
        
        # Time spent contributes to engagement
        if "time_spent" in feedback:
            time_spent = feedback["time_spent"]
            if time_spent > 600:  # >10 minutes
                engagement_score += 3
            elif time_spent > 180:  # >3 minutes
                engagement_score += 2
            elif time_spent > 60:  # >1 minute
                engagement_score += 1
        
        # Interaction frequency
        if "interaction_count" in feedback:
            count = feedback["interaction_count"]
            if count > 15:
                engagement_score += 3
            elif count > 5:
                engagement_score += 2
            elif count > 2:
                engagement_score += 1
        
        # Completion behavior
        if "completion_rate" in feedback:
            rate = feedback["completion_rate"]
            if rate > 0.9:
                engagement_score += 3
            elif rate > 0.7:
                engagement_score += 2
            elif rate > 0.5:
                engagement_score += 1
        
        if engagement_score >= 7:
            return "high"
        elif engagement_score >= 4:
            return "medium"
        else:
            return "low"
    
    def _detect_learning_signals(self, feedback: Dict[str, Any], 
                               context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect learning-related signals from behavior."""
        signals = []
        
        # Improvement over time
        if "performance_trend" in feedback:
            trend = feedback["performance_trend"]
            if trend == "improving":
                signals.append({
                    "signal": "learning_progress",
                    "strength": "strong",
                    "evidence": "performance_improvement"
                })
        
        # Error correction
        if "error_correction_rate" in feedback:
            rate = feedback["error_correction_rate"]
            if rate > 0.7:
                signals.append({
                    "signal": "adaptive_learning",
                    "strength": "strong",
                    "evidence": "high_error_correction"
                })
        
        # Exploration behavior
        if "exploration_rate" in feedback:
            rate = feedback["exploration_rate"]
            if 0.3 < rate < 0.7:  # Balanced exploration
                signals.append({
                    "signal": "balanced_learning",
                    "strength": "medium",
                    "evidence": "optimal_exploration"
                })
        
        return signals
    
    def _check_adaptation_triggers(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check analysis for adaptation triggers."""
        triggers = []
        
        farmer_id = analysis["farmer_id"]
        feedback_category = analysis.get("feedback_category")
        
        # Check explicit feedback triggers
        if feedback_category == "explicit":
            action = analysis.get("action_taken")
            rating = analysis.get("rating", 0)
            sentiment = analysis.get("sentiment", {}).get("polarity", "neutral")
            
            # Low rating trigger
            if rating <= 2:
                triggers.append({
                    "trigger": AdaptationTrigger.FEEDBACK_EXPLICIT.value,
                    "action": "rejected" if action == "rejected" else "low_rating",
                    "rating": rating,
                    "sentiment": sentiment,
                    "confidence": analysis.get("confidence", 0.5)
                })
            
            # Rejection trigger
            if action == "rejected":
                triggers.append({
                    "trigger": AdaptationTrigger.FEEDBACK_EXPLICIT.value,
                    "action": "rejected",
                    "rating": rating,
                    "sentiment": sentiment,
                    "confidence": analysis.get("confidence", 0.5)
                })
        
        # Check implicit feedback triggers
        elif feedback_category == "implicit":
            engagement = analysis.get("engagement_level", "low")
            indicators = analysis.get("behavioral_indicators", [])
            
            # Low engagement trigger
            if engagement == "low":
                triggers.append({
                    "trigger": AdaptationTrigger.FEEDBACK_IMPLICIT.value,
                    "pattern": "low_engagement",
                    "engagement_level": engagement,
                    "confidence": 0.6
                })
            
            # Check for specific behavioral patterns
            for indicator in indicators:
                if indicator.get("type") == "completion_behavior" and indicator.get("category") == "low":
                    triggers.append({
                        "trigger": AdaptationTrigger.FEEDBACK_IMPLICIT.value,
                        "pattern": "low_completion",
                        "value": indicator.get("value"),
                        "confidence": 0.7
                    })
        
        # Get learning signals
        learning_signals = analysis.get("learning_signals", [])
        for signal in learning_signals:
            if signal.get("signal") == "learning_progress" and signal.get("strength") == "strong":
                triggers.append({
                    "trigger": AdaptationTrigger.PERFORMANCE_CHANGE.value,
                    "pattern": "rapid_learner",
                    "evidence": signal.get("evidence"),
                    "confidence": 0.8
                })
        
        return triggers
    
    def apply_adaptation(self, farmer_id: str, trigger: Dict[str, Any], 
                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply adaptation rules based on trigger.
        
        Args:
            farmer_id: Farmer identifier
            trigger: Trigger information
            context: Full context for adaptation
        
        Returns:
            List of applied adaptations
        """
        print(f"🔄 Applying adaptation for farmer: {farmer_id}")
        
        applied_adaptations = []
        
        # Find matching rules
        matching_rules = []
        for rule in self.adaptation_rules:
            if rule.should_apply(trigger):
                matching_rules.append(rule)
        
        # Apply matching rules
        for rule in matching_rules:
            adaptation_result = rule.apply(context)
            adaptation_result["farmer_id"] = farmer_id
            adaptation_result["rule_id"] = id(rule)  # Simple identifier
            
            # Store in history
            self.adaptation_history[farmer_id].append({
                "type": "adaptation_applied",
                "timestamp": datetime.utcnow().isoformat(),
                "adaptation": adaptation_result,
                "trigger": trigger,
                "context": context
            })
            
            applied_adaptations.append(adaptation_result)
            
            # Execute the adaptation action
            self._execute_adaptation_action(farmer_id, rule.action, rule.parameters, context)
        
        # If no rules matched, consider creating new rule
        if not applied_adaptations and trigger.get("confidence", 0) > 0.7:
            new_rule = self._create_adaptation_rule(trigger, context)
            if new_rule:
                self.adaptation_rules.append(new_rule)
                applied_adaptations.append({
                    "rule_applied": False,
                    "new_rule_created": True,
                    "rule_description": f"New rule for {trigger.get('trigger')}",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return applied_adaptations
    
    def _execute_adaptation_action(self, farmer_id: str, action: AdaptationAction,
                                 parameters: Dict[str, Any], context: Dict[str, Any]):
        """Execute specific adaptation action."""
        action_handlers = {
            AdaptationAction.ADJUST_RECOMMENDATION_WEIGHT: self._adjust_recommendation_weight,
            AdaptationAction.MODIFY_COMMUNICATION_STYLE: self._modify_communication_style,
            AdaptationAction.CHANGE_TIMING_FREQUENCY: self._change_timing_frequency,
            AdaptationAction.ADJUST_COMPLEXITY: self._adjust_complexity,
            AdaptationAction.UPDATE_PERSONALIZATION: self._update_personalization,
            AdaptationAction.ADD_NEW_CONTENT: self._add_new_content,
            AdaptationAction.REMOVE_INEFFECTIVE: self._remove_ineffective,
            AdaptationAction.INTRODUCE_VARIATION: self._introduce_variation,
        }
        
        handler = action_handlers.get(action)
        if handler:
            handler(farmer_id, parameters, context)
    
    def _adjust_recommendation_weight(self, farmer_id: str, 
                                    parameters: Dict[str, Any], context: Dict[str, Any]):
        """Adjust recommendation weights for farmer."""
        adjustment = parameters.get("adjustment", 0)
        scope = parameters.get("scope", "all")
        
        # Get or create personalization profile
        profile = self.personalization_profiles[farmer_id]
        
        if "recommendation_weights" not in profile:
            profile["recommendation_weights"] = {}
        
        # Apply adjustment
        if scope == "all":
            profile["recommendation_weights"]["global_adjustment"] = adjustment
        elif scope == "similar_recommendations":
            # Apply to recent recommendation types
            recent_type = context.get("recommendation_type")
            if recent_type:
                profile["recommendation_weights"][f"type_{recent_type}"] = adjustment
        
        print(f"  📈 Adjusted recommendation weights for {farmer_id}: {adjustment}")
    
    def _modify_communication_style(self, farmer_id: str, 
                                  parameters: Dict[str, Any], context: Dict[str, Any]):
        """Modify communication style preferences."""
        profile = self.personalization_profiles[farmer_id]
        
        if "communication_preferences" not in profile:
            profile["communication_preferences"] = {}
        
        # Update preferences
        for key, value in parameters.items():
            profile["communication_preferences"][key] = value
        
        print(f"  💬 Modified communication style for {farmer_id}")
    
    def _change_timing_frequency(self, farmer_id: str, 
                               parameters: Dict[str, Any], context: Dict[str, Any]):
        """Change timing and frequency of communications."""
        profile = self.personalization_profiles[farmer_id]
        
        if "timing_preferences" not in profile:
            profile["timing_preferences"] = {}
        
        # Update timing preferences
        profile["timing_preferences"].update(parameters)
        
        print(f"  ⏰ Changed timing/frequency for {farmer_id}")
    
    def _adjust_complexity(self, farmer_id: str, 
                          parameters: Dict[str, Any], context: Dict[str, Any]):
        """Adjust content complexity level."""
        profile = self.personalization_profiles[farmer_id]
        
        if "complexity_level" not in profile:
            profile["complexity_level"] = {"current": 0.5, "target": 0.5}
        
        # Apply complexity adjustment
        current = profile["complexity_level"]["current"]
        adjustment = parameters.get("complexity_increase", 0) or parameters.get("complexity_decrease", 0)
        
        new_complexity = max(0, min(1, current + adjustment))
        profile["complexity_level"]["current"] = new_complexity
        profile["complexity_level"]["target"] = new_complexity
        
        # Store adjustment parameters
        profile["complexity_level"]["last_adjustment"] = {
            "amount": adjustment,
            "reason": context.get("trigger"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        print(f"  📊 Adjusted complexity for {farmer_id}: {new_complexity:.2f}")
    
    def _update_personalization(self, farmer_id: str, 
                              parameters: Dict[str, Any], context: Dict[str, Any]):
        """Update general personalization settings."""
        profile = self.personalization_profiles[farmer_id]
        profile.update(parameters)
        
        print(f"  🎯 Updated personalization for {farmer_id}")
    
    def _add_new_content(self, farmer_id: str, 
                        parameters: Dict[str, Any], context: Dict[str, Any]):
        """Add new content based on learning needs."""
        profile = self.personalization_profiles[farmer_id]
        
        if "content_queue" not in profile:
            profile["content_queue"] = []
        
        # Add new content to queue
        new_content = {
            "type": parameters.get("content_type", "learning_material"),
            "topic": parameters.get("topic", "general"),
            "priority": parameters.get("priority", "medium"),
            "added_at": datetime.utcnow().isoformat()
        }
        
        profile["content_queue"].append(new_content)
        
        print(f"  📚 Added new content for {farmer_id}")
    
    def _remove_ineffective(self, farmer_id: str, 
                           parameters: Dict[str, Any], context: Dict[str, Any]):
        """Remove ineffective content or approaches."""
        # Mark content as ineffective
        if "ineffective_content" not in self.personalization_profiles[farmer_id]:
            self.personalization_profiles[farmer_id]["ineffective_content"] = []
        
        ineffective_entry = {
            "content_id": parameters.get("content_id"),
            "reason": parameters.get("reason", "low_effectiveness"),
            "removed_at": datetime.utcnow().isoformat(),
            "evidence": context.get("evidence", {})
        }
        
        self.personalization_profiles[farmer_id]["ineffective_content"].append(ineffective_entry)
        
        print(f"  🗑️  Marked content as ineffective for {farmer_id}")
    
    def _introduce_variation(self, farmer_id: str, 
                            parameters: Dict[str, Any], context: Dict[str, Any]):
        """Introduce variation in content or approach."""
        profile = self.personalization_profiles[farmer_id]
        
        if "variation_settings" not in profile:
            profile["variation_settings"] = {"enabled": True, "variation_level": 0.3}
        
        # Update variation settings
        profile["variation_settings"].update(parameters)
        
        print(f"  🔄 Introduced variation for {farmer_id}")
    
    def _create_adaptation_rule(self, trigger: Dict[str, Any], 
                              context: Dict[str, Any]) -> Optional[AdaptationRule]:
        """Create new adaptation rule based on trigger pattern."""
        trigger_type = trigger.get("trigger")
        pattern = trigger.get("pattern", "")
        
        # Only create rules for patterns with high confidence
        if trigger.get("confidence", 0) < 0.7:
            return None
        
        # Map patterns to appropriate actions
        rule_mapping = {
            "rapid_learner": (
                AdaptationAction.ADJUST_COMPLEXITY,
                {"complexity_increase": 0.2, "pace_increase": 0.25}
            ),
            "low_engagement": (
                AdaptationAction.CHANGE_TIMING_FREQUENCY,
                {"frequency_decrease": 0.3, "simplify_content": True}
            ),
            "high_success_rate": (
                AdaptationAction.INTRODUCE_VARIATION,
                {"variation_level": 0.4, "new_challenges": True}
            )
        }
        
        if pattern in rule_mapping:
            action, parameters = rule_mapping[pattern]
            
            # Create condition from trigger
            condition = {
                "trigger": trigger_type,
                "pattern": pattern,
                "confidence": "high"
            }
            
            return AdaptationRule(
                trigger=AdaptationTrigger(trigger_type),
                condition=condition,
                action=action,
                parameters=parameters,
                weight=0.6,  # Start with moderate weight
                last_applied=datetime.utcnow()
            )
        
        return None
    
    def _update_personalization_profile(self, farmer_id: str, analysis: Dict[str, Any]):
        """Update personalization profile based on feedback analysis."""
        if farmer_id not in self.personalization_profiles:
            self.personalization_profiles[farmer_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "last_updated": datetime.utcnow().isoformat(),
                "adaptation_count": 0,
                "feedback_history": []
            }
        
        profile = self.personalization_profiles[farmer_id]
        profile["last_updated"] = datetime.utcnow().isoformat()
        profile["adaptation_count"] = profile.get("adaptation_count", 0) + 1
        
        # Store feedback analysis
        if "feedback_history" not in profile:
            profile["feedback_history"] = []
        
        profile["feedback_history"].append({
            "timestamp": analysis["timestamp"],
            "feedback_type": analysis["feedback_type"],
            "key_insights": self._extract_key_insights(analysis),
            "trigger_count": len(analysis.get("adaptation_triggers", []))
        })
        
        # Limit history size
        if len(profile["feedback_history"]) > 100:
            profile["feedback_history"] = profile["feedback_history"][-50:]
    
    def _extract_key_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract key insights from feedback analysis."""
        insights = []
        
        if analysis.get("feedback_category") == "explicit":
            action = analysis.get("action_taken")
            rating = analysis.get("rating", 0)
            sentiment = analysis.get("sentiment", {}).get("polarity")
            
            if action:
                insights.append(f"Action: {action}")
            if rating:
                insights.append(f"Rating: {rating}/5")
            if sentiment:
                insights.append(f"Sentiment: {sentiment}")
        
        elif analysis.get("feedback_category") == "implicit":
            engagement = analysis.get("engagement_level")
            if engagement:
                insights.append(f"Engagement: {engagement}")
            
            signals = analysis.get("learning_signals", [])
            for signal in signals[:2]:  # Top 2 signals
                insights.append(f"Learning: {signal.get('signal')}")
        
        return insights
    
    def evolve_personalization(self, farmer_id: str) -> Dict[str, Any]:
        """
        Evolve personalization settings based on accumulated learning.
        
        Features:
        1. Gradually increase recommendation complexity
        2. Adapt communication style preferences
        3. Adjust timing and frequency of advice
        4. Customize information depth
        """
        print(f"🌱 Evolving personalization for farmer: {farmer_id}")
        
        if farmer_id not in self.personalization_profiles:
            return {"error": "No personalization profile found"}
        
        profile = self.personalization_profiles[farmer_id]
        history = self.adaptation_history.get(farmer_id, [])
        
        # Get behavioral profile for context
        try:
            behavioral_profile = behavioral_analyzer.get_behavioral_profile(farmer_id)
            behavioral_context = {
                "risk_tolerance": behavioral_profile.risk_tolerance,
                "learning_style": behavioral_profile.learning_style.value,
                "decision_confidence": behavioral_profile.decision_confidence,
                "adaptability_score": behavioral_profile.adaptability_score
            }
        except:
            behavioral_context = {}
        
        evolution_plan = {
            "farmer_id": farmer_id,
            "timestamp": datetime.utcnow().isoformat(),
            "behavioral_context": behavioral_context,
            "current_profile": profile.copy(),
            "evolution_changes": []
        }
        
        # 1. Gradually increase recommendation complexity
        current_complexity = profile.get("complexity_level", {}).get("current", 0.5)
        adaptation_count = profile.get("adaptation_count", 0)
        
        if adaptation_count > 10 and current_complexity < 0.8:
            complexity_increase = min(0.15, (adaptation_count - 10) * 0.01)
            new_complexity = current_complexity + complexity_increase
            
            evolution_plan["evolution_changes"].append({
                "aspect": "complexity",
                "change": f"Increase from {current_complexity:.2f} to {new_complexity:.2f}",
                "rationale": f"Based on {adaptation_count} successful adaptations",
                "implementation": "gradual_over_2_weeks"
            })
            
            # Update profile
            if "complexity_level" not in profile:
                profile["complexity_level"] = {}
            profile["complexity_level"]["target"] = new_complexity
        
        # 2. Adapt communication style based on behavioral profile
        if behavioral_context:
            learning_style = behavioral_context.get("learning_style")
            
            if learning_style and "communication_preferences" not in profile:
                profile["communication_preferences"] = {}
            
            if learning_style == "visual":
                evolution_plan["evolution_changes"].append({
                    "aspect": "communication_style",
                    "change": "Increase visual content by 30%",
                    "rationale": "Visual learner preference detected",
                    "implementation": "immediate"
                })
                profile["communication_preferences"]["visual_bias"] = 0.3
            
            elif learning_style == "auditory":
                evolution_plan["evolution_changes"].append({
                    "aspect": "communication_style",
                    "change": "Add audio explanations",
                    "rationale": "Auditory learner preference",
                    "implementation": "within_1_week"
                })
                profile["communication_preferences"]["audio_enabled"] = True
        
        # 3. Adjust timing and frequency based on engagement patterns
        engagement_pattern = self._analyze_engagement_pattern(history)
        
        if engagement_pattern:
            optimal_timing = self._calculate_optimal_timing(engagement_pattern)
            
            evolution_plan["evolution_changes"].append({
                "aspect": "timing_frequency",
                "change": f"Adjust to {optimal_timing['frequency']} frequency",
                "rationale": f"Based on {engagement_pattern['pattern']} engagement",
                "implementation": "gradual_adjustment"
            })
            
            if "timing_preferences" not in profile:
                profile["timing_preferences"] = {}
            profile["timing_preferences"].update(optimal_timing)
        
        # 4. Customize information depth based on performance
        performance_level = self._assess_performance_level(farmer_id)
        
        if performance_level == "advanced":
            evolution_plan["evolution_changes"].append({
                "aspect": "information_depth",
                "change": "Increase technical detail by 25%",
                "rationale": "Advanced performance level detected",
                "implementation": "selective_content"
            })
            profile["information_depth"] = "detailed"
        
        elif performance_level == "beginner":
            evolution_plan["evolution_changes"].append({
                "aspect": "information_depth",
                "change": "Simplify explanations, focus on fundamentals",
                "rationale": "Beginner level support needed",
                "implementation": "all_content"
            })
            profile["information_depth"] = "simplified"
        
        # 5. Update success metrics tracking
        profile["evolution_milestones"] = profile.get("evolution_milestones", [])
        profile["evolution_milestones"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "changes_applied": len(evolution_plan["evolution_changes"]),
            "complexity_level": profile.get("complexity_level", {}).get("target", 0.5)
        })
        
        evolution_plan["updated_profile"] = profile.copy()
        
        return evolution_plan
    
    def _analyze_engagement_pattern(self, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze engagement patterns from history."""
        if len(history) < 5:
            return None
        
        # Analyze timing patterns
        timestamps = []
        for entry in history[-10:]:  # Last 10 entries
            if "timestamp" in entry:
                try:
                    ts = datetime.fromisoformat(entry["timestamp"])
                    timestamps.append(ts)
                except:
                    pass
        
        if len(timestamps) < 3:
            return None
        
        # Calculate time between engagements
        timestamps.sort()
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # Hours
            intervals.append(interval)
        
        avg_interval = np.mean(intervals) if intervals else 0
        
        if avg_interval < 24:
            pattern = "daily_engagement"
        elif avg_interval < 168:  # 7 days
            pattern = "weekly_engagement"
        else:
            pattern = "occasional_engagement"
        
        return {
            "pattern": pattern,
            "avg_interval_hours": round(avg_interval, 1),
            "consistency": "high" if np.std(intervals) < avg_interval * 0.5 else "medium"
        }
    
    def _calculate_optimal_timing(self, engagement_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing based on engagement pattern."""
        pattern = engagement_pattern["pattern"]
        
        if pattern == "daily_engagement":
            return {
                "frequency": "daily",
                "optimal_times": ["morning", "evening"],
                "notification_type": "push",
                "content_volume": "moderate"
            }
        elif pattern == "weekly_engagement":
            return {
                "frequency": "2-3_times_weekly",
                "optimal_times": ["weekday_mornings"],
                "notification_type": "scheduled",
                "content_volume": "substantial"
            }
        else:  # occasional_engagement
            return {
                "frequency": "weekly",
                "optimal_times": ["weekends"],
                "notification_type": "email",
                "content_volume": "comprehensive"
            }
    
    def _assess_performance_level(self, farmer_id: str) -> str:
        """Assess farmer performance level."""
        # Try to get game performance data
        try:
            learning_report = game_data_processor.get_farmer_learning_report(farmer_id)
            
            if "error" not in learning_report:
                avg_score = learning_report.get("learning_statistics", {}).get("avg_session_score", 0)
                
                if avg_score > 80:
                    return "advanced"
                elif avg_score > 60:
                    return "intermediate"
                else:
                    return "beginner"
        except:
            pass
        
        # Fallback to adaptation count
        profile = self.personalization_profiles.get(farmer_id, {})
        adaptation_count = profile.get("adaptation_count", 0)
        
        if adaptation_count > 20:
            return "advanced"
        elif adaptation_count > 10:
            return "intermediate"
        else:
            return "beginner"
    
    def get_adaptation_report(self, farmer_id: str, timeframe_days: int = 30) -> Dict[str, Any]:
        """Get adaptation report for a farmer."""
        if farmer_id not in self.adaptation_history:
            return {"error": "No adaptation history found"}
        
        # Filter history by timeframe
        cutoff_date = datetime.utcnow() - timedelta(days=timeframe_days)
        recent_history = [
            h for h in self.adaptation_history[farmer_id]
            if datetime.fromisoformat(h["timestamp"]) > cutoff_date
        ]
        
        if not recent_history:
            return {"error": "No recent adaptations"}
        
        # Analyze adaptations
        adaptation_types = defaultdict(int)
        triggers = defaultdict(int)
        outcomes = {"success": 0, "failure": 0, "neutral": 0}
        
        for entry in recent_history:
            if entry.get("type") == "adaptation_applied":
                adaptation = entry.get("adaptation", {})
                action = adaptation.get("action", "unknown")
                outcome = adaptation.get("outcome", "neutral")
                trigger = entry.get("trigger", {}).get("trigger", "unknown")
                
                adaptation_types[action] += 1
                triggers[trigger] += 1
                outcomes[outcome] += 1
        
        # Calculate effectiveness
        total_adaptations = sum(adaptation_types.values())
        effectiveness = outcomes["success"] / total_adaptations if total_adaptations > 0 else 0
        
        # Get current personalization profile
        profile = self.personalization_profiles.get(farmer_id, {})
        
        report = {
            "farmer_id": farmer_id,
            "timeframe_days": timeframe_days,
            "analysis_period": {
                "start": cutoff_date.isoformat(),
                "end": datetime.utcnow().isoformat()
            },
            "adaptation_statistics": {
                "total_adaptations": total_adaptations,
                "adaptation_types": dict(adaptation_types),
                "common_triggers": dict(triggers),
                "outcome_distribution": outcomes,
                "effectiveness_rate": round(effectiveness, 3)
            },
            "current_personalization": {
                "complexity_level": profile.get("complexity_level", {}).get("current", "unknown"),
                "communication_preferences": profile.get("communication_preferences", {}),
                "timing_preferences": profile.get("timing_preferences", {}),
                "adaptation_count": profile.get("adaptation_count", 0)
            },
            "recommendations": self._generate_adaptation_recommendations(adaptation_types, effectiveness)
        }
        
        return report
    
    def _generate_adaptation_recommendations(self, adaptation_types: Dict[str, int], 
                                           effectiveness: float) -> List[Dict[str, Any]]:
        """Generate recommendations based on adaptation patterns."""
        recommendations = []
        
        # Analyze adaptation frequency
        total = sum(adaptation_types.values())
        if total > 20:
            recommendations.append({
                "type": "system_stability",
                "title": "Reduce adaptation frequency",
                "description": "High adaptation frequency may indicate system instability",
                "priority": "medium",
                "suggestion": "Increase threshold for adaptations"
            })
        
        # Check effectiveness
        if effectiveness < 0.5:
            recommendations.append({
                "type": "effectiveness_improvement",
                "title": "Improve adaptation effectiveness",
                "description": f"Current effectiveness rate is {effectiveness:.1%}",
                "priority": "high",
                "suggestion": "Review and adjust adaptation rules"
            })
        
        # Check for over-reliance on certain adaptations
        if len(adaptation_types) < 3 and total > 10:
            recommendations.append({
                "type": "adaptation_diversity",
                "title": "Increase adaptation diversity",
                "description": "Limited adaptation types being applied",
                "priority": "low",
                "suggestion": "Explore additional adaptation strategies"
            })
        
        return recommendations
    
    def reset_farmer_adaptations(self, farmer_id: str) -> Dict[str, Any]:
        """Reset adaptations for a farmer (for testing or fresh start)."""
        if farmer_id in self.adaptation_history:
            self.adaptation_history[farmer_id] = []
        
        if farmer_id in self.personalization_profiles:
            self.personalization_profiles[farmer_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "last_updated": datetime.utcnow().isoformat(),
                "adaptation_count": 0,
                "reset_count": self.personalization_profiles[farmer_id].get("reset_count", 0) + 1
            }
        
        return {
            "farmer_id": farmer_id,
            "action": "reset_complete",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "All adaptations and personalization reset"
        }


# Create singleton instance for easy import
adaptation_manager = AdaptationManager()