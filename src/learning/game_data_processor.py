"""
Game Data Processor - Week 5-6 Implementation
Process gamification data to train AI models
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import json
from dataclasses import dataclass
from collections import defaultdict

# Import behavioral analyzer
from .behavioral_analyzer import behavioral_analyzer, LearningStyle, BehavioralProfile


@dataclass
class GameEvent:
    """Represents a game event/decision point."""
    event_id: str
    farmer_id: str
    game_type: str
    decision_point: str
    timestamp: datetime
    choices: List[str]
    selected_choice: str
    decision_time_seconds: float
    outcome: Dict[str, Any]  # Score, rewards, etc.
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class GameSession:
    """Represents a complete game session."""
    def __init__(self, session_id: str, farmer_id: str, game_type: str):
        self.session_id = session_id
        self.farmer_id = farmer_id
        self.game_type = game_type
        self.start_time = datetime.utcnow()
        self.end_time = None
        self.events: List[GameEvent] = []
        self.session_score = 0.0
        self.completion_status = "in_progress"
        self.learning_objectives = []
        
    def add_event(self, event: GameEvent):
        """Add a game event to the session."""
        self.events.append(event)
        
    def finalize(self, final_score: float = None):
        """Finalize the game session."""
        self.end_time = datetime.utcnow()
        self.completion_status = "completed"
        
        if final_score is not None:
            self.session_score = final_score
        elif self.events:
            # Calculate average score from events
            scores = [e.outcome.get("score", 0) for e in self.events]
            self.session_score = np.mean(scores) if scores else 0.0
            
    def duration_seconds(self) -> float:
        """Calculate session duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "farmer_id": self.farmer_id,
            "game_type": self.game_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds(),
            "session_score": self.session_score,
            "completion_status": self.completion_status,
            "event_count": len(self.events),
            "learning_objectives": self.learning_objectives
        }


class GameDataProcessor:
    """
    Process gamification data to train AI models.
    
    Features:
    1. Game Event Parser
    2. Behavioral Feature Extraction
    3. Model Training Pipeline
    4. Learning Velocity Measurement
    """
    
    def __init__(self):
        self.sessions: Dict[str, GameSession] = {}
        self.event_history = defaultdict(list)
        self.learning_progress = defaultdict(dict)
        
        print("✅ GameDataProcessor initialized")
    
    def process_game_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete game session.
        
        Args:
            session_data: {
                "session_id": "session_123",
                "farmer_id": "farmer_001",
                "game_type": "cash_crops_introduction",
                "events": [list of game events],
                "final_score": 85.5,
                "learning_objectives": ["understand_fertilizer", "learn_pest_control"]
            }
        
        Returns:
            Processed results with behavioral insights and recommendations
        """
        print(f"🎮 Processing game session: {session_data.get('session_id')}")
        
        # Create session object
        session = GameSession(
            session_id=session_data["session_id"],
            farmer_id=session_data["farmer_id"],
            game_type=session_data["game_type"]
        )
        
        # Process events
        events_data = session_data.get("events", [])
        for event_data in events_data:
            # Ensure farmer_id is in event data
            if "farmer_id" not in event_data:
                event_data["farmer_id"] = session_data["farmer_id"]
                
            event = self._parse_game_event(event_data)
            session.add_event(event)
            
            # Update behavioral analyzer with game decision
            self._update_behavioral_analyzer(event)
        
        # Finalize session
        session.finalize(session_data.get("final_score"))
        session.learning_objectives = session_data.get("learning_objectives", [])
        
        # Store session
        self.sessions[session.session_id] = session
        
        # Extract behavioral features
        behavioral_features = self.extract_behavioral_features(session)
        
        # Update farmer's behavioral profile
        updated_profile = self.update_farmer_behavioral_profile(session.farmer_id, session)
        
        # Adjust difficulty levels
        difficulty_recommendation = self.adjust_difficulty_levels(session.farmer_id, session)
        
        # Identify learning opportunities
        learning_opportunities = self.identify_learning_opportunities(session)
        
        # Update recommendation personalization
        personalization_update = self.update_recommendation_personalization(session.farmer_id, session)
        
        return {
            "session_summary": session.to_dict(),
            "behavioral_features": behavioral_features,
            "updated_profile": updated_profile.to_dict() if updated_profile else None,
            "difficulty_recommendation": difficulty_recommendation,
            "learning_opportunities": learning_opportunities,
            "personalization_update": personalization_update,
            "learning_velocity": self.calculate_learning_velocity(session.farmer_id)
        }
    
    def _parse_game_event(self, event_data: Dict[str, Any]) -> GameEvent:
        """Parse raw event data into GameEvent object."""
        # Parse timestamp
        timestamp_str = event_data.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except:
                timestamp = datetime.utcnow()
        else:
            timestamp = datetime.utcnow()
        
        # Get farmer_id with fallback
        farmer_id = event_data.get("farmer_id", "unknown")
        
        return GameEvent(
            event_id=event_data.get("event_id", f"event_{datetime.utcnow().timestamp()}"),
            farmer_id=farmer_id,
            game_type=event_data.get("game_type", "unknown"),
            decision_point=event_data.get("decision_point", "unknown"),
            timestamp=timestamp,
            choices=event_data.get("choices", []),
            selected_choice=event_data.get("selected_choice", ""),
            decision_time_seconds=event_data.get("decision_time_seconds", 0.0),
            outcome=event_data.get("outcome", {}),
            metadata=event_data.get("metadata", {})
        )
    
    def _update_behavioral_analyzer(self, event: GameEvent):
        """Update behavioral analyzer with game decision data."""
        # Map game decision to behavioral decision format
        decision_data = {
            "decision_type": f"game_{event.game_type}",
            "action": self._map_game_choice_to_action(event.selected_choice, event.outcome),
            "timing": self._determine_decision_timing(event.decision_time_seconds),
            "risk_level": self._assess_game_risk_level(event),
            "decision_time_seconds": event.decision_time_seconds,
            "outcome_confidence": self._calculate_outcome_confidence(event.outcome)
        }
        
        # Update behavioral analyzer
        behavioral_analyzer.analyze_decision_pattern(event.farmer_id, decision_data)
    
    def _map_game_choice_to_action(self, choice: str, outcome: Dict[str, Any]) -> str:
        """Map game choice to behavioral action."""
        outcome_score = outcome.get("score", 0)
        
        if "optimal" in choice.lower() or outcome_score > 80:
            return "accepted"
        elif "suboptimal" in choice.lower() or outcome_score < 50:
            return "rejected"
        else:
            return "modified"
    
    def _determine_decision_timing(self, decision_time_seconds: float) -> str:
        """Determine decision timing category."""
        if decision_time_seconds < 10:
            return "immediate"
        elif decision_time_seconds < 60:
            return "prompt"
        elif decision_time_seconds < 300:
            return "deliberate"
        else:
            return "delayed"
    
    def _assess_game_risk_level(self, event: GameEvent) -> str:
        """Assess risk level of game decision."""
        # Simple heuristic based on choice count and outcome variance
        choice_count = len(event.choices)
        outcome_score = event.outcome.get("score", 50)
        
        if choice_count <= 2:
            # Binary choice = lower risk
            return "low"
        elif choice_count >= 5:
            # Many choices = higher risk
            if outcome_score < 40:
                return "high"
            else:
                return "medium"
        else:
            return "medium"
    
    def _calculate_outcome_confidence(self, outcome: Dict[str, Any]) -> float:
        """Calculate confidence level from outcome."""
        score = outcome.get("score", 50)
        return score / 100.0  # Normalize to 0-1
    
    # ... (rest of the file remains the same, all methods should work)
    
    def extract_behavioral_features(self, session: GameSession) -> Dict[str, Any]:
        """Extract behavioral features from game session."""
        if not session.events:
            return {"error": "No events in session"}
        
        events = session.events
        
        # Calculate various behavioral metrics
        decision_times = [e.decision_time_seconds for e in events]
        outcome_scores = [e.outcome.get("score", 0) for e in events]
        choices_made = [e.selected_choice for e in events]
        
        # Decision timing analysis
        timing_stats = {
            "avg_decision_time": np.mean(decision_times) if decision_times else 0,
            "std_decision_time": np.std(decision_times) if len(decision_times) > 1 else 0,
            "quick_decisions": sum(1 for t in decision_times if t < 10),
            "deliberate_decisions": sum(1 for t in decision_times if t > 30)
        }
        
        # Risk-reward tradeoff patterns
        risk_reward_patterns = []
        for i, event in enumerate(events):
            if i > 0:
                prev_score = events[i-1].outcome.get("score", 50)
                current_score = event.outcome.get("score", 50)
                risk_taken = abs(len(event.choices) - len(events[i-1].choices))
                
                pattern = {
                    "risk_level": risk_taken,
                    "score_change": current_score - prev_score,
                    "risk_reward_ratio": (current_score - prev_score) / max(risk_taken, 1)
                }
                risk_reward_patterns.append(pattern)
        
        # Problem-solving approaches
        problem_solving_approaches = self._analyze_problem_solving_approaches(events)
        
        # Error correction speed
        error_correction_speed = self._calculate_error_correction_speed(events)
        
        # Learning velocity during session
        session_learning_velocity = self._calculate_session_learning_velocity(events)
        
        return {
            "timing_analysis": timing_stats,
            "risk_reward_patterns": risk_reward_patterns[:5],  # Limit output
            "problem_solving_approaches": problem_solving_approaches,
            "error_correction_speed": error_correction_speed,
            "session_learning_velocity": session_learning_velocity,
            "decision_consistency": self._calculate_decision_consistency(choices_made),
            "score_progression": self._calculate_score_progression(outcome_scores)
        }
    
    def _analyze_problem_solving_approaches(self, events: List[GameEvent]) -> Dict[str, Any]:
        """Analyze problem-solving approaches from game events."""
        if len(events) < 3:
            return {"approach": "unknown", "confidence": 0.0}
        
        # Simple heuristic analysis
        exploration_count = 0
        exploitation_count = 0
        systematic_count = 0
        
        for i in range(1, len(events)):
            prev_choice = events[i-1].selected_choice
            current_choice = events[i].selected_choice
            prev_score = events[i-1].outcome.get("score", 50)
            current_score = events[i].outcome.get("score", 50)
            
            if current_choice != prev_choice and current_score <= prev_score:
                exploration_count += 1
            elif current_choice == prev_choice and current_score > prev_score:
                exploitation_count += 1
            elif i > 1 and events[i-2].selected_choice != current_choice:
                systematic_count += 1
        
        total = len(events) - 1
        if total == 0:
            return {"approach": "unknown", "confidence": 0.0}
        
        approaches = {
            "exploration": exploration_count / total,
            "exploitation": exploitation_count / total,
            "systematic": systematic_count / total
        }
        
        dominant_approach = max(approaches.items(), key=lambda x: x[1])
        
        return {
            "approach": dominant_approach[0],
            "confidence": dominant_approach[1],
            "distribution": approaches
        }
    
    def _calculate_error_correction_speed(self, events: List[GameEvent]) -> Dict[str, Any]:
        """Calculate how quickly farmer corrects errors."""
        if len(events) < 2:
            return {"avg_correction_time": 0, "correction_rate": 0}
        
        correction_times = []
        correction_count = 0
        
        for i in range(1, len(events)):
            prev_score = events[i-1].outcome.get("score", 50)
            current_score = events[i].outcome.get("score", 50)
            
            if prev_score < 50 and current_score > prev_score + 10:
                # Significant improvement after poor performance
                time_diff = (events[i].timestamp - events[i-1].timestamp).total_seconds()
                correction_times.append(time_diff)
                correction_count += 1
        
        return {
            "avg_correction_time": np.mean(correction_times) if correction_times else 0,
            "correction_rate": correction_count / (len(events) - 1) if len(events) > 1 else 0,
            "total_corrections": correction_count
        }
    
    def _calculate_session_learning_velocity(self, events: List[GameEvent]) -> float:
        """Calculate learning velocity within a single session."""
        if len(events) < 4:
            return 0.0
        
        # Split events into halves
        half_point = len(events) // 2
        first_half = events[:half_point]
        second_half = events[half_point:]
        
        # Calculate average scores
        first_avg = np.mean([e.outcome.get("score", 50) for e in first_half])
        second_avg = np.mean([e.outcome.get("score", 50) for e in second_half])
        
        # Calculate velocity (improvement per event)
        improvement = second_avg - first_avg
        velocity = improvement / len(second_half)
        
        return round(velocity, 3)
    
    def _calculate_decision_consistency(self, choices: List[str]) -> float:
        """Calculate consistency of decisions."""
        if len(choices) < 2:
            return 1.0
        
        unique_choices = len(set(choices))
        consistency = 1 - (unique_choices / len(choices))
        
        return round(consistency, 3)
    
    def _calculate_score_progression(self, scores: List[float]) -> Dict[str, Any]:
        """Calculate progression of scores over time."""
        if len(scores) < 2:
            return {"trend": "stable", "slope": 0, "r_squared": 0}
        
        # Calculate linear trend
        x = list(range(len(scores)))
        slope, intercept = np.polyfit(x, scores, 1)
        
        # Calculate R-squared
        y_pred = [slope * xi + intercept for xi in x]
        ss_res = sum((scores[i] - y_pred[i]) ** 2 for i in range(len(scores)))
        ss_tot = sum((scores[i] - np.mean(scores)) ** 2 for i in range(len(scores)))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend
        if slope > 2:
            trend = "improving"
        elif slope < -2:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "slope": round(slope, 3),
            "r_squared": round(r_squared, 3),
            "final_improvement": scores[-1] - scores[0] if scores else 0
        }
    
    def update_farmer_behavioral_profile(self, farmer_id: str, session: GameSession) -> Optional[BehavioralProfile]:
        """Update farmer's behavioral profile based on game performance."""
        try:
            profile = behavioral_analyzer.get_behavioral_profile(farmer_id)
            
            # Update based on game performance
            if session.events:
                avg_score = np.mean([e.outcome.get("score", 50) for e in session.events])
                
                # Update decision confidence based on game performance
                if avg_score > 70:
                    profile.decision_confidence = min(profile.decision_confidence + 5, 100)
                elif avg_score < 40:
                    profile.decision_confidence = max(profile.decision_confidence - 3, 0)
                
                # Update learning velocity
                session_velocity = self._calculate_session_learning_velocity(session.events)
                profile.learning_velocity = (profile.learning_velocity + session_velocity) / 2
                
                # Update adaptability based on diverse choices
                unique_choices = len(set(e.selected_choice for e in session.events))
                if unique_choices > len(session.events) / 2:
                    profile.adaptability_score = min(profile.adaptability_score + 3, 100)
            
            return profile
            
        except Exception as e:
            print(f"⚠️ Error updating behavioral profile: {e}")
            return None
    
    def adjust_difficulty_levels(self, farmer_id: str, session: GameSession) -> Dict[str, Any]:
        """Adjust game difficulty levels based on performance."""
        if not session.events:
            return {"action": "maintain", "reason": "no_events"}
        
        avg_score = np.mean([e.outcome.get("score", 50) for e in session.events])
        completion_rate = len([e for e in session.events if e.outcome.get("completed", False)]) / len(session.events)
        
        # Get behavioral profile for context
        profile = behavioral_analyzer.get_behavioral_profile(farmer_id)
        
        recommendation = {
            "current_performance": {
                "avg_score": round(avg_score, 1),
                "completion_rate": round(completion_rate, 3)
            },
            "behavioral_context": {
                "risk_tolerance": profile.risk_tolerance,
                "decision_confidence": profile.decision_confidence
            }
        }
        
        # Difficulty adjustment logic
        if avg_score > 80 and completion_rate > 0.9:
            recommendation.update({
                "action": "increase",
                "reason": "excellent_performance",
                "suggested_increase": "15%",
                "new_challenges": ["time_pressure", "resource_constraints"]
            })
        elif avg_score < 40 or completion_rate < 0.5:
            recommendation.update({
                "action": "decrease",
                "reason": "struggling_with_content",
                "suggested_decrease": "20%",
                "additional_support": ["hints", "simplified_instructions"]
            })
        else:
            recommendation.update({
                "action": "maintain",
                "reason": "appropriate_challenge_level",
                "suggested_variation": "different_scenarios",
                "focus_areas": ["decision_explanation", "alternative_strategies"]
            })
        
        return recommendation
    
    def identify_learning_opportunities(self, session: GameSession) -> List[Dict[str, Any]]:
        """Identify specific learning opportunities from game session."""
        opportunities = []
        
        if not session.events:
            return opportunities
        
        # Analyze weak areas
        low_score_events = [e for e in session.events if e.outcome.get("score", 0) < 50]
        
        for event in low_score_events[:3]:  # Limit to top 3 opportunities
            opportunity = {
                "decision_point": event.decision_point,
                "score": event.outcome.get("score", 0),
                "selected_choice": event.selected_choice,
                "optimal_choice": self._identify_optimal_choice(event),
                "learning_gap": self._identify_learning_gap(event),
                "recommended_content": self._suggest_learning_content(event),
                "priority": "high" if event.outcome.get("score", 0) < 30 else "medium"
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    def _identify_optimal_choice(self, event: GameEvent) -> str:
        """Identify what would have been the optimal choice."""
        # Simple heuristic - in real implementation, this would use game logic
        choices = event.choices
        if not choices:
            return ""
        
        # For demonstration, return first choice or a heuristic
        if "optimal" in choices[0].lower():
            return choices[0]
        elif len(choices) > 1:
            return choices[1]  # Assume second choice is often optimal for variety
        else:
            return choices[0]
    
    def _identify_learning_gap(self, event: GameEvent) -> str:
        """Identify the specific learning gap."""
        decision_point = event.decision_point.lower()
        
        gap_mapping = {
            "fertilizer": "nutrient_management",
            "pesticide": "pest_identification",
            "irrigation": "water_requirements",
            "planting": "timing_considerations",
            "harvesting": "ripeness_indicators"
        }
        
        for keyword, gap in gap_mapping.items():
            if keyword in decision_point:
                return gap
        
        return "general_decision_making"
    
    def _suggest_learning_content(self, event: GameEvent) -> List[str]:
        """Suggest learning content to address gap."""
        gap = self._identify_learning_gap(event)
        
        content_mapping = {
            "nutrient_management": ["soil_testing_video", "fertilizer_calculator_tool"],
            "pest_identification": ["pest_identification_guide", "ipm_strategies"],
            "water_requirements": ["irrigation_scheduling_app", "drought_tolerant_varieties"],
            "timing_considerations": ["planting_calendar", "weather_forecasting"],
            "general_decision_making": ["decision_framework", "risk_assessment_guide"]
        }
        
        return content_mapping.get(gap, ["general_farming_best_practices"])
    
    def update_recommendation_personalization(self, farmer_id: str, session: GameSession) -> Dict[str, Any]:
        """Update recommendation personalization based on game behavior."""
        profile = behavioral_analyzer.get_behavioral_profile(farmer_id)
        
        personalization_update = {
            "farmer_id": farmer_id,
            "game_type": session.game_type,
            "performance_summary": {
                "avg_score": round(np.mean([e.outcome.get("score", 50) for e in session.events]), 1) if session.events else 0,
                "decision_count": len(session.events)
            },
            "behavioral_insights": {
                "preferred_learning_style": profile.learning_style.value,
                "risk_preference": profile.stress_response.value,
                "decision_speed": "fast" if session.events and np.mean([e.decision_time_seconds for e in session.events]) < 20 else "deliberate"
            },
            "personalization_adjustments": self._generate_personalization_adjustments(profile, session)
        }
        
        return personalization_update
    
    def _generate_personalization_adjustments(self, profile: BehavioralProfile, session: GameSession) -> List[Dict[str, Any]]:
        """Generate specific personalization adjustments."""
        adjustments = []
        
        # Based on learning style
        if profile.learning_style == LearningStyle.VISUAL:
            adjustments.append({
                "aspect": "content_delivery",
                "adjustment": "increase_visual_content",
                "magnitude": "30%",
                "rationale": "visual_learner_detected"
            })
        
        # Based on risk tolerance
        if profile.risk_tolerance > 70:
            adjustments.append({
                "aspect": "recommendation_type",
                "adjustment": "include_more_innovative_options",
                "magnitude": "25%",
                "rationale": "high_risk_tolerance"
            })
        elif profile.risk_tolerance < 30:
            adjustments.append({
                "aspect": "recommendation_type",
                "adjustment": "focus_on_proven_practices",
                "magnitude": "40%",
                "rationale": "risk_averse_profile"
            })
        
        # Based on game performance
        if session.events:
            avg_score = np.mean([e.outcome.get("score", 50) for e in session.events])
            if avg_score > 75:
                adjustments.append({
                    "aspect": "complexity",
                    "adjustment": "increase_conceptual_depth",
                    "magnitude": "15%",
                    "rationale": "excellent_comprehension"
                })
        
        return adjustments
    
    def calculate_learning_velocity(self, farmer_id: str) -> Dict[str, Any]:
        """Calculate learning velocity across multiple sessions."""
        # Get all sessions for this farmer
        farmer_sessions = [s for s in self.sessions.values() if s.farmer_id == farmer_id]
        
        if len(farmer_sessions) < 2:
            return {"error": "insufficient_sessions", "required": 2, "available": len(farmer_sessions)}
        
        # Sort sessions by time
        farmer_sessions.sort(key=lambda s: s.start_time)
        
        # Calculate improvement over time
        early_sessions = farmer_sessions[:len(farmer_sessions)//2]
        late_sessions = farmer_sessions[len(farmer_sessions)//2:]
        
        early_avg = np.mean([s.session_score for s in early_sessions])
        late_avg = np.mean([s.session_score for s in late_sessions])
        
        # Calculate velocity (improvement per session)
        improvement = late_avg - early_avg
        velocity = improvement / max(len(late_sessions), 1)
        
        return {
            "total_sessions": len(farmer_sessions),
            "early_period_avg": round(early_avg, 2),
            "late_period_avg": round(late_avg, 2),
            "total_improvement": round(improvement, 2),
            "learning_velocity": round(velocity, 3),
            "velocity_category": self._categorize_learning_velocity(velocity),
            "trend": "improving" if improvement > 0 else "stable" if improvement == 0 else "declining"
        }
    
    def _categorize_learning_velocity(self, velocity: float) -> str:
        """Categorize learning velocity."""
        if velocity > 2:
            return "rapid"
        elif velocity > 0.5:
            return "moderate"
        elif velocity > -0.5:
            return "stable"
        else:
            return "slow"
    
    def get_farmer_learning_report(self, farmer_id: str) -> Dict[str, Any]:
        """Generate comprehensive learning report for a farmer."""
        # Get behavioral profile
        profile = behavioral_analyzer.get_behavioral_profile(farmer_id)
        
        # Get all sessions for this farmer
        farmer_sessions = [s for s in self.sessions.values() if s.farmer_id == farmer_id]
        
        # Calculate various metrics
        if farmer_sessions:
            session_scores = [s.session_score for s in farmer_sessions]
            game_types = list(set(s.game_type for s in farmer_sessions))
            
            # Calculate strengths and weaknesses
            strengths, weaknesses = self._analyze_strengths_weaknesses(farmer_sessions)
            
            report = {
                "farmer_id": farmer_id,
                "behavioral_profile_summary": {
                    "risk_tolerance": profile.risk_tolerance,
                    "learning_style": profile.learning_style.value,
                    "decision_confidence": profile.decision_confidence
                },
                "learning_statistics": {
                    "total_sessions": len(farmer_sessions),
                    "avg_session_score": round(np.mean(session_scores), 2),
                    "best_score": round(max(session_scores), 2) if session_scores else 0,
                    "game_types_played": game_types,
                    "total_learning_time_hours": round(sum(s.duration_seconds() for s in farmer_sessions) / 3600, 1)
                },
                "learning_velocity": self.calculate_learning_velocity(farmer_id),
                "strengths": strengths[:3],  # Top 3 strengths
                "weaknesses": weaknesses[:3],  # Top 3 areas for improvement
                "personalized_learning_plan": self._generate_learning_plan(profile, strengths, weaknesses),
                "recommended_next_steps": self._recommend_next_steps(profile, farmer_sessions)
            }
        else:
            report = {
                "farmer_id": farmer_id,
                "error": "no_game_data",
                "message": "No game sessions found for this farmer",
                "recommendation": "Complete at least one game session to generate learning report"
            }
        
        return report
    
    def _analyze_strengths_weaknesses(self, sessions: List[GameSession]) -> Tuple[List[Dict], List[Dict]]:
        """Analyze strengths and weaknesses from game sessions."""
        # Simplified analysis - in real implementation, this would analyze specific skills
        strengths = []
        weaknesses = []
        
        # Group events by decision point
        all_events = []
        for session in sessions:
            all_events.extend(session.events)
        
        # Analyze by decision point
        decision_points = {}
        for event in all_events:
            dp = event.decision_point
            if dp not in decision_points:
                decision_points[dp] = []
            decision_points[dp].append(event.outcome.get("score", 50))
        
        # Categorize as strength or weakness
        for dp, scores in decision_points.items():
            avg_score = np.mean(scores)
            
            if avg_score > 70:
                strengths.append({
                    "skill_area": dp,
                    "avg_score": round(avg_score, 1),
                    "consistency": round(np.std(scores), 1) if len(scores) > 1 else 0,
                    "data_points": len(scores)
                })
            elif avg_score < 50:
                weaknesses.append({
                    "skill_area": dp,
                    "avg_score": round(avg_score, 1),
                    "improvement_needed": round(70 - avg_score, 1),
                    "data_points": len(scores)
                })
        
        # Sort strengths by score (descending) and weaknesses by improvement needed (descending)
        strengths.sort(key=lambda x: x["avg_score"], reverse=True)
        weaknesses.sort(key=lambda x: x["improvement_needed"], reverse=True)
        
        return strengths, weaknesses
    
    def _generate_learning_plan(self, profile: BehavioralProfile, 
                               strengths: List[Dict], weaknesses: List[Dict]) -> Dict[str, Any]:
        """Generate personalized learning plan."""
        plan = {
            "timeframe_weeks": 4,
            "focus_areas": [w["skill_area"] for w in weaknesses[:2]],  # Focus on top 2 weaknesses
            "reinforcement_areas": [s["skill_area"] for s in strengths[:2]],  # Reinforce top 2 strengths
            "learning_approach": self._determine_learning_approach(profile),
            "recommended_games": self._recommend_games_based_on_profile(profile, weaknesses),
            "success_metrics": {
                "target_score_improvement": 20,
                "target_confidence_increase": 15,
                "target_completion_rate": 0.8
            }
        }
        
        return plan
    
    def _determine_learning_approach(self, profile: BehavioralProfile) -> str:
        """Determine optimal learning approach based on profile."""
        if profile.learning_style == LearningStyle.KINESTHETIC:
            return "hands_on_practice"
        elif profile.learning_style == LearningStyle.VISUAL:
            return "demonstration_based"
        elif profile.learning_style == LearningStyle.AUDITORY:
            return "discussion_based"
        else:
            return "mixed_method"
    
    def _recommend_games_based_on_profile(self, profile: BehavioralProfile, 
                                         weaknesses: List[Dict]) -> List[Dict[str, Any]]:
        """Recommend games based on behavioral profile and weaknesses."""
        recommendations = []
        
        # Base recommendations on weaknesses
        for weakness in weaknesses[:2]:  # Top 2 weaknesses
            skill_area = weakness["skill_area"]
            
            game_mapping = {
                "nutrient_management": "fertilizer_calculator_game",
                "pest_identification": "pest_detective_game",
                "water_requirements": "irrigation_strategy_game",
                "timing_considerations": "planting_calendar_game",
                "general_decision_making": "farm_strategy_simulator"
            }
            
            game_type = game_mapping.get(skill_area.lower(), "general_farming_game")
            
            recommendations.append({
                "game_type": game_type,
                "skill_focus": skill_area,
                "difficulty": "beginner" if weakness["avg_score"] < 40 else "intermediate",
                "expected_duration_minutes": 15,
                "rationale": f"Addresses weakness in {skill_area} (avg score: {weakness['avg_score']})"
            })
        
        # Add one game based on learning style
        style_game = {
            LearningStyle.VISUAL: "interactive_infographic_game",
            LearningStyle.AUDITORY: "podcast_quiz_game",
            LearningStyle.KINESTHETIC: "physical_simulation_game",
            LearningStyle.READ_WRITE: "document_analysis_game",
            LearningStyle.MULTIMODAL: "multi_approach_farming_game"
        }.get(profile.learning_style, "general_farming_game")
        
        recommendations.append({
            "game_type": style_game,
            "skill_focus": "learning_style_reinforcement",
            "difficulty": "adaptive",
            "expected_duration_minutes": 20,
            "rationale": f"Matches your {profile.learning_style.value} learning style"
        })
        
        return recommendations
    
    def _recommend_next_steps(self, profile: BehavioralProfile, 
                             sessions: List[GameSession]) -> List[Dict[str, Any]]:
        """Recommend next steps for the farmer."""
        next_steps = []
        
        # Based on number of sessions
        session_count = len(sessions)
        if session_count == 0:
            next_steps.append({
                "step": "complete_first_game",
                "priority": "high",
                "description": "Start with a beginner-friendly game to establish baseline",
                "expected_time": "15 minutes"
            })
        elif session_count < 3:
            next_steps.append({
                "step": "explore_different_game_types",
                "priority": "medium",
                "description": "Try different game formats to identify preferences",
                "expected_time": "45 minutes"
            })
        else:
            # Based on performance
            if sessions:
                recent_score = sessions[-1].session_score
                if recent_score > 80:
                    next_steps.append({
                        "step": "tackle_advanced_challenges",
                        "priority": "medium",
                        "description": "You're ready for more complex scenarios",
                        "expected_time": "30 minutes"
                    })
            
            # Based on behavioral profile
            if profile.risk_tolerance < 40:
                next_steps.append({
                    "step": "risk_awareness_exercise",
                    "priority": "low",
                    "description": "Practice assessing and managing agricultural risks",
                    "expected_time": "20 minutes"
                })
        
        return next_steps


# Create singleton instance for easy import
game_data_processor = GameDataProcessor()