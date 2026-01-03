"""
Recommendation Engine Core - Generate personalized farming recommendations
Week 3-4 Implementation - Based on AFIE Blueprint
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass
import math

# Import our models
from src.models.recommendation import (
    Recommendation, RecommendationType, PriorityLevel, ImpactType,
    ImpactEstimate, ResourceRequirement, ActionStep, Alternative,
    RecommendationBatch, create_operational_recommendation, create_strategic_recommendation
)
from src.knowledge.agricultural_rules import AgriculturalKnowledgeBase, AgriculturalRule
from src.models.farm import FarmDNA  # Assuming this exists from Week 1-2


class RecommendationContext:
    """Context for recommendation generation."""
    
    def __init__(self):
        self.timestamp = datetime.utcnow()
        self.weather: Dict[str, Any] = {}
        self.market_prices: Dict[str, float] = {}
        self.season: str = "growing"
        self.current_activities: List[str] = []
        self.upcoming_events: List[Dict[str, Any]] = []
        self.recent_recommendations: List[str] = []  # IDs of recent recommendations
        self.farmer_feedback: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "weather": self.weather,
            "market_prices": self.market_prices,
            "season": self.season,
            "current_activities": self.current_activities,
            "upcoming_events": self.upcoming_events,
            "recent_recommendations": self.recent_recommendations,
            "farmer_feedback": self.farmer_feedback
        }


@dataclass
class RecommendationConstraints:
    """Constraints for recommendation generation."""
    max_recommendations: int = 5
    max_cost: Optional[float] = None
    max_labor_hours: Optional[float] = None
    time_horizon_days: int = 30
    priority_threshold: PriorityLevel = PriorityLevel.LOW
    farmer_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.farmer_preferences is None:
            self.farmer_preferences = {}


class RecommendationScore:
    """Scoring for recommendations."""
    
    def __init__(self):
        self.relevance_score: float = 0.0  # 0-100
        self.impact_score: float = 0.0     # 0-100
        self.feasibility_score: float = 0.0  # 0-100
        self.urgency_score: float = 0.0    # 0-100
        self.confidence_score: float = 0.0  # 0-100
        
    def overall_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate overall score with optional weights."""
        if weights is None:
            weights = {
                'relevance': 0.3,
                'impact': 0.3,
                'feasibility': 0.2,
                'urgency': 0.1,
                'confidence': 0.1
            }
        
        total = 0.0
        for key, weight in weights.items():
            score = getattr(self, f"{key}_score", 0.0)
            total += score * weight
        
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevance_score": self.relevance_score,
            "impact_score": self.impact_score,
            "feasibility_score": self.feasibility_score,
            "urgency_score": self.urgency_score,
            "confidence_score": self.confidence_score,
            "overall_score": self.overall_score()
        }


class RecommendationEngine:
    """
    Core recommendation engine for generating personalized farming recommendations.
    """
    
    def __init__(self, knowledge_base: Optional[AgriculturalKnowledgeBase] = None):
        self.knowledge_base = knowledge_base or AgriculturalKnowledgeBase()
        self.recommendation_history: Dict[str, List[Recommendation]] = {}
        print("✅ RecommendationEngine initialized")
    
    def generate_recommendations(
        self,
        farm_dna: FarmDNA,
        context: RecommendationContext,
        constraints: RecommendationConstraints
    ) -> RecommendationBatch:
        """
        Generate personalized recommendations for a farm.
        
        Args:
            farm_dna: The farm's DNA profile
            context: Current context (weather, market, etc.)
            constraints: Generation constraints
            
        Returns:
            Batch of personalized recommendations
        """
        print(f"🎯 Generating recommendations for farm: {farm_dna.identity.name}")
        
        # Create recommendation batch
        batch = RecommendationBatch(
            farm_dna_id=farm_dna.identity.farm_id,
            context=context.to_dict()
        )
        
        # Get all potential recommendations
        all_recommendations = self._generate_all_potential_recommendations(
            farm_dna, context
        )
        
        # Score and filter recommendations
        scored_recommendations = []
        for recommendation in all_recommendations:
            # Score the recommendation
            score = self._score_recommendation(recommendation, farm_dna, context, constraints)
            recommendation.confidence_score = score.overall_score() / 100.0
            
            # Apply constraints
            if self._passes_constraints(recommendation, constraints):
                scored_recommendations.append((recommendation, score))
        
        # Sort by score and limit
        scored_recommendations.sort(key=lambda x: x[1].overall_score(), reverse=True)
        scored_recommendations = scored_recommendations[:constraints.max_recommendations]
        
        # Add to batch
        for recommendation, score in scored_recommendations:
            # Set priority based on score
            overall_score = score.overall_score()
            if overall_score >= 80:
                recommendation.priority = PriorityLevel.CRITICAL
            elif overall_score >= 60:
                recommendation.priority = PriorityLevel.HIGH
            elif overall_score >= 40:
                recommendation.priority = PriorityLevel.MEDIUM
            else:
                recommendation.priority = PriorityLevel.LOW
            
            # Set explanation
            recommendation.explanation = self._generate_explanation(recommendation, score)
            
            batch.add_recommendation(recommendation)
        
        # Set batch themes
        batch.themes = self._identify_batch_themes(batch)
        batch.seasonal_focus = context.season
        
        print(f"✅ Generated {len(batch.recommendations)} recommendations")
        print(f"  • Themes: {batch.themes}")
        print(f"  • Estimated total value: ${batch.get_total_estimated_value():.2f}")
        
        return batch
    
    def _generate_all_potential_recommendations(
        self,
        farm_dna: FarmDNA,
        context: RecommendationContext
    ) -> List[Recommendation]:
        """Generate all potential recommendations before filtering."""
        recommendations = []
        
        # Get farm-specific information
        main_crop = farm_dna.typology.specialization[0] if farm_dna.typology.specialization else "maize"
        farm_size = farm_dna.identity.size_category
        challenges = farm_dna.constraints
        
        # 1. Operational recommendations based on current context
        operational_recs = self._generate_operational_recommendations(
            farm_dna, context, main_crop
        )
        recommendations.extend(operational_recs)
        
        # 2. Strategic recommendations based on farm DNA
        strategic_recs = self._generate_strategic_recommendations(farm_dna, main_crop)
        recommendations.extend(strategic_recs)
        
        # 3. Educational recommendations based on farmer profile
        educational_recs = self._generate_educational_recommendations(farm_dna, main_crop)
        recommendations.extend(educational_recs)
        
        # 4. Market recommendations based on context
        market_recs = self._generate_market_recommendations(farm_dna, context)
        recommendations.extend(market_recs)
        
        # 5. Financial recommendations based on constraints
        financial_recs = self._generate_financial_recommendations(farm_dna, challenges)
        recommendations.extend(financial_recs)
        
        # 6. Environmental recommendations
        environmental_recs = self._generate_environmental_recommendations(farm_dna)
        recommendations.extend(environmental_recs)
        
        return recommendations
    
    def _generate_operational_recommendations(
        self,
        farm_dna: FarmDNA,
        context: RecommendationContext,
        main_crop: str
    ) -> List[Recommendation]:
        """Generate operational (immediate action) recommendations."""
        recommendations = []
        
        # Get agricultural rules for the main crop
        rules = self.knowledge_base.get_rules_for_crop(main_crop)
        
        # Check for time-sensitive operations
        if context.season == "planting":
            planting_rules = [r for r in rules if r.practice == "planting"]
            for rule in planting_rules:
                rec = create_operational_recommendation(
                    title=f"Plant {main_crop} now",
                    description=rule.description,
                    farm_dna_id=farm_dna.identity.farm_id,
                    context=context.to_dict()
                )
                
                # Add action steps from rule
                for i, instruction in enumerate(rule.instructions[:3], 1):
                    rec.add_action_step(
                        description=f"Step {i}: {instruction}",
                        detailed_instructions=instruction,
                        estimated_duration_minutes=30
                    )
                
                # Add resource requirements
                if "resource_requirements" in rule.to_dict():
                    for req_type, req_details in rule.resource_requirements.items():
                        rec.add_resource_requirement(
                            resource_type=req_type,
                            quantity=req_details.get("amount", 0),
                            unit=req_details.get("unit", "units")
                        )
                
                # Add impact estimates
                rec.add_impact_estimate(
                    impact_type=ImpactType.YIELD_INCREASE,
                    magnitude=20.0,  # Estimated 20% yield increase
                    unit="percent",
                    confidence=rule.success_rate,
                    time_horizon="season"
                )
                
                recommendations.append(rec)
        
        # Weather-based recommendations
        if context.weather.get("forecast") == "heavy_rain":
            if "harvest" in context.current_activities:
                rec = create_operational_recommendation(
                    title="Harvest before heavy rain",
                    description="Heavy rain forecasted. Harvest crops now to prevent damage.",
                    farm_dna_id=farm_dna.identity.farm_id,
                    context=context.to_dict()
                )
                
                rec.add_action_step(
                    description="Check crop maturity",
                    detailed_instructions="Inspect crops to see if they're ready for harvest",
                    estimated_duration_minutes=15
                )
                rec.add_action_step(
                    description="Organize harvesting team",
                    detailed_instructions="Gather labor and equipment for immediate harvest",
                    estimated_duration_minutes=30
                )
                rec.add_action_step(
                    description="Begin harvesting",
                    detailed_instructions="Start harvesting mature crops immediately",
                    estimated_duration_minutes=120
                )
                
                rec.add_impact_estimate(
                    impact_type=ImpactType.RISK_REDUCTION,
                    magnitude=40.0,
                    unit="percent",
                    confidence=0.85,
                    time_horizon="immediate",
                    evidence_source="Weather forecast"
                )
                
                recommendations.append(rec)
        
        # Pest monitoring recommendations
        if context.season == "growing":
            pest_rules = [r for r in rules if r.practice == "pest_control"]
            for rule in pest_rules[:2]:  # Take first 2 pest rules
                rec = create_operational_recommendation(
                    title=f"Monitor for {rule.conditions.get('pest_type', 'pests')}",
                    description=rule.description,
                    farm_dna_id=farm_dna.identity.farm_id,
                    context=context.to_dict()
                )
                
                for i, instruction in enumerate(rule.instructions[:2], 1):
                    rec.add_action_step(
                        description=f"Monitoring step {i}",
                        detailed_instructions=instruction,
                        estimated_duration_minutes=20
                    )
                
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_strategic_recommendations(
        self,
        farm_dna: FarmDNA,
        main_crop: str
    ) -> List[Recommendation]:
        """Generate strategic (long-term) recommendations."""
        recommendations = []
        
        # Based on farm constraints
        if farm_dna.constraints.water_access_score < 40:
            rec = create_strategic_recommendation(
                title="Switch to drought-resistant varieties",
                description="Your farm has low water access. Consider switching to drought-resistant crop varieties.",
                farm_dna_id=farm_dna.identity.farm_id,
                context={"constraint": "water_access"}
            )
            
            rec.add_action_step(
                description="Research drought-resistant varieties",
                detailed_instructions="Look for varieties suitable for your region and soil type",
                estimated_duration_minutes=60
            )
            rec.add_action_step(
                description="Calculate seed requirements",
                detailed_instructions="Determine how much seed you'll need for next season",
                estimated_duration_minutes=30
            )
            rec.add_action_step(
                description="Source seeds",
                detailed_instructions="Contact local suppliers or agricultural extension officers",
                estimated_duration_minutes=90
            )
            
            rec.add_impact_estimate(
                impact_type=ImpactType.YIELD_INCREASE,
                magnitude=15.0,
                unit="percent",
                confidence=0.75,
                time_horizon="next_season"
            )
            rec.add_impact_estimate(
                impact_type=ImpactType.RISK_REDUCTION,
                magnitude=30.0,
                unit="percent",
                confidence=0.80,
                time_horizon="long_term"
            )
            
            # Add alternative
            rec.add_alternative(
                title="Improve irrigation system",
                description="Instead of changing varieties, invest in better irrigation",
                differences={"approach": "infrastructure vs biological"},
                advantages=["Can keep current varieties", "Improves all crops"],
                disadvantages=["Higher initial cost", "Requires maintenance"],
                cost_difference=500.0  # More expensive
            )
            
            recommendations.append(rec)
        
        # Crop rotation recommendations
        if len(farm_dna.typology.specialization) == 1:  # Monoculture
            rec = create_strategic_recommendation(
                title="Implement crop rotation",
                description="Rotating crops can improve soil health and reduce pest pressure.",
                farm_dna_id=farm_dna.identity.farm_id,
                context={"practice": "crop_rotation"}
            )
            
            # Suggest rotation crops based on main crop
            if main_crop == "maize":
                rotation_crops = ["beans", "soybeans", "groundnuts"]
            elif main_crop == "beans":
                rotation_crops = ["maize", "sorghum", "millet"]
            else:
                rotation_crops = ["legumes", "cereals"]
            
            rec.add_action_step(
                description="Plan rotation sequence",
                detailed_instructions=f"Consider rotating with: {', '.join(rotation_crops)}",
                estimated_duration_minutes=45
            )
            
            rec.add_impact_estimate(
                impact_type=ImpactType.SUSTAINABILITY,
                magnitude=25.0,
                unit="percent",
                confidence=0.85,
                time_horizon="long_term"
            )
            
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_educational_recommendations(
        self,
        farm_dna: FarmDNA,
        main_crop: str
    ) -> List[Recommendation]:
        """Generate educational recommendations."""
        recommendations = []
        
        # Based on biggest challenge from onboarding
        biggest_challenge = getattr(farm_dna.identity, 'biggest_challenge', 'water_access')
        
        challenge_topics = {
            "water_access": ["water_management", "drought_tolerance", "irrigation"],
            "capital_access": ["financial_planning", "microfinance", "cost_reduction"],
            "market_access": ["market_strategies", "price_negotiation", "value_addition"],
            "pest_disease": ["integrated_pest_mgmt", "disease_prevention", "biocontrol"],
            "soil_fertility": ["soil_health", "fertilizer_use", "composting"]
        }
        
        topics = challenge_topics.get(biggest_challenge, ["general_farming"])
        
        for topic in topics[:2]:  # Limit to 2 topics
            rec = Recommendation(
                recommendation_type=RecommendationType.EDUCATIONAL,
                title=f"Learn about {topic.replace('_', ' ')}",
                description=f"Educational content to help with {biggest_challenge.replace('_', ' ')}",
                farm_dna_id=farm_dna.identity.farm_id,
                context={"learning_topic": topic}
            )
            
            rec.priority = PriorityLevel.INFORMATIONAL
            
            # Suggest learning activities based on farmer's mobile literacy
            mobile_literacy = getattr(farm_dna.identity, 'mobile_literacy', 'basic')
            
            if mobile_literacy == "none":
                formats = ["audio_guide", "voice_message", "radio_program"]
            elif mobile_literacy == "basic":
                formats = ["video", "simple_text", "images"]
            else:
                formats = ["interactive_module", "detailed_article", "online_course"]
            
            for i, format in enumerate(formats[:2], 1):
                rec.add_action_step(
                    description=f"Access {format} content",
                    detailed_instructions=f"Find educational content about {topic} in {format} format",
                    estimated_duration_minutes=15
                )
            
            rec.add_impact_estimate(
                impact_type=ImpactType.RISK_REDUCTION,
                magnitude=10.0,
                unit="percent",
                confidence=0.70,
                time_horizon="long_term"
            )
            
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_market_recommendations(
        self,
        farm_dna: FarmDNA,
        context: RecommendationContext
    ) -> List[Recommendation]:
        """Generate market recommendations."""
        recommendations = []
        
        main_crop = farm_dna.typology.specialization[0] if farm_dna.typology.specialization else "maize"
        
        # Price-based recommendations
        if main_crop in context.market_prices:
            current_price = context.market_prices[main_crop]
            
            # Simple price trend logic (in reality would use historical data)
            if current_price > 100:  # Assuming 100 is threshold for "good price"
                rec = Recommendation(
                    recommendation_type=RecommendationType.MARKET,
                    title=f"Sell {main_crop} now",
                    description=f"Current market price ({current_price}) is favorable for selling.",
                    farm_dna_id=farm_dna.identity.farm_id,
                    context={"current_price": current_price, "crop": main_crop}
                )
                
                rec.add_action_step(
                    description="Check crop readiness",
                    detailed_instructions="Ensure crops are ready for harvest and sale",
                    estimated_duration_minutes=30
                )
                rec.add_action_step(
                    description="Contact buyers",
                    detailed_instructions="Reach out to local markets or aggregators",
                    estimated_duration_minutes=60
                )
                rec.add_action_step(
                    description="Arange transportation",
                    detailed_instructions="Organize transport to market",
                    estimated_duration_minutes=45
                )
                
                rec.add_impact_estimate(
                    impact_type=ImpactType.COST_REDUCTION,  # Actually revenue increase
                    magnitude=current_price * 0.1,  # 10% better than average
                    unit="USD",
                    confidence=0.65,
                    time_horizon="immediate"
                )
                
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_financial_recommendations(
        self,
        farm_dna: FarmDNA,
        challenges: Any  # ResourceConstraints from FarmDNA
    ) -> List[Recommendation]:
        """Generate financial recommendations."""
        recommendations = []
        
        # Capital access recommendations
        if hasattr(challenges, 'capital_access_score') and challenges.capital_access_score < 50:
            rec = Recommendation(
                recommendation_type=RecommendationType.FINANCIAL,
                title="Apply for agricultural microloan",
                description="Improve capital access through microloan programs.",
                farm_dna_id=farm_dna.identity.farm_id,
                context={"constraint": "capital_access"}
            )
            
            rec.add_action_step(
                description="Research loan options",
                detailed_instructions="Look for agricultural microloan programs in your area",
                estimated_duration_minutes=60
            )
            rec.add_action_step(
                description="Prepare application documents",
                detailed_instructions="Gather farm records, identification, and business plan",
                estimated_duration_minutes=90
            )
            rec.add_action_step(
                description="Submit application",
                detailed_instructions="Apply to suitable microloan program",
                estimated_duration_minutes=30
            )
            
            rec.add_impact_estimate(
                impact_type=ImpactType.MARKET_ACCESS,  # Through increased investment
                magnitude=25.0,
                unit="percent",
                confidence=0.60,
                time_horizon="medium_term"
            )
            
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_environmental_recommendations(
        self,
        farm_dna: FarmDNA
    ) -> List[Recommendation]:
        """Generate environmental sustainability recommendations."""
        recommendations = []
        
        # Soil conservation recommendations
        if farm_dna.environment.soil_organic_matter < 2.0:  # Low organic matter
            rec = Recommendation(
                recommendation_type=RecommendationType.ENVIRONMENTAL,
                title="Improve soil organic matter",
                description="Add organic matter to improve soil health and water retention.",
                farm_dna_id=farm_dna.identity.farm_id,
                context={"issue": "low_organic_matter"}
            )
            
            rec.add_action_step(
                description="Start composting",
                detailed_instructions="Collect organic waste and create compost pile",
                estimated_duration_minutes=60
            )
            rec.add_action_step(
                description="Apply compost to fields",
                detailed_instructions="Spread 5-10 tons of compost per hectare",
                estimated_duration_minutes=120
            )
            rec.add_action_step(
                description="Plant cover crops",
                detailed_instructions="Sow legumes or grasses as cover crops",
                estimated_duration_minutes=90
            )
            
            rec.add_impact_estimate(
                impact_type=ImpactType.SUSTAINABILITY,
                magnitude=30.0,
                unit="percent",
                confidence=0.80,
                time_horizon="long_term"
            )
            rec.add_impact_estimate(
                impact_type=ImpactType.YIELD_INCREASE,
                magnitude=10.0,
                unit="percent",
                confidence=0.70,
                time_horizon="next_season"
            )
            
            recommendations.append(rec)
        
        return recommendations
    
    def _score_recommendation(
        self,
        recommendation: Recommendation,
        farm_dna: FarmDNA,
        context: RecommendationContext,
        constraints: RecommendationConstraints
    ) -> RecommendationScore:
        """Score a recommendation on multiple dimensions."""
        score = RecommendationScore()
        
        # 1. Relevance score (0-100)
        score.relevance_score = self._calculate_relevance_score(recommendation, farm_dna)
        
        # 2. Impact score (0-100)
        score.impact_score = self._calculate_impact_score(recommendation)
        
        # 3. Feasibility score (0-100)
        score.feasibility_score = self._calculate_feasibility_score(recommendation, farm_dna, constraints)
        
        # 4. Urgency score (0-100)
        score.urgency_score = self._calculate_urgency_score(recommendation, context)
        
        # 5. Confidence score (0-100)
        score.confidence_score = self._calculate_confidence_score(recommendation)
        
        return score
    
    def _calculate_relevance_score(
        self,
        recommendation: Recommendation,
        farm_dna: FarmDNA
    ) -> float:
        """Calculate how relevant the recommendation is to this farm."""
        relevance = 50.0  # Base score
        
        # Check if recommendation addresses farm's biggest challenge
        biggest_challenge = getattr(farm_dna.identity, 'biggest_challenge', None)
        if biggest_challenge:
            challenge_keywords = {
                "water_access": ["water", "irrigation", "drought"],
                "capital_access": ["loan", "finance", "capital", "cost"],
                "market_access": ["market", "sell", "price", "buyer"],
                "pest_disease": ["pest", "disease", "spray", "control"],
                "soil_fertility": ["soil", "fertility", "compost", "organic"]
            }
            
            keywords = challenge_keywords.get(biggest_challenge, [])
            for keyword in keywords:
                if (keyword in recommendation.title.lower() or 
                    keyword in recommendation.description.lower()):
                    relevance += 20
                    break
        
        # Check if matches farm typology
        if farm_dna.typology.typology_type.value == "smallholder":
            if recommendation.type in [RecommendationType.OPERATIONAL, RecommendationType.EDUCATIONAL]:
                relevance += 15
        else:  # Commercial farm
            if recommendation.type in [RecommendationType.STRATEGIC, RecommendationType.MARKET]:
                relevance += 15
        
        # Check if matches current season
        # (Season would come from context in real implementation)
        
        return min(max(relevance, 0), 100)
    
    def _calculate_impact_score(self, recommendation: Recommendation) -> float:
        """Calculate potential impact score."""
        if not recommendation.expected_impacts:
            return 30.0
        
        # Calculate weighted impact score
        total_impact = 0.0
        total_weight = 0.0
        
        for impact in recommendation.expected_impacts:
            # Different weights for different impact types
            impact_weights = {
                ImpactType.YIELD_INCREASE: 1.0,
                ImpactType.COST_REDUCTION: 0.9,
                ImpactType.RISK_REDUCTION: 0.8,
                ImpactType.QUALITY_IMPROVEMENT: 0.7,
                ImpactType.SUSTAINABILITY: 0.6,
                ImpactType.MARKET_ACCESS: 0.8,
                ImpactType.TIME_SAVINGS: 0.5
            }
            
            weight = impact_weights.get(impact.impact_type, 0.5)
            magnitude = impact.magnitude
            confidence = impact.confidence
            
            # Normalize magnitude based on unit
            if impact.unit == "percent":
                normalized_magnitude = magnitude / 100.0
            else:
                normalized_magnitude = min(magnitude / 1000.0, 1.0)  # Rough normalization
            
            impact_value = normalized_magnitude * confidence * weight * 100
            
            total_impact += impact_value
            total_weight += weight
        
        if total_weight > 0:
            return min(total_impact / total_weight, 100)
        return 50.0
    
    def _calculate_feasibility_score(
        self,
        recommendation: Recommendation,
        farm_dna: FarmDNA,
        constraints: RecommendationConstraints
    ) -> float:
        """Calculate feasibility score based on farm constraints."""
        feasibility = 70.0  # Base score
        
        # Check cost feasibility
        total_cost = recommendation.calculate_total_cost()
        if constraints.max_cost and total_cost > constraints.max_cost:
            feasibility -= 30
        
        # Check labor feasibility
        total_labor = recommendation.calculate_total_labor()
        if constraints.max_labor_hours and total_labor > constraints.max_labor_hours:
            feasibility -= 25
        
        # Check against farm constraints
        if hasattr(farm_dna.constraints, 'capital_access_score'):
            if farm_dna.constraints.capital_access_score < 30 and total_cost > 100:
                feasibility -= 20
        
        if hasattr(farm_dna.constraints, 'labor_availability_score'):
            if farm_dna.constraints.labor_availability_score < 30 and total_labor > 20:
                feasibility -= 20
        
        # Check time feasibility
        if recommendation.type == RecommendationType.OPERATIONAL:
            # Operational recommendations need to be completed quickly
            if len(recommendation.actions) > 5:
                feasibility -= 10
        
        return min(max(feasibility, 0), 100)
    
    def _calculate_urgency_score(
        self,
        recommendation: Recommendation,
        context: RecommendationContext
    ) -> float:
        """Calculate urgency score based on context."""
        urgency = 50.0  # Base score
        
        # Time-sensitive recommendations get higher urgency
        if recommendation.priority == PriorityLevel.CRITICAL:
            urgency += 40
        elif recommendation.priority == PriorityLevel.HIGH:
            urgency += 20
        
        # Weather-dependent urgency
        if context.weather.get("alert") == "severe":
            urgency += 30
        
        # Market opportunity urgency
        if "price_spike" in context.farmer_feedback.get("market_conditions", ""):
            if recommendation.type == RecommendationType.MARKET:
                urgency += 25
        
        # Season-based urgency
        if context.season == "harvest" and recommendation.type == RecommendationType.OPERATIONAL:
            urgency += 15
        
        return min(max(urgency, 0), 100)
    
    def _calculate_confidence_score(self, recommendation: Recommendation) -> float:
        """Calculate confidence score based on evidence and validation."""
        confidence = 60.0  # Base score
        
        # Higher confidence for recommendations with more detailed action steps
        if len(recommendation.actions) >= 3:
            confidence += 10
        
        # Higher confidence for recommendations with impact estimates
        if recommendation.expected_impacts:
            confidence += 15
        
        # Higher confidence for recommendations with alternatives
        if recommendation.alternatives:
            confidence += 5
        
        return min(max(confidence, 0), 100)
    
    def _passes_constraints(
        self,
        recommendation: Recommendation,
        constraints: RecommendationConstraints
    ) -> bool:
        """Check if recommendation passes all constraints."""
        # Check priority threshold
        priority_order = {
            PriorityLevel.CRITICAL: 0,
            PriorityLevel.HIGH: 1,
            PriorityLevel.MEDIUM: 2,
            PriorityLevel.LOW: 3,
            PriorityLevel.INFORMATIONAL: 4
        }
        
        rec_priority = priority_order[recommendation.priority]
        threshold_priority = priority_order[constraints.priority_threshold]
        
        if rec_priority > threshold_priority:
            return False
        
        # Check cost constraint
        if constraints.max_cost is not None:
            total_cost = recommendation.calculate_total_cost()
            if total_cost > constraints.max_cost:
                return False
        
        # Check labor constraint
        if constraints.max_labor_hours is not None:
            total_labor = recommendation.calculate_total_labor()
            if total_labor > constraints.max_labor_hours:
                return False
        
        # Check farmer preferences
        if constraints.farmer_preferences:
            # Example: Farmer prefers no chemical pesticides
            if constraints.farmer_preferences.get("no_chemicals", False):
                if "spray" in recommendation.description.lower() and "chemical" in recommendation.description.lower():
                    return False
        
        return True
    
    def _generate_explanation(
        self,
        recommendation: Recommendation,
        score: RecommendationScore
    ) -> str:
        """Generate explanation for why recommendation was made."""
        explanations = []
        
        if score.relevance_score >= 70:
            explanations.append("Highly relevant to your farm's current situation")
        
        if score.impact_score >= 70:
            explanations.append("Has significant potential positive impact")
        
        if score.feasibility_score >= 70:
            explanations.append("Feasible given your available resources")
        elif score.feasibility_score <= 40:
            explanations.append("May require additional resources")
        
        if score.urgency_score >= 70:
            explanations.append("Time-sensitive action required")
        
        if not explanations:
            explanations.append("Based on agricultural best practices for your farm type")
        
        return "Recommended because: " + "; ".join(explanations)
    
    def _identify_batch_themes(self, batch: RecommendationBatch) -> List[str]:
        """Identify common themes in a recommendation batch."""
        theme_keywords = {
            "water_management": ["water", "irrigation", "drought", "rain"],
            "financial_planning": ["loan", "cost", "finance", "capital", "money"],
            "market_strategies": ["market", "sell", "price", "buyer", "sale"],
            "pest_control": ["pest", "disease", "spray", "control", "insect"],
            "soil_health": ["soil", "fertility", "compost", "organic", "nutrient"],
            "crop_management": ["plant", "harvest", "fertilizer", "variety", "rotation"]
        }
        
        theme_counts = {theme: 0 for theme in theme_keywords.keys()}
        all_text = " ".join([
            rec.title + " " + rec.description
            for rec in batch.recommendations
        ]).lower()
        
        for theme, keywords in theme_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    theme_counts[theme] += 1
        
        # Get top 3 themes
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:3] if count > 0]
    
    def log_recommendation_feedback(
        self,
        recommendation_id: str,
        farm_dna_id: str,
        accepted: bool,
        implemented: bool,
        farmer_rating: Optional[int] = None,
        notes: Optional[str] = None
    ) -> None:
        """Log farmer feedback on recommendations."""
        # In a real implementation, this would save to database
        # For now, just print feedback
        print(f"📝 Recommendation feedback logged:")
        print(f"  • Recommendation: {recommendation_id}")
        print(f"  • Farm: {farm_dna_id}")
        print(f"  • Accepted: {accepted}")
        print(f"  • Implemented: {implemented}")
        print(f"  • Rating: {farmer_rating}/5")
        if notes:
            print(f"  • Notes: {notes}")
        
        # Could use this feedback to improve future recommendations
        # e.g., adjust success rates of underlying rules


# Singleton instance for easy access
recommendation_engine = RecommendationEngine()


# Quick test function
def test_recommendation_engine() -> None:
    """Test the recommendation engine."""
    print("🧪 Testing Recommendation Engine...")
    
    # Create a mock FarmDNA (simplified)
    class MockFarmDNA:
        class Identity:
            farm_id = "test_farm_123"
            name = "Test Farm"
            size_category = "smallholder"
            biggest_challenge = "water_access"
            mobile_literacy = "intermediate"
        
        class Typology:
            specialization = ["maize"]
            typology_type = type('Enum', (), {'value': 'smallholder'})()
        
        class Constraints:
            water_access_score = 30
            capital_access_score = 40
            labor_availability_score = 50
            max_capital_per_season = 1000.0
        
        class Environment:
            soil_organic_matter = 1.5
        
        identity = Identity()
        typology = Typology()
        constraints = Constraints()
        environment = Environment()
    
    # Create context
    context = RecommendationContext()
    context.weather = {"forecast": "clear", "temperature": 25}
    context.market_prices = {"maize": 120.0}
    context.season = "growing"
    context.current_activities = ["weeding", "irrigation"]
    
    # Create constraints
    constraints = RecommendationConstraints(
        max_recommendations=3,
        max_cost=500.0,
        max_labor_hours=40.0,
        time_horizon_days=14
    )
    
    # Create engine and generate recommendations
    engine = RecommendationEngine()
    farm_dna = MockFarmDNA()
    
    batch = engine.generate_recommendations(farm_dna, context, constraints)
    
    print(f"\n✅ Generated {len(batch.recommendations)} recommendations:")
    for rec in batch.recommendations:
        print(f"\n📋 {rec.type.value.upper()}: {rec.title}")
        print(f"   Priority: {rec.priority.value}")
        print(f"   Confidence: {rec.confidence_score:.0%}")
        print(f"   Explanation: {rec.explanation[:80]}...")
    
    print(f"\n🎯 Batch themes: {batch.themes}")
    print(f"💰 Estimated total value: ${batch.get_total_estimated_value():.2f}")


if __name__ == "__main__":
    test_recommendation_engine()
