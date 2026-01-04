"""
Risk Intelligence - Week 7-8 Implementation
Multi-dimensional risk assessment and mitigation
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class RiskType(Enum):
    """Types of agricultural risks."""
    WEATHER = "weather"
    MARKET = "market"
    OPERATIONAL = "operational"
    BIOLOGICAL = "biological"
    FINANCIAL = "financial"
    POLICY = "policy"
    ENVIRONMENTAL = "environmental"


class RiskSeverity(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskProbability(Enum):
    """Risk probability levels."""
    RARE = "rare"        # <10%
    UNLIKELY = "unlikely" # 10-30%
    POSSIBLE = "possible" # 30-50%
    LIKELY = "likely"    # 50-70%
    VERY_LIKELY = "very_likely" # >70%


@dataclass
class RiskDimension:
    """Dimension of risk assessment."""
    risk_type: RiskType
    indicators: List[str]
    data_sources: List[str]
    monitoring_frequency: str  # daily, weekly, monthly
    impact_metrics: List[str]
    mitigation_strategies: List[str]


@dataclass
class RiskAssessment:
    """Complete risk assessment for a farm."""
    farm_id: str
    assessment_date: datetime
    time_horizon_days: int
    risk_dimensions: Dict[RiskType, Dict[str, Any]]
    overall_risk_score: float
    highest_risks: List[Dict[str, Any]]
    mitigation_plan: Dict[str, Any]
    early_warnings: List[Dict[str, Any]]


@dataclass
class EarlyWarning:
    """Early warning signal for risk."""
    risk_type: RiskType
    indicator: str
    current_value: float
    threshold: float
    severity: RiskSeverity
    confidence: float  # 0-1
    recommended_actions: List[str]
    escalation_protocol: str


class RiskIntelligence:
    """
    Multi-dimensional risk assessment and mitigation system.
    
    Features:
    1. Risk dimension models for different risk types
    2. Correlation matrix for cascading risks
    3. Early warning system with alerts
    4. Mitigation effectiveness analysis
    """
    
    def __init__(self):
        self.risk_models = self._initialize_risk_models()
        self.correlation_matrix = self._build_correlation_matrix()
        self.early_warnings = []
        self.risk_history = []
        
        print("✅ RiskIntelligence initialized")
    
    def _initialize_risk_models(self) -> Dict[RiskType, RiskDimension]:
        """Initialize models for different risk dimensions."""
        models = {}
        
        # Weather Risk Model
        models[RiskType.WEATHER] = RiskDimension(
            risk_type=RiskType.WEATHER,
            indicators=[
                "rainfall_deviation",
                "temperature_extremes", 
                "drought_index",
                "flood_probability",
                "frost_days",
                "storm_frequency"
            ],
            data_sources=["weather_stations", "satellite", "climate_models"],
            monitoring_frequency="daily",
            impact_metrics=["yield_loss_percent", "crop_failure_probability", "input_cost_increase"],
            mitigation_strategies=[
                "irrigation_systems",
                "drought_resistant_varieties",
                "crop_insurance",
                "weather_hedging"
            ]
        )
        
        # Market Risk Model
        models[RiskType.MARKET] = RiskDimension(
            risk_type=RiskType.MARKET,
            indicators=[
                "price_volatility",
                "market_access",
                "demand_shocks",
                "supply_chain_disruption",
                "export_restrictions",
                "input_price_fluctuation"
            ],
            data_sources=["market_reports", "price_feeds", "trade_data"],
            monitoring_frequency="weekly",
            impact_metrics=["revenue_loss", "profit_margin_erosion", "inventory_carrying_costs"],
            mitigation_strategies=[
                "forward_contracts",
                "diversified_markets",
                "storage_facilities",
                "price_insurance"
            ]
        )
        
        # Operational Risk Model
        models[RiskType.OPERATIONAL] = RiskDimension(
            risk_type=RiskType.OPERATIONAL,
            indicators=[
                "equipment_failure_rate",
                "labor_availability",
                "input_supply_reliability",
                "fuel_price_volatility",
                "transportation_disruption",
                "energy_outages"
            ],
            data_sources=["maintenance_logs", "labor_reports", "supplier_data"],
            monitoring_frequency="weekly",
            impact_metrics=["production_delays", "cost_overruns", "quality_issues"],
            mitigation_strategies=[
                "preventive_maintenance",
                "labor_training",
                "multiple_suppliers",
                "backup_equipment"
            ]
        )
        
        # Biological Risk Model
        models[RiskType.BIOLOGICAL] = RiskDimension(
            risk_type=RiskType.BIOLOGICAL,
            indicators=[
                "pest_infestation_level",
                "disease_prevalence",
                "weed_pressure",
                "soil_health_decline",
                "biodiversity_loss",
                "invasive_species"
            ],
            data_sources=["field_scouting", "lab_tests", "regional_alerts"],
            monitoring_frequency="weekly",
            impact_metrics=["yield_loss", "input_cost_increase", "crop_rejection_rate"],
            mitigation_strategies=[
                "integrated_pest_management",
                "crop_rotation",
                "resistant_varieties",
                "biological_controls"
            ]
        )
        
        # Financial Risk Model
        models[RiskType.FINANCIAL] = RiskDimension(
            risk_type=RiskType.FINANCIAL,
            indicators=[
                "debt_to_asset_ratio",
                "liquidity_ratio",
                "interest_rate_exposure",
                "exchange_rate_risk",
                "credit_availability",
                "insurance_coverage"
            ],
            data_sources=["financial_statements", "bank_reports", "insurance_policies"],
            monitoring_frequency="monthly",
            impact_metrics=["cash_flow_disruption", "loan_default_risk", "investment_delay"],
            mitigation_strategies=[
                "diversified_financing",
                "risk_reserves",
                "hedging_instruments",
                "insurance_portfolio"
            ]
        )
        
        return models
    
    def _build_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build correlation matrix between different risk types."""
        # Expert-estimated correlations between risk types
        correlations = {
            "weather_biological": 0.7,   # Weather affects pests/diseases
            "weather_market": 0.4,       # Weather affects prices
            "weather_operational": 0.6,  # Weather affects operations
            "market_financial": 0.8,     # Market affects finances
            "operational_financial": 0.7, # Operations affect finances
            "biological_operational": 0.5, # Pests affect operations
            "biological_market": 0.3,    # Pests affect market perception
            "weather_financial": 0.5,    # Weather affects finances
            "market_operational": 0.4,   # Market affects operations
            "biological_financial": 0.6  # Pests affect finances
        }
        
        # Create full matrix
        risk_types = [rt.value for rt in RiskType]
        matrix = {rt1: {rt2: 0.0 for rt2 in risk_types} for rt1 in risk_types}
        
        # Set self-correlations to 1.0
        for rt in risk_types:
            matrix[rt][rt] = 1.0
        
        # Set correlations
        for pair, correlation in correlations.items():
            rt1, rt2 = pair.split("_")
            matrix[rt1][rt2] = correlation
            matrix[rt2][rt1] = correlation
        
        return matrix
    
    def assess_farm_risk(self, 
                        farm_context: Dict[str, Any],
                        time_horizon_days: int = 90) -> RiskAssessment:
        """
        Perform comprehensive risk assessment for a farm.
        
        Args:
            farm_context: Farm-specific context and data
            time_horizon_days: Time horizon for risk assessment
        
        Returns:
            Complete risk assessment with scores and recommendations
        """
        print(f"🔍 Assessing farm risk for {farm_context.get('farm_id', 'unknown')}")
        
        # Assess each risk dimension
        risk_dimensions = {}
        dimension_scores = {}
        
        for risk_type, model in self.risk_models.items():
            dimension_assessment = self._assess_risk_dimension(
                risk_type, model, farm_context, time_horizon_days
            )
            risk_dimensions[risk_type] = dimension_assessment
            dimension_scores[risk_type] = dimension_assessment.get("risk_score", 0)
        
        # Calculate overall risk score (weighted average)
        overall_score = self._calculate_overall_risk_score(dimension_scores)
        
        # Identify highest risks
        highest_risks = self._identify_highest_risks(risk_dimensions)
        
        # Generate mitigation plan
        mitigation_plan = self._generate_mitigation_plan(risk_dimensions, farm_context)
        
        # Check for early warnings
        early_warnings = self._check_early_warnings(farm_context)
        
        # Store in history
        assessment = RiskAssessment(
            farm_id=farm_context.get("farm_id", "unknown"),
            assessment_date=datetime.utcnow(),
            time_horizon_days=time_horizon_days,
            risk_dimensions=risk_dimensions,
            overall_risk_score=overall_score,
            highest_risks=highest_risks,
            mitigation_plan=mitigation_plan,
            early_warnings=early_warnings
        )
        
        self.risk_history.append(assessment)
        
        return {
            "assessment_id": f"risk_assess_{datetime.utcnow().timestamp()}",
            "farm_id": farm_context.get("farm_id", "unknown"),
            "assessment_date": datetime.utcnow().isoformat(),
            "time_horizon_days": time_horizon_days,
            "overall_risk_score": round(overall_score, 2),
            "risk_level": self._get_risk_level(overall_score),
            "risk_dimensions": risk_dimensions,
            "highest_risks": highest_risks,
            "mitigation_plan": mitigation_plan,
            "early_warnings": early_warnings,
            "correlation_analysis": self._analyze_correlations(risk_dimensions),
            "risk_scenarios": self._generate_risk_scenarios(risk_dimensions, farm_context),
            "insurance_recommendations": self._generate_insurance_recommendations(risk_dimensions),
            "monitoring_recommendations": self._generate_monitoring_recommendations(risk_dimensions)
        }
    
    def _assess_risk_dimension(self, 
                              risk_type: RiskType,
                              model: RiskDimension,
                              farm_context: Dict[str, Any],
                              time_horizon_days: int) -> Dict[str, Any]:
        """Assess a specific risk dimension."""
        # Get risk indicators from farm context
        indicators = self._extract_risk_indicators(risk_type, farm_context)
        
        # Calculate risk score for each indicator
        indicator_scores = {}
        for indicator in model.indicators:
            value = indicators.get(indicator, 0)
            score = self._calculate_indicator_score(risk_type, indicator, value, time_horizon_days)
            indicator_scores[indicator] = {
                "value": value,
                "score": score,
                "weight": self._get_indicator_weight(risk_type, indicator),
                "threshold": self._get_indicator_threshold(risk_type, indicator)
            }
        
        # Calculate overall dimension score (weighted average)
        total_weight = 0
        weighted_score = 0
        for indicator, data in indicator_scores.items():
            weight = data["weight"]
            score = data["score"]
            weighted_score += score * weight
            total_weight += weight
        
        dimension_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Identify key risks within dimension
        key_risks = []
        for indicator, data in indicator_scores.items():
            if data["score"] > 0.6:  # High risk threshold
                key_risks.append({
                    "indicator": indicator,
                    "score": data["score"],
                    "value": data["value"],
                    "explanation": self._get_risk_explanation(risk_type, indicator, data["value"])
                })
        
        # Get mitigation strategies
        applicable_mitigations = self._get_applicable_mitigations(
            risk_type, indicator_scores, farm_context
        )
        
        return {
            "risk_type": risk_type.value,
            "risk_score": round(dimension_score, 2),
            "risk_level": self._get_risk_level(dimension_score),
            "indicator_scores": indicator_scores,
            "key_risks": key_risks,
            "mitigation_strategies": applicable_mitigations,
            "monitoring_requirements": model.monitoring_frequency,
            "data_sources": model.data_sources
        }
    
    def _extract_risk_indicators(self, 
                                risk_type: RiskType,
                                farm_context: Dict[str, Any]) -> Dict[str, float]:
        """Extract risk indicators from farm context."""
        indicators = {}
        
        if risk_type == RiskType.WEATHER:
            indicators = {
                "rainfall_deviation": farm_context.get("rainfall_deviation", 0),
                "temperature_extremes": farm_context.get("temperature_extremes", 0),
                "drought_index": farm_context.get("drought_index", 0),
                "flood_probability": farm_context.get("flood_probability", 0),
                "frost_days": farm_context.get("frost_days", 0),
                "storm_frequency": farm_context.get("storm_frequency", 0)
            }
        elif risk_type == RiskType.MARKET:
            indicators = {
                "price_volatility": farm_context.get("price_volatility", 0),
                "market_access": farm_context.get("market_access_score", 0),
                "demand_shocks": farm_context.get("demand_shock_probability", 0),
                "supply_chain_disruption": farm_context.get("supply_chain_risk", 0),
                "export_restrictions": farm_context.get("export_restriction_risk", 0),
                "input_price_fluctuation": farm_context.get("input_price_volatility", 0)
            }
        elif risk_type == RiskType.OPERATIONAL:
            indicators = {
                "equipment_failure_rate": farm_context.get("equipment_failure_rate", 0),
                "labor_availability": farm_context.get("labor_availability_score", 0),
                "input_supply_reliability": farm_context.get("input_supply_reliability", 0),
                "fuel_price_volatility": farm_context.get("fuel_price_volatility", 0),
                "transportation_disruption": farm_context.get("transportation_risk", 0),
                "energy_outages": farm_context.get("energy_outage_frequency", 0)
            }
        elif risk_type == RiskType.BIOLOGICAL:
            indicators = {
                "pest_infestation_level": farm_context.get("pest_infestation_level", 0),
                "disease_prevalence": farm_context.get("disease_prevalence", 0),
                "weed_pressure": farm_context.get("weed_pressure", 0),
                "soil_health_decline": farm_context.get("soil_health_decline", 0),
                "biodiversity_loss": farm_context.get("biodiversity_loss", 0),
                "invasive_species": farm_context.get("invasive_species_risk", 0)
            }
        elif risk_type == RiskType.FINANCIAL:
            indicators = {
                "debt_to_asset_ratio": farm_context.get("debt_to_asset_ratio", 0),
                "liquidity_ratio": farm_context.get("liquidity_ratio", 0),
                "interest_rate_exposure": farm_context.get("interest_rate_exposure", 0),
                "exchange_rate_risk": farm_context.get("exchange_rate_risk", 0),
                "credit_availability": farm_context.get("credit_availability_score", 0),
                "insurance_coverage": farm_context.get("insurance_coverage_ratio", 0)
            }
        
        return indicators
    
    def _calculate_indicator_score(self, 
                                  risk_type: RiskType,
                                  indicator: str,
                                  value: float,
                                  time_horizon_days: int) -> float:
        """Calculate risk score for a specific indicator."""
        # Simplified scoring logic
        # In practice, this would use statistical models
        
        thresholds = {
            "weather": {
                "rainfall_deviation": (0.7, 0.9),  # (medium_threshold, high_threshold)
                "temperature_extremes": (3, 5),
                "drought_index": (0.5, 0.7),
                "flood_probability": (0.3, 0.5),
                "frost_days": (2, 5),
                "storm_frequency": (2, 4)
            },
            "market": {
                "price_volatility": (0.2, 0.3),
                "market_access": (0.6, 0.4),  # Lower is worse for access
                "demand_shocks": (0.3, 0.5),
                "supply_chain_disruption": (0.4, 0.6),
                "export_restrictions": (0.3, 0.5),
                "input_price_fluctuation": (0.15, 0.25)
            },
            "operational": {
                "equipment_failure_rate": (0.1, 0.2),
                "labor_availability": (0.7, 0.5),  # Lower is worse
                "input_supply_reliability": (0.8, 0.6),  # Lower is worse
                "fuel_price_volatility": (0.15, 0.25),
                "transportation_disruption": (0.3, 0.5),
                "energy_outages": (2, 4)
            },
            "biological": {
                "pest_infestation_level": (0.3, 0.5),
                "disease_prevalence": (0.2, 0.4),
                "weed_pressure": (0.4, 0.6),
                "soil_health_decline": (0.3, 0.5),
                "biodiversity_loss": (0.4, 0.6),
                "invasive_species": (0.3, 0.5)
            },
            "financial": {
                "debt_to_asset_ratio": (0.6, 0.8),
                "liquidity_ratio": (1.5, 1.0),  # Lower is worse
                "interest_rate_exposure": (0.2, 0.3),
                "exchange_rate_risk": (0.15, 0.25),
                "credit_availability": (0.7, 0.5),  # Lower is worse
                "insurance_coverage": (0.7, 0.5)  # Lower is worse
            }
        }
        
        risk_type_str = risk_type.value
        if risk_type_str in thresholds and indicator in thresholds[risk_type_str]:
            low_thresh, high_thresh = thresholds[risk_type_str][indicator]
            
            # Some indicators: higher value = higher risk
            # Others: lower value = higher risk (inverse)
            inverse_indicators = ["market_access", "labor_availability", 
                                "input_supply_reliability", "liquidity_ratio",
                                "credit_availability", "insurance_coverage"]
            
            if indicator in inverse_indicators:
                # Inverse relationship
                if value <= high_thresh:
                    return 0.8 + (high_thresh - value) / high_thresh * 0.2
                elif value <= low_thresh:
                    return 0.4 + (low_thresh - value) / low_thresh * 0.4
                else:
                    return 0.0 + (value - low_thresh) / (1 - low_thresh) * 0.4
            else:
                # Direct relationship
                if value >= high_thresh:
                    return 0.8 + (value - high_thresh) / (1 - high_thresh) * 0.2
                elif value >= low_thresh:
                    return 0.4 + (value - low_thresh) / (high_thresh - low_thresh) * 0.4
                else:
                    return 0.0 + value / low_thresh * 0.4
        
        return 0.5  # Default medium risk
    
    def _get_indicator_weight(self, risk_type: RiskType, indicator: str) -> float:
        """Get weight for an indicator in risk calculation."""
        weights = {
            "weather": {
                "rainfall_deviation": 0.3,
                "temperature_extremes": 0.2,
                "drought_index": 0.25,
                "flood_probability": 0.15,
                "frost_days": 0.05,
                "storm_frequency": 0.05
            },
            "market": {
                "price_volatility": 0.3,
                "market_access": 0.25,
                "demand_shocks": 0.15,
                "supply_chain_disruption": 0.15,
                "export_restrictions": 0.1,
                "input_price_fluctuation": 0.05
            },
            "operational": {
                "equipment_failure_rate": 0.2,
                "labor_availability": 0.25,
                "input_supply_reliability": 0.2,
                "fuel_price_volatility": 0.15,
                "transportation_disruption": 0.1,
                "energy_outages": 0.1
            },
            "biological": {
                "pest_infestation_level": 0.3,
                "disease_prevalence": 0.25,
                "weed_pressure": 0.2,
                "soil_health_decline": 0.15,
                "biodiversity_loss": 0.05,
                "invasive_species": 0.05
            },
            "financial": {
                "debt_to_asset_ratio": 0.25,
                "liquidity_ratio": 0.25,
                "interest_rate_exposure": 0.15,
                "exchange_rate_risk": 0.1,
                "credit_availability": 0.15,
                "insurance_coverage": 0.1
            }
        }
        
        risk_type_str = risk_type.value
        if risk_type_str in weights and indicator in weights[risk_type_str]:
            return weights[risk_type_str][indicator]
        
        return 1.0  # Default weight
    
    def _get_indicator_threshold(self, risk_type: RiskType, indicator: str) -> Dict[str, float]:
        """Get thresholds for an indicator."""
        thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8
        }
        
        # Some indicators have specific thresholds
        specific_thresholds = {
            "weather": {
                "rainfall_deviation": {"low": 0.3, "medium": 0.6, "high": 0.8},
                "drought_index": {"low": 0.4, "medium": 0.6, "high": 0.8}
            }
        }
        
        risk_type_str = risk_type.value
        if risk_type_str in specific_thresholds and indicator in specific_thresholds[risk_type_str]:
            return specific_thresholds[risk_type_str][indicator]
        
        return thresholds
    
    def _get_risk_explanation(self, risk_type: RiskType, indicator: str, value: float) -> str:
        """Get explanation for a risk indicator."""
        explanations = {
            "weather": {
                "rainfall_deviation": f"Rainfall deviation of {value*100:.0f}% from normal",
                "drought_index": f"Drought index at {value:.2f} (0-1 scale)",
                "frost_days": f"{value} frost days expected"
            },
            "market": {
                "price_volatility": f"Price volatility at {value*100:.0f}%",
                "market_access": f"Market access score: {value:.2f}/1.0"
            },
            "operational": {
                "equipment_failure_rate": f"Equipment failure rate: {value*100:.0f}%",
                "labor_availability": f"Labor availability score: {value:.2f}/1.0"
            }
        }
        
        risk_type_str = risk_type.value
        if risk_type_str in explanations and indicator in explanations[risk_type_str]:
            return explanations[risk_type_str][indicator]
        
        return f"{indicator.replace('_', ' ').title()}: {value}"
    
    def _get_applicable_mitigations(self, 
                                   risk_type: RiskType,
                                   indicator_scores: Dict[str, Any],
                                   farm_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get applicable mitigation strategies for risks."""
        mitigations = []
        
        # Get high-risk indicators
        high_risk_indicators = [
            indicator for indicator, data in indicator_scores.items()
            if data["score"] > 0.6
        ]
        
        # Map indicators to mitigation strategies
        mitigation_map = {
            "weather": {
                "rainfall_deviation": ["irrigation_systems", "water_harvesting"],
                "drought_index": ["drought_resistant_varieties", "mulching"],
                "flood_probability": ["drainage_systems", "raised_beds"],
                "frost_days": ["frost_protection", "delayed_planting"]
            },
            "market": {
                "price_volatility": ["forward_contracts", "price_insurance"],
                "market_access": ["multiple_markets", "collective_marketing"],
                "supply_chain_disruption": ["local_storage", "alternative_transport"]
            },
            "operational": {
                "equipment_failure_rate": ["preventive_maintenance", "backup_equipment"],
                "labor_availability": ["labor_training", "mechanization"],
                "fuel_price_volatility": ["fuel_efficiency", "renewable_energy"]
            },
            "biological": {
                "pest_infestation_level": ["integrated_pest_management", "biological_controls"],
                "disease_prevalence": ["resistant_varieties", "crop_rotation"],
                "weed_pressure": ["mechanical_weeding", "cover_crops"]
            },
            "financial": {
                "debt_to_asset_ratio": ["debt_restructuring", "equity_financing"],
                "liquidity_ratio": ["cash_reserves", "credit_lines"],
                "insurance_coverage": ["comprehensive_insurance", "risk_pooling"]
            }
        }
        
        risk_type_str = risk_type.value
        if risk_type_str in mitigation_map:
            for indicator in high_risk_indicators:
                if indicator in mitigation_map[risk_type_str]:
                    for strategy in mitigation_map[risk_type_str][indicator]:
                        mitigations.append({
                            "strategy": strategy,
                            "target_indicator": indicator,
                            "expected_impact": "high" if indicator_scores[indicator]["score"] > 0.8 else "medium",
                            "implementation_cost": self._estimate_mitigation_cost(strategy, farm_context),
                            "time_to_implement": self._estimate_implementation_time(strategy)
                        })
        
        # Deduplicate
        unique_mitigations = []
        seen = set()
        for mitigation in mitigations:
            key = mitigation["strategy"]
            if key not in seen:
                seen.add(key)
                unique_mitigations.append(mitigation)
        
        return unique_mitigations
    
    def _estimate_mitigation_cost(self, strategy: str, farm_context: Dict[str, Any]) -> str:
        """Estimate cost of implementing a mitigation strategy."""
        cost_ranges = {
            "irrigation_systems": "high",
            "drought_resistant_varieties": "low",
            "forward_contracts": "low",
            "preventive_maintenance": "medium",
            "integrated_pest_management": "medium",
            "debt_restructuring": "low",
            "comprehensive_insurance": "medium"
        }
        
        return cost_ranges.get(strategy, "medium")
    
    def _estimate_implementation_time(self, strategy: str) -> str:
        """Estimate time to implement a mitigation strategy."""
        time_estimates = {
            "irrigation_systems": "weeks",
            "drought_resistant_varieties": "season",
            "forward_contracts": "days",
            "preventive_maintenance": "days",
            "integrated_pest_management": "weeks",
            "debt_restructuring": "weeks",
            "comprehensive_insurance": "days"
        }
        
        return time_estimates.get(strategy, "weeks")
    
    def _calculate_overall_risk_score(self, dimension_scores: Dict[RiskType, float]) -> float:
        """Calculate overall risk score from dimension scores."""
        # Weight different risk dimensions
        weights = {
            RiskType.WEATHER: 0.25,
            RiskType.MARKET: 0.20,
            RiskType.OPERATIONAL: 0.20,
            RiskType.BIOLOGICAL: 0.20,
            RiskType.FINANCIAL: 0.15
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for risk_type, score in dimension_scores.items():
            weight = weights.get(risk_type, 0.2)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to risk level."""
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "very_low"
    
    def _identify_highest_risks(self, risk_dimensions: Dict[RiskType, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify highest risks across all dimensions."""
        highest_risks = []
        
        for risk_type, dimension in risk_dimensions.items():
            dimension_score = dimension.get("risk_score", 0)
            risk_level = dimension.get("risk_level", "medium")
            
            if risk_level in ["high", "critical"]:
                highest_risks.append({
                    "risk_type": risk_type.value,
                    "risk_score": dimension_score,
                    "risk_level": risk_level,
                    "key_indicators": [
                        {
                            "indicator": ind,
                            "score": data["score"],
                            "value": data["value"]
                        }
                        for ind, data in dimension.get("indicator_scores", {}).items()
                        if data["score"] > 0.6
                    ],
                    "top_mitigation": dimension.get("mitigation_strategies", [])[:2]
                })
        
        # Sort by risk score descending
        highest_risks.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return highest_risks[:5]  # Return top 5
    
    def _generate_mitigation_plan(self, 
                                 risk_dimensions: Dict[RiskType, Dict[str, Any]],
                                 farm_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive mitigation plan."""
        # Collect all mitigation strategies
        all_mitigations = []
        for dimension in risk_dimensions.values():
            all_mitigations.extend(dimension.get("mitigation_strategies", []))
        
        # Group by implementation priority
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for mitigation in all_mitigations:
            impact = mitigation.get("expected_impact", "medium")
            cost = mitigation.get("implementation_cost", "medium")
            
            if impact == "high" and cost in ["low", "medium"]:
                high_priority.append(mitigation)
            elif impact == "high" and cost == "high":
                medium_priority.append(mitigation)
            else:
                low_priority.append(mitigation)
        
        # Estimate total cost
        cost_estimate = self._estimate_total_mitigation_cost(all_mitigations, farm_context)
        
        # Generate timeline
        timeline = self._generate_mitigation_timeline(all_mitigations)
        
        return {
            "total_mitigations": len(all_mitigations),
            "high_priority_actions": high_priority,
            "medium_priority_actions": medium_priority,
            "low_priority_actions": low_priority,
            "cost_estimate": cost_estimate,
            "implementation_timeline": timeline,
            "expected_risk_reduction": self._estimate_risk_reduction(all_mitigations),
            "success_metrics": self._define_success_metrics(risk_dimensions)
        }
    
    def _estimate_total_mitigation_cost(self, 
                                       mitigations: List[Dict[str, Any]],
                                       farm_context: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate total cost of mitigation plan."""
        farm_size = farm_context.get("farm_size_hectares", 2.5)
        
        # Simplified cost estimation
        cost_factors = {
            "low": 100,
            "medium": 500,
            "high": 2000
        }
        
        total_cost = 0
        for mitigation in mitigations:
            cost_level = mitigation.get("implementation_cost", "medium")
            total_cost += cost_factors.get(cost_level, 500)
        
        # Scale by farm size
        total_cost *= farm_size / 2.5
        
        return {
            "estimated_total": round(total_cost, 2),
            "cost_per_hectare": round(total_cost / farm_size, 2),
            "affordability": "affordable" if total_cost < farm_context.get("budget_limit", 5000) * 0.2 else "review_needed",
            "financing_options": ["savings", "microcredit", "cooperative_fund"] if total_cost > 1000 else ["savings"]
        }
    
    def _generate_mitigation_timeline(self, mitigations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate implementation timeline for mitigations."""
        timeline = []
        current_date = datetime.utcnow()
        
        # Group by implementation time
        immediate = []
        short_term = []
        medium_term = []
        long_term = []
        
        for mitigation in mitigations:
            time_to_implement = mitigation.get("time_to_implement", "weeks")
            
            if time_to_implement == "days":
                immediate.append(mitigation)
            elif time_to_implement == "weeks":
                short_term.append(mitigation)
            elif time_to_implement == "season":
                medium_term.append(mitigation)
            else:
                long_term.append(mitigation)
        
        # Create timeline entries
        if immediate:
            timeline.append({
                "phase": "immediate",
                "start_date": current_date.isoformat(),
                "end_date": (current_date + timedelta(days=7)).isoformat(),
                "actions": [m["strategy"] for m in immediate[:3]]
            })
        
        if short_term:
            timeline.append({
                "phase": "short_term",
                "start_date": (current_date + timedelta(days=7)).isoformat(),
                "end_date": (current_date + timedelta(days=30)).isoformat(),
                "actions": [m["strategy"] for m in short_term[:5]]
            })
        
        if medium_term:
            timeline.append({
                "phase": "medium_term",
                "start_date": (current_date + timedelta(days=30)).isoformat(),
                "end_date": (current_date + timedelta(days=90)).isoformat(),
                "actions": [m["strategy"] for m in medium_term[:3]]
            })
        
        return timeline
    
    def _estimate_risk_reduction(self, mitigations: List[Dict[str, Any]]) -> float:
        """Estimate overall risk reduction from mitigations."""
        total_reduction = 0
        
        for mitigation in mitigations:
            impact = mitigation.get("expected_impact", "medium")
            if impact == "high":
                total_reduction += 0.15
            elif impact == "medium":
                total_reduction += 0.08
            else:
                total_reduction += 0.03
        
        # Cap at 50% reduction
        return min(total_reduction, 0.5)
    
    def _define_success_metrics(self, risk_dimensions: Dict[RiskType, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Define success metrics for mitigation plan."""
        metrics = []
        
        for risk_type, dimension in risk_dimensions.items():
            if dimension.get("risk_level") in ["high", "critical"]:
                metrics.append({
                    "risk_type": risk_type.value,
                    "target_score": round(dimension.get("risk_score", 0) * 0.7, 2),  # 30% reduction
                    "measurement_frequency": dimension.get("monitoring_requirements", "weekly"),
                    "data_sources": dimension.get("data_sources", []),
                    "success_threshold": "30% reduction"
                })
        
        return metrics
    
    def _check_early_warnings(self, farm_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for early warning signals."""
        warnings = []
        
        # Check weather warnings
        if farm_context.get("drought_index", 0) > 0.7:
            warnings.append({
                "risk_type": "weather",
                "indicator": "drought_index",
                "current_value": farm_context.get("drought_index", 0),
                "threshold": 0.7,
                "severity": "high",
                "confidence": 0.8,
                "message": "High drought risk detected",
                "recommended_actions": ["Activate irrigation", "Apply mulch", "Consider drought-tolerant varieties"],
                "escalation_protocol": "Notify farm manager and check water reserves"
            })
        
        # Check pest warnings
        if farm_context.get("pest_infestation_level", 0) > 0.6:
            warnings.append({
                "risk_type": "biological",
                "indicator": "pest_infestation_level",
                "current_value": farm_context.get("pest_infestation_level", 0),
                "threshold": 0.6,
                "severity": "medium",
                "confidence": 0.7,
                "message": "Elevated pest pressure detected",
                "recommended_actions": ["Scout fields", "Prepare control measures", "Monitor neighboring farms"],
                "escalation_protocol": "Increase monitoring frequency and prepare IPM plan"
            })
        
        # Check market warnings
        if farm_context.get("price_volatility", 0) > 0.25:
            warnings.append({
                "risk_type": "market",
                "indicator": "price_volatility",
                "current_value": farm_context.get("price_volatility", 0),
                "threshold": 0.25,
                "severity": "medium",
                "confidence": 0.6,
                "message": "High market price volatility",
                "recommended_actions": ["Consider forward contracts", "Review storage options", "Diversify buyers"],
                "escalation_protocol": "Consult market advisor and review hedging options"
            })
        
        # Store warnings
        self.early_warnings.extend(warnings)
        
        return warnings
    
    def _analyze_correlations(self, risk_dimensions: Dict[RiskType, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between different risk dimensions."""
        correlations = []
        
        risk_types = list(risk_dimensions.keys())
        for i, rt1 in enumerate(risk_types):
            for rt2 in risk_types[i+1:]:
                correlation = self.correlation_matrix.get(rt1.value, {}).get(rt2.value, 0)
                if correlation > 0.4:  # Only report significant correlations
                    correlations.append({
                        "risk_pair": f"{rt1.value}_{rt2.value}",
                        "correlation": correlation,
                        "interpretation": self._interpret_correlation(rt1, rt2, correlation),
                        "implication": self._get_correlation_implication(rt1, rt2)
                    })
        
        # Identify cascading risk scenarios
        cascading_scenarios = self._identify_cascading_scenarios(risk_dimensions)
        
        return {
            "significant_correlations": correlations,
            "cascading_risk_scenarios": cascading_scenarios,
            "compound_risk_probability": self._calculate_compound_risk(risk_dimensions),
            "risk_mitigation_synergies": self._identify_mitigation_synergies(correlations)
        }
    
    def _interpret_correlation(self, rt1: RiskType, rt2: RiskType, correlation: float) -> str:
        """Interpret correlation between risk types."""
        interpretations = {
            ("weather", "biological"): "Weather conditions strongly affect pest and disease prevalence",
            ("weather", "market"): "Weather shocks can cause significant price volatility",
            ("market", "financial"): "Market downturns directly impact farm finances",
            ("operational", "financial"): "Operational disruptions lead to financial losses"
        }
        
        key1 = (rt1.value, rt2.value)
        key2 = (rt2.value, rt1.value)
        
        if key1 in interpretations:
            return interpretations[key1]
        elif key2 in interpretations:
            return interpretations[key2]
        
        return f"Moderate correlation between {rt1.value} and {rt2.value} risks"
    
    def _get_correlation_implication(self, rt1: RiskType, rt2: RiskType) -> str:
        """Get implication of correlation for risk management."""
        implications = {
            ("weather", "biological"): "Weather monitoring should trigger biological risk assessments",
            ("weather", "market"): "Weather forecasts should inform marketing decisions",
            ("market", "financial"): "Market risk hedging should consider financial exposure",
            ("operational", "financial"): "Operational continuity plans should include financial buffers"
        }
        
        key1 = (rt1.value, rt2.value)
        key2 = (rt2.value, rt1.value)
        
        if key1 in implications:
            return implications[key1]
        elif key2 in implications:
            return implications[key2]
        
        return f"Consider combined {rt1.value} and {rt2.value} risk mitigation"
    
    def _identify_cascading_scenarios(self, risk_dimensions: Dict[RiskType, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential cascading risk scenarios."""
        scenarios = []
        
        # Drought → Pest outbreak → Market impact
        if (risk_dimensions.get(RiskType.WEATHER, {}).get("risk_level") in ["high", "critical"] and
            risk_dimensions.get(RiskType.BIOLOGICAL, {}).get("risk_score", 0) > 0.5):
            scenarios.append({
                "scenario": "drought_pest_cascade",
                "trigger": "Severe drought",
                "cascade": "Drought stress → Pest susceptibility → Yield loss → Price volatility",
                "probability": 0.4,
                "impact": "high",
                "mitigation": "Integrated water and pest management"
            })
        
        # Market crash → Financial stress → Operational cuts
        if (risk_dimensions.get(RiskType.MARKET, {}).get("risk_level") in ["high", "critical"] and
            risk_dimensions.get(RiskType.FINANCIAL, {}).get("risk_score", 0) > 0.6):
            scenarios.append({
                "scenario": "market_financial_cascade",
                "trigger": "Market price collapse",
                "cascade": "Price drop → Revenue loss → Credit constraints → Input reduction → Yield decline",
                "probability": 0.3,
                "impact": "critical",
                "mitigation": "Diversified income streams and financial reserves"
            })
        
        # Equipment failure → Operational delay → Missed market window
        if (risk_dimensions.get(RiskType.OPERATIONAL, {}).get("risk_score", 0) > 0.7 and
            risk_dimensions.get(RiskType.MARKET, {}).get("risk_score", 0) > 0.5):
            scenarios.append({
                "scenario": "operational_market_cascade",
                "trigger": "Critical equipment failure",
                "cascade": "Equipment down → Planting delay → Late harvest → Missed premium market window",
                "probability": 0.25,
                "impact": "high",
                "mitigation": "Preventive maintenance and equipment redundancy"
            })
        
        return scenarios
    
    def _calculate_compound_risk(self, risk_dimensions: Dict[RiskType, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate compound risk probability."""
        # Simplified compound risk calculation
        high_risk_count = sum(
            1 for dimension in risk_dimensions.values()
            if dimension.get("risk_level") in ["high", "critical"]
        )
        
        if high_risk_count >= 3:
            compound_probability = 0.4
            impact = "critical"
        elif high_risk_count == 2:
            compound_probability = 0.25
            impact = "high"
        elif high_risk_count == 1:
            compound_probability = 0.1
            impact = "medium"
        else:
            compound_probability = 0.05
            impact = "low"
        
        return {
            "compound_probability": compound_probability,
            "impact_level": impact,
            "contributing_risks": [
                rt.value for rt, dimension in risk_dimensions.items()
                if dimension.get("risk_level") in ["high", "critical"]
            ],
            "recommendation": "Implement integrated risk management" if compound_probability > 0.2 else "Monitor risk correlations"
        }
    
    def _identify_mitigation_synergies(self, correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify synergies between mitigation strategies."""
        synergies = []
        
        for correlation in correlations:
            if correlation["correlation"] > 0.6:
                rt1, rt2 = correlation["risk_pair"].split("_")
                
                synergies.append({
                    "risk_pair": correlation["risk_pair"],
                    "synergistic_mitigations": [
                        f"Integrated {rt1}-{rt2} monitoring",
                        f"Combined {rt1}-{rt2} insurance products",
                        f"Coordinated {rt1}-{rt2} response plans"
                    ],
                    "expected_benefit": "30-50% higher effectiveness than separate mitigations",
                    "implementation_approach": "Cross-functional risk management team"
                })
        
        return synergies
    
    def _generate_risk_scenarios(self, 
                                risk_dimensions: Dict[RiskType, Dict[str, Any]],
                                farm_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk scenarios for stress testing."""
        scenarios = []
        
        # Scenario 1: Severe drought
        scenarios.append({
            "scenario_name": "severe_drought",
            "probability": 0.2,
            "impact": {
                "weather_risk_increase": 0.4,
                "biological_risk_increase": 0.3,
                "market_risk_increase": 0.2,
                "yield_impact": "-40%",
                "financial_impact": "-50% revenue"
            },
            "trigger_conditions": ["rainfall < 50% normal", "drought_index > 0.8"],
            "mitigation_effectiveness": {
                "irrigation_systems": "high",
                "drought_resistant_varieties": "medium",
                "crop_insurance": "high"
            },
            "recommended_preparations": [
                "Test irrigation systems",
                "Secure water rights",
                "Review insurance coverage"
            ]
        })
        
        # Scenario 2: Market crash
        scenarios.append({
            "scenario_name": "market_crash",
            "probability": 0.15,
            "impact": {
                "market_risk_increase": 0.6,
                "financial_risk_increase": 0.5,
                "operational_risk_increase": 0.3,
                "price_impact": "-60%",
                "financial_impact": "Liquidity crisis"
            },
            "trigger_conditions": ["global oversupply", "trade restrictions", "demand collapse"],
            "mitigation_effectiveness": {
                "forward_contracts": "high",
                "diversified_markets": "medium",
                "financial_reserves": "high"
            },
            "recommended_preparations": [
                "Diversify buyer portfolio",
                "Build cash reserves",
                "Explore value-added products"
            ]
        })
        
        # Scenario 3: Pest epidemic
        scenarios.append({
            "scenario_name": "pest_epidemic",
            "probability": 0.25,
            "impact": {
                "biological_risk_increase": 0.7,
                "operational_risk_increase": 0.4,
                "financial_risk_increase": 0.3,
                "yield_impact": "-60%",
                "cost_impact": "+40% inputs"
            },
            "trigger_conditions": ["warm wet weather", "neighboring outbreaks", "resistance development"],
            "mitigation_effectiveness": {
                "integrated_pest_management": "high",
                "resistant_varieties": "medium",
                "early_detection": "high"
            },
            "recommended_preparations": [
                "Establish monitoring network",
                "Stock biological controls",
                "Train field scouts"
            ]
        })
        
        return scenarios
    
    def _generate_insurance_recommendations(self, 
                                          risk_dimensions: Dict[RiskType, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insurance recommendations based on risk assessment."""
        recommendations = []
        
        # Weather insurance
        if risk_dimensions.get(RiskType.WEATHER, {}).get("risk_level") in ["high", "critical"]:
            recommendations.append({
                "insurance_type": "weather_index",
                "coverage": "drought, excess rain, temperature extremes",
                "premium_estimate": "3-5% of insured value",
                "payout_triggers": ["rainfall deficit", "temperature thresholds"],
                "providers": ["ACRE Africa", "Pula", "government_schemes"],
                "priority": "high"
            })
        
        # Crop insurance
        if (risk_dimensions.get(RiskType.BIOLOGICAL, {}).get("risk_score", 0) > 0.6 or
            risk_dimensions.get(RiskType.WEATHER, {}).get("risk_score", 0) > 0.6):
            recommendations.append({
                "insurance_type": "multi_peril_crop",
                "coverage": "yield loss from weather, pests, diseases",
                "premium_estimate": "5-8% of expected revenue",
                "payout_triggers": ["yield shortfall", "area-based indices"],
                "providers": ["local_insurers", "international_reinsurers"],
                "priority": "medium"
            })
        
        # Price insurance
        if risk_dimensions.get(RiskType.MARKET, {}).get("risk_score", 0) > 0.7:
            recommendations.append({
                "insurance_type": "price_revenue",
                "coverage": "price drops, revenue shortfalls",
                "premium_estimate": "2-4% of expected revenue",
                "payout_triggers": ["market price below threshold", "revenue shortfall"],
                "providers": ["commodity_exchanges", "specialized_insurers"],
                "priority": "medium"
            })
        
        # Equipment insurance
        if risk_dimensions.get(RiskType.OPERATIONAL, {}).get("risk_score", 0) > 0.6:
            recommendations.append({
                "insurance_type": "equipment_breakdown",
                "coverage": "repair, replacement, business interruption",
                "premium_estimate": "1-2% of equipment value",
                "payout_triggers": ["mechanical failure", "accidental damage"],
                "providers": ["general_insurers", "equipment_dealers"],
                "priority": "low"
            })
        
        return recommendations
    
    def _generate_monitoring_recommendations(self, 
                                           risk_dimensions: Dict[RiskType, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate monitoring recommendations based on risk assessment."""
        monitoring = {
            "high_priority_monitoring": [],
            "medium_priority_monitoring": [],
            "data_collection_tools": [],
            "alert_thresholds": []
        }
        
        for risk_type, dimension in risk_dimensions.items():
            risk_level = dimension.get("risk_level", "medium")
            
            if risk_level in ["high", "critical"]:
                monitoring["high_priority_monitoring"].extend(
                    f"{risk_type.value}: {indicator}" 
                    for indicator, data in dimension.get("indicator_scores", {}).items()
                    if data["score"] > 0.6
                )
            else:
                monitoring["medium_priority_monitoring"].extend(
                    f"{risk_type.value}: {indicator}" 
                    for indicator, data in dimension.get("indicator_scores", {}).items()
                    if data["score"] > 0.4
                )
        
        # Recommend tools based on monitoring needs
        if monitoring["high_priority_monitoring"]:
            monitoring["data_collection_tools"] = [
                "weather_station",
                "soil_moisture_sensors",
                "pest_traps",
                "market_price_feeds",
                "financial_tracking_app"
            ]
        
        # Set alert thresholds
        monitoring["alert_thresholds"] = [
            {"indicator": "drought_index", "threshold": 0.7, "action": "activate_water_plan"},
            {"indicator": "pest_infestation_level", "threshold": 0.6, "action": "initiate_ipm"},
            {"indicator": "price_volatility", "threshold": 0.25, "action": "review_marketing"},
            {"indicator": "equipment_failure_rate", "threshold": 0.15, "action": "schedule_maintenance"}
        ]
        
        return monitoring
    
    def monitor_early_warnings(self, 
                              farm_id: str,
                              monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor early warning signals and generate alerts.
        
        Args:
            farm_id: Farm identifier
            monitoring_data: Current monitoring data
        
        Returns:
            Early warning alerts and recommendations
        """
        print(f"👁️ Monitoring early warnings for farm: {farm_id}")
        
        alerts = []
        updated_warnings = []
        
        # Check each risk dimension
        for risk_type in RiskType:
            indicators = self._extract_risk_indicators(risk_type, monitoring_data)
            
            for indicator, value in indicators.items():
                threshold = self._get_indicator_threshold(risk_type, indicator)
                high_threshold = threshold.get("high", 0.8)
                
                if value >= high_threshold:
                    # Check if this is a new warning or escalation
                    existing_warning = next(
                        (w for w in self.early_warnings 
                         if w.get("indicator") == indicator and w.get("farm_id") == farm_id),
                        None
                    )
                    
                    if existing_warning:
                        # Update existing warning
                        if value > existing_warning.get("current_value", 0):
                            existing_warning["severity"] = self._get_escalated_severity(
                                existing_warning.get("severity", "medium")
                            )
                            existing_warning["current_value"] = value
                            alerts.append({
                                "type": "escalation",
                                "message": f"⚠️ {indicator.replace('_', ' ').title()} has worsened",
                                "previous_value": existing_warning.get("current_value"),
                                "current_value": value,
                                "severity": existing_warning["severity"]
                            })
                    else:
                        # New warning
                        warning = EarlyWarning(
                            risk_type=risk_type,
                            indicator=indicator,
                            current_value=value,
                            threshold=high_threshold,
                            severity=RiskSeverity.HIGH if value > high_threshold * 1.2 else RiskSeverity.MEDIUM,
                            confidence=0.8,
                            recommended_actions=self._get_warning_actions(risk_type, indicator),
                            escalation_protocol="Notify farm manager immediately"
                        )
                        
                        self.early_warnings.append({
                            "farm_id": farm_id,
                            **warning.__dict__
                        })
                        
                        alerts.append({
                            "type": "new",
                            "message": f"🚨 New {risk_type.value} warning: {indicator.replace('_', ' ')}",
                            "current_value": value,
                            "threshold": high_threshold,
                            "severity": warning.severity.value,
                            "recommended_actions": warning.recommended_actions
                        })
        
        # Generate summary
        if alerts:
            status = "active_warnings"
            priority = "high" if any(a.get("severity") == "critical" for a in alerts) else "medium"
        else:
            status = "all_clear"
            priority = "low"
        
        return {
            "monitoring_id": f"monitor_{datetime.utcnow().timestamp()}",
            "farm_id": farm_id,
            "monitoring_time": datetime.utcnow().isoformat(),
            "status": status,
            "priority": priority,
            "alerts": alerts,
            "active_warnings": [
                w for w in self.early_warnings 
                if w.get("farm_id") == farm_id and w.get("severity") in ["high", "critical"]
            ],
            "recommended_actions": self._consolidate_actions(alerts),
            "next_monitoring_schedule": self._get_next_monitoring_schedule(alerts)
        }
    
    def _get_escalated_severity(self, current_severity: str) -> str:
        """Get escalated severity level."""
        escalation_map = {
            "low": "medium",
            "medium": "high",
            "high": "critical",
            "critical": "critical"
        }
        return escalation_map.get(current_severity, "medium")
    
    def _get_warning_actions(self, risk_type: RiskType, indicator: str) -> List[str]:
        """Get recommended actions for a warning."""
        action_map = {
            "weather": {
                "drought_index": ["Check irrigation", "Apply mulch", "Reduce planting area"],
                "flood_probability": ["Clear drainage", "Move equipment", "Prepare sandbags"],
                "frost_days": ["Cover sensitive crops", "Use frost protection", "Delay harvest"]
            },
            "biological": {
                "pest_infestation_level": ["Scout fields", "Deploy traps", "Prepare controls"],
                "disease_prevalence": ["Remove infected plants", "Apply fungicide", "Improve air circulation"]
            },
            "market": {
                "price_volatility": ["Review contracts", "Consider storage", "Diversify buyers"],
                "supply_chain_disruption": ["Find alternatives", "Build inventory", "Contact buyers"]
            }
        }
        
        risk_type_str = risk_type.value
        if risk_type_str in action_map and indicator in action_map[risk_type_str]:
            return action_map[risk_type_str][indicator]
        
        return ["Monitor closely", "Review risk management plan", "Consult advisor"]
    
    def _consolidate_actions(self, alerts: List[Dict[str, Any]]) -> List[str]:
        """Consolidate recommended actions from multiple alerts."""
        all_actions = []
        
        for alert in alerts:
            if "recommended_actions" in alert:
                all_actions.extend(alert["recommended_actions"])
        
        # Remove duplicates and keep unique
        unique_actions = []
        seen = set()
        for action in all_actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)
        
        return unique_actions[:5]  # Return top 5 most frequent
    
    def _get_next_monitoring_schedule(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get next monitoring schedule based on current alerts."""
        if any(a.get("severity") == "critical" for a in alerts):
            frequency = "hourly"
            next_check = datetime.utcnow() + timedelta(hours=1)
        elif any(a.get("severity") == "high" for a in alerts):
            frequency = "4_hourly"
            next_check = datetime.utcnow() + timedelta(hours=4)
        elif alerts:
            frequency = "daily"
            next_check = datetime.utcnow() + timedelta(days=1)
        else:
            frequency = "weekly"
            next_check = datetime.utcnow() + timedelta(days=7)
        
        return {
            "frequency": frequency,
            "next_check": next_check.isoformat(),
            "monitoring_points": ["weather", "pests", "market_prices", "equipment_status"]
        }


# Create global instance
risk_intelligence = RiskIntelligence()