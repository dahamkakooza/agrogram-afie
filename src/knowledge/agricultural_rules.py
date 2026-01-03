"""
Agricultural Rules Knowledge Base - Central repository of farming rules and best practices
Week 3-4 Implementation - Based on AFIE Blueprint
"""
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, field


class ClimateZone(str, Enum):
    """Climate zones for regional adaptation."""
    TROPICAL = "tropical"
    SUBTROPICAL = "subtropical"
    TEMPERATE = "temperate"
    ARID = "arid"
    SEMI_ARID = "semi_arid"
    HUMID = "humid"


class SoilType(str, Enum):
    """Soil types for adaptation."""
    SANDY = "sandy"
    CLAY = "clay"
    LOAM = "loam"
    SILT = "silt"
    PEAT = "peat"
    CHALK = "chalk"


class Season(str, Enum):
    """Agricultural seasons."""
    PLANTING = "planting"
    GROWING = "growing"
    HARVEST = "harvest"
    FALLOW = "fallow"


@dataclass
class AgriculturalRule:
    """A single agricultural rule or best practice."""
    rule_id: str
    crop: str
    practice: str  # e.g., "planting", "fertilizer", "pest_control"
    description: str
    instructions: List[str]
    
    # Conditions for application
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Regional adaptations
    climate_zones: List[ClimateZone] = field(default_factory=list)
    soil_types: List[SoilType] = field(default_factory=list)
    seasons: List[Season] = field(default_factory=list)
    
    # Resource requirements
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Expected outcomes
    expected_impacts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Validation metrics
    success_rate: float = 0.0  # 0.0 to 1.0
    validation_source: str = ""  # "FAO", "University", "Government", "Farmer_validated"
    last_validated: Optional[datetime] = None
    
    # Regional variations
    regional_variations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            "rule_id": self.rule_id,
            "crop": self.crop,
            "practice": self.practice,
            "description": self.description,
            "instructions": self.instructions,
            "conditions": self.conditions,
            "climate_zones": [cz.value for cz in self.climate_zones],
            "soil_types": [st.value for st in self.soil_types],
            "seasons": [s.value for s in self.seasons],
            "resource_requirements": self.resource_requirements,
            "expected_impacts": self.expected_impacts,
            "validation": {
                "success_rate": self.success_rate,
                "source": self.validation_source,
                "last_validated": self.last_validated.isoformat() if self.last_validated else None
            },
            "regional_variations": self.regional_variations
        }
    
    def matches_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Check if rule matches given conditions."""
        for key, value in self.conditions.items():
            if key not in conditions:
                return False
            if conditions[key] != value:
                return False
        return True


class AgriculturalKnowledgeBase:
    """
    Central repository of farming rules and best practices.
    """
    
    def __init__(self):
        self.rules: Dict[str, AgriculturalRule] = {}
        self.crop_rules: Dict[str, List[str]] = {}  # crop -> list of rule_ids
        self.regional_adaptations: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with base knowledge
        self._initialize_base_rules()
        self._initialize_regional_adaptations()
        
        print("✅ AgriculturalKnowledgeBase initialized with rules")
    
    def _initialize_base_rules(self) -> None:
        """Initialize base agricultural rules."""
        
        # ===== MAIZE RULES =====
        maize_rules = [
            AgriculturalRule(
                rule_id="maize_planting_basic",
                crop="maize",
                practice="planting",
                description="Basic maize planting guidelines",
                instructions=[
                    "Plant when soil temperature reaches 10°C",
                    "Plant seeds 2-5 cm deep",
                    "Space rows 75-90 cm apart",
                    "Space plants 20-30 cm within rows",
                    "Use 20-25 kg seeds per hectare"
                ],
                conditions={"soil_temp": ">10°C", "season": "rainy"},
                climate_zones=[ClimateZone.TROPICAL, ClimateZone.SUBTROPICAL],
                soil_types=[SoilType.LOAM, SoilType.SILT],
                seasons=[Season.PLANTING],
                resource_requirements={
                    "seeds": {"amount": 20, "unit": "kg/ha"},
                    "labor": {"amount": 8, "unit": "hours/ha"}
                },
                expected_impacts=[
                    {"impact": "germination_rate", "value": "85-95%"},
                    {"impact": "establishment", "value": "good"}
                ],
                success_rate=0.85,
                validation_source="FAO",
                last_validated=datetime(2024, 1, 15)
            ),
            
            AgriculturalRule(
                rule_id="maize_fertilizer_basic",
                crop="maize",
                practice="fertilizer",
                description="Basic maize fertilizer application",
                instructions=[
                    "Apply NPK fertilizer at planting",
                    "Use 100-50-50 kg/ha NPK ratio",
                    "Apply top dressing at 6 weeks after planting",
                    "Use CAN or Urea for top dressing",
                    "Incorporate fertilizer into soil"
                ],
                conditions={"growth_stage": "vegetative"},
                climate_zones=[ClimateZone.TROPICAL, ClimateZone.SUBTROPICAL],
                soil_types=[SoilType.LOAM, SoilType.SILT, SoilType.CLAY],
                seasons=[Season.GROWING],
                resource_requirements={
                    "fertilizer": {"amount": 100, "unit": "kg/ha NPK"},
                    "labor": {"amount": 4, "unit": "hours/ha"}
                },
                expected_impacts=[
                    {"impact": "yield_increase", "value": "20-40%"},
                    {"impact": "plant_height", "value": "increased"}
                ],
                success_rate=0.78,
                validation_source="University",
                last_validated=datetime(2024, 2, 10)
            ),
            
            AgriculturalRule(
                rule_id="maize_water_management",
                crop="maize",
                practice="water_management",
                description="Maize water requirements",
                instructions=[
                    "Maize requires 500-800 mm water per season",
                    "Critical periods: flowering and grain filling",
                    "Avoid water stress during tasseling",
                    "Use mulching to conserve moisture",
                    "Irrigate if rainfall is insufficient"
                ],
                conditions={"water_availability": "<500mm"},
                climate_zones=[ClimateZone.ARID, ClimateZone.SEMI_ARID],
                soil_types=[SoilType.SANDY, SoilType.LOAM],
                seasons=[Season.GROWING],
                resource_requirements={
                    "water": {"amount": 500, "unit": "mm/season"},
                    "labor": {"amount": 6, "unit": "hours/ha"}
                },
                expected_impacts=[
                    {"impact": "water_use_efficiency", "value": "improved"},
                    {"impact": "drought_tolerance", "value": "increased"}
                ],
                success_rate=0.82,
                validation_source="Government",
                last_validated=datetime(2024, 3, 5)
            ),
            
            AgriculturalRule(
                rule_id="maize_pest_control_fall_armyworm",
                crop="maize",
                practice="pest_control",
                description="Fall armyworm management",
                instructions=[
                    "Monitor fields weekly for egg masses",
                    "Use pheromone traps for early detection",
                    "Apply biopesticides (Bt, neem) at first sign",
                    "Hand-pick larvae in small infestations",
                    "Use chemical pesticides only as last resort"
                ],
                conditions={"pest_pressure": "high", "pest_type": "fall_armyworm"},
                climate_zones=[ClimateZone.TROPICAL, ClimateZone.SUBTROPICAL],
                soil_types=[],
                seasons=[Season.GROWING],
                resource_requirements={
                    "pesticides": {"amount": 1, "unit": "kg/ha"},
                    "labor": {"amount": 12, "unit": "hours/ha"}
                },
                expected_impacts=[
                    {"impact": "pest_damage_reduction", "value": "60-80%"},
                    {"impact": "yield_loss_prevention", "value": "15-30%"}
                ],
                success_rate=0.75,
                validation_source="Research",
                last_validated=datetime(2024, 4, 20)
            )
        ]
        
        # ===== BEANS RULES =====
        beans_rules = [
            AgriculturalRule(
                rule_id="beans_planting_intercropping",
                crop="beans",
                practice="planting",
                description="Beans intercropping with maize",
                instructions=[
                    "Plant beans 2-3 weeks after maize",
                    "Use 2 rows beans between maize rows",
                    "Plant beans 5-10 cm deep",
                    "Space bean plants 10-15 cm apart",
                    "Use inoculant for nitrogen fixation"
                ],
                conditions={"cropping_system": "intercropping", "companion_crop": "maize"},
                climate_zones=[ClimateZone.TROPICAL],
                soil_types=[SoilType.LOAM],
                seasons=[Season.PLANTING],
                resource_requirements={
                    "seeds": {"amount": 15, "unit": "kg/ha"},
                    "inoculant": {"amount": 0.5, "unit": "kg/ha"}
                },
                expected_impacts=[
                    {"impact": "nitrogen_fixation", "value": "40-60 kg/ha"},
                    {"impact": "land_productivity", "value": "increased"}
                ],
                success_rate=0.88,
                validation_source="FAO",
                last_validated=datetime(2024, 1, 20)
            )
        ]
        
        # ===== COFFEE RULES =====
        coffee_rules = [
            AgriculturalRule(
                rule_id="coffee_pruning_young",
                crop="coffee",
                practice="pruning",
                description="Pruning young coffee plants",
                instructions=[
                    "Prune in dry season",
                    "Remove suckers from base",
                    "Maintain single stem for first 2 years",
                    "Cut back to 30-40 cm after first year",
                    "Remove diseased or damaged branches"
                ],
                conditions={"plant_age": "young", "season": "dry"},
                climate_zones=[ClimateZone.TROPICAL],
                soil_types=[],
                seasons=[Season.GROWING],
                resource_requirements={
                    "tools": {"type": "pruning_shears"},
                    "labor": {"amount": 16, "unit": "hours/ha"}
                },
                expected_impacts=[
                    {"impact": "plant_health", "value": "improved"},
                    {"impact": "future_yield", "value": "increased"}
                ],
                success_rate=0.90,
                validation_source="University",
                last_validated=datetime(2024, 2, 15)
            )
        ]
        
        # Add all rules to knowledge base
        all_rules = maize_rules + beans_rules + coffee_rules
        
        for rule in all_rules:
            self.add_rule(rule)
    
    def _initialize_regional_adaptations(self) -> None:
        """Initialize regional adaptation rules."""
        
        # East Africa adaptations
        self.regional_adaptations["east_africa"] = {
            "climate_zone": ClimateZone.TROPICAL,
            "adaptations": {
                "maize": {
                    "planting_time": "With first rains (March/April or September/October)",
                    "varieties": ["Drought-tolerant varieties recommended"],
                    "spacing": "Reduce spacing in drier areas",
                    "fertilizer": "Use organic manure where available"
                },
                "beans": {
                    "intercropping": "Common with maize and bananas",
                    "varieties": "Climbing beans in highlands, bush beans in lowlands"
                },
                "coffee": {
                    "shade": "Use Grevillea or banana for shade",
                    "processing": "Wet processing for Arabica"
                }
            },
            "common_practices": [
                "Conservation agriculture",
                "Integrated pest management",
                "Water harvesting"
            ]
        }
        
        # West Africa adaptations
        self.regional_adaptations["west_africa"] = {
            "climate_zone": ClimateZone.TROPICAL,
            "adaptations": {
                "maize": {
                    "planting_time": "June-July for main season",
                    "varieties": ["Early maturing varieties recommended"],
                    "pest_management": "Focus on stem borer control"
                }
            }
        }
        
        # Southern Africa adaptations
        self.regional_adaptations["southern_africa"] = {
            "climate_zone": ClimateZone.SUBTROPICAL,
            "adaptations": {
                "maize": {
                    "planting_time": "November-December",
                    "water_management": "Critical due to erratic rainfall",
                    "drought_tolerance": "Essential trait for varieties"
                }
            }
        }
    
    def add_rule(self, rule: AgriculturalRule) -> None:
        """Add a rule to the knowledge base."""
        self.rules[rule.rule_id] = rule
        
        # Index by crop
        if rule.crop not in self.crop_rules:
            self.crop_rules[rule.crop] = []
        self.crop_rules[rule.crop].append(rule.rule_id)
        
        # Update last validated timestamp
        rule.last_validated = datetime.utcnow()
    
    def get_rules_for_crop(self, crop: str, practice: Optional[str] = None) -> List[AgriculturalRule]:
        """Get all rules for a specific crop and optionally practice."""
        if crop not in self.crop_rules:
            return []
        
        rules = []
        for rule_id in self.crop_rules[crop]:
            rule = self.rules[rule_id]
            if practice is None or rule.practice == practice:
                rules.append(rule)
        
        return rules
    
    def get_matching_rules(
        self,
        crop: str,
        conditions: Dict[str, Any],
        region: Optional[str] = None
    ) -> List[AgriculturalRule]:
        """Get rules that match specific conditions."""
        matching_rules = []
        
        for rule in self.get_rules_for_crop(crop):
            if rule.matches_conditions(conditions):
                # Apply regional adaptations if available
                adapted_rule = self._apply_regional_adaptations(rule, region)
                matching_rules.append(adapted_rule)
        
        return matching_rules
    
    def _apply_regional_adaptations(
        self,
        rule: AgriculturalRule,
        region: Optional[str]
    ) -> AgriculturalRule:
        """Apply regional adaptations to a rule."""
        if not region or region not in self.regional_adaptations:
            return rule
        
        # Create a copy of the rule with adaptations
        adapted_rule = AgriculturalRule(
            rule_id=f"{rule.rule_id}_{region}",
            crop=rule.crop,
            practice=rule.practice,
            description=rule.description,
            instructions=rule.instructions.copy(),
            conditions=rule.conditions.copy(),
            climate_zones=rule.climate_zones.copy(),
            soil_types=rule.soil_types.copy(),
            seasons=rule.seasons.copy(),
            resource_requirements=rule.resource_requirements.copy(),
            expected_impacts=rule.expected_impacts.copy(),
            success_rate=rule.success_rate,
            validation_source=rule.validation_source,
            last_validated=rule.last_validated,
            regional_variations=rule.regional_variations.copy()
        )
        
        # Apply region-specific adaptations
        region_data = self.regional_adaptations[region]
        if rule.crop in region_data["adaptations"]:
            crop_adaptations = region_data["adaptations"][rule.crop]
            
            # Add adaptation notes to description
            adapted_rule.description += f" (Adapted for {region})"
            
            # Add regional instructions
            for key, value in crop_adaptations.items():
                if isinstance(value, list):
                    for item in value:
                        adapted_rule.instructions.append(f"{key}: {item}")
                else:
                    adapted_rule.instructions.append(f"{key}: {value}")
        
        return adapted_rule
    
    def get_rule_by_id(self, rule_id: str) -> Optional[AgriculturalRule]:
        """Get a rule by its ID."""
        return self.rules.get(rule_id)
    
    def search_rules(
        self,
        query: str,
        crop: Optional[str] = None,
        practice: Optional[str] = None
    ) -> List[AgriculturalRule]:
        """Search rules by text query."""
        results = []
        query_lower = query.lower()
        
        for rule_id, rule in self.rules.items():
            # Filter by crop and practice if specified
            if crop and rule.crop != crop:
                continue
            if practice and rule.practice != practice:
                continue
            
            # Search in text fields
            if (query_lower in rule.description.lower() or
                query_lower in rule.practice.lower() or
                any(query_lower in instr.lower() for instr in rule.instructions)):
                results.append(rule)
        
        return results
    
    def validate_rule_with_farmer_feedback(
        self,
        rule_id: str,
        farmer_id: str,
        success: bool,
        notes: Optional[str] = None
    ) -> None:
        """Update rule validation with farmer feedback."""
        if rule_id not in self.rules:
            return
        
        rule = self.rules[rule_id]
        
        # Simple success rate update (would be more sophisticated in production)
        if success:
            rule.success_rate = min(1.0, rule.success_rate + 0.01)
        else:
            rule.success_rate = max(0.0, rule.success_rate - 0.02)
        
        rule.last_validated = datetime.utcnow()
        
        # In a real system, we'd store detailed farmer feedback
        print(f"✅ Updated rule {rule_id} with farmer feedback: success={success}")
    
    def get_crop_suitability(
        self,
        crop: str,
        climate_zone: ClimateZone,
        soil_type: SoilType
    ) -> Dict[str, Any]:
        """Get crop suitability for given conditions."""
        rules = self.get_rules_for_crop(crop)
        
        if not rules:
            return {"suitable": False, "reason": "No rules found for crop"}
        
        # Check climate zone suitability
        climate_suitable = any(
            climate_zone in rule.climate_zones
            for rule in rules
        )
        
        # Check soil type suitability
        soil_suitable = any(
            not rule.soil_types or soil_type in rule.soil_types
            for rule in rules
        )
        
        suitability_score = 0
        if climate_suitable:
            suitability_score += 50
        if soil_suitable:
            suitability_score += 50
        
        # Get specific recommendations
        recommendations = []
        if not climate_suitable:
            recommendations.append(f"Crop not typically grown in {climate_zone.value} climate")
        if not soil_suitable:
            recommendations.append(f"May need soil amendments for {soil_type.value} soil")
        
        return {
            "suitable": climate_suitable and soil_suitable,
            "suitability_score": suitability_score,
            "climate_suitable": climate_suitable,
            "soil_suitable": soil_suitable,
            "recommendations": recommendations,
            "applicable_rules": len(rules)
        }
    
    def to_json(self) -> str:
        """Export knowledge base to JSON."""
        data = {
            "rules": {
                rule_id: rule.to_dict()
                for rule_id, rule in self.rules.items()
            },
            "crop_rules": self.crop_rules,
            "regional_adaptations": self.regional_adaptations,
            "metadata": {
                "total_rules": len(self.rules),
                "crops_covered": list(self.crop_rules.keys()),
                "regions_covered": list(self.regional_adaptations.keys()),
                "export_date": datetime.utcnow().isoformat()
            }
        }
        
        return json.dumps(data, indent=2)
    
    def save_to_file(self, filepath: str) -> None:
        """Save knowledge base to file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        print(f"✅ Saved knowledge base to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'AgriculturalKnowledgeBase':
        """Load knowledge base from file."""
        kb = cls()
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Clear existing rules
            kb.rules = {}
            kb.crop_rules = {}
            
            # Load rules
            for rule_id, rule_data in data.get("rules", {}).items():
                rule = AgriculturalRule(
                    rule_id=rule_data["rule_id"],
                    crop=rule_data["crop"],
                    practice=rule_data["practice"],
                    description=rule_data["description"],
                    instructions=rule_data["instructions"],
                    conditions=rule_data["conditions"],
                    climate_zones=[ClimateZone(cz) for cz in rule_data["climate_zones"]],
                    soil_types=[SoilType(st) for st in rule_data["soil_types"]],
                    seasons=[Season(s) for s in rule_data["seasons"]],
                    resource_requirements=rule_data["resource_requirements"],
                    expected_impacts=rule_data["expected_impacts"],
                    success_rate=rule_data["validation"]["success_rate"],
                    validation_source=rule_data["validation"]["source"],
                    last_validated=datetime.fromisoformat(rule_data["validation"]["last_validated"])
                    if rule_data["validation"]["last_validated"] else None,
                    regional_variations=rule_data.get("regional_variations", {})
                )
                kb.rules[rule_id] = rule
                
                # Index by crop
                if rule.crop not in kb.crop_rules:
                    kb.crop_rules[rule.crop] = []
                kb.crop_rules[rule.crop].append(rule_id)
            
            # Load regional adaptations
            kb.regional_adaptations = data.get("regional_adaptations", {})
            
            print(f"✅ Loaded knowledge base from {filepath}")
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️  Could not load from {filepath}: {e}. Using default rules.")
        
        return kb


# Example rule templates for quick creation
def create_fertilizer_rule_template(crop: str, npk_ratio: str) -> AgriculturalRule:
    """Create a fertilizer rule template for a crop."""
    return AgriculturalRule(
        rule_id=f"{crop}_fertilizer_template",
        crop=crop,
        practice="fertilizer",
        description=f"{crop.capitalize()} fertilizer application",
        instructions=[
            f"Apply {npk_ratio} kg/ha NPK fertilizer",
            "Incorporate into soil before planting",
            "Apply top dressing at appropriate growth stage",
            "Adjust based on soil test results"
        ],
        conditions={"growth_stage": "vegetative"},
        resource_requirements={
            "fertilizer": {"amount": float(npk_ratio.split('-')[0]), "unit": "kg/ha"}
        },
        expected_impacts=[
            {"impact": "yield_increase", "value": "15-30%"},
            {"impact": "plant_health", "value": "improved"}
        ],
        success_rate=0.75,
        validation_source="Template"
    )


def create_pest_control_rule_template(crop: str, pest: str) -> AgriculturalRule:
    """Create a pest control rule template."""
    return AgriculturalRule(
        rule_id=f"{crop}_pest_{pest}",
        crop=crop,
        practice="pest_control",
        description=f"{pest.replace('_', ' ').title()} control for {crop}",
        instructions=[
            f"Monitor for {pest} regularly",
            "Use integrated pest management approach",
            "Apply control measures at first sign",
            "Rotate control methods to prevent resistance"
        ],
        conditions={"pest_type": pest},
        expected_impacts=[
            {"impact": "pest_damage_reduction", "value": "50-70%"},
            {"impact": "yield_loss_prevention", "value": "10-25%"}
        ],
        success_rate=0.70,
        validation_source="Template"
    )


# Singleton instance for easy access
agricultural_kb = AgriculturalKnowledgeBase()


# Quick test function
def test_knowledge_base() -> None:
    """Test the agricultural knowledge base."""
    kb = AgriculturalKnowledgeBase()
    
    print("\n📊 Knowledge Base Summary:")
    print(f"Total rules: {len(kb.rules)}")
    print(f"Crops covered: {list(kb.crop_rules.keys())}")
    
    print("\n🌽 Maize Rules:")
    maize_rules = kb.get_rules_for_crop("maize")
    for rule in maize_rules:
        print(f"  • {rule.practice}: {rule.description}")
    
    print("\n🔍 Search for 'fertilizer':")
    results = kb.search_rules("fertilizer")
    for rule in results[:3]:  # Show first 3
        print(f"  • {rule.crop}: {rule.description}")
    
    print("\n🌍 East Africa adaptations for maize:")
    east_africa_rules = kb.get_matching_rules(
        crop="maize",
        conditions={"soil_temp": ">10°C"},
        region="east_africa"
    )
    for rule in east_africa_rules[:2]:
        print(f"  • {rule.description}")
        print(f"    Instructions: {rule.instructions[:2]}")


if __name__ == "__main__":
    test_knowledge_base()
