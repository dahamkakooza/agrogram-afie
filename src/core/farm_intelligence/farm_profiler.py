"""
Farm Profiler System - Creates comprehensive farm DNA profile from minimal data
Week 1-2 Implementation - Based on AFIE Blueprint
"""
import numpy as np
from typing import Dict, Any, List, Optional
import uuid
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

from src.models.farm import (
    FarmDNA, FarmIdentity, FarmTypology, ResourceConstraints,
    EnvironmentalContext, FarmSizeCategory, FarmTypologyType,
    SoilType, IrrigationType
)


class FarmProfiler:
    """Creates comprehensive farm DNA profile from minimal data"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize Farm Profiler with optional pre-trained models.
        
        Args:
            model_path: Path to saved ML models
        """
        self.model_path = model_path or "./models/farm_profiler"
        
        # Initialize classifiers
        self.typology_classifier = self._initialize_typology_classifier()
        
        # Agricultural databases (simplified - would come from external sources)
        self.soil_db = self._initialize_soil_database()
        self.crop_suitability_db = self._initialize_crop_suitability()
        self.climate_zones = self._initialize_climate_zones()
        
        print(f"✅ FarmProfiler initialized with model path: {self.model_path}")
    
    def _initialize_typology_classifier(self) -> RandomForestClassifier:
        """Initialize or load farm typology classifier"""
        model_file = os.path.join(self.model_path, "typology_classifier.pkl")
        
        if os.path.exists(model_file):
            # Load pre-trained model
            with open(model_file, 'rb') as f:
                classifier = pickle.load(f)
            print(f"✓ Loaded pre-trained typology classifier from {model_file}")
        else:
            # Initialize new classifier with simplified training data
            # Features: [size_hectares, labor_count, capital_score, market_distance, rainfall, altitude]
            X_train = np.array([
                [2, 3, 2, 15, 800, 800],    # Smallholder
                [50, 10, 4, 5, 1200, 500],   # Commercial
                [20, 5, 3, 20, 600, 1200],   # Livestock
                [5, 4, 4, 10, 900, 600],     # Organic
                [15, 6, 3, 12, 1000, 700],   # Mixed
                [10, 3, 2, 25, 1500, 300],   # Agroforestry
                [3, 2, 3, 8, 1100, 400],     # Horticulture
                [8, 4, 3, 15, 0, 200],       # Aquaculture
            ])
            
            # Labels: 0=smallholder, 1=commercial, 2=livestock, 3=organic, 
            # 4=mixed, 5=agroforestry, 6=horticulture, 7=aquaculture
            y_train = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            classifier.fit(X_train, y_train)
            print("✓ Initialized new typology classifier with synthetic data")
            
            # Save the model
            os.makedirs(self.model_path, exist_ok=True)
            with open(model_file, 'wb') as f:
                pickle.dump(classifier, f)
            print(f"✓ Saved typology classifier to {model_file}")
        
        return classifier
    
    def _initialize_soil_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize soil type database"""
        return {
            'sandy': {
                'drainage': 'excellent',
                'fertility': 'low',
                'water_holding': 'poor',
                'suitable_crops': ['groundnuts', 'watermelon', 'sweet potatoes']
            },
            'clay': {
                'drainage': 'poor',
                'fertility': 'high',
                'water_holding': 'good',
                'suitable_crops': ['rice', 'wheat', 'cabbage']
            },
            'loam': {
                'drainage': 'good',
                'fertility': 'high',
                'water_holding': 'good',
                'suitable_crops': ['maize', 'beans', 'vegetables', 'fruits']
            },
            'silt': {
                'drainage': 'moderate',
                'fertility': 'moderate',
                'water_holding': 'good',
                'suitable_crops': ['potatoes', 'carrots', 'onions']
            }
        }
    
    def _initialize_crop_suitability(self) -> Dict[str, Dict[str, Any]]:
        """Initialize crop suitability database"""
        return {
            'maize': {
                'rainfall_min': 500,
                'rainfall_max': 1200,
                'temp_min': 15,
                'temp_max': 30,
                'soil_preference': ['loam', 'clay'],
                'growing_days': 90
            },
            'beans': {
                'rainfall_min': 400,
                'rainfall_max': 1000,
                'temp_min': 18,
                'temp_max': 25,
                'soil_preference': ['loam', 'silt'],
                'growing_days': 70
            },
            'coffee': {
                'rainfall_min': 1000,
                'rainfall_max': 2000,
                'temp_min': 18,
                'temp_max': 24,
                'soil_preference': ['volcanic', 'loam'],
                'growing_days': 240
            },
            'banana': {
                'rainfall_min': 1200,
                'rainfall_max': 2500,
                'temp_min': 20,
                'temp_max': 30,
                'soil_preference': ['loam', 'clay'],
                'growing_days': 300
            }
        }
    
    def _initialize_climate_zones(self) -> Dict[str, Dict[str, Any]]:
        """Initialize climate zone database"""
        return {
            'tropical': {
                'temp_range': (20, 30),
                'rainfall': 'high',
                'seasons': ['wet', 'dry'],
                'suitable_crops': ['banana', 'coffee', 'cocoa', 'sugarcane']
            },
            'subtropical': {
                'temp_range': (15, 25),
                'rainfall': 'moderate',
                'seasons': ['spring', 'summer', 'fall', 'winter'],
                'suitable_crops': ['citrus', 'avocado', 'tea']
            },
            'temperate': {
                'temp_range': (10, 20),
                'rainfall': 'moderate',
                'seasons': ['spring', 'summer', 'fall', 'winter'],
                'suitable_crops': ['wheat', 'barley', 'apples', 'grapes']
            }
        }
    
    def classify_farm_typology(self, farm_data: Dict[str, Any]) -> FarmTypology:
        """
        Classify farm type using Random Forest classifier.
        
        Args:
            farm_data: Dictionary containing farm features
            
        Returns:
            FarmTypology object
        """
        # Extract features for classification
        features = np.array([[
            farm_data.get('size_hectares', 2),
            farm_data.get('labor_count', 3),
            farm_data.get('capital_score', 2),
            farm_data.get('market_distance_km', 15),
            farm_data.get('annual_rainfall_mm', 800),
            farm_data.get('altitude_m', 800)
        ]])
        
        # Make prediction
        prediction = self.typology_classifier.predict(features)[0]
        
        # Map prediction to FarmTypologyType enum
        type_map = {
            0: FarmTypologyType.SMALLHOLDER,
            1: FarmTypologyType.COMMERCIAL,
            2: FarmTypologyType.LIVESTOCK,
            3: FarmTypologyType.ORGANIC,
            4: FarmTypologyType.MIXED,
            5: FarmTypologyType.AGROFORESTRY,
            6: FarmTypologyType.HORTICULTURE,
            7: FarmTypologyType.AQUACULTURE
        }
        
        farm_type = type_map.get(prediction, FarmTypologyType.SMALLHOLDER)
        
        # Get specialization from data or infer
        specialization = farm_data.get('specialization', ['mixed'])
        if specialization == ['mixed'] and farm_type == FarmTypologyType.LIVESTOCK:
            specialization = ['cattle', 'goats']
        elif specialization == ['mixed'] and farm_type == FarmTypologyType.HORTICULTURE:
            specialization = ['vegetables', 'fruits']
        
        return FarmTypology(
            typology_type=farm_type,
            sub_type=f"{farm_type.value}_general",
            specialization=specialization,
            farming_system=farm_data.get('farming_system', 'rainfed'),
            technology_level=farm_data.get('technology_level', 'intermediate'),
            market_orientation=farm_data.get('market_orientation', 'local'),
            sustainability_practices=farm_data.get('sustainability_practices', ['crop_rotation']),
            certifications=farm_data.get('certifications', [])
        )
    
    def analyze_resource_constraints(self, farm_data: Dict[str, Any]) -> ResourceConstraints:
        """
        Analyze resource constraints based on farm data.
        
        Args:
            farm_data: Dictionary containing farm resource information
            
        Returns:
            ResourceConstraints object
        """
        # Calculate scores based on available data
        size_hectares = farm_data.get('size_hectares', 2)
        labor_count = farm_data.get('labor_count', 3)
        market_distance = farm_data.get('market_distance_km', 15)
        water_access = farm_data.get('water_access', 'rainfed')
        
        # Score calculations (simplified)
        capital_score = max(0, min(100, farm_data.get('capital_score', 50)))
        labor_score = max(0, min(100, (labor_count / 10) * 100))
        equipment_score = max(0, min(100, farm_data.get('equipment_score', 50)))
        market_score = max(0, min(100, 100 - (market_distance / 50) * 100))
        water_score = 80 if water_access in ['river', 'well', 'irrigation'] else 50
        land_score = 90 if size_hectares > 5 else 70
        
        # Calculate constraints
        max_capital = farm_data.get('annual_budget', 100000)
        max_labor = labor_count * 8  # Assuming 8 hours per worker
        
        return ResourceConstraints(
            capital_access_score=capital_score,
            labor_availability_score=labor_score,
            equipment_access_score=equipment_score,
            market_proximity_score=market_score,
            water_access_score=water_score,
            land_availability_score=land_score,
            max_capital_per_season=max_capital,
            max_labor_hours_per_day=max_labor,
            equipment_limitations=farm_data.get('equipment_limitations', []),
            water_restrictions=farm_data.get('water_restrictions', [])
        )
    
    def build_environmental_context(self, location_data: Dict[str, Any]) -> EnvironmentalContext:
        """
        Build environmental context from location data.
        
        Args:
            location_data: Dictionary containing location information
            
        Returns:
            EnvironmentalContext object
        """
        # Get or infer soil type
        soil_type_str = location_data.get('soil_type', 'loam')
        soil_type = SoilType(soil_type_str) if soil_type_str in SoilType.__members__ else SoilType.LOAM
        
        # Get climate data
        rainfall = location_data.get('annual_rainfall_mm', 1000)
        temperature = location_data.get('average_temperature_c', 22)
        altitude = location_data.get('altitude_m', 800)
        
        # Determine climate zone
        if temperature > 20 and rainfall > 800:
            climate_zone = 'tropical'
        elif temperature > 15:
            climate_zone = 'subtropical'
        else:
            climate_zone = 'temperate'
        
        # Determine growing season
        if climate_zone == 'tropical':
            growing_days = 365
        else:
            growing_days = 180  # 6 months
        
        # Determine irrigation type based on rainfall
        if rainfall > 1200:
            irrigation = IrrigationType.RAINFED
        elif rainfall > 600:
            irrigation = IrrigationType.SURFACE
        else:
            irrigation = IrrigationType.DRIP
        
        return EnvironmentalContext(
            soil_type=soil_type,
            soil_ph=location_data.get('soil_ph', 6.5),
            soil_organic_matter=location_data.get('soil_organic_matter', 2.5),
            elevation_meters=altitude,
            average_rainfall_mm=rainfall,
            average_temperature_c=temperature,
            growing_season_days=growing_days,
            irrigation_type=irrigation,
            water_source=location_data.get('water_source', 'rain'),
            microclimate_zones=[climate_zone],
            topography=location_data.get('topography', ['flat'])
        )
    
    def create_farm_dna_from_minimal(self, minimal_data: Dict[str, Any]) -> FarmDNA:
        """
        Create comprehensive Farm DNA from minimal data points (5 points).
        
        Args:
            minimal_data: Dictionary with 5 key data points:
                - primary_language
                - farm_size_category  
                - main_crop_animal
                - biggest_challenge
                - mobile_literacy
                
        Returns:
            Complete FarmDNA object
        """
        print("🎯 Creating FarmDNA from minimal data points...")
        
        # Extract the 5 data points
        language = minimal_data.get('primary_language', 'en')
        size_category = minimal_data.get('farm_size_category', 'smallholder')
        main_crop = minimal_data.get('main_crop_animal', 'maize')
        challenge = minimal_data.get('biggest_challenge', 'water_access')
        mobile_literacy = minimal_data.get('mobile_literacy', 'basic')
        
        print(f"  • Language: {language}")
        print(f"  • Size category: {size_category}")
        print(f"  • Main crop: {main_crop}")
        print(f"  • Challenge: {challenge}")
        print(f"  • Mobile literacy: {mobile_literacy}")
        
        # Map size category to hectares
        size_map = {
            'micro': 0.5,
            'smallholder': 2.0,
            'medium': 10.0,
            'commercial': 50.0,
            'large_scale': 200.0
        }
        size_hectares = size_map.get(size_category, 2.0)
        
        # Map mobile literacy to technology level
        tech_map = {
            'none': 'basic',
            'basic': 'intermediate',
            'intermediate': 'advanced',
            'advanced': 'advanced'
        }
        technology_level = tech_map.get(mobile_literacy, 'intermediate')
        
        # Map challenge to resource scores
        challenge_map = {
            'water_access': {'water_score': 30, 'capital_score': 40},
            'capital_access': {'water_score': 70, 'capital_score': 20},
            'market_access': {'market_score': 30, 'capital_score': 50},
            'labor_shortage': {'labor_score': 30, 'capital_score': 60},
            'pest_disease': {'capital_score': 50, 'equipment_score': 40}
        }
        challenge_impact = challenge_map.get(challenge, {'water_score': 50, 'capital_score': 50})
        
        # Create farm data for classifiers
        farm_data = {
            'size_hectares': size_hectares,
            'labor_count': 3,  # Default family size
            'capital_score': challenge_impact.get('capital_score', 50),
            'market_distance_km': 15,  # Default
            'annual_rainfall_mm': 1000,  # Default tropical
            'altitude_m': 800,  # Default
            'specialization': [main_crop],
            'farming_system': 'rainfed',
            'technology_level': technology_level,
            'market_orientation': 'local',
            'sustainability_practices': ['crop_rotation'],
            'annual_budget': 50000,
            'equipment_score': challenge_impact.get('equipment_score', 50),
            'water_access': 'rainfed' if challenge != 'water_access' else 'limited'
        }
        
        # Create location data
        location_data = {
            'soil_type': 'loam',  # Default
            'soil_ph': 6.5,
            'soil_organic_matter': 2.5,
            'annual_rainfall_mm': 1000,
            'average_temperature_c': 22,
            'altitude_m': 800,
            'water_source': 'rain'
        }
        
        # Create Farm Identity
        identity = FarmIdentity(
            farmer_id=uuid.UUID(minimal_data.get('farmer_id') or minimal_data.get('farm_id') or minimal_data.get('id') or str(uuid.uuid4())),
            name=minimal_data.get('farm_name', 'New Farm'),
            location=minimal_data.get('location', 'Uganda'),
            latitude=minimal_data.get('latitude', 0.0),
            longitude=minimal_data.get('longitude', 32.0),
            size_hectares=size_hectares,
            size_category=FarmSizeCategory(size_category),
            established_year=minimal_data.get('established_year', 2020),
            registration_number=minimal_data.get('registration_number')
        )
        
        # Create components using profiler
        typology = self.classify_farm_typology(farm_data)
        constraints = self.analyze_resource_constraints(farm_data)
        environment = self.build_environmental_context(location_data)
        
        # Override constraint scores based on challenge
        if 'water_score' in challenge_impact:
            constraints.water_access_score = challenge_impact['water_score']
        if 'labor_score' in challenge_impact:
            constraints.labor_availability_score = challenge_impact['labor_score']
        if 'market_score' in challenge_impact:
            constraints.market_proximity_score = challenge_impact['market_score']
        
        print("✅ FarmDNA created successfully!")
        print(f"  • Farm type: {typology.typology_type}")
        print(f"  • Water access score: {constraints.water_access_score}/100")
        print(f"  • Capital access score: {constraints.capital_access_score}/100")
        
        return FarmDNA(
            identity=identity,
            typology=typology,
            constraints=constraints,
            environment=environment,
            goals=minimal_data.get('goals', ['increase yield', 'improve income']),
            preferences={},
            historical_performance=[],
            version=1,
            confidence_score=0.7  # Initial confidence
        )


# Singleton instance for easy access
farm_profiler = FarmProfiler()
