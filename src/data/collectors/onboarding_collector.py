"""
Onboarding Data Collector - Collects initial 5 data points and triggers gamification
Week 1-2 Implementation - Based on AFIE Blueprint
"""
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import uuid
from datetime import datetime


class OnboardingStep(str, Enum):
    """Onboarding steps enumeration."""
    LANGUAGE = "language"
    FARM_SIZE = "farm_size"
    MAIN_CROP = "main_crop"
    CHALLENGE = "challenge"
    MOBILE_LITERACY = "mobile_literacy"


class DifficultyLevel(str, Enum):
    """Game difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class LearningStyle(str, Enum):
    """Farmer learning styles."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READ_WRITE = "read_write"


class OnboardingCollector:
    """
    Collects initial 5 data points and triggers appropriate gamification.

    Data Points:
    1. Primary language → Communication preference
    2. Farm size category → Resource constraint level
    3. Main crop/animal → Specialization complexity
    4. Biggest challenge → Pain point priority
    5. Mobile literacy → Interface complexity tolerance
    """

    def __init__(self):
        """Initialize onboarding collector with default mappings."""
        self.language_map = self._initialize_language_map()
        self.size_map = self._initialize_size_map()
        self.crop_map = self._initialize_crop_map()
        self.challenge_map = self._initialize_challenge_map()
        self.literacy_map = self._initialize_literacy_map()

        print("✅ OnboardingCollector initialized")

    def _initialize_language_map(self) -> Dict[str, Dict[str, Any]]:
        """Map languages to communication preferences."""
        return {
            'en': {
                'communication_style': 'direct',
                'formality': 'medium',
                'preferred_channels': ['text', 'video'],
                'game_language': 'english'
            },
            'sw': {
                'communication_style': 'relational',
                'formality': 'low',
                'preferred_channels': ['audio', 'video'],
                'game_language': 'swahili'
            },
            'fr': {
                'communication_style': 'formal',
                'formality': 'high',
                'preferred_channels': ['text', 'audio'],
                'game_language': 'french'
            },
            'lg': {
                'communication_style': 'relational',
                'formality': 'low',
                'preferred_channels': ['audio', 'local_video'],
                'game_language': 'luganda'
            }
        }

    def _initialize_size_map(self) -> Dict[str, Dict[str, Any]]:
        """Map farm sizes to resource constraint levels."""
        return {
            'micro': {
                'resource_level': 'very_constrained',
                'labor_scale': 'family_only',
                'capital_access': 'very_limited',
                'risk_tolerance': 'very_low'
            },
            'smallholder': {
                'resource_level': 'constrained',
                'labor_scale': 'family_plus',
                'capital_access': 'limited',
                'risk_tolerance': 'low'
            },
            'medium': {
                'resource_level': 'moderate',
                'labor_scale': 'hired_labor',
                'capital_access': 'moderate',
                'risk_tolerance': 'medium'
            },
            'commercial': {
                'resource_level': 'adequate',
                'labor_scale': 'team',
                'capital_access': 'good',
                'risk_tolerance': 'high'
            },
            'large_scale': {
                'resource_level': 'abundant',
                'labor_scale': 'organization',
                'capital_access': 'excellent',
                'risk_tolerance': 'very_high'
            }
        }

    def _initialize_crop_map(self) -> Dict[str, Dict[str, Any]]:
        """Map crops/animals to specialization complexity."""
        return {
            'maize': {
                'complexity': 'low',
                'knowledge_required': 'basic',
                'risk_level': 'medium',
                'learning_curve': 'gentle',
                'game_category': 'cereals'
            },
            'beans': {
                'complexity': 'low',
                'knowledge_required': 'basic',
                'risk_level': 'low',
                'learning_curve': 'gentle',
                'game_category': 'legumes'
            },
            'coffee': {
                'complexity': 'high',
                'knowledge_required': 'advanced',
                'risk_level': 'high',
                'learning_curve': 'steep',
                'game_category': 'cash_crops'
            },
            'banana': {
                'complexity': 'medium',
                'knowledge_required': 'intermediate',
                'risk_level': 'medium',
                'learning_curve': 'moderate',
                'game_category': 'fruits'
            },
            'cattle': {
                'complexity': 'high',
                'knowledge_required': 'advanced',
                'risk_level': 'high',
                'learning_curve': 'steep',
                'game_category': 'livestock'
            },
            'goats': {
                'complexity': 'medium',
                'knowledge_required': 'intermediate',
                'risk_level': 'medium',
                'learning_curve': 'moderate',
                'game_category': 'livestock'
            },
            'poultry': {
                'complexity': 'low',
                'knowledge_required': 'basic',
                'risk_level': 'low',
                'learning_curve': 'gentle',
                'game_category': 'poultry'
            }
        }

    def _initialize_challenge_map(self) -> Dict[str, Dict[str, Any]]:
        """Map challenges to pain point priorities."""
        return {
            'water_access': {
                'priority': 'critical',
                'urgency': 'high',
                'impact': 'severe',
                'game_focus': 'water_management',
                'recommended_games': ['rainwater_harvesting', 'irrigation_planning']
            },
            'capital_access': {
                'priority': 'high',
                'urgency': 'medium',
                'impact': 'severe',
                'game_focus': 'financial_planning',
                'recommended_games': ['budget_allocation', 'loan_management']
            },
            'market_access': {
                'priority': 'medium',
                'urgency': 'low',
                'impact': 'moderate',
                'game_focus': 'market_strategy',
                'recommended_games': ['price_prediction', 'transport_optimization']
            },
            'labor_shortage': {
                'priority': 'high',
                'urgency': 'medium',
                'impact': 'moderate',
                'game_focus': 'labor_management',
                'recommended_games': ['task_scheduling', 'skill_allocation']
            },
            'pest_disease': {
                'priority': 'critical',
                'urgency': 'high',
                'impact': 'severe',
                'game_focus': 'pest_management',
                'recommended_games': ['integrated_pest_mgmt', 'disease_detection']
            },
            'soil_fertility': {
                'priority': 'medium',
                'urgency': 'low',
                'impact': 'moderate',
                'game_focus': 'soil_health',
                'recommended_games': ['soil_testing', 'fertilizer_calculation']
            }
        }

    def _initialize_literacy_map(self) -> Dict[str, Dict[str, Any]]:
        """Map mobile literacy to interface complexity tolerance."""
        return {
            'none': {
                'interface_complexity': 'very_simple',
                'interaction_type': 'voice_based',
                'visual_elements': 'minimal',
                'text_amount': 'none',
                'game_interface': 'audio_guided'
            },
            'basic': {
                'interface_complexity': 'simple',
                'interaction_type': 'touch_based',
                'visual_elements': 'basic',
                'text_amount': 'minimal',
                'game_interface': 'icon_based'
            },
            'intermediate': {
                'interface_complexity': 'moderate',
                'interaction_type': 'touch_text',
                'visual_elements': 'standard',
                'text_amount': 'moderate',
                'game_interface': 'text_guided'
            },
            'advanced': {
                'interface_complexity': 'complex',
                'interaction_type': 'multi_modal',
                'visual_elements': 'rich',
                'text_amount': 'substantial',
                'game_interface': 'full_featured'
            }
        }

    def collect_onboarding_data(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect and process onboarding responses.

        Args:
            responses: Dictionary with 5 data points:
                - primary_language
                - farm_size_category
                - main_crop_animal
                - biggest_challenge
                - mobile_literacy

        Returns:
            Dictionary with processed onboarding data and gamification triggers
        """
        print("📋 Collecting onboarding data...")

        # Validate required fields
        required_fields = ['primary_language', 'farm_size_category', 'main_crop_animal',
                          'biggest_challenge', 'mobile_literacy']

        for field in required_fields:
            if field not in responses:
                raise ValueError(f"Missing required field: {field}")

        # Extract responses
        language = responses['primary_language']
        size_category = responses['farm_size_category']
        main_crop = responses['main_crop_animal']
        challenge = responses['biggest_challenge']
        literacy = responses['mobile_literacy']

        # Get mappings
        language_info = self.language_map.get(language, self.language_map['en'])
        size_info = self.size_map.get(size_category, self.size_map['smallholder'])
        crop_info = self.crop_map.get(main_crop, self.crop_map['maize'])
        challenge_info = self.challenge_map.get(challenge, self.challenge_map['water_access'])
        literacy_info = self.literacy_map.get(literacy, self.literacy_map['basic'])

        # Determine learning style (simplified based on language and literacy)
        if literacy == 'none' or language in ['lg', 'sw']:
            learning_style = LearningStyle.AUDITORY
        elif literacy == 'basic':
            learning_style = LearningStyle.VISUAL
        elif literacy == 'intermediate':
            learning_style = LearningStyle.KINESTHETIC
        else:
            learning_style = LearningStyle.READ_WRITE

        # Determine initial difficulty level
        if crop_info['complexity'] == 'high' or challenge_info['priority'] == 'critical':
            difficulty = DifficultyLevel.BEGINNER
        elif crop_info['complexity'] == 'medium' or size_info['resource_level'] == 'constrained':
            difficulty = DifficultyLevel.INTERMEDIATE
        else:
            difficulty = DifficultyLevel.ADVANCED

        # Create personalized onboarding journey
        onboarding_journey = self._create_onboarding_journey(
            language_info, size_info, crop_info, challenge_info, literacy_info,
            learning_style, difficulty, main_crop
        )

        # Create gamification triggers
        gamification_triggers = self._create_gamification_triggers(
            crop_info, challenge_info, literacy_info, learning_style, difficulty
        )

        # Compile results
        result = {
            'onboarding_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'collected_responses': responses,
            'processed_data': {
                'communication_preferences': language_info,
                'resource_constraint_level': size_info,
                'specialization_complexity': crop_info,
                'pain_point_priority': challenge_info,
                'interface_complexity_tolerance': literacy_info
            },
            'farmer_profile': {
                'learning_style': learning_style,
                'initial_difficulty': difficulty,
                'risk_tolerance': size_info['risk_tolerance'],
                'technology_affinity': literacy_info['interface_complexity']
            },
            'onboarding_journey': onboarding_journey,
            'gamification_triggers': gamification_triggers,
            'next_steps': [
                {
                    'step': 'farm_dna_creation',
                    'action': 'create_farm_profile',
                    'data_required': responses
                },
                {
                    'step': 'initial_game_assignment',
                    'action': 'start_first_game',
                    'game': gamification_triggers['initial_games'][0]
                }
            ]
        }

        print("✅ Onboarding data collected and processed")
        print(f"  • Learning style: {learning_style}")
        print(f"  • Initial difficulty: {difficulty}")
        print(f"  • First game: {gamification_triggers['initial_games'][0]}")

        return result

    def _create_onboarding_journey(self, language_info: Dict[str, Any],
                                  size_info: Dict[str, Any],
                                  crop_info: Dict[str, Any],
                                  challenge_info: Dict[str, Any],
                                  literacy_info: Dict[str, Any],
                                  learning_style: LearningStyle,
                                  difficulty: DifficultyLevel,
                                  main_crop: str) -> List[Dict[str, Any]]:
        """Create personalized onboarding journey."""

        journey = [
            {
                'step': 1,
                'title': 'Welcome & Orientation',
                'description': 'Introduction to AFIE system',
                'duration_minutes': 5,
                'content_type': language_info['preferred_channels'][0],
                'learning_style': learning_style.value,
                'difficulty': difficulty.value
            },
            {
                'step': 2,
                'title': 'Farm Setup',
                'description': 'Setting up your farm profile',
                'duration_minutes': 10,
                'content_type': literacy_info['game_interface'],
                'focus': 'farm_basics',
                'personalized_for': size_info['resource_level']
            },
            {
                'step': 3,
                'title': f"{crop_info['game_category'].title()} Basics",
                'description': f"Introduction to {main_crop} farming",
                'duration_minutes': 15,
                'content_type': 'interactive_game',
                'game_category': crop_info['game_category'],
                'complexity': crop_info['complexity']
            },
            {
                'step': 4,
                'title': f"Addressing {challenge_info['game_focus'].replace('_', ' ').title()}",
                'description': 'Learn to manage your biggest challenge',
                'duration_minutes': 20,
                'content_type': 'problem_solving_game',
                'game_focus': challenge_info['game_focus'],
                'priority': challenge_info['priority']
            },
            {
                'step': 5,
                'title': 'System Navigation',
                'description': 'Learn to use the AFIE interface',
                'duration_minutes': 10,
                'content_type': literacy_info['game_interface'],
                'interface_complexity': literacy_info['interface_complexity']
            }
        ]

        return journey

    def _create_gamification_triggers(self, crop_info: Dict[str, Any],
                                     challenge_info: Dict[str, Any],
                                     literacy_info: Dict[str, Any],
                                     learning_style: LearningStyle,
                                     difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Create gamification triggers based on onboarding data."""

        # Determine initial game assignments
        initial_games = []

        # Add crop-specific game
        initial_games.append(f"{crop_info['game_category']}_introduction")

        # Add challenge-specific game
        if challenge_info['recommended_games']:
            initial_games.append(challenge_info['recommended_games'][0])

        # Add literacy-appropriate interface game
        if literacy_info['game_interface'] == 'audio_guided':
            initial_games.append('voice_navigation_tutorial')
        elif literacy_info['game_interface'] == 'icon_based':
            initial_games.append('icon_recognition_game')

        # Add learning-style specific game
        if learning_style == LearningStyle.VISUAL:
            initial_games.append('visual_patterns_game')
        elif learning_style == LearningStyle.AUDITORY:
            initial_games.append('audio_instructions_game')
        elif learning_style == LearningStyle.KINESTHETIC:
            initial_games.append('interactive_simulation')

        return {
            'initial_games': initial_games[:3],  # Limit to 3 initial games
            'difficulty_level': difficulty.value,
            'reward_frequency': 'frequent' if difficulty == DifficultyLevel.BEGINNER else 'moderate',
            'progress_tracking': {
                'milestones': ['onboarding_complete', 'first_game_complete', 'first_recommendation'],
                'reward_points': [100, 200, 500]
            },
            'personalization': {
                'interface_style': literacy_info['game_interface'],
                'content_delivery': learning_style.value,
                'challenge_level': difficulty.value
            }
        }

    def get_minimal_data_for_farm_profiler(self, onboarding_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract minimal data needed for farm profiler from onboarding data.

        Args:
            onboarding_data: Full onboarding data from collect_onboarding_data()

        Returns:
            Minimal data dictionary for farm profiler
        """
        responses = onboarding_data['collected_responses']

        return {
            'primary_language': responses['primary_language'],
            'farm_size_category': responses['farm_size_category'],
            'main_crop_animal': responses['main_crop_animal'],
            'biggest_challenge': responses['biggest_challenge'],
            'mobile_literacy': responses['mobile_literacy'],
            'farmer_id': onboarding_data.get('onboarding_id', str(uuid.uuid4())),
            'farm_name': f"{responses.get('farmer_name', 'Farmer')}'s Farm"
        }


# Singleton instance for easy access
onboarding_collector = OnboardingCollector()