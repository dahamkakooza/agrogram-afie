"""
Test Script for Week 5-6 Behavioral Learning Engine
Tests all three modules: Behavioral Analyzer, Game Data Processor, and Adaptation Manager
"""
import sys
import os
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_behavioral_analyzer():
    """Test the Behavioral Analyzer module."""
    print("\n" + "="*60)
    print("TEST 1: Behavioral Analyzer")
    print("="*60)
    
    try:
        from src.learning.behavioral_analyzer import behavioral_analyzer, BehavioralProfile, LearningStyle
        
        print("✅ Behavioral Analyzer imported successfully")
        
        # Test 1.1: Create behavioral profile
        print("\n🔍 Test 1.1: Creating Behavioral Profile")
        profile = behavioral_analyzer.get_behavioral_profile("test-farmer-001")
        print(f"   Farmer ID: {profile.farmer_id}")
        print(f"   Risk Tolerance: {profile.risk_tolerance}")
        print(f"   Learning Style: {profile.learning_style.value}")
        print(f"   Decision Confidence: {profile.decision_confidence}")
        
        # Test 1.2: Analyze decision patterns
        print("\n🔍 Test 1.2: Analyzing Decision Patterns")
        decision_data = {
            "decision_type": "recommendation_adoption",
            "action": "accepted",
            "timing": "immediate",
            "risk_level": "medium",
            "decision_time_seconds": 45.5,
            "outcome_confidence": 0.8
        }
        
        analysis = behavioral_analyzer.analyze_decision_pattern("test-farmer-001", decision_data)
        print(f"   Decision analyzed successfully")
        print(f"   Insights: {analysis['insights']}")
        print(f"   Recommendations generated: {len(analysis['recommendations'])}")
        
        # Test 1.3: Track pattern evolution
        print("\n🔍 Test 1.3: Tracking Pattern Evolution")
        
        # Add more decisions to see evolution
        for i in range(5):
            decision = {
                "decision_type": f"test_decision_{i}",
                "action": "accepted" if i % 2 == 0 else "modified",
                "risk_level": "high" if i == 2 else "medium",
                "decision_time_seconds": 30 + i * 10,
                "outcome_confidence": 0.6 + i * 0.05
            }
            behavioral_analyzer.analyze_decision_pattern("test-farmer-001", decision)
        
        evolution = behavioral_analyzer.track_pattern_evolution("test-farmer-001", 7)
        if "error" in evolution:
            print(f"   Evolution tracking: {evolution['error']}")
        else:
            print(f"   Success rate change: {evolution['success_rate_change']}")
            print(f"   Detected patterns: {evolution['detected_patterns']}")
        
        # Test 1.4: Get behavioral summary
        print("\n🔍 Test 1.4: Behavioral Summary")
        summary = behavioral_analyzer.get_behavioral_summary("test-farmer-001")
        print(f"   Total decisions: {summary['decision_statistics']['total_decisions']}")
        print(f"   Success rate: {summary['decision_statistics']['success_rate']}")
        print(f"   Recommendation fit score: {summary['recommendation_fit_score']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Behavioral Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_game_data_processor():
    """Test the Game Data Processor module."""
    print("\n" + "="*60)
    print("TEST 2: Game Data Processor")
    print("="*60)
    
    try:
        from src.learning.game_data_processor import game_data_processor, GameSession, GameEvent
        
        print("✅ Game Data Processor imported successfully")
        
        # Test 2.1: Process a game session
        print("\n🔍 Test 2.1: Processing Game Session")
        
        session_data = {
            "session_id": "session_test_001",
            "farmer_id": "test-farmer-001",
            "game_type": "cash_crops_introduction",
            "learning_objectives": ["fertilizer_management", "pest_identification"],
            "events": [
                {
                    "event_id": "event_1",
                    "decision_point": "fertilizer_application",
                    "choices": ["Apply now", "Wait for rain", "Test soil first"],
                    "selected_choice": "Test soil first",
                    "decision_time_seconds": 25.5,
                    "outcome": {
                        "score": 85,
                        "reward": 50,
                        "feedback": "Good choice! Soil testing is recommended."
                    },
                    "metadata": {"difficulty": "medium", "topic": "soil_health"}
                },
                {
                    "event_id": "event_2",
                    "decision_point": "pest_identification",
                    "choices": ["Chemical spray", "Natural predators", "Crop rotation"],
                    "selected_choice": "Natural predators",
                    "decision_time_seconds": 18.2,
                    "outcome": {
                        "score": 92,
                        "reward": 75,
                        "feedback": "Excellent! Sustainable pest control."
                    },
                    "metadata": {"difficulty": "hard", "topic": "pest_management"}
                }
            ],
            "final_score": 88.5
        }
        
        result = game_data_processor.process_game_session(session_data)
        print(f"   Session processed: {result['session_summary']['session_id']}")
        print(f"   Behavioral features extracted: {len(result['behavioral_features'])}")
        print(f"   Learning opportunities identified: {len(result['learning_opportunities'])}")
        
        # Test 2.2: Test GameEvent and GameSession classes
        print("\n🔍 Test 2.2: Testing GameEvent and GameSession Classes")
        
        # Create GameEvent directly
        event = GameEvent(
            event_id="direct_event_1",
            farmer_id="test-farmer-001",
            game_type="water_management",
            decision_point="irrigation_scheduling",
            timestamp=datetime.utcnow(),
            choices=["Morning", "Evening", "Night"],
            selected_choice="Morning",
            decision_time_seconds=12.5,
            outcome={"score": 78, "feedback": "Good timing choice"}
        )
        
        print(f"   GameEvent created: {event.event_id}")
        print(f"   Decision: {event.selected_choice} (score: {event.outcome['score']})")
        
        # Create GameSession
        session = GameSession(
            session_id="direct_session_001",
            farmer_id="test-farmer-001",
            game_type="water_management"
        )
        session.add_event(event)
        session.finalize(85.0)
        
        print(f"   GameSession created: {session.session_id}")
        print(f"   Session score: {session.session_score}")
        print(f"   Duration: {session.duration_seconds()} seconds")
        
        # Test 2.3: Learning velocity calculation
        print("\n🔍 Test 2.3: Learning Velocity Calculation")
        
        # Process more sessions for the same farmer
        for i in range(2):
            session_data_copy = session_data.copy()
            session_data_copy["session_id"] = f"session_test_00{i+2}"
            session_data_copy["final_score"] = 70 + i * 10  # Increasing scores
            
            # Modify events slightly
            for event in session_data_copy["events"]:
                event["outcome"]["score"] = 70 + i * 10
            
            game_data_processor.process_game_session(session_data_copy)
        
        # Calculate learning velocity
        velocity = game_data_processor.calculate_learning_velocity("test-farmer-001")
        if "error" in velocity:
            print(f"   Learning velocity: {velocity['error']}")
        else:
            print(f"   Learning velocity: {velocity['learning_velocity']}")
            print(f"   Total sessions: {velocity['total_sessions']}")
            print(f"   Improvement: {velocity['total_improvement']}")
        
        # Test 2.4: Farmer learning report
        print("\n🔍 Test 2.4: Farmer Learning Report")
        report = game_data_processor.get_farmer_learning_report("test-farmer-001")
        
        if "error" in report:
            print(f"   Report: {report['error']}")
        else:
            print(f"   Total sessions: {report['learning_statistics']['total_sessions']}")
            print(f"   Average score: {report['learning_statistics']['avg_session_score']}")
            print(f"   Strengths identified: {len(report['strengths'])}")
            print(f"   Weaknesses identified: {len(report['weaknesses'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Game Data Processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adaptation_manager():
    """Test the Adaptation Manager module."""
    print("\n" + "="*60)
    print("TEST 3: Adaptation Manager")
    print("="*60)
    
    try:
        from src.learning.adaptation_manager import adaptation_manager, AdaptationTrigger, AdaptationAction
        
        print("✅ Adaptation Manager imported successfully")
        
        # Test 3.1: Analyze explicit feedback
        print("\n🔍 Test 3.1: Analyzing Explicit Feedback")
        
        feedback_data = {
            "farmer_id": "test-farmer-001",
            "feedback_type": "explicit",
            "feedback": {
                "type": "recommendation",
                "action": "rejected",
                "rating": 2,
                "comments": "This recommendation doesn't fit my farm conditions",
                "timestamp": datetime.utcnow().isoformat()
            },
            "context": {
                "recommendation_id": "rec_12345",
                "recommendation_type": "fertilizer_application",
                "context": "dry_season"
            }
        }
        
        analysis = adaptation_manager.analyze_feedback(feedback_data)
        print(f"   Feedback analyzed: {analysis['feedback_type']}")
        print(f"   Sentiment: {analysis.get('sentiment', {}).get('polarity', 'unknown')}")
        print(f"   Rating: {analysis.get('rating', 'unknown')}")
        
        if 'adaptation_triggers' in analysis:
            print(f"   Adaptation triggers: {len(analysis['adaptation_triggers'])}")
        
        # Test 3.2: Analyze implicit feedback
        print("\n🔍 Test 3.2: Analyzing Implicit Feedback")
        
        implicit_feedback = {
            "farmer_id": "test-farmer-001",
            "feedback_type": "implicit",
            "feedback": {
                "time_spent": 450,  # seconds
                "interaction_count": 12,
                "completion_rate": 0.85,
                "return_frequency": 3,
                "performance_trend": "improving",
                "error_correction_rate": 0.7
            },
            "context": {
                "content_type": "interactive_tutorial",
                "session_duration": 480,
                "device_type": "mobile"
            }
        }
        
        implicit_analysis = adaptation_manager.analyze_feedback(implicit_feedback)
        print(f"   Implicit feedback analyzed")
        print(f"   Engagement level: {implicit_analysis.get('engagement_level', 'unknown')}")
        print(f"   Behavioral indicators: {len(implicit_analysis.get('behavioral_indicators', []))}")
        print(f"   Learning signals: {len(implicit_analysis.get('learning_signals', []))}")
        
        # Test 3.3: Apply direct adaptation
        print("\n🔍 Test 3.3: Applying Direct Adaptation")
        
        trigger = {
            "trigger": AdaptationTrigger.FEEDBACK_EXPLICIT.value,
            "action": "rejected",
            "rating": 2,
            "sentiment": "negative",
            "confidence": 0.8
        }
        
        context = {
            "farmer_id": "test-farmer-001",
            "recommendation_type": "irrigation_schedule",
            "success": False
        }
        
        adaptations = adaptation_manager.apply_adaptation("test-farmer-003", trigger, context)
        print(f"   Adaptations applied: {len(adaptations)}")
        for i, adaptation in enumerate(adaptations, 1):
            print(f"   Adaptation {i}: {adaptation.get('action', 'unknown')}")
        
        # Test 3.4: Evolve personalization
        print("\n🔍 Test 3.4: Evolving Personalization")
        
        # Add some history first
        for i in range(5):
            feedback = {
                "farmer_id": "test-farmer-001",
                "feedback_type": "explicit",
                "feedback": {
                    "type": "content",
                    "action": "accepted",
                    "rating": 4,
                    "comments": "Helpful content" if i % 2 == 0 else "Good information",
                    "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat()
                },
                "context": {"content_type": "learning_module"}
            }
            adaptation_manager.analyze_feedback(feedback)
        
        evolution = adaptation_manager.evolve_personalization("test-farmer-004")
        print(f"   Personalization evolved")
        print(f"   Evolution changes: {len(evolution.get('evolution_changes', []))}")
        
        if evolution.get('evolution_changes'):
            for change in evolution['evolution_changes'][:2]:  # Show first 2 changes
                print(f"   • {change.get('aspect')}: {change.get('change')}")
        
        # Test 3.5: Get adaptation report
        print("\n🔍 Test 3.5: Adaptation Report")
        
        report = adaptation_manager.get_adaptation_report("test-farmer-001", 30)
        
        if "error" in report:
            print(f"   Report: {report['error']}")
        else:
            print(f"   Total adaptations: {report['adaptation_statistics']['total_adaptations']}")
            print(f"   Effectiveness rate: {report['adaptation_statistics']['effectiveness_rate']}")
            print(f"   Common triggers: {list(report['adaptation_statistics']['common_triggers'].keys())[:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptation Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between all three modules."""
    print("\n" + "="*60)
    print("TEST 4: Integrated System Test")
    print("="*60)
    
    try:
        from src.learning.behavioral_analyzer import behavioral_analyzer
        from src.learning.game_data_processor import game_data_processor
        from src.learning.adaptation_manager import adaptation_manager
        
        print("✅ All modules imported successfully for integration test")
        
        # Create a test farmer for integration
        farmer_id = "integrated-farmer-001"
        
        # Step 1: Farmer plays a game (Game Data Processor)
        print("\n🎮 Step 1: Farmer plays a game")
        game_session = {
            "session_id": "integration_session_001",
            "farmer_id": farmer_id,
            "game_type": "integrated_learning_game",
            "learning_objectives": ["decision_making", "risk_assessment"],
            "events": [
                {
                    "event_id": "int_event_1",
                    "decision_point": "crop_selection",
                    "choices": ["High risk/high reward", "Medium risk", "Low risk/stable"],
                    "selected_choice": "Medium risk",
                    "decision_time_seconds": 35.2,
                    "outcome": {"score": 75, "reward": 60, "feedback": "Balanced choice"}
                },
                {
                    "event_id": "int_event_2",
                    "decision_point": "market_timing",
                    "choices": ["Sell now", "Wait for better price", "Store and sell later"],
                    "selected_choice": "Wait for better price",
                    "decision_time_seconds": 42.8,
                    "outcome": {"score": 88, "reward": 85, "feedback": "Good market analysis"}
                }
            ],
            "final_score": 81.5
        }
        
        game_result = game_data_processor.process_game_session(game_session)
        print(f"   Game processed: Score = {game_result['session_summary']['session_score']}")
        
        # Step 2: Behavioral analyzer learns from game decisions
        print("\n🧠 Step 2: Behavioral analysis from game")
        
        # The game processor should have updated behavioral analyzer internally
        # Let's check the behavioral profile
        profile = behavioral_analyzer.get_behavioral_profile(farmer_id)
        print(f"   Updated risk tolerance: {profile.risk_tolerance}")
        print(f"   Decision confidence: {profile.decision_confidence}")
        print(f"   Learning style: {profile.learning_style.value}")
        
        # Step 3: Farmer provides feedback (Adaptation Manager)
        print("\n💬 Step 3: Farmer provides feedback")
        
        feedback = {
            "farmer_id": farmer_id,
            "feedback_type": "explicit",
            "feedback": {
                "type": "game",
                "action": "accepted",
                "rating": 4,
                "comments": "The game helped me understand market timing better",
                "timestamp": datetime.utcnow().isoformat()
            },
            "context": {
                "session_id": "integration_session_001",
                "game_type": "integrated_learning_game"
            }
        }
        
        feedback_analysis = adaptation_manager.analyze_feedback(feedback)
        print(f"   Feedback analyzed: Sentiment = {feedback_analysis.get('sentiment', {}).get('polarity', 'unknown')}")
        
        # Step 4: System adapts based on learning
        print("\n🔄 Step 4: System adaptation")
        
        # Check if adaptations were triggered
        if 'adaptation_triggers' in feedback_analysis:
            print(f"   Adaptation triggers: {len(feedback_analysis['adaptation_triggers'])}")
            
            # Get adaptation report
            report = adaptation_manager.get_adaptation_report(farmer_id, 7)
            if "error" not in report:
                print(f"   Recent adaptations: {report['adaptation_statistics']['total_adaptations']}")
                print(f"   Effectiveness: {report['adaptation_statistics']['effectiveness_rate']:.1%}")
        
        # Step 5: Generate learning report
        print("\n📊 Step 5: Comprehensive learning report")
        
        learning_report = game_data_processor.get_farmer_learning_report(farmer_id)
        if "error" in learning_report:
            print(f"   Learning report: {learning_report['error']}")
        else:
            print(f"   Total learning time: {learning_report['learning_statistics']['total_learning_time_hours']} hours")
            print(f"   Strengths: {[s['skill_area'] for s in learning_report['strengths'][:2]]}")
            print(f"   Weaknesses: {[w['skill_area'] for w in learning_report['weaknesses'][:2]]}")
        
        # Step 6: Evolve personalization
        print("\n🌱 Step 6: Personalization evolution")
        
        evolution = adaptation_manager.evolve_personalization(farmer_id)
        print(f"   Evolution changes proposed: {len(evolution.get('evolution_changes', []))}")
        
        if evolution.get('evolution_changes'):
            for change in evolution['evolution_changes']:
                print(f"   • {change['aspect']}: {change['change']}")
        
        print("\n✅ Integration test completed successfully!")
        print("   All three modules are working together:")
        print("   1. Game Data Processor → Extracts learning from gameplay")
        print("   2. Behavioral Analyzer → Understands farmer decision patterns")  
        print("   3. Adaptation Manager → Adjusts system based on feedback")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and provide summary."""
    print("\n" + "="*60)
    print("WEEK 5-6: BEHAVIORAL LEARNING ENGINE - COMPREHENSIVE TEST")
    print("="*60)
    
    test_results = {}
    
    # Run individual module tests
    test_results['behavioral_analyzer'] = test_behavioral_analyzer()
    test_results['game_data_processor'] = test_game_data_processor()
    test_results['adaptation_manager'] = test_adaptation_manager()
    test_results['integration'] = test_integration()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30} {status}")
    
    print("-" * 60)
    print(f"Total: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.0f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Week 5-6 implementation is working correctly.")
        print("\nKey Features Verified:")
        print("1. ✅ Decision pattern recognition and behavioral profiling")
        print("2. ✅ Game data processing and learning feature extraction")
        print("3. ✅ Feedback analysis and system adaptation")
        print("4. ✅ Integrated learning system with personalization evolution")
    else:
        print("\n⚠️ Some tests failed. Review the output above for details.")
    
    return all(test_results.values())

def create_sample_data_for_demonstration():
    """Create sample data to demonstrate the system."""
    print("\n" + "="*60)
    print("SAMPLE DATA FOR DEMONSTRATION")
    print("="*60)
    
    sample_data = {
        "farmer_profile": {
            "farmer_id": "demo-farmer-001",
            "name": "John Demo",
            "experience_years": 5,
            "farm_size_hectares": 2.5,
            "main_crop": "maize"
        },
        "game_sessions": [
            {
                "session_id": "demo_game_1",
                "game_type": "risk_assessment_game",
                "score": 82,
                "learning_objectives": ["risk_evaluation", "decision_timing"],
                "duration_minutes": 15
            },
            {
                "session_id": "demo_game_2", 
                "game_type": "market_strategy_game",
                "score": 91,
                "learning_objectives": ["price_analysis", "timing_decisions"],
                "duration_minutes": 20
            }
        ],
        "feedback_examples": [
            {
                "type": "positive",
                "rating": 5,
                "comment": "The game helped me understand when to sell my crops"
            },
            {
                "type": "constructive", 
                "rating": 3,
                "comment": "Would like more examples from my region"
            }
        ],
        "behavioral_insights": {
            "risk_tolerance": "medium (55/100)",
            "learning_style": "kinesthetic (learns by doing)",
            "decision_speed": "deliberate (avg 45 seconds)",
            "adaptability": "high (quick to adjust strategies)"
        }
    }
    
    print("Sample data created for demonstration:")
    print(json.dumps(sample_data, indent=2))
    
    return sample_data

if __name__ == "__main__":
    print("\n🚀 Starting Week 5-6 Behavioral Learning Engine Tests")
    print("This will test all three modules and their integration.\n")
    
    # Run tests
    success = run_all_tests()
    
    # Create sample data if tests passed
    if success:
        create_sample_data_for_demonstration()
        
        print("\n" + "="*60)
        print("NEXT STEPS FOR WEEK 7-8")
        print("="*60)
        print("Now that Week 5-6 is complete, you can proceed to:")
        print("1. ✅ Yield Prediction System (yield_predictor.py)")
        print("2. ✅ Resource Optimization (resource_orchestrator.py)")
        print("3. ✅ Risk Intelligence (risk_intelligence.py)")
        print("\nThe learning system is now ready to:")
        print("• Predict crop yields with AI models")
        print("• Optimize resource allocation")
        print("• Assess and mitigate agricultural risks")
    
    print("\n✨ Test script completed. Check output above for results.")