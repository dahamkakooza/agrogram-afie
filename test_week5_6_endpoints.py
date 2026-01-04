# test_week5_6_endpoints_final.py
"""
Final test script for Week 5-6 Learning Endpoints
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_learning_endpoints():
    """Test all Week 5-6 learning endpoints."""
    
    print("🧪 Testing Week 5-6 Learning Endpoints (FINAL)...\n")
    print("=" * 60)
    
    test_farmer_id = f"test-farmer-{int(time.time())}"
    
    # Test 1: Learning System Health
    print("1. Testing /learning/health")
    try:
        response = requests.get(f"{BASE_URL}/learning/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ System Status: {data.get('status')}")
            print(f"   ✓ Active Farmers: {data.get('components', {}).get('active_farmers')}")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
    print()
    
    # Test 2: Learning System Status
    print("2. Testing /learning/status")
    try:
        response = requests.get(f"{BASE_URL}/learning/status")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ System: {data.get('system')}")
            print(f"   ✓ Week: {data.get('week_implementation')}")
            print(f"   ✓ Endpoints: {len(data.get('endpoints_available', []))} available")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
    print()
    
    # Test 3: Create Behavioral Profile
    print("3. Testing /learning/behavioral-profile/{farmer_id}")
    try:
        response = requests.get(f"{BASE_URL}/learning/behavioral-profile/{test_farmer_id}")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Farmer ID: {data.get('farmer_id')}")
            print(f"   ✓ Risk Tolerance: {data.get('risk_tolerance')}")
            print(f"   ✓ Learning Style: {data.get('learning_style')}")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
    print()
    
    # Test 4: Process Game Session
    print("4. Testing /learning/game-session")
    try:
        game_data = {
            "session_id": f"game-session-{int(time.time())}",
            "farmer_id": test_farmer_id,
            "game_type": "risk_assessment_game",
            "events": [
                {
                    "event_id": "event-1",
                    "farmer_id": test_farmer_id,
                    "decision_point": "market_timing",
                    "choices": ["Sell now", "Wait 1 week", "Store"],
                    "selected_choice": "Wait 1 week",
                    "decision_time_seconds": 25.5,
                    "outcome": {"score": 85, "reward": 50},
                    "metadata": {"difficulty": "medium"}
                }
            ],
            "final_score": 85.0,
            "learning_objectives": ["market_timing"]
        }
        
        response = requests.post(
            f"{BASE_URL}/learning/game-session",
            json=game_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Session ID: {data.get('session_id')}")
            print(f"   ✓ Processing Status: {data.get('processing_status')}")
            print(f"   ✓ Behavioral Insights: {len(data.get('behavioral_insights', {}))} items")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
    print()
    
    # Test 5: Submit Farmer Feedback
    print("5. Testing /learning/feedback")
    try:
        feedback_data = {
            "farmer_id": test_farmer_id,
            "feedback_type": "explicit",
            "feedback": {
                "type": "recommendation",
                "action": "accepted",
                "rating": 4,
                "comments": "Helpful game for learning market timing"
            },
            "context": {
                "game_type": "risk_assessment_game",
                "session_id": "recent_session"
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/learning/feedback",
            json=feedback_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Feedback Type: {data.get('feedback_type')}")
            print(f"   ✓ Adaptation Applied: {data.get('adaptation_applied')}")
            print(f"   ✓ Analysis Results: {bool(data.get('analysis_results'))}")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
    print()
    
    # Test 6: Test Modules Integration
    print("6. Testing /learning/test-modules")
    try:
        response = requests.post(f"{BASE_URL}/learning/test-modules")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Status: {data.get('status')}")
            print(f"   ✓ Modules Tested: {len(data.get('modules_tested', {}))}")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
    print()
    
    # Test 7: System Learning Insights
    print("7. Testing /learning/system-learning-insights")
    try:
        response = requests.get(f"{BASE_URL}/learning/system-learning-insights")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ System Status: {data.get('system_status')}")
            print(f"   ✓ Total Farmers: {data.get('total_farmers_analyzed')}")
            print(f"   ✓ Total Sessions: {data.get('total_game_sessions')}")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Week 5-6 API endpoints test complete!")
    print(f"📊 Test Farmer ID: {test_farmer_id}")
    print("🔗 OpenAPI Docs: http://localhost:8000/api/v1/docs")
    print("=" * 60)

if __name__ == "__main__":
    # Check if server is running
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=2)
        if health_response.status_code == 200:
            print("✅ AFIE server is running")
            test_learning_endpoints()
        else:
            print("❌ AFIE server responded with error")
    except requests.exceptions.ConnectionError:
        print("❌ AFIE server is not running. Start it with:")
        print("   python -m uvicorn src.main:app --reload")
        print("\nThen run this test again.")