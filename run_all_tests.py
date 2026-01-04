"""
Test Runner for Week 7-8 Modules
Run all tests in sequence
"""
import subprocess
import sys
import os

def run_test(script_name):
    """Run a test script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} PASSED")
            return True
        else:
            print(f"❌ {script_name} FAILED")
            print(f"Error output:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Run all tests."""
    test_scripts = [
        "test_yield_predictor.py",
        "test_resource_orchestrator.py",
        "test_risk_intelligence.py",
        "test_etl_pipeline.py",
        "test_model_manager.py"
    ]
    
    print("🚀 Starting Week 7-8 Comprehensive Tests")
    print("="*60)
    
    results = {}
    for script in test_scripts:
        if os.path.exists(script):
            results[script] = run_test(script)
        else:
            print(f"❌ {script} not found!")
            results[script] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for script, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {script}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Week 7-8 implementation is ready.")
        print("\n📋 Next steps:")
        print("1. Create API endpoints for each module")
        print("2. Build integration workflows")
        print("3. Add database persistence")
        print("4. Create user dashboards")
        print("5. Deploy to staging environment")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please fix the issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())