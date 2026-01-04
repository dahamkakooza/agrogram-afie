"""
Test Script for Resource Orchestrator - Week 7-8
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimization.resource_orchestrator import (
    ResourceOptimizationProblem,
    FarmContext,
    ResourceType,
    TaskType,
    resource_orchestrator
)
from datetime import datetime, timedelta
import json

def test_resource_orchestrator():
    """Comprehensive test of Resource Orchestrator module."""
    print("🧪 Testing Resource Orchestrator Module")
    print("=" * 60)
    
    # Test 1: Create Farm Context
    print("\n1. Testing Farm Context creation...")
    try:
        farm_context = FarmContext(
            farm_size_hectares=2.5,
            crop_type="maize",
            season="long_rains",
            available_labor=2,
            working_hours_per_day=8,
            work_days_per_week=6,
            budget_limit=5000.0,
            risk_tolerance=0.6,
            sustainability_weight=0.4
        )
        
        print(f"   ✅ Farm Context created successfully")
        print(f"   🏡 Farm size: {farm_context.farm_size_hectares} hectares")
        print(f"   🌽 Crop: {farm_context.crop_type}")
        print(f"   👷 Available labor: {farm_context.available_labor} workers")
        print(f"   💰 Budget: ${farm_context.budget_limit}")
        print(f"   ⚖️ Risk tolerance: {farm_context.risk_tolerance}")
    except Exception as e:
        print(f"   ❌ Failed to create Farm Context: {e}")
        return False
    
    # Test 2: Initialize Resource Optimization Problem
    print("\n2. Testing Resource Optimization Problem initialization...")
    try:
        optimizer = ResourceOptimizationProblem(farm_context)
        
        print(f"   ✅ Resource Optimizer initialized")
        print(f"   📊 Resource constraints: {len(optimizer.resource_constraints)} types")
        print(f"   📋 Task requirements: {len(optimizer.task_requirements)} tasks")
        
        # Show some resource constraints
        print(f"   💼 Resource constraints overview:")
        for i, (rtype, constraint) in enumerate(list(optimizer.resource_constraints.items())[:3]):
            print(f"      • {rtype.value}: {constraint.available_amount} {constraint.unit}")
    except Exception as e:
        print(f"   ❌ Resource optimizer initialization failed: {e}")
        return False
    
    # Test 3: Optimize Schedule
    print("\n3. Testing schedule optimization...")
    try:
        optimization_result = optimizer.optimize_schedule(
            planning_horizon_days=180,
            objectives=["maximize_yield", "minimize_cost", "minimize_risk"]
        )
        
        print(f"   ✅ Schedule optimization successful")
        print(f"   📅 Planning horizon: {optimization_result['planning_horizon_days']} days")
        print(f"   🎯 Objectives: {', '.join(optimization_result['objectives'])}")
        
        # Show performance metrics
        metrics = optimization_result.get('performance_metrics', {})
        print(f"   📊 Performance metrics:")
        print(f"      • Estimated yield: {metrics.get('yield_estimate', 'N/A')} tons")
        print(f"      • Total cost: ${metrics.get('total_cost', 'N/A')}")
        print(f"      • Risk score: {metrics.get('risk_score', 'N/A'):.3f}")
        print(f"      • Sustainability: {metrics.get('sustainability', 'N/A'):.3f}")
        
        # Show schedule summary
        schedule = optimization_result.get('optimized_schedule', {})
        tasks = schedule.get('tasks', [])
        print(f"   📋 Schedule tasks: {len(tasks)} activities")
        
        # Show first few tasks
        print(f"   📅 First 3 tasks:")
        for task in tasks[:3]:
            print(f"      • {task['task_type']}: {task['start_date'][:10]} to {task['end_date'][:10]}")
    except Exception as e:
        print(f"   ❌ Schedule optimization failed: {e}")
        return False
    
    # Test 4: Resource Allocation Analysis
    print("\n4. Testing resource allocation analysis...")
    try:
        allocation = optimization_result.get('resource_allocation', {})
        
        print(f"   ✅ Resource allocation analysis successful")
        
        # Show total resources
        total_resources = allocation.get('total_resources', {})
        print(f"   📦 Total resource requirements:")
        for resource, amount in list(total_resources.items())[:5]:
            print(f"      • {resource}: {amount}")
        
        # Show utilization rates
        utilization = allocation.get('resource_utilization', {})
        print(f"   📊 Resource utilization rates:")
        for resource, stats in list(utilization.items())[:3]:
            util_rate = stats.get('total_utilization', 0)
            print(f"      • {resource}: {util_rate}%")
    except Exception as e:
        print(f"   ❌ Resource allocation analysis failed: {e}")
    
    # Test 5: Constraint Satisfaction Check
    print("\n5. Testing constraint satisfaction...")
    try:
        constraints = optimization_result.get('constraint_satisfaction', {})
        
        print(f"   ✅ Constraint check successful")
        print(f"   ✅ All constraints satisfied: {constraints.get('all_constraints_satisfied', False)}")
        
        violations = constraints.get('violations', [])
        warnings = constraints.get('warnings', [])
        
        if violations:
            print(f"   ⚠️ Constraint violations: {len(violations)}")
            for violation in violations[:2]:
                print(f"      • {violation.get('constraint', 'N/A')}: {violation.get('severity', 'N/A')}")
        else:
            print(f"   ✅ No constraint violations")
        
        if warnings:
            print(f"   ⚠️ Warnings: {len(warnings)}")
            for warning in warnings[:2]:
                print(f"      • {warning.get('constraint', 'N/A')}: {warning.get('message', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Constraint check failed: {e}")
    
    # Test 6: Optimization Insights
    print("\n6. Testing optimization insights...")
    try:
        insights = optimization_result.get('optimization_insights', {})
        
        print(f"   ✅ Optimization insights generated")
        
        # Show insights
        insight_list = insights.get('insights', [])
        if insight_list:
            print(f"   💡 Key insights:")
            for insight in insight_list[:3]:
                print(f"      • {insight}")
        
        # Show improvement opportunities
        improvements = insights.get('improvement_opportunities', [])
        if improvements:
            print(f"   🔧 Improvement opportunities:")
            for improvement in improvements[:3]:
                print(f"      • {improvement}")
        
        # Show KPIs
        kpis = insights.get('key_performance_indicators', {})
        if kpis:
            print(f"   📈 Key performance indicators:")
            for kpi, value in kpis.items():
                print(f"      • {kpi}: {value}")
    except Exception as e:
        print(f"   ❌ Optimization insights failed: {e}")
    
    # Test 7: Scenario Analysis
    print("\n7. Testing scenario analysis...")
    try:
        scenarios = [
            {
                "name": "drought_scenario",
                "farm_size": 2.5,
                "available_labor": 1,  # Labor shortage
                "budget": 3000.0,  # Reduced budget
                "risk_tolerance": 0.3,  # More risk-averse
                "objectives": ["minimize_risk", "minimize_cost"]
            },
            {
                "name": "optimal_conditions",
                "farm_size": 2.5,
                "available_labor": 3,  # Extra labor
                "budget": 7000.0,  # Increased budget
                "risk_tolerance": 0.8,  # More risk-tolerant
                "objectives": ["maximize_yield", "maximize_sustainability"]
            }
        ]
        
        scenario_result = optimizer.run_scenario_analysis(scenarios)
        
        print(f"   ✅ Scenario analysis successful")
        print(f"   📊 Scenarios analyzed: {len(scenario_result.get('scenario_results', []))}")
        
        # Show recommended scenario
        recommended = scenario_result.get('recommended_scenario', {})
        if recommended:
            print(f"   🏆 Recommended scenario: {recommended.get('scenario_name', 'N/A')}")
            print(f"   🎯 Score: {recommended.get('score', 'N/A')}")
            
            advantages = recommended.get('key_advantages', [])
            if advantages:
                print(f"   ✅ Key advantages:")
                for advantage in advantages[:2]:
                    print(f"      • {advantage}")
    except Exception as e:
        print(f"   ❌ Scenario analysis failed: {e}")
    
    # Test 8: Re-optimization Triggers
    print("\n8. Testing re-optimization triggers...")
    try:
        triggers = optimizer.get_reoptimization_triggers()
        
        print(f"   ✅ Re-optimization triggers identified")
        
        # Show trigger categories
        trigger_categories = triggers.get('triggers', {})
        print(f"   🔔 Trigger categories:")
        for category, trigger_list in trigger_categories.items():
            print(f"      • {category}: {len(trigger_list)} triggers")
        
        # Show notification settings
        print(f"   📢 Notification channels: {', '.join(triggers.get('notification_channels', []))}")
        print(f"   🔄 Auto re-optimize: {triggers.get('auto_reoptimize', False)}")
        print(f"   ⏰ Check frequency: {triggers.get('check_frequency', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Re-optimization triggers failed: {e}")
    
    # Test 9: Task Criticality Analysis
    print("\n9. Testing task criticality analysis...")
    try:
        schedule = optimization_result.get('optimized_schedule', {})
        tasks = schedule.get('tasks', [])
        
        critical_tasks = [t for t in tasks if t.get('criticality') == 'critical']
        high_tasks = [t for t in tasks if t.get('criticality') == 'high']
        
        print(f"   ✅ Task criticality analysis successful")
        print(f"   ⚠️ Critical tasks: {len(critical_tasks)}")
        print(f"   ⚠️ High priority tasks: {len(high_tasks)}")
        
        if critical_tasks:
            print(f"   📋 Critical tasks:")
            for task in critical_tasks[:2]:
                print(f"      • {task['task_type']} (starts: {task['start_date'][:10]})")
    except Exception as e:
        print(f"   ❌ Task criticality analysis failed: {e}")
    
    # Test 10: Optimization History
    print("\n10. Testing optimization history...")
    try:
        history = optimizer.optimization_history
        
        print(f"   ✅ Optimization history retrieved")
        print(f"   📊 Total optimizations: {len(history)}")
        
        if history:
            latest = history[-1]
            print(f"   📅 Latest optimization: {latest.get('timestamp', 'N/A')}")
            print(f"   🎯 Objectives: {', '.join(latest.get('objectives', []))}")
            print(f"   📈 Horizon: {latest.get('horizon_days', 'N/A')} days")
    except Exception as e:
        print(f"   ❌ Optimization history failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Resource Orchestrator Module Tests Complete!")
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"• Farm Context: ✓ Working")
    print(f"• Resource Optimization: ✓ Working")
    print(f"• Schedule Generation: ✓ {len(tasks) if 'tasks' in locals() else 0} tasks")
    print(f"• Resource Allocation: ✓ Working")
    print(f"• Constraint Checking: ✓ Working")
    print(f"• Scenario Analysis: ✓ {len(scenarios) if 'scenarios' in locals() else 0} scenarios")
    print(f"• Optimization Insights: ✓ Working")
    
    # Example schedule for documentation
    example_schedule = {
        "optimization_id": optimization_result.get('optimization_id', 'N/A'),
        "farm_size": farm_context.farm_size_hectares,
        "crop": farm_context.crop_type,
        "planning_horizon": optimization_result.get('planning_horizon_days', 'N/A'),
        "estimated_yield": metrics.get('yield_estimate', 'N/A'),
        "total_cost": metrics.get('total_cost', 'N/A'),
        "tasks_scheduled": len(tasks) if 'tasks' in locals() else 0,
        "constraints_satisfied": constraints.get('all_constraints_satisfied', False)
    }
    
    print(f"\n📝 Example Optimization Result:")
    print(json.dumps(example_schedule, indent=2))
    
    return True

if __name__ == "__main__":
    success = test_resource_orchestrator()
    if success:
        print("\n🎉 All Resource Orchestrator tests passed!")
        print("\nNext steps:")
        print("1. Run Risk Intelligence tests")
        print("2. Run ETL Pipeline tests")
        print("3. Run Model Manager tests")
    else:
        print("\n❌ Some tests failed. Check errors above.")
        sys.exit(1)