"""
Resource Orchestrator - Week 7-8 Implementation
Optimize resource allocation across farm operations
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import itertools
from collections import defaultdict


class ResourceType(Enum):
    """Types of farm resources."""
    LABOR = "labor"
    EQUIPMENT = "equipment"
    WATER = "water"
    FERTILIZER = "fertilizer"
    PESTICIDE = "pesticide"
    SEEDS = "seeds"
    FUEL = "fuel"
    CAPITAL = "capital"


class TaskType(Enum):
    """Types of farm tasks."""
    LAND_PREPARATION = "land_preparation"
    PLANTING = "planting"
    IRRIGATION = "irrigation"
    FERTILIZATION = "fertilization"
    WEEDING = "weeding"
    PEST_CONTROL = "pest_control"
    HARVESTING = "harvesting"
    POST_HARVEST = "post_harvest"
    MARKETING = "marketing"


@dataclass
class ResourceConstraint:
    """Constraint on resource availability."""
    resource_type: ResourceType
    available_amount: float
    unit: str
    cost_per_unit: float = 0.0
    refill_interval_days: Optional[int] = None
    refill_amount: Optional[float] = None


@dataclass
class TaskRequirement:
    """Resource requirements for a task."""
    task_type: TaskType
    duration_days: float
    labor_hours_per_day: float
    equipment_hours: Optional[float] = None
    material_requirements: Dict[ResourceType, float] = None
    dependencies: List[TaskType] = None
    optimal_timing: Optional[Dict[str, Any]] = None
    flexibility_days: int = 3
    
    def __post_init__(self):
        if self.material_requirements is None:
            self.material_requirements = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.optimal_timing is None:
            self.optimal_timing = {}


@dataclass
class FarmContext:
    """Context for optimization."""
    farm_size_hectares: float
    crop_type: str
    season: str
    available_labor: int  # Number of workers
    working_hours_per_day: int = 8
    work_days_per_week: int = 6
    budget_limit: float = 10000.0
    risk_tolerance: float = 0.5  # 0-1 scale
    sustainability_weight: float = 0.3  # 0-1 scale


class ResourceOptimizationProblem:
    """
    Constraint-based optimization for farm resource allocation.
    
    Features:
    1. Mixed integer programming for discrete decisions
    2. Multi-objective optimization
    3. Real-time re-optimization
    4. Scenario planning
    """
    
    def __init__(self, farm_context: FarmContext):
        self.context = farm_context
        self.resource_constraints = self._initialize_default_constraints()
        self.task_requirements = self._initialize_task_requirements()
        self.optimization_history = []
        
        print(f"✅ ResourceOptimizationProblem initialized for {farm_context.crop_type}")
    
    def _initialize_default_constraints(self) -> Dict[ResourceType, ResourceConstraint]:
        """Initialize default resource constraints."""
        constraints = {}
        
        # Labor constraint
        total_labor_hours = (
            self.context.available_labor * 
            self.context.working_hours_per_day * 
            self.context.work_days_per_week * 4  # Monthly
        )
        constraints[ResourceType.LABOR] = ResourceConstraint(
            resource_type=ResourceType.LABOR,
            available_amount=total_labor_hours,
            unit="hours/month",
            cost_per_unit=5.0  # $ per hour
        )
        
        # Equipment constraint (simplified)
        constraints[ResourceType.EQUIPMENT] = ResourceConstraint(
            resource_type=ResourceType.EQUIPMENT,
            available_amount=160,  # hours/month
            unit="hours/month",
            cost_per_unit=15.0  # $ per hour
        )
        
        # Water constraint
        constraints[ResourceType.WATER] = ResourceConstraint(
            resource_type=ResourceType.WATER,
            available_amount=1000000,  # liters/month
            unit="liters/month",
            cost_per_unit=0.001  # $ per liter
        )
        
        # Fertilizer constraint
        constraints[ResourceType.FERTILIZER] = ResourceConstraint(
            resource_type=ResourceType.FERTILIZER,
            available_amount=500,  # kg
            unit="kg",
            cost_per_unit=1.5,  # $ per kg
            refill_interval_days=30,
            refill_amount=500
        )
        
        # Budget constraint
        constraints[ResourceType.CAPITAL] = ResourceConstraint(
            resource_type=ResourceType.CAPITAL,
            available_amount=self.context.budget_limit,
            unit="USD",
            cost_per_unit=1.0
        )
        
        return constraints
    
    def _initialize_task_requirements(self) -> Dict[TaskType, TaskRequirement]:
        """Initialize task requirements based on crop type."""
        requirements = {}
        
        # Base requirements (will be adjusted by crop)
        base_requirements = {
            TaskType.LAND_PREPARATION: TaskRequirement(
                task_type=TaskType.LAND_PREPARATION,
                duration_days=5,
                labor_hours_per_day=6,
                equipment_hours=20,
                material_requirements={ResourceType.FUEL: 50},  # liters
                dependencies=[],
                optimal_timing={"days_after_season_start": 0}
            ),
            TaskType.PLANTING: TaskRequirement(
                task_type=TaskType.PLANTING,
                duration_days=3,
                labor_hours_per_day=8,
                equipment_hours=15,
                material_requirements={
                    ResourceType.SEEDS: 25  # kg per hectare (will be scaled)
                },
                dependencies=[TaskType.LAND_PREPARATION],
                optimal_timing={"days_after_land_prep": 2}
            ),
            TaskType.FERTILIZATION: TaskRequirement(
                task_type=TaskType.FERTILIZATION,
                duration_days=2,
                labor_hours_per_day=4,
                equipment_hours=10,
                material_requirements={
                    ResourceType.FERTILIZER: 100  # kg per hectare
                },
                dependencies=[TaskType.PLANTING],
                optimal_timing={"days_after_planting": 14}
            ),
            TaskType.WEEDING: TaskRequirement(
                task_type=TaskType.WEEDING,
                duration_days=4,
                labor_hours_per_day=6,
                equipment_hours=0,  # Manual weeding
                material_requirements={},
                dependencies=[TaskType.PLANTING],
                optimal_timing={"days_after_planting": 21}
            ),
            TaskType.IRRIGATION: TaskRequirement(
                task_type=TaskType.IRRIGATION,
                duration_days=90,  # Ongoing
                labor_hours_per_day=1,
                equipment_hours=2,
                material_requirements={ResourceType.WATER: 10000},  # liters/day
                dependencies=[TaskType.PLANTING],
                optimal_timing={"start_days_after_planting": 1}
            ),
            TaskType.PEST_CONTROL: TaskRequirement(
                task_type=TaskType.PEST_CONTROL,
                duration_days=2,
                labor_hours_per_day=4,
                equipment_hours=8,
                material_requirements={ResourceType.PESTICIDE: 5},  # liters
                dependencies=[TaskType.PLANTING],
                optimal_timing={"trigger": "pest_monitoring"}
            ),
            TaskType.HARVESTING: TaskRequirement(
                task_type=TaskType.HARVESTING,
                duration_days=7,
                labor_hours_per_day=10,
                equipment_hours=40,
                material_requirements={},
                dependencies=[TaskType.PLANTING],
                optimal_timing={"days_after_planting": 120}
            )
        }
        
        # Adjust for crop type
        crop_adjustments = self._get_crop_adjustments()
        
        for task_type, requirement in base_requirements.items():
            # Create a copy
            adjusted = TaskRequirement(
                task_type=requirement.task_type,
                duration_days=requirement.duration_days,
                labor_hours_per_day=requirement.labor_hours_per_day,
                equipment_hours=requirement.equipment_hours,
                material_requirements=requirement.material_requirements.copy(),
                dependencies=requirement.dependencies.copy(),
                optimal_timing=requirement.optimal_timing.copy(),
                flexibility_days=requirement.flexibility_days
            )
            
            # Apply crop-specific adjustments
            if self.context.crop_type in crop_adjustments.get(task_type, {}):
                adjustments = crop_adjustments[task_type][self.context.crop_type]
                for key, value in adjustments.items():
                    if hasattr(adjusted, key):
                        if key == "material_requirements":
                            adjusted.material_requirements.update(value)
                        else:
                            setattr(adjusted, key, value)
            
            # Scale by farm size
            self._scale_requirements(adjusted)
            
            requirements[task_type] = adjusted
        
        return requirements
    
    def _get_crop_adjustments(self) -> Dict[TaskType, Dict[str, Dict[str, Any]]]:
        """Get crop-specific adjustments for task requirements."""
        return {
            TaskType.LAND_PREPARATION: {
                "maize": {"duration_days": 5, "labor_hours_per_day": 6},
                "rice": {"duration_days": 7, "labor_hours_per_day": 8},
                "vegetables": {"duration_days": 4, "labor_hours_per_day": 5}
            },
            TaskType.PLANTING: {
                "maize": {"duration_days": 3, "labor_hours_per_day": 8},
                "rice": {"duration_days": 5, "labor_hours_per_day": 10},
                "vegetables": {"duration_days": 4, "labor_hours_per_day": 6}
            },
            TaskType.FERTILIZATION: {
                "maize": {"material_requirements": {ResourceType.FERTILIZER: 150}},
                "rice": {"material_requirements": {ResourceType.FERTILIZER: 120}},
                "vegetables": {"material_requirements": {ResourceType.FERTILIZER: 80}}
            },
            TaskType.IRRIGATION: {
                "maize": {"material_requirements": {ResourceType.WATER: 8000}},
                "rice": {"material_requirements": {ResourceType.WATER: 15000}},
                "vegetables": {"material_requirements": {ResourceType.WATER: 5000}}
            }
        }
    
    def _scale_requirements(self, requirement: TaskRequirement):
        """Scale requirements by farm size."""
        scale_factor = self.context.farm_size_hectares
        
        # Scale labor (non-linear scaling for efficiency)
        labor_scale = 0.8 + 0.2 * scale_factor  # More efficient at larger scale
        requirement.labor_hours_per_day *= labor_scale
        
        # Scale materials linearly
        for resource_type, amount in requirement.material_requirements.items():
            requirement.material_requirements[resource_type] = amount * scale_factor
        
        # Scale equipment (some efficiency gains)
        if requirement.equipment_hours:
            equipment_scale = 0.7 + 0.3 * scale_factor
            requirement.equipment_hours *= equipment_scale
    
    def optimize_schedule(self, 
                         planning_horizon_days: int = 180,
                         objectives: List[str] = None) -> Dict[str, Any]:
        """
        Optimize resource allocation and task scheduling.
        
        Args:
            planning_horizon_days: Number of days to plan for
            objectives: List of objectives to optimize
        
        Returns:
            Optimized schedule and resource allocation
        """
        print(f"⚙️ Optimizing schedule for {planning_horizon_days} days")
        
        if objectives is None:
            objectives = ["maximize_yield", "minimize_cost", "minimize_risk"]
        
        # Generate candidate schedules
        candidate_schedules = self._generate_candidate_schedules(planning_horizon_days)
        
        # Evaluate each schedule
        evaluated_schedules = []
        for schedule in candidate_schedules[:20]:  # Limit for performance
            evaluation = self._evaluate_schedule(schedule, objectives)
            evaluated_schedules.append((schedule, evaluation))
        
        # Select best schedule based on objectives
        best_schedule, best_evaluation = self._select_best_schedule(
            evaluated_schedules, objectives
        )
        
        # Generate optimization insights
        insights = self._generate_optimization_insights(best_schedule, best_evaluation)
        
        # Store in history
        self.optimization_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "horizon_days": planning_horizon_days,
            "objectives": objectives,
            "best_schedule": best_schedule,
            "evaluation": best_evaluation
        })
        
        return {
            "optimization_id": f"opt_{datetime.utcnow().timestamp()}",
            "status": "optimized",
            "planning_horizon_days": planning_horizon_days,
            "objectives": objectives,
            "optimized_schedule": best_schedule,
            "resource_allocation": self._extract_resource_allocation(best_schedule),
            "performance_metrics": best_evaluation,
            "constraint_satisfaction": self._check_constraints(best_schedule),
            "optimization_insights": insights,
            "recommendations": self._generate_recommendations(best_schedule, best_evaluation),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_candidate_schedules(self, horizon_days: int) -> List[Dict[str, Any]]:
        """Generate candidate task schedules."""
        schedules = []
        
        # Generate schedule variants with different task timings
        base_schedule = self._create_base_schedule(horizon_days)
        schedules.append(base_schedule)
        
        # Create variant: early planting
        early_schedule = self._create_variant_schedule(base_schedule, "early", 7)
        schedules.append(early_schedule)
        
        # Create variant: late planting
        late_schedule = self._create_variant_schedule(base_schedule, "late", -7)
        schedules.append(late_schedule)
        
        # Create variant: intensive weeding
        intensive_schedule = self._create_intensive_schedule(base_schedule)
        schedules.append(intensive_schedule)
        
        # Create variant: conservative resources
        conservative_schedule = self._create_conservative_schedule(base_schedule)
        schedules.append(conservative_schedule)
        
        # Add some random variations
        for i in range(5):
            random_schedule = self._create_random_variant(base_schedule)
            schedules.append(random_schedule)
        
        return schedules
    
    def _create_base_schedule(self, horizon_days: int) -> Dict[str, Any]:
        """Create base schedule based on optimal timings."""
        schedule = {
            "tasks": [],
            "start_date": datetime.utcnow().isoformat(),
            "end_date": (datetime.utcnow() + timedelta(days=horizon_days)).isoformat(),
            "resource_allocations": defaultdict(list)
        }
        
        current_date = datetime.utcnow()
        
        # Schedule tasks in dependency order
        scheduled_tasks = set()
        
        for task_type in [TaskType.LAND_PREPARATION, TaskType.PLANTING, 
                         TaskType.FERTILIZATION, TaskType.WEEDING,
                         TaskType.IRRIGATION, TaskType.PEST_CONTROL,
                         TaskType.HARVESTING]:
            
            if task_type not in self.task_requirements:
                continue
            
            requirement = self.task_requirements[task_type]
            
            # Calculate start date based on dependencies
            if not requirement.dependencies:
                # No dependencies, use optimal timing
                start_date = current_date + timedelta(
                    days=requirement.optimal_timing.get("days_after_season_start", 0)
                )
            else:
                # Start after last dependency completes
                dep_end_dates = []
                for dep in requirement.dependencies:
                    dep_task = next((t for t in schedule["tasks"] 
                                   if t["task_type"] == dep.value), None)
                    if dep_task:
                        dep_end_date = datetime.fromisoformat(dep_task["end_date"])
                        dep_end_dates.append(dep_end_date)
                
                if dep_end_dates:
                    start_date = max(dep_end_dates) + timedelta(
                        days=requirement.optimal_timing.get("days_after_last_dependency", 1)
                    )
                else:
                    start_date = current_date + timedelta(days=14)
            
            # Add flexibility based on context
            if self.context.risk_tolerance < 0.3:
                # Risk-averse: schedule earlier
                start_date -= timedelta(days=requirement.flexibility_days)
            
            # Calculate end date
            end_date = start_date + timedelta(days=requirement.duration_days)
            
            # Create task entry
            task_entry = {
                "task_id": f"{task_type.value}_{len(schedule['tasks'])}",
                "task_type": task_type.value,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "duration_days": requirement.duration_days,
                "resource_requirements": {
                    "labor_hours": requirement.labor_hours_per_day * requirement.duration_days,
                    "equipment_hours": requirement.equipment_hours or 0,
                    "materials": requirement.material_requirements
                },
                "dependencies": [dep.value for dep in requirement.dependencies],
                "criticality": self._calculate_task_criticality(task_type)
            }
            
            schedule["tasks"].append(task_entry)
            
            # Add to resource allocations
            for day_offset in range(requirement.duration_days):
                day_date = start_date + timedelta(days=day_offset)
                schedule["resource_allocations"][day_date.isoformat()[:10]].append({
                    "task": task_type.value,
                    "labor_hours": requirement.labor_hours_per_day,
                    "equipment_hours": requirement.equipment_hours or 0,
                    "materials": {
                        res_type.value: amount / requirement.duration_days
                        for res_type, amount in requirement.material_requirements.items()
                    }
                })
        
        return schedule
    
    def _create_variant_schedule(self, base_schedule: Dict[str, Any], 
                                variant_type: str, days_offset: int) -> Dict[str, Any]:
        """Create schedule variant by shifting tasks."""
        variant = base_schedule.copy()
        variant["tasks"] = []
        variant["variant_type"] = variant_type
        
        for task in base_schedule["tasks"]:
            task_copy = task.copy()
            start_date = datetime.fromisoformat(task["start_date"])
            end_date = datetime.fromisoformat(task["end_date"])
            
            # Apply offset
            if variant_type == "early":
                # Can only move earlier if not dependent on weather/season
                if task["task_type"] not in ["planting", "harvesting"]:
                    start_date += timedelta(days=days_offset)
                    end_date += timedelta(days=days_offset)
            elif variant_type == "late":
                # Can move later within limits
                max_offset = min(days_offset, 14)  # Don't move too late
                start_date += timedelta(days=max_offset)
                end_date += timedelta(days=max_offset)
            
            task_copy["start_date"] = start_date.isoformat()
            task_copy["end_date"] = end_date.isoformat()
            variant["tasks"].append(task_copy)
        
        return variant
    
    def _create_intensive_schedule(self, base_schedule: Dict[str, Any]) -> Dict[str, Any]:
        """Create intensive management schedule."""
        intensive = base_schedule.copy()
        intensive["variant_type"] = "intensive"
        intensive["tasks"] = []
        
        for task in base_schedule["tasks"]:
            task_copy = task.copy()
            
            # Increase resource allocation for key tasks
            if task["task_type"] in ["fertilization", "irrigation", "pest_control"]:
                # Increase by 20%
                task_copy["resource_requirements"]["labor_hours"] *= 1.2
                if task_copy["resource_requirements"].get("equipment_hours"):
                    task_copy["resource_requirements"]["equipment_hours"] *= 1.2
                
                # Increase materials
                if "materials" in task_copy["resource_requirements"]:
                    for material in task_copy["resource_requirements"]["materials"]:
                        task_copy["resource_requirements"]["materials"][material] *= 1.2
            
            intensive["tasks"].append(task_copy)
        
        return intensive
    
    def _create_conservative_schedule(self, base_schedule: Dict[str, Any]) -> Dict[str, Any]:
        """Create conservative resource schedule."""
        conservative = base_schedule.copy()
        conservative["variant_type"] = "conservative"
        conservative["tasks"] = []
        
        for task in base_schedule["tasks"]:
            task_copy = task.copy()
            
            # Reduce resource allocation for non-critical tasks
            if task["task_type"] not in ["planting", "harvesting"]:
                # Reduce by 15%
                task_copy["resource_requirements"]["labor_hours"] *= 0.85
                if task_copy["resource_requirements"].get("equipment_hours"):
                    task_copy["resource_requirements"]["equipment_hours"] *= 0.85
                
                # Reduce materials
                if "materials" in task_copy["resource_requirements"]:
                    for material in task_copy["resource_requirements"]["materials"]:
                        task_copy["resource_requirements"]["materials"][material] *= 0.85
            
            conservative["tasks"].append(task_copy)
        
        return conservative
    
    def _create_random_variant(self, base_schedule: Dict[str, Any]) -> Dict[str, Any]:
        """Create random schedule variant."""
        import random
        
        random_schedule = base_schedule.copy()
        random_schedule["variant_type"] = "random"
        random_schedule["tasks"] = []
        
        for task in base_schedule["tasks"]:
            task_copy = task.copy()
            
            # Random offset (-5 to +5 days)
            offset = random.randint(-5, 5)
            start_date = datetime.fromisoformat(task["start_date"])
            end_date = datetime.fromisoformat(task["end_date"])
            
            start_date += timedelta(days=offset)
            end_date += timedelta(days=offset)
            
            task_copy["start_date"] = start_date.isoformat()
            task_copy["end_date"] = end_date.isoformat()
            
            # Random resource adjustment
            if random.random() > 0.5:
                adjustment = 0.8 + random.random() * 0.4  # 0.8 to 1.2
                task_copy["resource_requirements"]["labor_hours"] *= adjustment
            
            random_schedule["tasks"].append(task_copy)
        
        return random_schedule
    
    def _calculate_task_criticality(self, task_type: TaskType) -> str:
        """Calculate criticality of a task."""
        critical_tasks = {TaskType.PLANTING, TaskType.HARVESTING}
        high_tasks = {TaskType.LAND_PREPARATION, TaskType.IRRIGATION}
        
        if task_type in critical_tasks:
            return "critical"
        elif task_type in high_tasks:
            return "high"
        else:
            return "medium"
    
    def _evaluate_schedule(self, schedule: Dict[str, Any], 
                          objectives: List[str]) -> Dict[str, float]:
        """Evaluate a schedule against objectives."""
        metrics = {}
        
        # Calculate total resource usage
        total_resources = self._calculate_total_resources(schedule)
        
        # Calculate costs
        total_cost = self._calculate_total_cost(schedule)
        metrics["total_cost"] = total_cost
        
        # Calculate yield estimate
        yield_estimate = self._estimate_yield(schedule)
        metrics["yield_estimate"] = yield_estimate
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(schedule)
        metrics["risk_score"] = risk_score
        
        # Calculate sustainability score
        sustainability = self._calculate_sustainability(schedule)
        metrics["sustainability"] = sustainability
        
        # Calculate efficiency
        efficiency = self._calculate_efficiency(schedule, total_resources)
        metrics["efficiency"] = efficiency
        
        # Calculate constraint violations
        violations = self._count_constraint_violations(schedule)
        metrics["constraint_violations"] = violations
        
        # Calculate objective scores
        objective_scores = {}
        for objective in objectives:
            if objective == "maximize_yield":
                objective_scores[objective] = yield_estimate
            elif objective == "minimize_cost":
                objective_scores[objective] = -total_cost  # Negative for minimization
            elif objective == "minimize_risk":
                objective_scores[objective] = -risk_score
            elif objective == "maximize_sustainability":
                objective_scores[objective] = sustainability
            elif objective == "maximize_efficiency":
                objective_scores[objective] = efficiency
        
        metrics["objective_scores"] = objective_scores
        
        # Calculate weighted score
        weights = {
            "maximize_yield": 0.4,
            "minimize_cost": 0.3,
            "minimize_risk": 0.2,
            "maximize_sustainability": 0.1
        }
        
        weighted_score = 0
        for objective, score in objective_scores.items():
            weight = weights.get(objective, 0.2)
            # Normalize scores to 0-1 range
            if objective == "maximize_yield":
                normalized = min(score / 10.0, 1.0)  # Cap at 10 tons/ha
            elif objective == "minimize_cost":
                normalized = max(1.0 - (score / 10000), 0)  # Cap at $10,000
            elif objective == "minimize_risk":
                normalized = max(1.0 - score, 0)
            else:
                normalized = min(score, 1.0)
            
            weighted_score += normalized * weight
        
        metrics["weighted_score"] = weighted_score
        
        return metrics
    
    def _calculate_total_resources(self, schedule: Dict[str, Any]) -> Dict[str, float]:
        """Calculate total resource usage for a schedule."""
        totals = defaultdict(float)
        
        for task in schedule["tasks"]:
            reqs = task["resource_requirements"]
            
            totals["labor_hours"] += reqs.get("labor_hours", 0)
            totals["equipment_hours"] += reqs.get("equipment_hours", 0)
            
            if "materials" in reqs:
                for material, amount in reqs["materials"].items():
                    totals[material] += amount
        
        return dict(totals)
    
    def _calculate_total_cost(self, schedule: Dict[str, Any]) -> float:
        """Calculate total cost of a schedule."""
        total_cost = 0.0
        
        for task in schedule["tasks"]:
            reqs = task["resource_requirements"]
            
            # Labor cost
            labor_hours = reqs.get("labor_hours", 0)
            labor_cost = labor_hours * self.resource_constraints[ResourceType.LABOR].cost_per_unit
            total_cost += labor_cost
            
            # Equipment cost
            equipment_hours = reqs.get("equipment_hours", 0)
            equipment_cost = equipment_hours * self.resource_constraints[ResourceType.EQUIPMENT].cost_per_unit
            total_cost += equipment_cost
            
            # Material costs
            if "materials" in reqs:
                for material_str, amount in reqs["materials"].items():
                    try:
                        resource_type = ResourceType(material_str)
                        if resource_type in self.resource_constraints:
                            unit_cost = self.resource_constraints[resource_type].cost_per_unit
                            total_cost += amount * unit_cost
                    except ValueError:
                        # Unknown material type
                        pass
        
        return round(total_cost, 2)
    
    def _estimate_yield(self, schedule: Dict[str, Any]) -> float:
        """Estimate yield for a schedule."""
        base_yield = 3.0  # tons/ha base
        
        # Adjust based on schedule characteristics
        adjustment = 1.0
        
        # Check if planting is at optimal time
        planting_task = next((t for t in schedule["tasks"] 
                            if t["task_type"] == "planting"), None)
        if planting_task:
            planting_date = datetime.fromisoformat(planting_task["start_date"])
            optimal_planting = datetime.utcnow() + timedelta(days=14)
            days_diff = abs((planting_date - optimal_planting).days)
            
            if days_diff <= 3:
                adjustment *= 1.1  # Optimal planting
            elif days_diff <= 7:
                adjustment *= 1.0  # Good planting
            else:
                adjustment *= 0.9 - (days_diff - 7) * 0.01  # Late planting penalty
        
        # Check resource allocation
        total_resources = self._calculate_total_resources(schedule)
        
        # Labor adequacy
        available_labor = self.resource_constraints[ResourceType.LABOR].available_amount
        labor_ratio = total_resources.get("labor_hours", 0) / available_labor
        
        if 0.7 <= labor_ratio <= 0.9:
            adjustment *= 1.05  # Optimal labor utilization
        elif labor_ratio > 1.0:
            adjustment *= 0.9  # Overworked
        
        # Fertilizer adequacy
        fertilizer_used = total_resources.get("fertilizer", 0)
        fertilizer_needed = 100 * self.context.farm_size_hectares  # kg
        
        if fertilizer_used >= fertilizer_needed * 0.8:
            adjustment *= 1.1  # Adequate fertilizer
        elif fertilizer_used < fertilizer_needed * 0.5:
            adjustment *= 0.8  # Insufficient fertilizer
        
        # Apply crop-specific adjustments
        crop_adjustments = {
            "maize": 1.0,
            "rice": 0.9,
            "vegetables": 1.2
        }
        crop_factor = crop_adjustments.get(self.context.crop_type, 1.0)
        
        final_yield = base_yield * adjustment * crop_factor * self.context.farm_size_hectares
        
        return round(final_yield, 2)
    
    def _calculate_risk_score(self, schedule: Dict[str, Any]) -> float:
        """Calculate risk score for a schedule."""
        risk_score = 0.0
        
        # Time-based risks
        for task in schedule["tasks"]:
            task_type = task["task_type"]
            
            # Check if task is scheduled during risky periods
            start_date = datetime.fromisoformat(task["start_date"])
            month = start_date.month
            
            # Simple season risk model
            if task_type in ["planting", "harvesting"]:
                if month in [12, 1, 2]:  # Peak rainy season in many regions
                    risk_score += 0.2
                elif month in [6, 7, 8]:  # Dry season
                    risk_score += 0.1
            
            # Dependency risk
            if len(task.get("dependencies", [])) > 2:
                risk_score += 0.05
        
        # Resource risk
        total_resources = self._calculate_total_resources(schedule)
        
        # Labor risk
        available_labor = self.resource_constraints[ResourceType.LABOR].available_amount
        labor_usage = total_resources.get("labor_hours", 0)
        
        if labor_usage > available_labor * 0.9:
            risk_score += 0.3  # High utilization risk
        elif labor_usage > available_labor:
            risk_score += 0.5  # Over-allocation risk
        
        # Budget risk
        total_cost = self._calculate_total_cost(schedule)
        if total_cost > self.context.budget_limit * 0.9:
            risk_score += 0.2
        
        # Normalize to 0-1 scale
        return min(risk_score, 1.0)
    
    def _calculate_sustainability(self, schedule: Dict[str, Any]) -> float:
        """Calculate sustainability score."""
        sustainability = 0.7  # Base score
        
        # Resource efficiency
        total_resources = self._calculate_total_resources(schedule)
        
        # Water efficiency
        water_used = total_resources.get("water", 0)
        water_per_hectare = water_used / self.context.farm_size_hectares
        
        if water_per_hectare < 5000:  # liters/ha
            sustainability += 0.1  # Efficient water use
        elif water_per_hectare > 15000:
            sustainability -= 0.1  # Excessive water use
        
        # Chemical use
        fertilizer_used = total_resources.get("fertilizer", 0)
        pesticide_used = total_resources.get("pesticide", 0)
        
        if fertilizer_used < 50 * self.context.farm_size_hectares:
            sustainability += 0.05  # Low fertilizer use
        if pesticide_used < 1 * self.context.farm_size_hectares:
            sustainability += 0.05  # Low pesticide use
        
        # Labor conditions (avoid overwork)
        max_daily_labor = 0
        for day, allocations in schedule.get("resource_allocations", {}).items():
            daily_labor = sum(a.get("labor_hours", 0) for a in allocations)
            max_daily_labor = max(max_daily_labor, daily_labor)
        
        if max_daily_labor <= self.context.working_hours_per_day * self.context.available_labor:
            sustainability += 0.05  # Reasonable work hours
        
        return min(sustainability, 1.0)
    
    def _calculate_efficiency(self, schedule: Dict[str, Any], 
                            total_resources: Dict[str, float]) -> float:
        """Calculate overall efficiency score."""
        efficiency = 0.5  # Base efficiency
        
        # Task duration efficiency
        total_duration = sum(t.get("duration_days", 0) for t in schedule["tasks"])
        
        # Ideal duration estimate
        ideal_duration = 150  # days for typical crop cycle
        
        duration_ratio = total_duration / ideal_duration
        if 0.9 <= duration_ratio <= 1.1:
            efficiency += 0.2  # Optimal duration
        elif duration_ratio > 1.3:
            efficiency -= 0.1  # Too long
        
        # Resource utilization efficiency
        available_labor = self.resource_constraints[ResourceType.LABOR].available_amount
        labor_used = total_resources.get("labor_hours", 0)
        
        if 0.7 <= labor_used / available_labor <= 0.9:
            efficiency += 0.2  # Optimal labor utilization
        elif labor_used / available_labor > 1.0:
            efficiency -= 0.3  # Over-utilization
        
        # Task overlap minimization
        max_concurrent_tasks = 0
        for allocations in schedule.get("resource_allocations", {}).values():
            concurrent_tasks = len(allocations)
            max_concurrent_tasks = max(max_concurrent_tasks, concurrent_tasks)
        
        if max_concurrent_tasks <= 3:
            efficiency += 0.1  # Good task distribution
        
        return min(efficiency, 1.0)
    
    def _count_constraint_violations(self, schedule: Dict[str, Any]) -> int:
        """Count constraint violations in schedule."""
        violations = 0
        
        # Check daily labor constraints
        for day, allocations in schedule.get("resource_allocations", {}).items():
            total_labor = sum(a.get("labor_hours", 0) for a in allocations)
            max_daily_labor = self.context.working_hours_per_day * self.context.available_labor
            
            if total_labor > max_daily_labor:
                violations += 1
        
        # Check budget constraint
        total_cost = self._calculate_total_cost(schedule)
        if total_cost > self.context.budget_limit:
            violations += 1
        
        # Check resource availability
        total_resources = self._calculate_total_resources(schedule)
        
        for resource_type, constraint in self.resource_constraints.items():
            resource_used = total_resources.get(resource_type.value, 0)
            if resource_used > constraint.available_amount:
                violations += 1
        
        return violations
    
    def _select_best_schedule(self, evaluated_schedules: List[Tuple[Dict, Dict]],
                             objectives: List[str]) -> Tuple[Dict, Dict]:
        """Select the best schedule based on objectives."""
        # Filter out schedules with constraint violations
        feasible_schedules = [
            (schedule, metrics) 
            for schedule, metrics in evaluated_schedules
            if metrics.get("constraint_violations", 0) == 0
        ]
        
        if not feasible_schedules:
            # If no feasible schedules, use the one with least violations
            evaluated_schedules.sort(key=lambda x: x[1].get("constraint_violations", 0))
            return evaluated_schedules[0]
        
        # Sort by weighted score
        feasible_schedules.sort(key=lambda x: x[1].get("weighted_score", 0), reverse=True)
        
        return feasible_schedules[0]
    
    def _extract_resource_allocation(self, schedule: Dict[str, Any]) -> Dict[str, Any]:
        """Extract resource allocation from schedule."""
        allocation = {
            "total_resources": self._calculate_total_resources(schedule),
            "daily_allocations": schedule.get("resource_allocations", {}),
            "peak_demand": self._calculate_peak_demand(schedule),
            "resource_utilization": self._calculate_resource_utilization(schedule)
        }
        
        return allocation
    
    def _calculate_peak_demand(self, schedule: Dict[str, Any]) -> Dict[str, float]:
        """Calculate peak resource demand."""
        peak = defaultdict(float)
        
        for day, allocations in schedule.get("resource_allocations", {}).items():
            daily_totals = defaultdict(float)
            
            for alloc in allocations:
                daily_totals["labor_hours"] += alloc.get("labor_hours", 0)
                daily_totals["equipment_hours"] += alloc.get("equipment_hours", 0)
                
                if "materials" in alloc:
                    for material, amount in alloc["materials"].items():
                        daily_totals[material] += amount
            
            # Update peaks
            for resource, amount in daily_totals.items():
                peak[resource] = max(peak[resource], amount)
        
        return dict(peak)
    
    def _calculate_resource_utilization(self, schedule: Dict[str, Any]) -> Dict[str, float]:
        """Calculate resource utilization rates."""
        utilization = {}
        total_resources = self._calculate_total_resources(schedule)
        peak_demand = self._calculate_peak_demand(schedule)
        
        for resource_type, constraint in self.resource_constraints.items():
            resource_key = resource_type.value
            total_used = total_resources.get(resource_key, 0)
            available = constraint.available_amount
            
            if available > 0:
                utilization[resource_key] = {
                    "total_utilization": round(total_used / available * 100, 1),
                    "peak_utilization": round(peak_demand.get(resource_key, 0) / available * 100, 1),
                    "available": available,
                    "used": total_used
                }
        
        return utilization
    
    def _check_constraints(self, schedule: Dict[str, Any]) -> Dict[str, Any]:
        """Check constraint satisfaction."""
        violations = []
        warnings = []
        
        total_cost = self._calculate_total_cost(schedule)
        if total_cost > self.context.budget_limit:
            violations.append({
                "constraint": "budget",
                "limit": self.context.budget_limit,
                "actual": total_cost,
                "severity": "high"
            })
        elif total_cost > self.context.budget_limit * 0.9:
            warnings.append({
                "constraint": "budget",
                "message": "Approaching budget limit",
                "utilization": round(total_cost / self.context.budget_limit * 100, 1)
            })
        
        # Check daily labor constraints
        for day, allocations in schedule.get("resource_allocations", {}).items():
            total_labor = sum(a.get("labor_hours", 0) for a in allocations)
            max_daily_labor = self.context.working_hours_per_day * self.context.available_labor
            
            if total_labor > max_daily_labor:
                violations.append({
                    "constraint": "daily_labor",
                    "day": day,
                    "limit": max_daily_labor,
                    "actual": total_labor,
                    "severity": "medium"
                })
            elif total_labor > max_daily_labor * 0.9:
                warnings.append({
                    "constraint": "daily_labor",
                    "day": day,
                    "message": "High labor utilization",
                    "utilization": round(total_labor / max_daily_labor * 100, 1)
                })
        
        return {
            "violations": violations,
            "warnings": warnings,
            "all_constraints_satisfied": len(violations) == 0
        }
    
    def _generate_optimization_insights(self, schedule: Dict[str, Any], 
                                       evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization insights."""
        insights = []
        improvements = []
        
        # Analyze resource utilization
        utilization = self._calculate_resource_utilization(schedule)
        
        for resource, stats in utilization.items():
            util_rate = stats.get("total_utilization", 0)
            
            if util_rate < 60:
                insights.append(f"Low utilization of {resource}: {util_rate}%")
                improvements.append(f"Consider reducing {resource} allocation")
            elif util_rate > 90:
                insights.append(f"High utilization of {resource}: {util_rate}%")
                improvements.append(f"Consider increasing {resource} availability")
        
        # Analyze task timing
        planting_task = next((t for t in schedule["tasks"] 
                            if t["task_type"] == "planting"), None)
        if planting_task:
            planting_date = datetime.fromisoformat(planting_task["start_date"])
            today = datetime.utcnow()
            days_until = (planting_date - today).days
            
            if days_until < 7:
                insights.append(f"Planting scheduled in {days_until} days - ensure preparation")
            elif days_until > 30:
                insights.append(f"Planting scheduled in {days_until} days - monitor conditions")
        
        # Analyze cost efficiency
        cost_per_ton = evaluation.get("total_cost", 0) / max(evaluation.get("yield_estimate", 1), 1)
        if cost_per_ton > 500:  # $ per ton
            insights.append(f"High cost per ton: ${cost_per_ton:.2f}")
            improvements.append("Review input costs and resource allocation")
        
        return {
            "insights": insights,
            "improvement_opportunities": improvements,
            "key_performance_indicators": {
                "cost_per_ton": round(cost_per_ton, 2),
                "yield_per_hectare": round(evaluation.get("yield_estimate", 0) / 
                                          self.context.farm_size_hectares, 2),
                "labor_productivity": round(evaluation.get("yield_estimate", 0) / 
                                           max(self._calculate_total_resources(schedule).get("labor_hours", 1), 1), 3)
            }
        }
    
    def _generate_recommendations(self, schedule: Dict[str, Any], 
                                 evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for the optimized schedule."""
        recommendations = []
        
        # Resource allocation recommendations
        utilization = self._calculate_resource_utilization(schedule)
        
        for resource, stats in utilization.items():
            util_rate = stats.get("total_utilization", 0)
            
            if util_rate < 50:
                recommendations.append({
                    "type": "resource_adjustment",
                    "resource": resource,
                    "action": "reduce",
                    "reason": f"Low utilization ({util_rate}%)",
                    "priority": "low"
                })
            elif util_rate > 95:
                recommendations.append({
                    "type": "resource_adjustment",
                    "resource": resource,
                    "action": "increase",
                    "reason": f"Near capacity ({util_rate}%)",
                    "priority": "high"
                })
        
        # Timing recommendations
        for task in schedule["tasks"]:
            if task.get("criticality") == "critical":
                start_date = datetime.fromisoformat(task["start_date"])
                days_until = (start_date - datetime.utcnow()).days
                
                if 0 < days_until <= 3:
                    recommendations.append({
                        "type": "timing_alert",
                        "task": task["task_type"],
                        "action": "prepare_immediately",
                        "reason": f"Starts in {days_until} days",
                        "priority": "critical"
                    })
        
        # Risk mitigation recommendations
        risk_score = evaluation.get("risk_score", 0)
        if risk_score > 0.6:
            recommendations.append({
                "type": "risk_mitigation",
                "action": "review_insurance",
                "reason": f"High risk score ({risk_score:.2f})",
                "priority": "high"
            })
        
        # Sustainability recommendations
        sustainability = evaluation.get("sustainability", 0)
        if sustainability < 0.5:
            recommendations.append({
                "type": "sustainability",
                "action": "reduce_chemical_inputs",
                "reason": f"Low sustainability score ({sustainability:.2f})",
                "priority": "medium"
            })
        
        return recommendations
    
    def run_scenario_analysis(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run scenario analysis for different conditions.
        
        Args:
            scenarios: List of scenario definitions
        
        Returns:
            Scenario analysis results
        """
        print(f"🔍 Running scenario analysis with {len(scenarios)} scenarios")
        
        results = []
        base_schedule = self.optimize_schedule(planning_horizon_days=180)
        
        for i, scenario in enumerate(scenarios):
            # Create scenario-specific context
            scenario_context = FarmContext(
                farm_size_hectares=scenario.get("farm_size", self.context.farm_size_hectares),
                crop_type=scenario.get("crop_type", self.context.crop_type),
                season=scenario.get("season", self.context.season),
                available_labor=scenario.get("available_labor", self.context.available_labor),
                budget_limit=scenario.get("budget", self.context.budget_limit),
                risk_tolerance=scenario.get("risk_tolerance", self.context.risk_tolerance)
            )
            
            # Create new optimization problem for scenario
            scenario_problem = ResourceOptimizationProblem(scenario_context)
            
            # Optimize for scenario
            scenario_result = scenario_problem.optimize_schedule(
                planning_horizon_days=scenario.get("horizon_days", 180),
                objectives=scenario.get("objectives", ["maximize_yield", "minimize_cost"])
            )
            
            results.append({
                "scenario_id": f"scenario_{i}",
                "scenario_name": scenario.get("name", f"Scenario {i}"),
                "assumptions": scenario,
                "optimization_result": scenario_result,
                "comparison_to_base": self._compare_to_base(scenario_result, base_schedule)
            })
        
        return {
            "scenario_analysis_id": f"scenario_analysis_{datetime.utcnow().timestamp()}",
            "base_scenario": base_schedule,
            "scenario_results": results,
            "recommended_scenario": self._select_recommended_scenario(results),
            "sensitivity_analysis": self._perform_sensitivity_analysis(results),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _compare_to_base(self, scenario_result: Dict[str, Any], 
                        base_schedule: Dict[str, Any]) -> Dict[str, Any]:
        """Compare scenario result to base schedule."""
        scenario_yield = scenario_result.get("performance_metrics", {}).get("yield_estimate", 0)
        base_yield = base_schedule.get("performance_metrics", {}).get("yield_estimate", 0)
        
        scenario_cost = scenario_result.get("performance_metrics", {}).get("total_cost", 0)
        base_cost = base_schedule.get("performance_metrics", {}).get("total_cost", 0)
        
        yield_change = ((scenario_yield - base_yield) / base_yield * 100) if base_yield > 0 else 0
        cost_change = ((scenario_cost - base_cost) / base_cost * 100) if base_cost > 0 else 0
        
        return {
            "yield_change_percent": round(yield_change, 1),
            "cost_change_percent": round(cost_change, 1),
            "net_benefit": round(scenario_yield - base_yield - (scenario_cost - base_cost) / 1000, 2),
            "recommendation": "preferable" if yield_change > 0 and cost_change < 10 else "consider"
        }
    
    def _select_recommended_scenario(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select recommended scenario based on multiple criteria."""
        if not results:
            return {}
        
        # Score each scenario
        scored_scenarios = []
        for result in results:
            comparison = result.get("comparison_to_base", {})
            metrics = result.get("optimization_result", {}).get("performance_metrics", {})
            
            # Calculate score
            score = 0
            score += comparison.get("yield_change_percent", 0) * 0.5
            score -= comparison.get("cost_change_percent", 0) * 0.3
            score -= metrics.get("risk_score", 0) * 20
            score += metrics.get("sustainability", 0) * 10
            
            scored_scenarios.append((score, result))
        
        # Select highest score
        scored_scenarios.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_result = scored_scenarios[0]
        
        return {
            "scenario_id": best_result.get("scenario_id"),
            "scenario_name": best_result.get("scenario_name"),
            "score": round(best_score, 2),
            "key_advantages": self._extract_advantages(best_result, scored_scenarios[1:]),
            "implementation_considerations": self._get_implementation_considerations(best_result)
        }
    
    def _extract_advantages(self, best_result: Dict[str, Any], 
                           other_results: List[Tuple[float, Dict]]) -> List[str]:
        """Extract key advantages of the best scenario."""
        advantages = []
        best_metrics = best_result.get("optimization_result", {}).get("performance_metrics", {})
        
        for score, other_result in other_results[:3]:  # Compare to top 3 alternatives
            other_metrics = other_result.get("optimization_result", {}).get("performance_metrics", {})
            
            if best_metrics.get("yield_estimate", 0) > other_metrics.get("yield_estimate", 0):
                advantage = f"Higher yield than {other_result.get('scenario_name')}"
                if advantage not in advantages:
                    advantages.append(advantage)
            
            if best_metrics.get("total_cost", 0) < other_metrics.get("total_cost", 0):
                advantage = f"Lower cost than {other_result.get('scenario_name')}"
                if advantage not in advantages:
                    advantages.append(advantage)
        
        return advantages[:3]  # Return top 3 advantages
    
    def _get_implementation_considerations(self, result: Dict[str, Any]) -> List[str]:
        """Get implementation considerations for a scenario."""
        considerations = []
        schedule = result.get("optimization_result", {}).get("optimized_schedule", {})
        
        # Check for resource peaks
        allocation = self._extract_resource_allocation(schedule)
        peak_demand = allocation.get("peak_demand", {})
        
        for resource, amount in peak_demand.items():
            constraint = self.resource_constraints.get(ResourceType(resource))
            if constraint and amount > constraint.available_amount * 0.8:
                considerations.append(f"High peak demand for {resource}: may require additional resources")
        
        # Check timing constraints
        for task in schedule.get("tasks", []):
            if task.get("criticality") == "critical":
                start_date = datetime.fromisoformat(task["start_date"])
                days_until = (start_date - datetime.utcnow()).days
                
                if days_until < 14:
                    considerations.append(
                        f"Critical task '{task['task_type']}' starts soon: ensure preparation"
                    )
        
        return considerations
    
    def _perform_sensitivity_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform sensitivity analysis on scenario results."""
        sensitivities = {}
        
        # Analyze sensitivity to key parameters
        parameters = ["available_labor", "budget", "risk_tolerance"]
        
        for param in parameters:
            param_values = []
            yield_values = []
            cost_values = []
            
            for result in results:
                scenario = result.get("assumptions", {})
                metrics = result.get("optimization_result", {}).get("performance_metrics", {})
                
                if param in scenario:
                    param_values.append(scenario[param])
                    yield_values.append(metrics.get("yield_estimate", 0))
                    cost_values.append(metrics.get("total_cost", 0))
            
            if len(param_values) >= 2:
                # Calculate sensitivity
                if max(param_values) - min(param_values) > 0:
                    yield_sensitivity = (max(yield_values) - min(yield_values)) / (max(param_values) - min(param_values))
                    cost_sensitivity = (max(cost_values) - min(cost_values)) / (max(param_values) - min(param_values))
                    
                    sensitivities[param] = {
                        "yield_sensitivity": round(yield_sensitivity, 3),
                        "cost_sensitivity": round(cost_sensitivity, 3),
                        "most_sensitive": abs(yield_sensitivity) > abs(cost_sensitivity)
                    }
        
        return {
            "parameter_sensitivities": sensitivities,
            "most_sensitive_parameter": max(sensitivities.items(), 
                                          key=lambda x: abs(x[1].get("yield_sensitivity", 0)))[0] 
                                          if sensitivities else None,
            "recommendations": self._generate_sensitivity_recommendations(sensitivities)
        }
    
    def _generate_sensitivity_recommendations(self, sensitivities: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on sensitivity analysis."""
        recommendations = []
        
        for param, sensitivity in sensitivities.items():
            yield_sens = sensitivity.get("yield_sensitivity", 0)
            
            if abs(yield_sens) > 0.5:  # High sensitivity
                if yield_sens > 0:
                    recommendations.append(
                        f"Increase {param.replace('_', ' ')} for significant yield improvement"
                    )
                else:
                    recommendations.append(
                        f"Monitor {param.replace('_', ' ')} closely as it strongly affects yield"
                    )
        
        return recommendations
    
    def get_reoptimization_triggers(self) -> Dict[str, Any]:
        """Get triggers for re-optimization."""
        triggers = {
            "weather_changes": [
                {"condition": "drought_forecast", "threshold": "rainfall < 50% normal"},
                {"condition": "excess_rain_forecast", "threshold": "rainfall > 150% normal"},
                {"condition": "frost_warning", "threshold": "temp < 5°C"}
            ],
            "market_changes": [
                {"condition": "price_increase", "threshold": ">20% price rise"},
                {"condition": "price_decrease", "threshold": ">20% price drop"},
                {"condition": "input_cost_change", "threshold": ">15% cost change"}
            ],
            "resource_changes": [
                {"condition": "labor_shortage", "threshold": "availability < 70%"},
                {"condition": "equipment_failure", "threshold": "any critical equipment"},
                {"condition": "budget_change", "threshold": ">10% budget change"}
            ],
            "biological_changes": [
                {"condition": "pest_outbreak", "threshold": "detected in region"},
                {"condition": "disease_outbreak", "threshold": "detected in region"},
                {"condition": "growth_stage_change", "threshold": "entered new stage"}
            ]
        }
        
        return {
            "triggers": triggers,
            "check_frequency": "daily",
            "auto_reoptimize": False,  # Manual approval recommended
            "notification_channels": ["in_app", "sms", "email"]
        }


# Create global instance
resource_orchestrator = ResourceOptimizationProblem(
    FarmContext(
        farm_size_hectares=2.5,
        crop_type="maize",
        season="long_rains",
        available_labor=2,
        budget_limit=5000.0
    )
)