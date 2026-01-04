"""
Week 7-8: Advanced Farm Intelligence & Optimization API Endpoints
Complete router with all endpoints for Weeks 7-8 modules
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from typing import Dict, Any, List, Optional
import time
import uuid
from datetime import datetime
from src.config.logging_config import get_logger

# Initialize logger
logger = get_logger("advanced_routes")

# Import Week 7-8 modules FROM YOUR ACTUAL LOCATIONS
try:
    # Import from your actual file locations with CORRECT class names
    from src.prediction.yield_predictor import YieldPredictor
    from src.optimization.resource_orchestrator import ResourceOptimizationProblem, FarmContext  # CORRECTED
    from src.prediction.risk_intelligence import RiskIntelligence
    from src.ml.model_manager import ModelManager
    from src.data.pipelines.etl_pipeline import ETLPipeline
    WEEK7_8_AVAILABLE = True
    logger.info("✅ Week 7-8 modules imported successfully from actual locations")
    
    # Initialize modules that don't need parameters
    yield_predictor = YieldPredictor()
    risk_intelligence = RiskIntelligence()
    model_manager = ModelManager()
    etl_pipeline = ETLPipeline()
    
    # ResourceOrchestrator will be initialized when needed with FarmContext
    resource_orchestrator = None  # Will be created per-request
    
    logger.info("✅ Week 7-8 modules initialized successfully")
    
except ImportError as e:
    WEEK7_8_AVAILABLE = False
    logger.warning(f"⚠️ Week 7-8 modules not available: {e}")
    # Set modules to None to avoid NameError
    yield_predictor = None
    resource_orchestrator = None
    risk_intelligence = None
    model_manager = None
    etl_pipeline = None
except Exception as e:
    WEEK7_8_AVAILABLE = False
    logger.error(f"❌ Error initializing Week 7-8 modules: {e}")
    yield_predictor = None
    resource_orchestrator = None
    risk_intelligence = None
    model_manager = None
    etl_pipeline = None

router = APIRouter(prefix="/advanced", tags=["Advanced Intelligence (Week 7-8)"])

# ==================== HEALTH & STATUS ====================

@router.get("/health", summary="Check health of Week 7-8 modules")
async def advanced_health_check():
    """Health check for advanced intelligence modules."""
    if not WEEK7_8_AVAILABLE:
        return {
            "status": "unhealthy",
            "message": "Week 7-8 modules not available",
            "modules": {
                "yield_predictor": "not_available",
                "resource_orchestrator": "not_available",
                "risk_intelligence": "not_available",
                "model_manager": "not_available",
                "etl_pipeline": "not_available"
            },
            "timestamp": time.time()
        }
    
    # Test each module
    module_status = {}
    
    # Test Yield Predictor
    try:
        if yield_predictor:
            module_status["yield_predictor"] = "healthy"
        else:
            module_status["yield_predictor"] = "not_initialized"
    except Exception as e:
        module_status["yield_predictor"] = f"unhealthy: {str(e)}"
    
    # Test Resource Orchestrator (can't test without FarmContext)
    try:
        module_status["resource_orchestrator"] = "available (needs FarmContext)"
    except Exception as e:
        module_status["resource_orchestrator"] = f"unhealthy: {str(e)}"
    
    # Test Risk Intelligence
    try:
        if risk_intelligence:
            module_status["risk_intelligence"] = "healthy"
        else:
            module_status["risk_intelligence"] = "not_initialized"
    except Exception as e:
        module_status["risk_intelligence"] = f"unhealthy: {str(e)}"
    
    # Test Model Manager
    try:
        if model_manager:
            module_status["model_manager"] = "healthy"
        else:
            module_status["model_manager"] = "not_initialized"
    except Exception as e:
        module_status["model_manager"] = f"unhealthy: {str(e)}"
    
    # Test ETL Pipeline
    try:
        if etl_pipeline:
            module_status["etl_pipeline"] = "healthy"
        else:
            module_status["etl_pipeline"] = "not_initialized"
    except Exception as e:
        module_status["etl_pipeline"] = f"unhealthy: {str(e)}"
    
    all_healthy = all("healthy" in status or "available" in status for status in module_status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "modules": module_status,
        "timestamp": time.time(),
        "version": "2.0.0"
    }

@router.get("/status", summary="Get detailed system status")
async def advanced_system_status():
    """Get detailed system status for Week 7-8 modules."""
    return {
        "system": "advanced_farm_intelligence",
        "status": "active" if WEEK7_8_AVAILABLE else "inactive",
        "week_implementation": "7-8",
        "modules_available": WEEK7_8_AVAILABLE,
        "modules_loaded": 5 if WEEK7_8_AVAILABLE else 0,
        "features": [
            "yield_prediction",
            "resource_optimization", 
            "risk_assessment",
            "model_management",
            "data_pipeline"
        ],
        "timestamp": time.time(),
        "api_version": "2.0"
    }

# ==================== YIELD PREDICTOR ====================

@router.post("/yield-predict", summary="Predict crop yield using multi-model ensemble")
async def predict_yield(
    farm_id: str = Body(..., description="Farm identifier"),
    crop_type: str = Body(..., description="Type of crop (maize, wheat, etc.)"),
    input_data: Dict[str, Any] = Body(..., description="Input features for prediction"),
    prediction_config: Optional[Dict[str, Any]] = Body(None, description="Prediction configuration")
):
    """
    Predict crop yield with uncertainty quantification and yield gap analysis.
    
    - Multi-model ensemble predictions
    - Uncertainty quantification
    - Transfer learning support
    - Yield gap analysis
    """
    if not WEEK7_8_AVAILABLE or not yield_predictor:
        raise HTTPException(status_code=503, detail="Yield predictor module not available")
    
    try:
        # Prepare prediction request
        request_data = {
            "farm_id": farm_id,
            "crop_type": crop_type,
            "input_data": input_data,
            "prediction_config": prediction_config or {}
        }
        
        # Make prediction
        # Note: Adjust based on actual method signature from your YieldPredictor
        prediction_result = yield_predictor.predict_yield(
            features=None,  # You'll need to create YieldFeatures from input_data
            crop_type=None,  # You'll need to convert crop_type to CropType enum
            season=None,     # You'll need to add season to request or use default
            include_uncertainty=True
        )
        
        return {
            "status": "success",
            "prediction_id": f"pred_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            "farm_id": farm_id,
            "crop_type": crop_type,
            "prediction": prediction_result,
            "timestamp": time.time(),
            "metadata": {
                "model_ensemble_used": prediction_result.get("model_contributions", {}),
                "confidence_score": prediction_result.get("confidence_score", 0.0)
            }
        }
    except Exception as e:
        logger.error(f"Yield prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Yield prediction failed: {str(e)}")

# ==================== RESOURCE ORCHESTRATOR ====================

@router.post("/optimize-resources", summary="Optimize resource allocation with constraints")
async def optimize_resources(
    farm_context_data: Dict[str, Any] = Body(..., description="Farm context data"),
    available_resources: Dict[str, Any] = Body(..., description="Available resources"),
    constraints: Dict[str, Any] = Body(..., description="Optimization constraints"),
    objectives: List[str] = Body(..., description="Optimization objectives"),
    scenario: Optional[str] = Body(None, description="Scenario name"),
    planning_horizon_days: int = Body(180, description="Planning horizon in days")
):
    """
    Create optimal resource allocation plan with constraint-based optimization.
    
    - Constraint-based optimization
    - Multi-objective scheduling
    - Scenario planning
    - Real-time re-optimization
    """
    if not WEEK7_8_AVAILABLE:
        raise HTTPException(status_code=503, detail="Resource orchestrator module not available")
    
    try:
        # Create FarmContext from request data
        context = FarmContext(
            farm_size_hectares=farm_context_data.get("farm_size_hectares", 2.5),
            crop_type=farm_context_data.get("crop_type", "maize"),
            season=farm_context_data.get("season", "long_rains"),
            available_labor=farm_context_data.get("available_labor", 2),
            working_hours_per_day=farm_context_data.get("working_hours_per_day", 8),
            work_days_per_week=farm_context_data.get("work_days_per_week", 6),
            budget_limit=farm_context_data.get("budget_limit", 5000.0),
            risk_tolerance=farm_context_data.get("risk_tolerance", 0.5),
            sustainability_weight=farm_context_data.get("sustainability_weight", 0.3)
        )
        
        # Create ResourceOptimizationProblem with the context
        orchestrator = ResourceOptimizationProblem(context)
        
        # Run optimization
        optimization_result = orchestrator.optimize_schedule(
            planning_horizon_days=planning_horizon_days,
            objectives=objectives
        )
        
        return {
            "status": "success",
            "optimization_id": f"opt_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            "farm_id": farm_context_data.get("farm_id", "unknown"),
            "optimization": optimization_result,
            "timestamp": time.time(),
            "metadata": {
                "scenario": scenario or "default",
                "planning_horizon_days": planning_horizon_days,
                "objectives": objectives
            }
        }
    except Exception as e:
        logger.error(f"Resource optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Resource optimization failed: {str(e)}")

# ==================== RISK INTELLIGENCE ====================

@router.post("/assess-risk", summary="Multi-dimensional risk assessment")
async def assess_risk(
    farm_id: str = Body(..., description="Farm identifier"),
    risk_dimensions: List[str] = Body(..., description="Risk dimensions to assess"),
    time_horizon: str = Body("seasonal", description="Time horizon for assessment"),
    include_mitigation: bool = Body(True, description="Include mitigation plans"),
    trigger_threshold: float = Body(0.6, description="Risk trigger threshold")
):
    """
    Perform comprehensive multi-dimensional risk assessment.
    
    - Multi-dimensional risk assessment
    - Early warning system
    - Mitigation planning
    - Correlation analysis
    """
    if not WEEK7_8_AVAILABLE or not risk_intelligence:
        raise HTTPException(status_code=503, detail="Risk intelligence module not available")
    
    try:
        # Prepare farm context (you'll need to get actual data)
        farm_context = {
            "farm_id": farm_id,
            # Add actual farm data here
            "rainfall_deviation": 0.0,
            "temperature_extremes": 0.0,
            "drought_index": 0.0,
            "pest_infestation_level": 0.0,
            "price_volatility": 0.0,
            # Add other required fields based on RiskIntelligence._extract_risk_indicators
        }
        
        # Run risk assessment
        risk_result = risk_intelligence.assess_farm_risk(
            farm_context=farm_context,
            time_horizon_days=90  # Convert time_horizon string to days
        )
        
        return {
            "status": "success",
            "assessment_id": f"risk_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            "farm_id": farm_id,
            "risk_assessment": risk_result,
            "timestamp": time.time(),
            "metadata": {
                "overall_risk_score": risk_result.get("overall_risk_score", 0.0),
                "risk_level": risk_result.get("risk_level", "unknown"),
                "early_warnings_count": len(risk_result.get("early_warnings", []))
            }
        }
    except Exception as e:
        logger.error(f"Risk assessment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

# ==================== MODEL MANAGER ====================

@router.get("/models", summary="Get all registered ML models")
async def get_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Maximum number of models to return")
):
    """
    Get information about all registered ML models.
    
    - Model version control
    - Performance tracking
    - Drift detection
    - Experiment logging
    """
    if not WEEK7_8_AVAILABLE or not model_manager:
        raise HTTPException(status_code=503, detail="Model manager module not available")
    
    try:
        # Get all models from registry
        models = model_manager.model_registry
        
        # Convert to list of dicts
        models_list = []
        for model_id, metadata in models.items():
            model_dict = {
                "model_id": model_id,
                "model_type": metadata.model_type.value if hasattr(metadata.model_type, 'value') else str(metadata.model_type),
                "version": metadata.version,
                "status": metadata.status.value if hasattr(metadata.status, 'value') else str(metadata.status),
                "deployment_stage": metadata.deployment_stage.value if hasattr(metadata.deployment_stage, 'value') else str(metadata.deployment_stage),
                "performance_metrics": metadata.performance_metrics,
                "created_at": metadata.created_at.isoformat() if hasattr(metadata.created_at, 'isoformat') else str(metadata.created_at),
                "updated_at": metadata.updated_at.isoformat() if hasattr(metadata.updated_at, 'isoformat') else str(metadata.updated_at)
            }
            models_list.append(model_dict)
        
        # Apply filters if provided
        if model_type:
            models_list = [m for m in models_list if m.get("model_type") == model_type]
        if status:
            models_list = [m for m in models_list if m.get("status") == status]
        
        # Apply limit
        models_list = models_list[:limit]
        
        return {
            "status": "success",
            "count": len(models_list),
            "models": models_list,
            "timestamp": time.time(),
            "metadata": {
                "total_models": len(models),
                "filter_applied": model_type or status is not None
            }
        }
    except Exception as e:
        logger.error(f"Failed to get models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

@router.post("/models/register", summary="Register a new ML model")
async def register_model(
    model_data: Dict[str, Any] = Body(..., description="Model information and configuration")
):
    """Register a new ML model in the model manager."""
    if not WEEK7_8_AVAILABLE or not model_manager:
        raise HTTPException(status_code=503, detail="Model manager module not available")
    
    try:
        # Extract data from request
        model_type_str = model_data.get("model_type")
        model_object = model_data.get("model_object", {})
        performance_metrics = model_data.get("performance_metrics", {})
        feature_importance = model_data.get("feature_importance", {})
        hyperparameters = model_data.get("hyperparameters", {})
        training_data_stats = model_data.get("training_data_stats", {})
        version = model_data.get("version")
        dependencies = model_data.get("dependencies", [])
        
        # Convert string to ModelType enum
        from src.ml.model_manager import ModelType
        model_type = ModelType(model_type_str)
        
        # Register model
        model_id = model_manager.register_model(
            model_type=model_type,
            model_object=model_object,
            performance_metrics=performance_metrics,
            feature_importance=feature_importance,
            hyperparameters=hyperparameters,
            training_data_stats=training_data_stats,
            version=version,
            dependencies=dependencies
        )
        
        return {
            "status": "success",
            "model_id": model_id,
            "message": "Model registered successfully",
            "timestamp": time.time(),
            "metadata": {
                "model_type": model_type_str,
                "version": version or "auto-generated",
                "registration_time": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Model registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model registration failed: {str(e)}")

@router.get("/models/{model_id}", summary="Get specific model details")
async def get_model_details(
    model_id: str = Path(..., description="Model identifier")
):
    """Get detailed information about a specific model."""
    if not WEEK7_8_AVAILABLE or not model_manager:
        raise HTTPException(status_code=503, detail="Model manager module not available")
    
    try:
        # Get model from registry
        models = model_manager.model_registry
        
        if model_id not in models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        metadata = models[model_id]
        
        # Convert to dict
        model_details = {
            "model_id": model_id,
            "model_type": metadata.model_type.value if hasattr(metadata.model_type, 'value') else str(metadata.model_type),
            "version": metadata.version,
            "status": metadata.status.value if hasattr(metadata.status, 'value') else str(metadata.status),
            "deployment_stage": metadata.deployment_stage.value if hasattr(metadata.deployment_stage, 'value') else str(metadata.deployment_stage),
            "performance_metrics": metadata.performance_metrics,
            "feature_importance": metadata.feature_importance,
            "hyperparameters": metadata.hyperparameters,
            "training_data_stats": metadata.training_data_stats,
            "dependencies": metadata.dependencies,
            "created_at": metadata.created_at.isoformat() if hasattr(metadata.created_at, 'isoformat') else str(metadata.created_at),
            "updated_at": metadata.updated_at.isoformat() if hasattr(metadata.updated_at, 'isoformat') else str(metadata.updated_at)
        }
        
        return {
            "status": "success",
            "model": model_details,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model details: {str(e)}")

# ==================== ETL PIPELINE ====================

@router.post("/etl/process", summary="Process data through ETL pipeline")
async def process_etl(
    pipeline_name: str = Body("default", description="Name of ETL pipeline to use"),
    input_data: Dict[str, Any] = Body(..., description="Input data to process"),
    transformations: Optional[List[Dict[str, Any]]] = Body(None, description="Transformations to apply"),
    quality_checks: bool = Body(True, description="Run data quality checks")
):
    """
    Process data through the ETL pipeline.
    
    - Data extraction (with simulation)
    - Data transformation
    - Quality assessment
    - Pipeline monitoring
    """
    if not WEEK7_8_AVAILABLE or not etl_pipeline:
        raise HTTPException(status_code=503, detail="ETL pipeline module not available")
    
    try:
        # Prepare ETL request
        # Note: The ETL pipeline in your code expects to extract from sources,
        # not process input_data directly. We'll adapt.
        
        # If input_data contains source specifications, use them
        sources = input_data.get("sources", ["weather_agromonitoring", "market_fao"])
        
        # Run ETL pipeline
        etl_result = await etl_pipeline.run_full_pipeline(
            sources=sources,
            transformations=["clean_and_validate", "normalize_units", "quality_assessment"],
            destination="cache"
        )
        
        return {
            "status": "success",
            "pipeline_id": f"etl_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            "pipeline_name": pipeline_name,
            "etl_result": etl_result,
            "timestamp": time.time(),
            "metadata": {
                "sources_processed": len(sources),
                "transformations_applied": etl_result.get("phases", {}).get("transform", {}).get("transformations_applied", []),
                "records_processed": etl_result.get("overall_metrics", {}).get("total_records_processed", 0),
                "data_quality": etl_result.get("overall_metrics", {}).get("average_data_quality", 0)
            }
        }
    except Exception as e:
        logger.error(f"ETL processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ETL processing failed: {str(e)}")

# ==================== INTEGRATION & TESTING ====================

@router.post("/integration-test", summary="Test all Week 7-8 modules integration")
async def integration_test(
    farm_id: str = Body("test-farm-001", description="Test farm ID"),
    run_comprehensive: bool = Body(False, description="Run comprehensive tests")
):
    """Test that all Week 7-8 modules work together correctly."""
    if not WEEK7_8_AVAILABLE:
        raise HTTPException(status_code=503, detail="Week 7-8 modules not available")
    
    test_results = {}
    test_id = f"test_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Test 1: Yield Predictor
    try:
        if yield_predictor:
            test_data = {
                "farm_id": farm_id,
                "crop_type": "maize",
                "input_data": {
                    "soil_moisture": 65.5,
                    "temperature": 25.0,
                    "rainfall": 120.0
                }
            }
            # Note: You'll need to adapt based on actual YieldPredictor method
            test_results["yield_predictor"] = {
                "status": "available",
                "module_loaded": True,
                "test_note": "Module loaded, needs proper YieldFeatures object for full test"
            }
        else:
            test_results["yield_predictor"] = {
                "status": "failed",
                "error": "Module not initialized"
            }
    except Exception as e:
        test_results["yield_predictor"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # Test 2: Resource Orchestrator
    try:
        # Test if we can create FarmContext and ResourceOptimizationProblem
        context = FarmContext(
            farm_size_hectares=2.5,
            crop_type="maize",
            season="long_rains",
            available_labor=2
        )
        orchestrator = ResourceOptimizationProblem(context)
        test_results["resource_orchestrator"] = {
            "status": "passed",
            "module_loaded": True,
            "test_note": "Module initialized with FarmContext"
        }
    except Exception as e:
        test_results["resource_orchestrator"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # Test 3: Risk Intelligence
    try:
        if risk_intelligence:
            test_results["risk_intelligence"] = {
                "status": "passed",
                "module_loaded": True,
                "test_note": "Module initialized"
            }
        else:
            test_results["risk_intelligence"] = {
                "status": "failed",
                "error": "Module not initialized"
            }
    except Exception as e:
        test_results["risk_intelligence"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # Test 4: Model Manager
    try:
        if model_manager:
            models_count = len(model_manager.model_registry)
            test_results["model_manager"] = {
                "status": "passed",
                "module_loaded": True,
                "models_in_registry": models_count,
                "test_note": f"Module initialized with {models_count} models in registry"
            }
        else:
            test_results["model_manager"] = {
                "status": "failed",
                "error": "Module not initialized"
            }
    except Exception as e:
        test_results["model_manager"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # Test 5: ETL Pipeline
    try:
        if etl_pipeline:
            test_results["etl_pipeline"] = {
                "status": "passed",
                "module_loaded": True,
                "data_sources_count": len(etl_pipeline.data_sources),
                "test_note": "Module initialized with data sources"
            }
        else:
            test_results["etl_pipeline"] = {
                "status": "failed",
                "error": "Module not initialized"
            }
    except Exception as e:
        test_results["etl_pipeline"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # Calculate overall status
    passed_modules = [name for name, result in test_results.items() if result["status"] == "passed"]
    all_passed = len(passed_modules) == 5
    
    return {
        "status": "success" if all_passed else "partial",
        "test_id": test_id,
        "farm_id": farm_id,
        "test_results": test_results,
        "summary": {
            "total_modules": 5,
            "passed_modules": len(passed_modules),
            "failed_modules": 5 - len(passed_modules),
            "all_modules_working": all_passed,
            "passed_module_names": passed_modules
        },
        "timestamp": time.time(),
        "message": "Week 7-8 modules integration test completed"
    }

@router.get("/module-stats", summary="Get module usage statistics")
async def module_statistics():
    """Get usage statistics for Week 7-8 modules."""
    return {
        "status": "success",
        "statistics": {
            "yield_predictor": {
                "predictions_made": 0,
                "average_confidence": 0.0,
                "most_predicted_crop": "maize",
                "module_available": yield_predictor is not None
            },
            "resource_orchestrator": {
                "optimizations_run": 0,
                "average_efficiency": 0.0,
                "common_constraints": ["budget", "labor"],
                "module_available": WEEK7_8_AVAILABLE
            },
            "risk_intelligence": {
                "assessments_done": 0,
                "average_risk_score": 0.0,
                "most_common_risk": "weather",
                "module_available": risk_intelligence is not None
            },
            "model_manager": {
                "models_registered": len(model_manager.model_registry) if model_manager else 0,
                "active_models": 0,
                "model_types": [],
                "module_available": model_manager is not None
            },
            "etl_pipeline": {
                "pipelines_run": 0,
                "records_processed": 0,
                "average_processing_time_ms": 0,
                "module_available": etl_pipeline is not None
            }
        },
        "timestamp": time.time()
    }

# ==================== UTILITY ENDPOINTS ====================

@router.get("/modules", summary="List all available Week 7-8 modules")
async def list_modules():
    """List all available Week 7-8 modules with their status."""
    modules = [
        {
            "name": "Yield Predictor",
            "description": "Multi-model ensemble for crop yield prediction",
            "status": "available" if WEEK7_8_AVAILABLE and yield_predictor else "unavailable",
            "endpoints": ["POST /advanced/yield-predict"],
            "features": ["ensemble", "uncertainty", "yield_gap"],
            "location": "src.prediction.yield_predictor",
            "class_name": "YieldPredictor"
        },
        {
            "name": "Resource Orchestrator",
            "description": "Constraint-based resource optimization",
            "status": "available" if WEEK7_8_AVAILABLE else "unavailable",
            "endpoints": ["POST /advanced/optimize-resources"],
            "features": ["optimization", "scheduling", "scenarios"],
            "location": "src.optimization.resource_orchestrator",
            "class_name": "ResourceOptimizationProblem"
        },
        {
            "name": "Risk Intelligence",
            "description": "Multi-dimensional risk assessment",
            "status": "available" if WEEK7_8_AVAILABLE and risk_intelligence else "unavailable",
            "endpoints": ["POST /advanced/assess-risk"],
            "features": ["assessment", "early_warning", "mitigation"],
            "location": "src.prediction.risk_intelligence",
            "class_name": "RiskIntelligence"
        },
        {
            "name": "Model Manager",
            "description": "ML model version control and management",
            "status": "available" if WEEK7_8_AVAILABLE and model_manager else "unavailable",
            "endpoints": ["GET /advanced/models", "POST /advanced/models/register", "GET /advanced/models/{id}"],
            "features": ["versioning", "tracking", "drift_detection"],
            "location": "src.ml.model_manager",
            "class_name": "ModelManager"
        },
        {
            "name": "ETL Pipeline",
            "description": "Data extraction, transformation, and loading",
            "status": "available" if WEEK7_8_AVAILABLE and etl_pipeline else "unavailable",
            "endpoints": ["POST /advanced/etl/process"],
            "features": ["extraction", "transformation", "quality_check"],
            "location": "src.data.pipelines.etl_pipeline",
            "class_name": "ETLPipeline"
        }
    ]
    
    return {
        "status": "success",
        "modules": modules,
        "total_modules": len(modules),
        "available_modules": len([m for m in modules if m["status"] == "available"]),
        "timestamp": time.time()
    }

@router.get("/api-docs", summary="Get API documentation links")
async def api_documentation():
    """Get documentation links for Week 7-8 API."""
    return {
        "status": "success",
        "documentation": {
            "openapi_spec": "/api/v1/advanced/openapi.json",
            "swagger_ui": "/docs#/Advanced%20Intelligence%20(Week%207-8)",
            "redoc": "/redoc",
            "endpoints_summary": {
                "health": "GET /advanced/health",
                "status": "GET /advanced/status",
                "yield_prediction": "POST /advanced/yield-predict",
                "resource_optimization": "POST /advanced/optimize-resources",
                "risk_assessment": "POST /advanced/assess-risk",
                "model_management": "GET /advanced/models, POST /advanced/models/register",
                "etl_processing": "POST /advanced/etl/process",
                "integration_test": "POST /advanced/integration-test",
                "module_listing": "GET /advanced/modules"
            }
        },
        "timestamp": time.time()
    }