"""
MAIN APPLICATION ENTRY POINT - UPDATED WITH DISEASE DIAGNOSIS
Adaptive Farm Intelligence Engine (AFIE) - Production Ready
"""
import asyncio
import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

# ─── Configuration ───────────────────────────────────────────────
from src.config.settings import settings
from src.config.logging_config import get_logger
from src.config.redis_config import get_redis_client, close_redis_connections

# ─── Database ────────────────────────────────────────────────────
from src.data.database.base import engine, get_db, create_tables
from src.data.database.models import Base


# ─── API routers ─────────────────────────────────────────────────
from src.api.routes import api_router
from src.api.routes.recommendations import router as recommendations_router
from src.api.routes.farm_intelligence import router as farm_intelligence_router
from src.api.routes.learning import router as learning_router            # WEEK 5-6
from src.api.routes.advanced import router as advanced_router            # WEEK 7-8
from src.api.routes.diagnosis import router as diagnosis_router          # NEW: Disease diagnosis

# Initialize logger
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Production-ready with proper error handling and resource management.
    """
    startup_success = True
    startup_errors = []

    # Redis variables (optional)
    redis_client = None
    redis_available = False

    # ============ STARTUP ============
    logger.info("=" * 60)
    logger.info("STARTING AFIE SERVICE - WEEK 7-8 + DISEASE DIAGNOSIS")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info("=" * 60)

    # ─── Database initialization ─────────────────────────────────
    try:
        logger.info("Initializing database connection...")
        async with engine.begin() as conn:
            # Create tables if they don't exist (includes diagnoses,
            # field_validation_images from models_diagnosis)
            await conn.run_sync(Base.metadata.create_all)
            logger.info("✓ Database tables verified/created")

            from sqlalchemy import text
            result = await conn.execute(text("SELECT version()"))
            db_version = result.scalar()
            logger.info(f"✓ Connected to PostgreSQL: {db_version}")
    except Exception as e:
        startup_success = False
        error_msg = f"Database initialization failed: {str(e)}"
        startup_errors.append(error_msg)
        logger.error(error_msg)
        if settings.debug:
            logger.error(traceback.format_exc())

    # ─── Redis initialization (optional) ─────────────────────────
    try:
        logger.info("Initializing Redis connection...")
        redis_client = await get_redis_client()
        await redis_client.ping()
        redis_available = True
        logger.info("✓ Redis connection established")
    except Exception as e:
        warning_msg = f"Redis connection failed (optional): {str(e)}"
        logger.warning(warning_msg)
        startup_errors.append(f"Warning: {warning_msg}")
        if settings.debug:
            logger.warning(traceback.format_exc())

    # ─── Disease model initialization (NEW) ──────────────────────
    if getattr(settings, "feature_disease_diagnosis", False):
        try:
            logger.info("Initializing disease diagnosis model...")
            from src.ml.disease_model import disease_model
            if disease_model.load():
                meta = disease_model.get_metadata()
                logger.info(
                    f"✓ Disease model ready: v{meta.get('version', 'unknown')} "
                    f"({meta.get('size_kb', '?')} KB, "
                    f"{meta.get('num_classes', '?')} classes)"
                )
            else:
                warning_msg = "Disease model failed to load - diagnosis module disabled"
                logger.warning(warning_msg)
                startup_errors.append(f"Warning: {warning_msg}")
        except Exception as e:
            warning_msg = f"Disease model initialization failed: {e}"
            logger.warning(warning_msg)
            startup_errors.append(f"Warning: {warning_msg}")
            if settings.debug:
                logger.warning(traceback.format_exc())
    else:
        logger.info("Disease diagnosis feature disabled via settings")

    # ─── Feature flags check ─────────────────────────────────────
    logger.info("Checking feature flags...")
    active_features = []
    if settings.feature_farm_profiler:
        active_features.append("farm_profiler")
    if settings.feature_recommendations:
        active_features.append("recommendations")
    if settings.feature_learning_system:
        active_features.append("learning_system")
    if settings.feature_predictions:
        active_features.append("yield_predictor")
    if settings.feature_optimization:
        active_features.append("resource_orchestrator")
    if settings.feature_risk_assessment:
        active_features.append("risk_intelligence")
    if settings.feature_model_management:
        active_features.append("model_manager")
    if settings.feature_etl:
        active_features.append("etl_pipeline")

    # Disease diagnosis feature flag
    if getattr(settings, "feature_disease_diagnosis", False):
        try:
            from src.ml.disease_model import disease_model
            if disease_model.is_loaded:
                active_features.append("disease_diagnosis")
        except Exception:
            pass

    if active_features:
        logger.info(f"✓ Active features: {', '.join(active_features)}")
    else:
        logger.warning("No features enabled - check feature flags")

    # ============ STARTUP COMPLETE ============
    if startup_success:
        logger.info("=" * 60)
        logger.info("✅ AFIE SERVICE STARTED SUCCESSFULLY")

        week7_8_features = [f for f in active_features if f in [
            "yield_predictor", "resource_orchestrator",
            "risk_intelligence", "model_manager", "etl_pipeline"
        ]]
        logger.info(f"✅ WEEK 7-8 MODULES INTEGRATED: {len(week7_8_features)}/5 active")

        if "disease_diagnosis" in active_features:
            logger.info("✅ DISEASE DIAGNOSIS: active (AgriYouthNet-Context+)")
        else:
            logger.info("⚠️  DISEASE DIAGNOSIS: inactive")

        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("❌ AFIE SERVICE STARTUP FAILED")
        for error in startup_errors:
            logger.error(f"  • {error}")
        logger.error("=" * 60)

    # Store state in app
    app.state.redis_available = redis_available
    app.state.redis_client = redis_client
    app.state.startup_errors = startup_errors
    app.state.active_features = active_features

    try:
        yield
    finally:
        # ============ SHUTDOWN ============
        logger.info("Shutting down AFIE service...")

        # Release disease model
        try:
            from src.ml.disease_model import disease_model
            if disease_model.is_loaded:
                disease_model.release()
                logger.info("✓ Disease model released")
        except Exception as e:
            logger.warning(f"Error releasing disease model: {e}")

        if redis_client:
            await close_redis_connections()
            logger.info("✓ Redis connections closed")

        await engine.dispose()
        logger.info("✓ Database connections closed")

        logger.info("✅ AFIE service shutdown complete")


# ─── FastAPI app ─────────────────────────────────────────────────
app = FastAPI(
    title="Adaptive Farm Intelligence Engine (AFIE) - Week 7-8 + Disease Diagnosis",
    description=(
        "AI system that learns each farm's unique personality and provides "
        "hyper-personalized recommendations with advanced intelligence, "
        "including ultra-lightweight plant disease diagnosis (AgriYouthNet-Context+)."
    ),
    version="2.1.0",
    openapi_url=f"{settings.api_prefix}/openapi.json" if settings.api_prefix else "/openapi.json",
    docs_url=f"{settings.api_prefix}/docs" if settings.api_prefix else "/docs",
    redoc_url=f"{settings.api_prefix}/redoc" if settings.api_prefix else "/redoc",
    lifespan=lifespan,
)

# ─── Middleware ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# ─── Routers ─────────────────────────────────────────────────────
app.include_router(api_router, prefix=settings.api_prefix)
app.include_router(farm_intelligence_router, prefix=settings.api_prefix)
app.include_router(recommendations_router, prefix=settings.api_prefix)
app.include_router(learning_router, prefix=settings.api_prefix)       # WEEK 5-6
app.include_router(advanced_router, prefix=settings.api_prefix)       # WEEK 7-8
app.include_router(diagnosis_router, prefix=settings.api_prefix)      # NEW


# ─── Root ────────────────────────────────────────────────────────
@app.get("/", tags=["info"], summary="Service Information")
async def root(request: Request):
    """Get service information and status."""
    return {
        "service": "Adaptive Farm Intelligence Engine (AFIE)",
        "version": "2.1.0",
        "status": "operational",
        "environment": settings.environment,
        "api_version": settings.api_version,
        "phase": "Week 7-8 + Disease Diagnosis",
        "active_features": app.state.active_features if hasattr(app.state, 'active_features') else [],
        "docs_url": f"{settings.api_prefix}/docs" if settings.api_prefix else "/docs",
        "openapi_url": f"{settings.api_prefix}/openapi.json" if settings.api_prefix else "/openapi.json",
    }


# ─── Health check ────────────────────────────────────────────────
@app.get("/health", tags=["health"], summary="Health Check")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "timestamp": time.time(), "version": "2.1.0"}


# ─── System status ───────────────────────────────────────────────
@app.get(f"{settings.api_prefix}/status", tags=["system"], summary="System Status")
async def system_status():
    """Get detailed system status."""
    disease_status = "inactive"
    if getattr(settings, "feature_disease_diagnosis", False):
        try:
            from src.ml.disease_model import disease_model
            disease_status = "active" if disease_model.is_loaded else "model_not_loaded"
        except Exception:
            disease_status = "error"

    return {
        "database": "connected",
        "redis": "unavailable" if not app.state.redis_available else "connected",
        "farm_profiler": "active" if settings.feature_farm_profiler else "inactive",
        "recommendations": "active" if settings.feature_recommendations else "inactive",
        "learning_system": "active" if settings.feature_learning_system else "inactive",
        "yield_predictor": "active" if settings.feature_predictions else "inactive",
        "resource_orchestrator": "active" if settings.feature_optimization else "inactive",
        "risk_intelligence": "active" if settings.feature_risk_assessment else "inactive",
        "model_manager": "active" if settings.feature_model_management else "inactive",
        "etl_pipeline": "active" if settings.feature_etl else "inactive",
        "disease_diagnosis": disease_status,
        "phase": "Week 7-8 + Disease Diagnosis",
        "timestamp": time.time(),
    }


# ─── Exception handlers ──────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed information."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body,
            "message": "Validation error. Please check your request data.",
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with logging."""
    logger.error(f"Unhandled exception: {str(exc)}")
    if settings.debug:
        logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc) if settings.debug else "Internal server error",
            "message": "An unexpected error occurred.",
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers if settings.is_production else 1,
        log_level=settings.log_level.lower(),
    )