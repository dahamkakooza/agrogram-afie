import re

# Read the file
with open('src/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The issue is the structure. The yield should be in its own try block.
# Let me restructure it properly
new_content = '''@asynccontextmanager
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
    logger.info("STARTING AFIE SERVICE")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info("=" * 60)
    
    # Database initialization
    try:
        logger.info("Initializing database connection...")
        async with engine.begin() as conn:
            # Create tables if they don't exist
            await conn.run_sync(Base.metadata.create_all)
            logger.info("✓ Database tables verified/created")
            
            # Test connection
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
    
    # Redis initialization (optional) - always try even if DB failed
    try:
        logger.info("Initializing Redis connection...")
        redis_client = await get_redis_client()
        await redis_client.ping()
        redis_available = True
        logger.info("✓ Redis connection established")
    except Exception as e:
        # Redis is optional for now - log warning but continue
        warning_msg = f"Redis connection failed (optional): {str(e)}"
        logger.warning(warning_msg)
        startup_errors.append(f"Warning: {warning_msg}")
        if settings.debug:
            logger.warning(traceback.format_exc())

    # Feature flags check
    logger.info("Checking feature flags...")
    active_features = []
    if settings.feature_farm_profiler:
        active_features.append("farm_profiler")
    if settings.feature_recommendations:
        active_features.append("recommendations")
    if settings.feature_learning_system:
        active_features.append("learning_system")
    if settings.feature_predictions:
        active_features.append("predictions")
    
    if active_features:
        logger.info(f"✓ Active features: {', '.join(active_features)}")
    else:
        logger.warning("No features enabled - check feature flags")

    # ============ STARTUP COMPLETE ============
    if startup_success:
        logger.info("=" * 60)
        logger.info("✅ AFIE SERVICE STARTED SUCCESSFULLY")
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
    
    try:
        yield
    finally:
        # ============ SHUTDOWN ============
        logger.info("Shutting down AFIE service...")
        
        # Close Redis connections if available
        if redis_client:
            await close_redis_connections()
            logger.info("✓ Redis connections closed")
        
        # Close database connections
        await engine.dispose()
        logger.info("✓ Database connections closed")
        
        logger.info("✅ AFIE service shutdown complete")'''

# Replace the entire lifespan function
pattern = r'@asynccontextmanager\s+async def lifespan\(app: FastAPI\):.*?(?=\n# Create FastAPI app)'
new_content_full = re.sub(pattern, new_content, content, flags=re.DOTALL)

# Write back
with open('src/main.py', 'w', encoding='utf-8') as f:
    f.write(new_content_full)

print("Fixed lifespan function structure")
