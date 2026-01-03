import re

# Read the current main.py
with open('src/main.py', 'r') as f:
    content = f.read()

# Replace the Redis initialization section
old_redis_section = '''    try:
        # 2. Redis initialization
        logger.info("Initializing Redis connection...")
        redis_client = await get_redis_client()
        await redis_client.ping()
        logger.info("✓ Redis connection established")
    except Exception as e:
        startup_success = False
        error_msg = f"Redis initialization failed: {str(e)}"
        startup_errors.append(error_msg)
        logger.error(error_msg)
        if settings.debug:
            logger.error(traceback.format_exc())'''

new_redis_section = '''    # 2. Redis initialization (optional)
    redis_client = None
    redis_available = False
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
            logger.warning(traceback.format_exc())'''

# Replace the section
content = content.replace(old_redis_section, new_redis_section)

# Write back
with open('src/main.py', 'w') as f:
    f.write(content)

print("Made Redis initialization optional")
