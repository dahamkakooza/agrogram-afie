import re

# Read with proper encoding
with open('src/main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the Redis section (lines 70-85)
start_line = None
for i, line in enumerate(lines):
    if '# 2. Redis initialization' in line and i > 65:
        start_line = i - 1  # Go to previous line (try:)
        break

if start_line is not None:
    # Replace lines 70-85
    new_lines = lines[:start_line]  # Keep everything before
    
    # Add new Redis section
    new_lines.extend([
        '        # 2. Redis initialization (optional)\n',
        '        redis_client = None\n',
        '        redis_available = False\n',
        '        try:\n',
        '            logger.info("Initializing Redis connection...")\n',
        '            redis_client = await get_redis_client()\n',
        '            await redis_client.ping()\n',
        '            redis_available = True\n',
        '            logger.info("✓ Redis connection established")\n',
        '        except Exception as e:\n',
        '            # Redis is optional for now - log warning but continue\n',
        '            warning_msg = f"Redis connection failed (optional): {str(e)}"\n',
        '            logger.warning(warning_msg)\n',
        '            startup_errors.append(f"Warning: {warning_msg}")\n',
        '            if settings.debug:\n',
        '                logger.warning(traceback.format_exc())\n'
    ])
    
    # Skip the old Redis section (find where it ends)
    # Look for the next section after Redis
    for i in range(start_line, len(lines)):
        if '# 3. Feature flags check' in lines[i]:
            # Add the remaining lines
            new_lines.extend(lines[i:])
            break
    
    # Write back
    with open('src/main.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("Made Redis initialization optional")
else:
    print("Could not find Redis initialization section")
