import re

# Read the recommendations file
with open('src/api/routes/recommendations.py', 'r') as f:
    content = f.read()

# Find and fix the get_farm_dna function
# The issue is with db = next(get_db()) - this doesn't work well with async

# Replace the problematic section
problem_section = '''    try:
        from src.data.database.base import get_db
        from src.data.utils.farm_dna_db import get_farm_from_database
        import uuid
        
        # Convert string to UUID
        farm_id = uuid.UUID(farm_dna_id) if isinstance(farm_dna_id, str) else farm_dna_id
        
        # Get database session
        db = next(get_db())
        
        # Fetch from database
        farm_dna = await get_farm_from_database(farm_id, db)'''

fixed_section = '''    try:
        from src.data.database.base import get_db
        from src.data.utils.farm_dna_db import get_farm_from_database
        import uuid
        
        # Convert string to UUID
        farm_id = uuid.UUID(farm_dna_id) if isinstance(farm_dna_id, str) else farm_dna_id
        
        # Get database session - use async for
        db_gen = get_db()
        db = await db_gen.__anext__()
        
        # Fetch from database
        farm_dna = await get_farm_from_database(farm_id, db)'''

content = content.replace(problem_section, fixed_section)

# Write back
with open('src/api/routes/recommendations.py', 'w') as f:
    f.write(content)

print("✅ Fixed async database session handling")
