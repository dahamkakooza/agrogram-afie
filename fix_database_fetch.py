import re

# Read the database utility file
with open('src/data/utils/farm_dna_db.py', 'r') as f:
    content = f.read()

# Find the get_farm_from_database function and fix it
# The issue is that await session.get() might not work as expected

# Replace the function with a corrected version
old_function = '''async def get_farm_from_database(farm_id: uuid.UUID, db_session) -> Optional[FarmDNA]:
    """
    Get FarmDNA from database.
    
    Args:
        farm_id: Farm ID to retrieve
        db_session: SQLAlchemy async session
        
    Returns:
        FarmDNA object or None if not found
    """
    farm_db = await db_session.get(FarmDB, farm_id)
    
    if not farm_db:
        return None
    
    return farm_db_to_farm_dna(farm_db)'''

new_function = '''async def get_farm_from_database(farm_id: uuid.UUID, db_session) -> Optional[FarmDNA]:
    """
    Get FarmDNA from database.
    
    Args:
        farm_id: Farm ID to retrieve
        db_session: SQLAlchemy async session
        
    Returns:
        FarmDNA object or None if not found
    """
    try:
        # Use query.get() instead of await session.get() for SQLAlchemy 2.0
        farm_db = db_session.get(FarmDB, farm_id)
        
        if not farm_db:
            print(f"Farm not found in database: {farm_id}")
            return None
        
        print(f"Found farm in database: {farm_db.name}")
        return farm_db_to_farm_dna(farm_db)
    except Exception as e:
        print(f"Error fetching farm from database: {e}")
        return None'''

# Replace in content
content = content.replace(old_function, new_function)

# Write back
with open('src/data/utils/farm_dna_db.py', 'w') as f:
    f.write(content)

print("✅ Fixed database fetch function")
