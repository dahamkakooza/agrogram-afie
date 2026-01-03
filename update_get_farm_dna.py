import re

# Read the file
with open('src/api/routes/recommendations.py', 'r') as f:
    content = f.read()

# Update the get_farm_dna function
old_function_pattern = r'async def get_farm_dna\(farm_dna_id: str\) -> Optional\[FarmDNA\]:.*?\n(?:    .*\n)*'
new_function = '''async def get_farm_dna(farm_dna_id: str) -> Optional[FarmDNA]:
    """Get farm DNA by ID - fetches from database."""
    try:
        from src.data.database.base import get_db
        from src.data.utils.farm_dna_db import get_farm_from_database
        import uuid
        
        # Convert string to UUID
        farm_id = uuid.UUID(farm_dna_id) if isinstance(farm_dna_id, str) else farm_dna_id
        
        # Get database session
        db = next(get_db())
        
        # Fetch from database
        farm_dna = await get_farm_from_database(farm_id, db)
        
        if farm_dna:
            print(f"✅ Retrieved FarmDNA from database for farm: {farm_dna.identity.name}")
            return farm_dna
        else:
            print(f"⚠️ Farm not found in database: {farm_id}")
            # Fallback: create from minimal data
            from src.core.farm_intelligence.farm_profiler import farm_profiler
            
            minimal_data = {
                'primary_language': 'en',
                'farm_size_category': 'smallholder',
                'main_crop_animal': 'maize',
                'biggest_challenge': 'water_access',
                'mobile_literacy': 'intermediate',
                'farmer_name': 'Unknown Farmer',
                'location': 'Unknown Location',
                'farm_name': f'Farm {farm_id}'
            }
            
            farm_dna = farm_profiler.create_farm_dna_from_minimal(minimal_data)
            farm_dna.identity.farm_id = farm_id
            print(f"⚠️ Created fallback FarmDNA for: {farm_id}")
            return farm_dna
            
    except Exception as e:
        print(f"❌ Error fetching FarmDNA: {e}")
        return None'''

# Replace the function
content = re.sub(old_function_pattern, new_function, content, flags=re.DOTALL)

# Write back
with open('src/api/routes/recommendations.py', 'w') as f:
    f.write(content)

print("✅ Updated get_farm_dna to fetch from database")
