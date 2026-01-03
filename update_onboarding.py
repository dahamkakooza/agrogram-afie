import re

# Read the file
with open('src/api/routes/farm_intelligence.py', 'r') as f:
    content = f.read()

# Add import for database utilities
import_statement = '''from src.data.utils.farm_dna_db import save_farm_to_database
from src.data.database.base import get_db'''

# Add import after existing imports
if 'from src.data.database.base import get_db' not in content:
    # Find where to insert the import
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'from src.core.farm_intelligence.farm_profiler import farm_profiler' in line:
            lines.insert(i + 1, 'from src.data.database.base import get_db')
            lines.insert(i + 2, 'from src.data.utils.farm_dna_db import save_farm_to_database')
            break
    content = '\n'.join(lines)

# Update the onboarding endpoint to save to database
# Find the onboarding endpoint function
pattern = r'(@router\.post\("/onboarding".*?)\nasync def process_onboarding.*?\n(    try:.*?\n        # 3\. Create FarmDNA from minimal data using ML-based profiler\n        farm_dna = farm_profiler\.create_farm_dna_from_minimal\(minimal_data\).*?\n        \n        return \{.*?\n            "farm_dna": farm_dna\.dict\(\).*?\n        \}.*?\n    except Exception as e:.*?\n        raise HTTPException\(.*?\n        \)'

replacement = '''@router.post("/onboarding", response_model=OnboardingResponse, status_code=201)
async def process_onboarding(request: OnboardingRequest):
    """
    Process onboarding data (5 points) and create initial farm profile.
    
    Expected data:
    - primary_language: "en", "sw", "fr", "lg"
    - farm_size_category: "micro", "smallholder", "medium", "commercial", "large_scale"
    - main_crop_animal: "maize", "beans", "coffee", "banana", "cattle", "goats", "poultry"
    - biggest_challenge: "water_access", "capital_access", "market_access", "labor_shortage", "pest_disease", "soil_fertility"
    - mobile_literacy: "none", "basic", "intermediate", "advanced"
    """
    try:
        print(f"🎯 Processing onboarding for {request.farmer_name}")
        
        # Convert to dict
        onboarding_data = request.dict()
        
        # 1. Process onboarding data
        processed_data = onboarding_collector.collect_onboarding_data(onboarding_data)
        
        # 2. Extract minimal data for farm profiler
        minimal_data = onboarding_collector.get_minimal_data_for_farm_profiler(processed_data)
        
        # 3. Create FarmDNA from minimal data using ML-based profiler
        farm_dna = farm_profiler.create_farm_dna_from_minimal(minimal_data)
        
        # 4. Save FarmDNA to database
        db = next(get_db())
        await save_farm_to_database(farm_dna, db)
        print(f"✓ Farm saved to database with ID: {farm_dna.identity.farm_id}")
        
        return {
            "status": "success",
            "onboarding_id": processed_data["onboarding_id"],
            "farmer_profile": processed_data["farmer_profile"],
            "gamification_triggers": processed_data["gamification_triggers"],
            "farm_dna": farm_dna.dict(),
            "next_steps": processed_data["next_steps"]
        }
        
    except Exception as e:
        print(f"❌ Onboarding error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Onboarding processing failed: {str(e)}"
        )'''

# Replace the function
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write back
with open('src/api/routes/farm_intelligence.py', 'w') as f:
    f.write(content)

print("✅ Updated onboarding endpoint to save to database")
