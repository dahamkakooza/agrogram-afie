with open('src/api/routes/recommendations.py', 'r') as f:
    lines = f.readlines()

# Find and replace the get_farm_dna function
with open('src/api/routes/recommendations.py', 'w') as f:
    in_function = False
    replaced = False
    
    for line in lines:
        if 'async def get_farm_dna' in line and not replaced:
            in_function = True
            # Write the new function that uses database
            f.write('''async def get_farm_dna(farm_dna_id: str) -> Optional[FarmDNA]:
    """Get farm DNA by ID - fetches from database."""
    from src.data.database.base import get_db
    from src.data.database.models import FarmDB
    from src.models.farm import FarmDNA, FarmIdentity, FarmTypology, ResourceConstraints, EnvironmentalContext
    import uuid
    
    try:
        # Convert string to UUID if needed
        if isinstance(farm_dna_id, str):
            farm_id = uuid.UUID(farm_dna_id)
        else:
            farm_id = farm_dna_id
        
        # Get database session
        db = next(get_db())
        
        # Fetch farm from database
        farm_db = db.query(FarmDB).filter(FarmDB.id == farm_id).first()
        
        if not farm_db:
            print(f"Farm not found in database: {farm_id}")
            return None
        
        # Convert FarmDB to FarmIdentity
        identity = FarmIdentity(
            farm_id=farm_db.id,
            farmer_id=farm_db.farmer_id,
            name=farm_db.name,
            location=farm_db.location,
            latitude=float(farm_db.latitude),
            longitude=float(farm_db.longitude),
            size_hectares=farm_db.size_hectares,
            size_category=farm_db.size_category,
            established_year=farm_db.established_year,
            registration_number=farm_db.registration_number
        )
        
        # Convert FarmDB to FarmTypology
        typology = FarmTypology(
            typology_type=farm_db.typology_type,
            sub_type=farm_db.sub_type,
            specialization=farm_db.specialization,
            farming_system=farm_db.farming_system,
            technology_level=farm_db.technology_level,
            market_orientation=farm_db.market_orientation,
            sustainability_practices=farm_db.sustainability_practices,
            certifications=farm_db.certifications
        )
        
        # Convert FarmDB to ResourceConstraints
        constraints = ResourceConstraints(
            capital_access_score=farm_db.capital_access_score,
            labor_availability_score=farm_db.labor_availability_score,
            equipment_access_score=farm_db.equipment_access_score,
            market_proximity_score=farm_db.market_proximity_score,
            water_access_score=farm_db.water_access_score,
            land_availability_score=farm_db.land_availability_score,
            max_capital_per_season=farm_db.max_capital_per_season,
            max_labor_hours_per_day=farm_db.max_labor_hours_per_day,
            equipment_limitations=farm_db.equipment_limitations,
            water_restrictions=farm_db.water_restrictions
        )
        
        # Convert FarmDB to EnvironmentalContext
        environment = EnvironmentalContext(
            soil_type=farm_db.soil_type,
            soil_ph=farm_db.soil_ph,
            soil_organic_matter=farm_db.soil_organic_matter,
            elevation_meters=farm_db.elevation_meters,
            average_rainfall_mm=farm_db.average_rainfall_mm,
            average_temperature_c=farm_db.average_temperature_c,
            growing_season_days=farm_db.growing_season_days,
            irrigation_type=farm_db.irrigation_type,
            water_source=farm_db.water_source,
            microclimate_zones=farm_db.microclimate_zones,
            topography=farm_db.topography
        )
        
        # Create FarmDNA
        farm_dna = FarmDNA(
            identity=identity,
            typology=typology,
            constraints=constraints,
            environment=environment,
            goals=farm_db.goals,
            preferences=farm_db.preferences or {},
            confidence_score=farm_db.confidence_score,
            version=farm_db.dna_version
        )
        
        print(f"✅ Retrieved FarmDNA from database for farm: {farm_db.name}")
        return farm_dna
        
    except Exception as e:
        print(f"❌ Error fetching FarmDNA from database: {e}")
        return None
''')
            replaced = True
            continue
        
        if in_function and line.strip() == '':
            in_function = False
        
        if not in_function or replaced:
            f.write(line)

print("✅ Updated get_farm_dna to fetch from database")
