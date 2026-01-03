"""
Repository for farm data operations.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID
from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from src.data.database.models import FarmDB, HistoricalPerformanceDB
from src.config.logging_config import get_logger

logger = get_logger("data.repositories.farm")


class FarmRepository:
    """Repository for farm data operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_farm_by_id(self, farm_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get farm by ID.
        
        Args:
            farm_id: Farm ID
        
        Returns:
            Farm data dictionary or None if not found
        """
        try:
            stmt = select(FarmDB).where(
                and_(
                    FarmDB.id == farm_id,
                    FarmDB.is_active == True
                )
            )
            result = await self.db.execute(stmt)
            farm = result.scalar_one_or_none()
            
            if farm:
                # Also load historical performance
                await self.db.refresh(farm, ["historical_performance"])
                return self._farm_to_dict(farm)
            return None
        except Exception as e:
            logger.error(f"Error getting farm {farm_id}: {e}")
            raise
    
    async def get_farm_by_identity(self, farmer_id: UUID, farm_name: str) -> Optional[Dict[str, Any]]:
        """
        Get farm by farmer ID and farm name.
        
        Args:
            farmer_id: Farmer ID
            farm_name: Farm name
        
        Returns:
            Farm data dictionary or None if not found
        """
        try:
            stmt = select(FarmDB).where(
                and_(
                    FarmDB.farmer_id == farmer_id,
                    func.lower(FarmDB.name) == farm_name.lower(),
                    FarmDB.is_active == True
                )
            )
            result = await self.db.execute(stmt)
            farm = result.scalar_one_or_none()
            
            if farm:
                await self.db.refresh(farm, ["historical_performance"])
                return self._farm_to_dict(farm)
            return None
        except Exception as e:
            logger.error(f"Error getting farm for farmer {farmer_id}, name {farm_name}: {e}")
            raise
    
    async def list_farms(
        self,
        farmer_id: Optional[UUID] = None,
        typology: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List farms with optional filtering.
        
        Args:
            farmer_id: Filter by farmer ID
            typology: Filter by farm typology
            skip: Number of records to skip
            limit: Maximum number of records to return
        
        Returns:
            List of farm data dictionaries
        """
        try:
            conditions = [FarmDB.is_active == True]
            
            if farmer_id:
                conditions.append(FarmDB.farmer_id == farmer_id)
            
            if typology:
                conditions.append(FarmDB.typology_type == typology)
            
            stmt = (
                select(FarmDB)
                .where(and_(*conditions))
                .offset(skip)
                .limit(limit)
                .order_by(FarmDB.created_at.desc())
            )
            
            result = await self.db.execute(stmt)
            farms = result.scalars().all()
            
            # Load historical performance for each farm
            farm_dicts = []
            for farm in farms:
                await self.db.refresh(farm, ["historical_performance"])
                farm_dicts.append(self._farm_to_dict(farm))
            
            return farm_dicts
        except Exception as e:
            logger.error(f"Error listing farms: {e}")
            raise
    
    async def create_farm(self, farm_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new farm.
        
        Args:
            farm_data: Farm data dictionary
        
        Returns:
            Created farm data dictionary
        """
        try:
            # Extract identity
            identity = farm_data.get("identity", {})
            typology = farm_data.get("typology", {})
            constraints = farm_data.get("constraints", {})
            environment = farm_data.get("environment", {})
            
            # Create farm database object
            farm = FarmDB(
                id=identity.get("farm_id"),
                farmer_id=identity.get("farmer_id"),
                user_id=identity.get("farmer_id"),  # Assuming farmer_id = user_id for now
                name=identity.get("name"),
                location=identity.get("location"),
                latitude=float(identity.get("latitude", 0)),
                longitude=float(identity.get("longitude", 0)),
                size_hectares=float(identity.get("size_hectares", 0)),
                size_category=identity.get("size_category", "smallholder"),
                established_year=identity.get("established_year", datetime.now().year),
                registration_number=identity.get("registration_number"),
                
                # Typology
                typology_type=typology.get("typology_type", "smallholder"),
                sub_type=typology.get("sub_type"),
                specialization=typology.get("specialization", []),
                farming_system=typology.get("farming_system", "monoculture"),
                technology_level=typology.get("technology_level", "traditional"),
                market_orientation=typology.get("market_orientation", "subsistence"),
                sustainability_practices=typology.get("sustainability_practices", []),
                certifications=typology.get("certifications", []),
                
                # Resource constraints
                capital_access_score=float(constraints.get("capital_access_score", 50)),
                labor_availability_score=float(constraints.get("labor_availability_score", 50)),
                equipment_access_score=float(constraints.get("equipment_access_score", 50)),
                market_proximity_score=float(constraints.get("market_proximity_score", 50)),
                water_access_score=float(constraints.get("water_access_score", 50)),
                land_availability_score=float(constraints.get("land_availability_score", 50)),
                max_capital_per_season=constraints.get("max_capital_per_season"),
                max_labor_hours_per_day=constraints.get("max_labor_hours_per_day"),
                equipment_limitations=constraints.get("equipment_limitations", []),
                water_restrictions=constraints.get("water_restrictions", []),
                
                # Environmental context
                soil_type=environment.get("soil_type", "loam"),
                soil_ph=float(environment.get("soil_ph", 7.0)),
                soil_organic_matter=float(environment.get("soil_organic_matter", 2.0)),
                elevation_meters=float(environment.get("elevation_meters", 0)),
                average_rainfall_mm=float(environment.get("average_rainfall_mm", 0)),
                average_temperature_c=float(environment.get("average_temperature_c", 25)),
                growing_season_days=environment.get("growing_season_days", 180),
                irrigation_type=environment.get("irrigation_type", "rainfed"),
                water_source=environment.get("water_source"),
                microclimate_zones=environment.get("microclimate_zones", []),
                topography=environment.get("topography", []),
                
                # Goals and preferences
                goals=farm_data.get("goals", []),
                preferences=farm_data.get("preferences", {}),
                
                # Metadata
                confidence_score=farm_data.get("confidence_score", 0.0),
                dna_version=farm_data.get("version", 1),
            )
            
            self.db.add(farm)
            await self.db.commit()
            await self.db.refresh(farm)
            
            # Add historical performance if provided
            historical_performance = farm_data.get("historical_performance", [])
            for perf in historical_performance:
                await self.add_performance_record(farm.id, perf)
            
            await self.db.refresh(farm, ["historical_performance"])
            return self._farm_to_dict(farm)
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating farm: {e}")
            raise
    
    async def update_farm(self, farm_id: UUID, farm_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update existing farm.
        
        Args:
            farm_id: Farm ID
            farm_data: Updated farm data
        
        Returns:
            Updated farm data dictionary
        """
        try:
            # Get existing farm
            stmt = select(FarmDB).where(
                and_(
                    FarmDB.id == farm_id,
                    FarmDB.is_active == True
                )
            )
            result = await self.db.execute(stmt)
            farm = result.scalar_one_or_none()
            
            if not farm:
                raise ValueError(f"Farm {farm_id} not found")
            
            # Update fields
            update_data = {}
            
            # Update identity if provided
            if "identity" in farm_data:
                identity = farm_data["identity"]
                update_data.update({
                    "name": identity.get("name", farm.name),
                    "location": identity.get("location", farm.location),
                    "latitude": float(identity.get("latitude", farm.latitude)),
                    "longitude": float(identity.get("longitude", farm.longitude)),
                    "size_hectares": float(identity.get("size_hectares", farm.size_hectares)),
                    "size_category": identity.get("size_category", farm.size_category),
                    "established_year": identity.get("established_year", farm.established_year),
                    "registration_number": identity.get("registration_number", farm.registration_number),
                })
            
            # Update typology if provided
            if "typology" in farm_data:
                typology = farm_data["typology"]
                update_data.update({
                    "typology_type": typology.get("typology_type", farm.typology_type),
                    "sub_type": typology.get("sub_type", farm.sub_type),
                    "specialization": typology.get("specialization", farm.specialization),
                    "farming_system": typology.get("farming_system", farm.farming_system),
                    "technology_level": typology.get("technology_level", farm.technology_level),
                    "market_orientation": typology.get("market_orientation", farm.market_orientation),
                    "sustainability_practices": typology.get("sustainability_practices", farm.sustainability_practices),
                    "certifications": typology.get("certifications", farm.certifications),
                })
            
            # Update constraints if provided
            if "constraints" in farm_data:
                constraints = farm_data["constraints"]
                update_data.update({
                    "capital_access_score": float(constraints.get("capital_access_score", farm.capital_access_score)),
                    "labor_availability_score": float(constraints.get("labor_availability_score", farm.labor_availability_score)),
                    "equipment_access_score": float(constraints.get("equipment_access_score", farm.equipment_access_score)),
                    "market_proximity_score": float(constraints.get("market_proximity_score", farm.market_proximity_score)),
                    "water_access_score": float(constraints.get("water_access_score", farm.water_access_score)),
                    "land_availability_score": float(constraints.get("land_availability_score", farm.land_availability_score)),
                    "max_capital_per_season": constraints.get("max_capital_per_season", farm.max_capital_per_season),
                    "max_labor_hours_per_day": constraints.get("max_labor_hours_per_day", farm.max_labor_hours_per_day),
                    "equipment_limitations": constraints.get("equipment_limitations", farm.equipment_limitations),
                    "water_restrictions": constraints.get("water_restrictions", farm.water_restrictions),
                })
            
            # Update environment if provided
            if "environment" in farm_data:
                environment = farm_data["environment"]
                update_data.update({
                    "soil_type": environment.get("soil_type", farm.soil_type),
                    "soil_ph": float(environment.get("soil_ph", farm.soil_ph)),
                    "soil_organic_matter": float(environment.get("soil_organic_matter", farm.soil_organic_matter)),
                    "elevation_meters": float(environment.get("elevation_meters", farm.elevation_meters)),
                    "average_rainfall_mm": float(environment.get("average_rainfall_mm", farm.average_rainfall_mm)),
                    "average_temperature_c": float(environment.get("average_temperature_c", farm.average_temperature_c)),
                    "growing_season_days": environment.get("growing_season_days", farm.growing_season_days),
                    "irrigation_type": environment.get("irrigation_type", farm.irrigation_type),
                    "water_source": environment.get("water_source", farm.water_source),
                    "microclimate_zones": environment.get("microclimate_zones", farm.microclimate_zones),
                    "topography": environment.get("topography", farm.topography),
                })
            
            # Update other fields
            if "goals" in farm_data:
                update_data["goals"] = farm_data["goals"]
            
            if "preferences" in farm_data:
                update_data["preferences"] = farm_data["preferences"]
            
            if "confidence_score" in farm_data:
                update_data["confidence_score"] = farm_data["confidence_score"]
            
            if "version" in farm_data:
                update_data["dna_version"] = farm_data["version"]
            
            # Update last profile update timestamp
            update_data["last_profile_update"] = datetime.utcnow()
            
            # Perform update
            stmt = (
                update(FarmDB)
                .where(FarmDB.id == farm_id)
                .values(**update_data)
            )
            await self.db.execute(stmt)
            await self.db.commit()
            
            # Get updated farm
            await self.db.refresh(farm, ["historical_performance"])
            return self._farm_to_dict(farm)
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating farm {farm_id}: {e}")
            raise
    
    async def delete_farm(self, farm_id: UUID) -> None:
        """
        Soft delete farm (mark as inactive).
        
        Args:
            farm_id: Farm ID
        """
        try:
            stmt = (
                update(FarmDB)
                .where(FarmDB.id == farm_id)
                .values(is_active=False, updated_at=datetime.utcnow())
            )
            await self.db.execute(stmt)
            await self.db.commit()
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deleting farm {farm_id}: {e}")
            raise
    
    async def add_performance_record(self, farm_id: UUID, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add historical performance record to farm.
        
        Args:
            farm_id: Farm ID
            performance_data: Performance data
        
        Returns:
            Added performance record dictionary
        """
        try:
            # Calculate derived fields
            profit = None
            if performance_data.get("total_revenue") is not None and performance_data.get("total_cost") is not None:
                profit = performance_data["total_revenue"] - performance_data["total_cost"]
            
            # Create performance record
            performance = HistoricalPerformanceDB(
                farm_id=farm_id,
                year=performance_data.get("year", datetime.now().year),
                season=performance_data.get("season", "main"),
                crops_grown=performance_data.get("crops_grown", []),
                total_yield_kg=float(performance_data.get("total_yield_kg", 0)),
                total_revenue=performance_data.get("total_revenue"),
                total_cost=performance_data.get("total_cost"),
                water_used_m3=performance_data.get("water_used_m3"),
                labor_hours=performance_data.get("labor_hours"),
                inputs_used=performance_data.get("inputs_used", {}),
                challenges_faced=performance_data.get("challenges_faced", []),
                successes=performance_data.get("successes", []),
                notes=performance_data.get("notes"),
                profit=profit,
                yield_per_hectare=performance_data.get("yield_per_hectare"),
                is_verified=performance_data.get("is_verified", False),
                verified_by=performance_data.get("verified_by"),
                verification_date=performance_data.get("verification_date"),
            )
            
            self.db.add(performance)
            await self.db.commit()
            await self.db.refresh(performance)
            
            return self._performance_to_dict(performance)
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error adding performance record for farm {farm_id}: {e}")
            raise
    
    def _farm_to_dict(self, farm: FarmDB) -> Dict[str, Any]:
        """Convert FarmDB object to dictionary."""
        return {
            "identity": {
                "farm_id": farm.id,
                "farmer_id": farm.farmer_id,
                "name": farm.name,
                "location": farm.location,
                "latitude": float(farm.latitude),
                "longitude": float(farm.longitude),
                "size_hectares": farm.size_hectares,
                "size_category": farm.size_category,
                "established_year": farm.established_year,
                "registration_number": farm.registration_number,
            },
            "typology": {
                "typology_type": farm.typology_type,
                "sub_type": farm.sub_type,
                "specialization": farm.specialization,
                "farming_system": farm.farming_system,
                "technology_level": farm.technology_level,
                "market_orientation": farm.market_orientation,
                "sustainability_practices": farm.sustainability_practices,
                "certifications": farm.certifications,
            },
            "constraints": {
                "capital_access_score": farm.capital_access_score,
                "labor_availability_score": farm.labor_availability_score,
                "equipment_access_score": farm.equipment_access_score,
                "market_proximity_score": farm.market_proximity_score,
                "water_access_score": farm.water_access_score,
                "land_availability_score": farm.land_availability_score,
                "max_capital_per_season": farm.max_capital_per_season,
                "max_labor_hours_per_day": farm.max_labor_hours_per_day,
                "equipment_limitations": farm.equipment_limitations,
                "water_restrictions": farm.water_restrictions,
            },
            "environment": {
                "soil_type": farm.soil_type,
                "soil_ph": farm.soil_ph,
                "soil_organic_matter": farm.soil_organic_matter,
                "elevation_meters": farm.elevation_meters,
                "average_rainfall_mm": farm.average_rainfall_mm,
                "average_temperature_c": farm.average_temperature_c,
                "growing_season_days": farm.growing_season_days,
                "irrigation_type": farm.irrigation_type,
                "water_source": farm.water_source,
                "microclimate_zones": farm.microclimate_zones,
                "topography": farm.topography,
            },
            "goals": farm.goals,
            "preferences": farm.preferences,
            "historical_performance": [
                self._performance_to_dict(perf) 
                for perf in farm.historical_performance
            ],
            "version": farm.dna_version,
            "created_at": farm.created_at,
            "updated_at": farm.updated_at,
            "confidence_score": farm.confidence_score,
        }
    
    def _performance_to_dict(self, performance: HistoricalPerformanceDB) -> Dict[str, Any]:
        """Convert HistoricalPerformanceDB object to dictionary."""
        return {
            "id": performance.id,
            "farm_id": performance.farm_id,
            "year": performance.year,
            "season": performance.season,
            "crops_grown": performance.crops_grown,
            "total_yield_kg": performance.total_yield_kg,
            "total_revenue": performance.total_revenue,
            "total_cost": performance.total_cost,
            "water_used_m3": performance.water_used_m3,
            "labor_hours": performance.labor_hours,
            "inputs_used": performance.inputs_used,
            "challenges_faced": performance.challenges_faced,
            "successes": performance.successes,
            "notes": performance.notes,
            "profit": performance.profit,
            "yield_per_hectare": performance.yield_per_hectare,
            "is_verified": performance.is_verified,
            "verified_by": performance.verified_by,
            "verification_date": performance.verification_date,
            "created_at": performance.created_at,
            "updated_at": performance.updated_at,
        }