"""
Farm management endpoints.
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.database import get_db
from src.config.logging_config import get_logger
from src.data.database.repositories.farm_repo import FarmRepository
from src.models.farm import FarmDNA, FarmIdentity, HistoricalPerformance

router = APIRouter()
logger = get_logger("api.farm")


@router.get("/", response_model=List[FarmDNA])
async def list_farms(
    farmer_id: Optional[UUID] = None,
    typology: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """
    List farms with optional filtering.
    
    Args:
        farmer_id: Filter by farmer ID
        typology: Filter by farm typology
        skip: Number of records to skip
        limit: Maximum number of records to return
    
    Returns:
        List of farm DNA profiles
    """
    try:
        repo = FarmRepository(db)
        farms = await repo.list_farms(
            farmer_id=farmer_id,
            typology=typology,
            skip=skip,
            limit=limit
        )
        return [FarmDNA.from_dict(farm) for farm in farms]
    except Exception as e:
        logger.error(f"Error listing farms: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list farms"
        )


@router.get("/{farm_id}", response_model=FarmDNA)
async def get_farm(
    farm_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get farm DNA profile by ID.
    
    Args:
        farm_id: Farm ID
    
    Returns:
        Farm DNA profile
    
    Raises:
        HTTPException: If farm not found
    """
    try:
        repo = FarmRepository(db)
        farm_data = await repo.get_farm_by_id(farm_id)
        
        if not farm_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Farm {farm_id} not found"
            )
        
        return FarmDNA.from_dict(farm_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting farm {farm_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get farm"
        )


@router.post("/", response_model=FarmDNA, status_code=status.HTTP_201_CREATED)
async def create_farm(
    farm_data: FarmDNA,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new farm DNA profile.
    
    Args:
        farm_data: Farm DNA data
    
    Returns:
        Created farm DNA profile
    """
    try:
        repo = FarmRepository(db)
        
        # Check if farm already exists
        existing = await repo.get_farm_by_identity(
            farmer_id=farm_data.identity.farmer_id,
            farm_name=farm_data.identity.name
        )
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Farm '{farm_data.identity.name}' already exists for this farmer"
            )
        
        # Create farm
        created_farm = await repo.create_farm(farm_data.to_dict())
        
        logger.info(f"Created farm {created_farm['identity']['farm_id']} for farmer {farm_data.identity.farmer_id}")
        
        return FarmDNA.from_dict(created_farm)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating farm: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create farm"
        )


@router.put("/{farm_id}", response_model=FarmDNA)
async def update_farm(
    farm_id: UUID,
    farm_data: FarmDNA,
    db: AsyncSession = Depends(get_db)
):
    """
    Update farm DNA profile.
    
    Args:
        farm_id: Farm ID
        farm_data: Updated farm DNA data
    
    Returns:
        Updated farm DNA profile
    
    Raises:
        HTTPException: If farm not found
    """
    try:
        repo = FarmRepository(db)
        
        # Check if farm exists
        existing = await repo.get_farm_by_id(farm_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Farm {farm_id} not found"
            )
        
        # Update farm
        updated_farm = await repo.update_farm(farm_id, farm_data.to_dict())
        
        logger.info(f"Updated farm {farm_id}")
        
        return FarmDNA.from_dict(updated_farm)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating farm {farm_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update farm"
        )


@router.delete("/{farm_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_farm(
    farm_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete farm DNA profile (soft delete).
    
    Args:
        farm_id: Farm ID
    
    Raises:
        HTTPException: If farm not found
    """
    try:
        repo = FarmRepository(db)
        
        # Check if farm exists
        existing = await repo.get_farm_by_id(farm_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Farm {farm_id} not found"
            )
        
        # Soft delete farm
        await repo.delete_farm(farm_id)
        
        logger.info(f"Deleted farm {farm_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting farm {farm_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete farm"
        )


@router.post("/{farm_id}/performance", response_model=HistoricalPerformance)
async def add_performance_record(
    farm_id: UUID,
    performance: HistoricalPerformance,
    db: AsyncSession = Depends(get_db)
):
    """
    Add historical performance record to farm.
    
    Args:
        farm_id: Farm ID
        performance: Performance data
    
    Returns:
        Added performance record
    
    Raises:
        HTTPException: If farm not found
    """
    try:
        repo = FarmRepository(db)
        
        # Check if farm exists
        existing = await repo.get_farm_by_id(farm_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Farm {farm_id} not found"
            )
        
        # Add performance record
        record = await repo.add_performance_record(farm_id, performance.model_dump())
        
        logger.info(f"Added performance record for farm {farm_id}, year {performance.year}")
        
        return HistoricalPerformance(**record)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding performance record for farm {farm_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add performance record"
        )