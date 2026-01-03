"""
API routes module for AFIE service.
"""
from fastapi import APIRouter

from .farm import router as farm_router
from .health import router as health_router

# Create main router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(health_router, tags=["health"])
api_router.include_router(farm_router, prefix="/farms", tags=["farms"])