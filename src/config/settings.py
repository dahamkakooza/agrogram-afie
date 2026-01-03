import os
from typing import List, Optional
from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = Field(default="AFIE", description="Application name")
    environment: str = Field(default="development", description="Runtime environment")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=4, description="Number of worker processes")
    reload: bool = Field(default=False, description="Auto-reload for development")
    
    # Database
    database_url: PostgresDsn = Field(
        default="postgresql://user:pass@localhost:5432/afie_db",
        description="PostgreSQL connection URL"
    )
    database_pool_size: int = Field(default=20, description="Database connection pool size")
    database_max_overflow: int = Field(default=10, description="Max overflow connections")
    database_echo: bool = Field(default=False, description="Echo SQL queries")
    
    # Redis
    redis_url: RedisDsn = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    redis_pool_size: int = Field(default=10, description="Redis connection pool size")
    redis_max_connections: int = Field(default=50, description="Max Redis connections")
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiry")
    
    # CORS
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    allowed_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods"
    )
    allowed_headers: List[str] = Field(
        default=["*"],
        description="Allowed HTTP headers"
    )
    
    
    
    
    
    # API Configuration
    api_prefix: str = Field(default="/api/v1", description="API URL prefix")
    api_title: str = Field(default="AFIE API", description="API documentation title")
    api_description: str = Field(default="Adaptive Farm Intelligence Engine API", description="API description")
    api_version: str = Field(default="0.1.0", description="API version")# API Configuration
    api_prefix: str = Field(default="/api/v1", description="API URL prefix")
    api_title: str = Field(default="AFIE API", description="API documentation title")
    api_description: str = Field(default="Adaptive Farm Intelligence Engine API", description="API description")
    api_version: str = Field(default="0.1.0", description="API version")# External APIs
    weather_api_key: Optional[str] = Field(default=None, description="Weather API key")
    soil_api_key: Optional[str] = Field(default=None, description="Soil data API key")
    market_api_key: Optional[str] = Field(default=None, description="Market data API key")
    satellite_api_key: Optional[str] = Field(default=None, description="Satellite imagery API key")
    
    # ML Configuration
    ml_models_path: str = Field(default="./models", description="Path to ML models")
    ml_cache_size: int = Field(default=1000, description="ML prediction cache size")
    ml_prediction_timeout: int = Field(default=10, description="ML prediction timeout in seconds")
    
    # Monitoring
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    jaeger_enabled: bool = Field(default=False, description="Enable Jaeger tracing")
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    
    # Feature Flags
    feature_farm_profiler: bool = Field(default=True, description="Enable farm profiler")
    feature_recommendations: bool = Field(default=False, description="Enable recommendations")
    feature_learning_system: bool = Field(default=False, description="Enable learning system")
    feature_predictions: bool = Field(default=False, description="Enable predictions")
    
    @field_validator("secret_key")
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key length and complexity."""
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        if v == "your-secret-key-change-in-production":
            raise ValueError("Please change the default SECRET_KEY in production")
        return v
    
    @field_validator("allowed_origins")
    def validate_origins(cls, v: List[str]) -> List[str]:
        """Validate CORS origins."""
        if "*" in v and len(v) > 1:
            raise ValueError("Cannot mix '*' with specific origins")
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment.lower() == "testing"


settings = Settings()