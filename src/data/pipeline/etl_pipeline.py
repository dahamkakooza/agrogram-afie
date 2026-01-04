"""
ETL Pipeline - Week 7-8 Implementation
Data collection, transformation, and loading pipeline
"""
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass
from enum import Enum
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("etl_pipeline")


class DataSourceType(Enum):
    """Types of data sources."""
    WEATHER_API = "weather_api"
    MARKET_API = "market_api"
    IOT_DEVICE = "iot_device"
    MANUAL_ENTRY = "manual_entry"
    WEB_SCRAPER = "web_scraper"
    GOVERNMENT_API = "government_api"


class DataQualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"  # Complete, accurate, timely
    GOOD = "good"           # Minor issues
    FAIR = "fair"           # Some missing/inaccurate data
    POOR = "poor"           # Major data issues
    UNUSABLE = "unusable"   # Cannot be used


@dataclass
class DataSource:
    """Data source configuration."""
    source_type: DataSourceType
    endpoint: str
    api_key: Optional[str] = None
    parameters: Dict[str, Any] = None
    refresh_rate_minutes: int = 60
    last_successful_fetch: Optional[datetime] = None
    success_rate: float = 1.0
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.last_successful_fetch is None:
            self.last_successful_fetch = datetime.utcnow()


@dataclass
class DataQualityCheck:
    """Data quality check configuration."""
    check_name: str
    check_type: str  # completeness, accuracy, consistency, timeliness
    threshold: float
    severity: str  # warning, error
    description: str


class ETLPipeline:
    """
    Extract-Transform-Load pipeline for agricultural data.
    
    Features:
    1. Multiple data source connectors
    2. Data quality validation
    3. Feature engineering
    4. Real-time and batch processing
    """
    
    def __init__(self):
        self.data_sources = self._initialize_data_sources()
        self.quality_checks = self._initialize_quality_checks()
        self.data_cache = {}
        self.processing_history = []
        self.data_quality_scores = {}
        
        logger.info("✅ ETLPipeline initialized")
    
    def _initialize_data_sources(self) -> Dict[str, DataSource]:
        """Initialize data source configurations."""
        sources = {}
        
        # Weather data source (mock)
        sources["weather_agromonitoring"] = DataSource(
            source_type=DataSourceType.WEATHER_API,
            endpoint="https://api.agromonitoring.com/agro/1.0/weather",
            parameters={"units": "metric", "lang": "en"},
            refresh_rate_minutes=30
        )
        
        # Market data source (mock)
        sources["market_fao"] = DataSource(
            source_type=DataSourceType.MARKET_API,
            endpoint="https://api.fao.org/market/prices",
            refresh_rate_minutes=1440  # Daily
        )
        
        # IoT device data (mock)
        sources["iot_field_sensors"] = DataSource(
            source_type=DataSourceType.IOT_DEVICE,
            endpoint="mqtt://iot-broker/field-sensors",
            refresh_rate_minutes=5
        )
        
        # Government agricultural data (mock)
        sources["gov_agriculture"] = DataSource(
            source_type=DataSourceType.GOVERNMENT_API,
            endpoint="https://api.agriculture.gov/data",
            refresh_rate_minutes=1440  # Daily
        )
        
        return sources
    
    def _initialize_quality_checks(self) -> List[DataQualityCheck]:
        """Initialize data quality checks."""
        checks = []
        
        # Completeness checks
        checks.append(DataQualityCheck(
            check_name="weather_data_completeness",
            check_type="completeness",
            threshold=0.95,  # 95% complete
            severity="error",
            description="Weather data must be at least 95% complete"
        ))
        
        checks.append(DataQualityCheck(
            check_name="market_data_completeness",
            check_type="completeness",
            threshold=0.90,
            severity="warning",
            description="Market data should be at least 90% complete"
        ))
        
        # Accuracy checks
        checks.append(DataQualityCheck(
            check_name="temperature_range_check",
            check_type="accuracy",
            threshold=0.98,
            severity="error",
            description="Temperature values must be within valid range (-50 to 60°C)"
        ))
        
        checks.append(DataQualityCheck(
            check_name="price_positive_check",
            check_type="accuracy",
            threshold=1.0,
            severity="error",
            description="Price values must be positive"
        ))
        
        # Consistency checks
        checks.append(DataQualityCheck(
            check_name="temporal_consistency",
            check_type="consistency",
            threshold=0.95,
            severity="warning",
            description="Data should be temporally consistent without large jumps"
        ))
        
        # Timeliness checks
        checks.append(DataQualityCheck(
            check_name="data_freshness",
            check_type="timeliness",
            threshold=0.95,
            severity="warning",
            description="Data should be fresh (within expected update window)"
        ))
        
        return checks
    
    async def extract_data(self, 
                          source_name: str,
                          force_refresh: bool = False) -> Dict[str, Any]:
        """
        Extract data from a source.
        
        Args:
            source_name: Name of data source
            force_refresh: Whether to force refresh even if not due
        
        Returns:
            Extracted data with metadata
        """
        logger.info(f"📥 Extracting data from {source_name}")
        
        source = self.data_sources.get(source_name)
        if not source:
            raise ValueError(f"Unknown data source: {source_name}")
        
        # Check if refresh is needed
        if not force_refresh and source.last_successful_fetch:
            time_since_fetch = datetime.utcnow() - source.last_successful_fetch
            minutes_since = time_since_fetch.total_seconds() / 60
            
            if minutes_since < source.refresh_rate_minutes:
                logger.info(f"Using cached data for {source_name}")
                return self.data_cache.get(source_name, {})
        
        try:
            # Simulate data extraction (in practice, this would make actual API calls)
            extracted_data = await self._simulate_extraction(source)
            
            # Update source metadata
            source.last_successful_fetch = datetime.utcnow()
            source.success_rate = 0.95  # Update success rate
            
            # Cache the data
            self.data_cache[source_name] = {
                "data": extracted_data,
                "metadata": {
                    "source": source_name,
                    "extraction_time": datetime.utcnow().isoformat(),
                    "data_points": len(extracted_data.get("records", [])),
                    "extraction_status": "success"
                }
            }
            
            logger.info(f"✅ Successfully extracted {len(extracted_data.get('records', []))} records from {source_name}")
            
            return self.data_cache[source_name]
            
        except Exception as e:
            logger.error(f"❌ Failed to extract data from {source_name}: {e}")
            source.success_rate = max(0.0, source.success_rate - 0.1)
            
            # Return cached data if available
            cached = self.data_cache.get(source_name)
            if cached:
                cached["metadata"]["extraction_status"] = "failed_using_cache"
                cached["metadata"]["extraction_error"] = str(e)
                return cached
            
            return {
                "data": {},
                "metadata": {
                    "source": source_name,
                    "extraction_time": datetime.utcnow().isoformat(),
                    "extraction_status": "failed",
                    "error": str(e)
                }
            }
    
    async def _simulate_extraction(self, source: DataSource) -> Dict[str, Any]:
        """Simulate data extraction from a source."""
        await asyncio.sleep(0.1)  # Simulate network delay
        
        source_type = source.source_type
        
        if source_type == DataSourceType.WEATHER_API:
            return self._simulate_weather_data()
        elif source_type == DataSourceType.MARKET_API:
            return self._simulate_market_data()
        elif source_type == DataSourceType.IOT_DEVICE:
            return self._simulate_iot_data()
        elif source_type == DataSourceType.GOVERNMENT_API:
            return self._simulate_government_data()
        else:
            return {"records": [], "source": source_type.value}
    
    def _simulate_weather_data(self) -> Dict[str, Any]:
        """Simulate weather data."""
        # Generate realistic weather data
        base_temp = 25.0
        base_rain = 2.5
        
        records = []
        for i in range(7):  # Last 7 days
            date = (datetime.utcnow() - timedelta(days=i)).date()
            
            # Add some variation
            temp_variation = np.random.normal(0, 2)
            rain_variation = max(0, np.random.normal(0, 1))
            
            records.append({
                "date": date.isoformat(),
                "temperature_avg": round(base_temp + temp_variation, 1),
                "temperature_min": round(base_temp + temp_variation - 5, 1),
                "temperature_max": round(base_temp + temp_variation + 5, 1),
                "rainfall_mm": round(max(0, base_rain + rain_variation), 1),
                "humidity_percent": round(60 + np.random.normal(0, 10), 1),
                "wind_speed_kmh": round(10 + np.random.normal(0, 5), 1),
                "solar_radiation_mj_m2": round(20 + np.random.normal(0, 5), 1)
            })
        
        return {
            "data_type": "weather",
            "location": {"lat": -1.2921, "lon": 36.8219},  # Nairobi
            "units": {"temperature": "°C", "rainfall": "mm"},
            "records": records,
            "summary": {
                "avg_temperature": round(np.mean([r["temperature_avg"] for r in records]), 1),
                "total_rainfall": round(sum(r["rainfall_mm"] for r in records), 1),
                "data_quality": "good"
            }
        }
    
    def _simulate_market_data(self) -> Dict[str, Any]:
        """Simulate market data."""
        crops = ["maize", "wheat", "rice", "beans", "potatoes"]
        
        records = []
        for crop in crops:
            base_price = {
                "maize": 450,
                "wheat": 500,
                "rice": 600,
                "beans": 800,
                "potatoes": 300
            }.get(crop, 400)
            
            # Add some price variation
            price_variation = np.random.normal(0, 0.1)  # 10% variation
            current_price = round(base_price * (1 + price_variation), 2)
            
            records.append({
                "crop": crop,
                "price_per_kg": current_price,
                "currency": "KES",
                "market": "Nairobi",
                "date": datetime.utcnow().date().isoformat(),
                "price_trend": "stable" if abs(price_variation) < 0.05 else 
                              "rising" if price_variation > 0 else "falling",
                "volume_traded_kg": np.random.randint(10000, 100000),
                "quality": "grade_a"
            })
        
        return {
            "data_type": "market_prices",
            "market": "Nairobi",
            "currency": "KES",
            "records": records,
            "summary": {
                "total_crops": len(crops),
                "price_range": {
                    "min": min(r["price_per_kg"] for r in records),
                    "max": max(r["price_per_kg"] for r in records),
                    "avg": round(np.mean([r["price_per_kg"] for r in records]), 2)
                },
                "update_frequency": "daily"
            }
        }
    
    def _simulate_iot_data(self) -> Dict[str, Any]:
        """Simulate IoT sensor data."""
        sensors = [
            {"id": "soil_moisture_1", "type": "soil_moisture", "unit": "%"},
            {"id": "soil_temperature_1", "type": "soil_temperature", "unit": "°C"},
            {"id": "air_temperature_1", "type": "air_temperature", "unit": "°C"},
            {"id": "humidity_1", "type": "humidity", "unit": "%"},
            {"id": "soil_ph_1", "type": "soil_ph", "unit": "pH"}
        ]
        
        records = []
        timestamp = datetime.utcnow()
        
        for sensor in sensors:
            sensor_type = sensor["type"]
            
            # Generate realistic sensor values
            if sensor_type == "soil_moisture":
                value = round(30 + np.random.normal(0, 5), 1)
                value = max(0, min(100, value))
            elif sensor_type == "soil_temperature":
                value = round(25 + np.random.normal(0, 2), 1)
            elif sensor_type == "air_temperature":
                value = round(27 + np.random.normal(0, 3), 1)
            elif sensor_type == "humidity":
                value = round(65 + np.random.normal(0, 10), 1)
                value = max(0, min(100, value))
            elif sensor_type == "soil_ph":
                value = round(6.5 + np.random.normal(0, 0.5), 1)
                value = max(3, min(9, value))
            else:
                value = 0.0
            
            records.append({
                "sensor_id": sensor["id"],
                "sensor_type": sensor_type,
                "value": value,
                "unit": sensor["unit"],
                "timestamp": timestamp.isoformat(),
                "location": {"lat": -1.2921, "lon": 36.8219, "field": "field_a"},
                "battery_level": round(80 + np.random.normal(0, 10), 1),
                "signal_strength": round(85 + np.random.normal(0, 15), 1)
            })
        
        return {
            "data_type": "iot_sensor",
            "farm_id": "farm_001",
            "total_sensors": len(sensors),
            "records": records,
            "summary": {
                "reading_time": timestamp.isoformat(),
                "active_sensors": len(sensors),
                "data_frequency": "5_minutes"
            }
        }
    
    def _simulate_government_data(self) -> Dict[str, Any]:
        """Simulate government agricultural data."""
        return {
            "data_type": "government_agriculture",
            "country": "Kenya",
            "records": [
                {
                    "indicator": "fertilizer_subsidy_rate",
                    "value": 0.3,
                    "unit": "proportion",
                    "effective_date": "2025-01-01",
                    "region": "national"
                },
                {
                    "indicator": "extension_services_coverage",
                    "value": 0.65,
                    "unit": "proportion",
                    "effective_date": "2025-01-01",
                    "region": "national"
                },
                {
                    "indicator": "crop_insurance_penetration",
                    "value": 0.15,
                    "unit": "proportion",
                    "effective_date": "2025-01-01",
                    "region": "national"
                },
                {
                    "indicator": "average_farm_size",
                    "value": 2.5,
                    "unit": "hectares",
                    "effective_date": "2024-01-01",
                    "region": "national"
                }
            ],
            "summary": {
                "data_year": 2025,
                "update_frequency": "annual",
                "source": "Ministry of Agriculture"
            }
        }
    
    async def extract_all_sources(self, 
                                 source_names: List[str] = None,
                                 parallel: bool = True) -> Dict[str, Any]:
        """
        Extract data from multiple sources.
        
        Args:
            source_names: List of source names (None for all)
            parallel: Whether to extract in parallel
        
        Returns:
            Combined data from all sources
        """
        if source_names is None:
            source_names = list(self.data_sources.keys())
        
        logger.info(f"📥 Extracting data from {len(source_names)} sources")
        
        if parallel:
            # Extract in parallel
            tasks = [self.extract_data(source) for source in source_names]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            extracted_data = {}
            for source, result in zip(source_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to extract from {source}: {result}")
                    extracted_data[source] = {
                        "data": {},
                        "metadata": {
                            "source": source,
                            "extraction_status": "failed",
                            "error": str(result)
                        }
                    }
                else:
                    extracted_data[source] = result
        else:
            # Extract sequentially
            extracted_data = {}
            for source in source_names:
                try:
                    extracted_data[source] = await self.extract_data(source)
                except Exception as e:
                    logger.error(f"Failed to extract from {source}: {e}")
                    extracted_data[source] = {
                        "data": {},
                        "metadata": {
                            "source": source,
                            "extraction_status": "failed",
                            "error": str(e)
                        }
                    }
        
        # Store in processing history
        processing_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "sources": source_names,
            "results": {k: v["metadata"]["extraction_status"] for k, v in extracted_data.items()}
        }
        self.processing_history.append(processing_record)
        
        return {
            "extraction_id": f"extract_{datetime.utcnow().timestamp()}",
            "timestamp": datetime.utcnow().isoformat(),
            "sources_extracted": source_names,
            "data": extracted_data,
            "summary": {
                "total_sources": len(source_names),
                "successful_sources": sum(1 for v in extracted_data.values() 
                                         if v["metadata"]["extraction_status"] == "success"),
                "failed_sources": sum(1 for v in extracted_data.values() 
                                     if v["metadata"]["extraction_status"] == "failed"),
                "using_cache_sources": sum(1 for v in extracted_data.values() 
                                          if "using_cache" in v["metadata"]["extraction_status"])
            }
        }
    
    def transform_data(self, 
                      extracted_data: Dict[str, Any],
                      transformations: List[str] = None) -> Dict[str, Any]:
        """
        Transform extracted data.
        
        Args:
            extracted_data: Data from extract phase
            transformations: List of transformations to apply
        
        Returns:
            Transformed data
        """
        logger.info("🔄 Transforming extracted data")
        
        if transformations is None:
            transformations = [
                "clean_and_validate",
                "normalize_units",
                "engineer_features",
                "temporal_alignment",
                "quality_assessment"
            ]
        
        transformed_data = {
            "raw_data": extracted_data,
            "transformations_applied": [],
            "transformed_records": {},
            "feature_vectors": {},
            "quality_scores": {}
        }
        
        # Apply each transformation
        for transform in transformations:
            if transform == "clean_and_validate":
                self._apply_cleaning(extracted_data, transformed_data)
                transformed_data["transformations_applied"].append(transform)
                
            elif transform == "normalize_units":
                self._apply_unit_normalization(extracted_data, transformed_data)
                transformed_data["transformations_applied"].append(transform)
                
            elif transform == "engineer_features":
                self._apply_feature_engineering(extracted_data, transformed_data)
                transformed_data["transformations_applied"].append(transform)
                
            elif transform == "temporal_alignment":
                self._apply_temporal_alignment(extracted_data, transformed_data)
                transformed_data["transformations_applied"].append(transform)
                
            elif transform == "quality_assessment":
                quality_scores = self._assess_data_quality(extracted_data)
                transformed_data["quality_scores"] = quality_scores
                transformed_data["transformations_applied"].append(transform)
        
        # Generate feature vectors for ML models
        transformed_data["feature_vectors"] = self._generate_feature_vectors(transformed_data)
        
        logger.info(f"✅ Applied {len(transformed_data['transformations_applied'])} transformations")
        
        return transformed_data
    
    def _apply_cleaning(self, extracted_data: Dict[str, Any], transformed_data: Dict[str, Any]):
        """Apply data cleaning and validation."""
        cleaned_records = {}
        
        for source, data in extracted_data.get("data", {}).items():
            raw_records = data.get("data", {}).get("records", [])
            
            if not raw_records:
                cleaned_records[source] = []
                continue
            
            # Apply cleaning based on data type
            source_type = self.data_sources.get(source, DataSource("unknown", "")).source_type
            
            if source_type == DataSourceType.WEATHER_API:
                cleaned = self._clean_weather_data(raw_records)
            elif source_type == DataSourceType.MARKET_API:
                cleaned = self._clean_market_data(raw_records)
            elif source_type == DataSourceType.IOT_DEVICE:
                cleaned = self._clean_iot_data(raw_records)
            else:
                cleaned = raw_records  # Keep as-is for unknown types
            
            cleaned_records[source] = cleaned
        
        transformed_data["cleaned_records"] = cleaned_records
    
    def _clean_weather_data(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean weather data."""
        cleaned = []
        
        for record in records:
            # Create a copy
            clean_record = record.copy()
            
            # Validate temperature range
            temp_keys = ["temperature_avg", "temperature_min", "temperature_max"]
            for key in temp_keys:
                if key in clean_record:
                    value = clean_record[key]
                    if not (-50 <= value <= 60):  # Valid temperature range
                        clean_record[key] = None
                        clean_record[f"{key}_quality_flag"] = "out_of_range"
            
            # Validate rainfall (non-negative)
            if "rainfall_mm" in clean_record and clean_record["rainfall_mm"] < 0:
                clean_record["rainfall_mm"] = 0
                clean_record["rainfall_quality_flag"] = "negative_corrected"
            
            # Validate humidity (0-100%)
            if "humidity_percent" in clean_record:
                humidity = clean_record["humidity_percent"]
                if not (0 <= humidity <= 100):
                    clean_record["humidity_percent"] = max(0, min(100, humidity))
                    clean_record["humidity_quality_flag"] = "clipped"
            
            cleaned.append(clean_record)
        
        return cleaned
    
    def _clean_market_data(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean market data."""
        cleaned = []
        
        for record in records:
            clean_record = record.copy()
            
            # Validate price (positive)
            if "price_per_kg" in clean_record and clean_record["price_per_kg"] <= 0:
                clean_record["price_per_kg"] = None
                clean_record["price_quality_flag"] = "invalid"
            
            # Validate volume (non-negative)
            if "volume_traded_kg" in clean_record and clean_record["volume_traded_kg"] < 0:
                clean_record["volume_traded_kg"] = 0
                clean_record["volume_quality_flag"] = "negative_corrected"
            
            # Ensure required fields
            required_fields = ["crop", "price_per_kg", "date"]
            missing_fields = [field for field in required_fields if field not in clean_record]
            if missing_fields:
                clean_record["quality_flag"] = f"missing_fields: {missing_fields}"
            
            cleaned.append(clean_record)
        
        return cleaned
    
    def _clean_iot_data(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean IoT sensor data."""
        cleaned = []
        
        for record in records:
            clean_record = record.copy()
            
            # Validate sensor values based on type
            sensor_type = clean_record.get("sensor_type", "")
            value = clean_record.get("value", 0)
            
            if sensor_type == "soil_moisture" and not (0 <= value <= 100):
                clean_record["value"] = max(0, min(100, value))
                clean_record["quality_flag"] = "clipped"
            
            elif sensor_type == "soil_ph" and not (3 <= value <= 9):
                clean_record["value"] = max(3, min(9, value))
                clean_record["quality_flag"] = "clipped"
            
            elif sensor_type == "humidity" and not (0 <= value <= 100):
                clean_record["value"] = max(0, min(100, value))
                clean_record["quality_flag"] = "clipped"
            
            # Check battery level
            if "battery_level" in clean_record and clean_record["battery_level"] < 10:
                clean_record["battery_warning"] = "low"
            
            # Check signal strength
            if "signal_strength" in clean_record and clean_record["signal_strength"] < 50:
                clean_record["signal_warning"] = "weak"
            
            cleaned.append(clean_record)
        
        return cleaned
    
    def _apply_unit_normalization(self, extracted_data: Dict[str, Any], transformed_data: Dict[str, Any]):
        """Normalize units across data sources."""
        normalized_records = {}
        
        for source, data in extracted_data.get("data", {}).items():
            records = transformed_data.get("cleaned_records", {}).get(source, [])
            
            if not records:
                normalized_records[source] = []
                continue
            
            normalized = []
            for record in records:
                norm_record = record.copy()
                
                # Convert temperatures to Celsius if needed
                temp_keys = [k for k in record.keys() if "temperature" in k.lower()]
                for key in temp_keys:
                    if key in norm_record and norm_record[key] is not None:
                        # Check if in Fahrenheit (assuming > 100 is Fahrenheit)
                        if norm_record[key] > 100:
                            norm_record[key] = (norm_record[key] - 32) * 5/9
                            norm_record[f"{key}_original_unit"] = "°F"
                            norm_record[f"{key}_converted"] = True
                
                # Convert rainfall units if needed
                if "rainfall_mm" in norm_record and norm_record["rainfall_mm"] is not None:
                    # Assuming millimeters, convert from inches if value is small
                    if norm_record["rainfall_mm"] < 10:  # Likely inches
                        norm_record["rainfall_mm"] = norm_record["rainfall_mm"] * 25.4
                        norm_record["rainfall_original_unit"] = "inches"
                        norm_record["rainfall_converted"] = True
                
                normalized.append(norm_record)
            
            normalized_records[source] = normalized
        
        transformed_data["normalized_records"] = normalized_records
    
    def _apply_feature_engineering(self, extracted_data: Dict[str, Any], transformed_data: Dict[str, Any]):
        """Apply feature engineering to create derived features."""
        engineered_records = {}
        
        for source, data in extracted_data.get("data", {}).items():
            records = transformed_data.get("normalized_records", {}).get(source, [])
            
            if not records:
                engineered_records[source] = []
                continue
            
            engineered = []
            for i, record in enumerate(records):
                eng_record = record.copy()
                
                # Weather-specific features
                if "temperature_avg" in record and "temperature_min" in record and "temperature_max" in record:
                    # Temperature range
                    eng_record["temperature_range"] = record["temperature_max"] - record["temperature_min"]
                    
                    # Growing degree days (base 10°C)
                    if record["temperature_avg"] > 10:
                        eng_record["growing_degree_days"] = record["temperature_avg"] - 10
                    else:
                        eng_record["growing_degree_days"] = 0
                
                # Rainfall features
                if "rainfall_mm" in record:
                    # Cumulative rainfall (if we have previous days)
                    if i > 0 and "rainfall_mm" in records[i-1]:
                        prev_record = records[i-1]
                        if "cumulative_rainfall" in prev_record:
                            eng_record["cumulative_rainfall"] = prev_record["cumulative_rainfall"] + record["rainfall_mm"]
                        else:
                            eng_record["cumulative_rainfall"] = record["rainfall_mm"]
                    else:
                        eng_record["cumulative_rainfall"] = record["rainfall_mm"]
                    
                    # Rain intensity (if we had duration data)
                    eng_record["rain_intensity"] = record["rainfall_mm"] / 24  # mm per hour assumption
                
                # Market-specific features
                if "price_per_kg" in record:
                    # Price change from previous day (if available)
                    if i > 0 and "price_per_kg" in records[i-1]:
                        prev_price = records[i-1]["price_per_kg"]
                        if prev_price and record["price_per_kg"]:
                            eng_record["price_change_percent"] = (
                                (record["price_per_kg"] - prev_price) / prev_price * 100
                            )
                    
                    # Price volatility (rolling standard deviation)
                    if i >= 4:  # Need at least 5 days for volatility
                        recent_prices = [
                            r.get("price_per_kg") 
                            for r in records[max(0, i-4):i+1]
                            if r.get("price_per_kg") is not None
                        ]
                        if len(recent_prices) >= 3:
                            eng_record["price_volatility"] = np.std(recent_prices)
                
                engineered.append(eng_record)
            
            engineered_records[source] = engineered
        
        transformed_data["engineered_records"] = engineered_records
    
    def _apply_temporal_alignment(self, extracted_data: Dict[str, Any], transformed_data: Dict[str, Any]):
        """Align data temporally across sources."""
        aligned_records = {}
        
        # Get all timestamps across sources
        all_timestamps = set()
        
        for source, records in transformed_data.get("engineered_records", {}).items():
            for record in records:
                timestamp = record.get("date") or record.get("timestamp")
                if timestamp:
                    all_timestamps.add(timestamp.split("T")[0])  # Keep only date part
        
        # Sort timestamps
        sorted_timestamps = sorted(all_timestamps)
        
        # Create aligned records for each timestamp
        for timestamp in sorted_timestamps:
            aligned_record = {"timestamp": timestamp}
            
            for source, records in transformed_data.get("engineered_records", {}).items():
                # Find record for this timestamp
                source_record = next(
                    (r for r in records 
                     if (r.get("date") or r.get("timestamp", "").split("T")[0]) == timestamp),
                    None
                )
                
                if source_record:
                    # Add source-specific data with prefix
                    for key, value in source_record.items():
                        if key not in ["date", "timestamp"]:
                            aligned_record[f"{source}_{key}"] = value
            
            if len(aligned_record) > 1:  # More than just timestamp
                if "aligned_records" not in transformed_data:
                    transformed_data["aligned_records"] = []
                transformed_data["aligned_records"].append(aligned_record)
    
    def _assess_data_quality(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality for each source."""
        quality_scores = {}
        
        for source, data in extracted_data.get("data", {}).items():
            records = data.get("data", {}).get("records", [])
            metadata = data.get("metadata", {})
            
            if not records:
                quality_scores[source] = {
                    "quality_level": DataQualityLevel.UNUSABLE.value,
                    "score": 0.0,
                    "issues": ["No data available"],
                    "recommendations": ["Check data source connectivity"]
                }
                continue
            
            # Calculate quality metrics
            completeness = self._calculate_completeness(records)
            accuracy = self._calculate_accuracy(records, source)
            consistency = self._calculate_consistency(records)
            timeliness = self._calculate_timeliness(metadata)
            
            # Overall score (weighted average)
            weights = {"completeness": 0.3, "accuracy": 0.3, "consistency": 0.2, "timeliness": 0.2}
            overall_score = (
                completeness * weights["completeness"] +
                accuracy * weights["accuracy"] +
                consistency * weights["consistency"] +
                timeliness * weights["timeliness"]
            )
            
            # Determine quality level
            if overall_score >= 0.9:
                quality_level = DataQualityLevel.EXCELLENT
            elif overall_score >= 0.8:
                quality_level = DataQualityLevel.GOOD
            elif overall_score >= 0.6:
                quality_level = DataQualityLevel.FAIR
            elif overall_score >= 0.4:
                quality_level = DataQualityLevel.POOR
            else:
                quality_level = DataQualityLevel.UNUSABLE
            
            # Identify issues
            issues = []
            if completeness < 0.9:
                issues.append(f"Low completeness: {completeness:.1%}")
            if accuracy < 0.9:
                issues.append(f"Accuracy concerns: {accuracy:.1%}")
            if consistency < 0.8:
                issues.append(f"Inconsistent data: {consistency:.1%}")
            if timeliness < 0.8:
                issues.append(f"Data freshness issues: {timeliness:.1%}")
            
            # Generate recommendations
            recommendations = []
            if completeness < 0.9:
                recommendations.append("Increase data collection frequency")
            if accuracy < 0.9:
                recommendations.append("Implement data validation rules")
            if "extraction_status" in metadata and metadata["extraction_status"] != "success":
                recommendations.append("Fix data source connectivity")
            
            quality_scores[source] = {
                "quality_level": quality_level.value,
                "score": round(overall_score, 3),
                "metrics": {
                    "completeness": round(completeness, 3),
                    "accuracy": round(accuracy, 3),
                    "consistency": round(consistency, 3),
                    "timeliness": round(timeliness, 3)
                },
                "issues": issues,
                "recommendations": recommendations,
                "data_points": len(records)
            }
        
        return quality_scores
    
    def _calculate_completeness(self, records: List[Dict[str, Any]]) -> float:
        """Calculate data completeness score."""
        if not records:
            return 0.0
        
        total_fields = 0
        missing_fields = 0
        
        for record in records:
            for key, value in record.items():
                total_fields += 1
                if value is None or value == "":
                    missing_fields += 1
        
        if total_fields == 0:
            return 0.0
        
        return 1.0 - (missing_fields / total_fields)
    
    def _calculate_accuracy(self, records: List[Dict[str, Any]], source: str) -> float:
        """Calculate data accuracy score."""
        if not records:
            return 0.0
        
        # Check for obvious outliers based on source type
        source_type = self.data_sources.get(source, DataSource("unknown", "")).source_type
        
        if source_type == DataSourceType.WEATHER_API:
            return self._check_weather_accuracy(records)
        elif source_type == DataSourceType.MARKET_API:
            return self._check_market_accuracy(records)
        elif source_type == DataSourceType.IOT_DEVICE:
            return self._check_iot_accuracy(records)
        else:
            return 0.8  # Default accuracy
    
    def _check_weather_accuracy(self, records: List[Dict[str, Any]]) -> float:
        """Check weather data accuracy."""
        valid_records = 0
        
        for record in records:
            # Check temperature ranges
            temp_ok = True
            for key in ["temperature_avg", "temperature_min", "temperature_max"]:
                if key in record and record[key] is not None:
                    if not (-50 <= record[key] <= 60):
                        temp_ok = False
            
            # Check rainfall (non-negative)
            rain_ok = True
            if "rainfall_mm" in record and record["rainfall_mm"] is not None:
                if record["rainfall_mm"] < 0:
                    rain_ok = False
            
            if temp_ok and rain_ok:
                valid_records += 1
        
        return valid_records / len(records) if records else 0.0
    
    def _check_market_accuracy(self, records: List[Dict[str, Any]]) -> float:
        """Check market data accuracy."""
        valid_records = 0
        
        for record in records:
            # Check price (positive)
            price_ok = True
            if "price_per_kg" in record and record["price_per_kg"] is not None:
                if record["price_per_kg"] <= 0:
                    price_ok = False
            
            # Check volume (non-negative)
            volume_ok = True
            if "volume_traded_kg" in record and record["volume_traded_kg"] is not None:
                if record["volume_traded_kg"] < 0:
                    volume_ok = False
            
            if price_ok and volume_ok:
                valid_records += 1
        
        return valid_records / len(records) if records else 0.0
    
    def _check_iot_accuracy(self, records: List[Dict[str, Any]]) -> float:
        """Check IoT data accuracy."""
        valid_records = 0
        
        for record in records:
            sensor_type = record.get("sensor_type", "")
            value = record.get("value", 0)
            
            # Check based on sensor type
            if sensor_type == "soil_moisture" and (0 <= value <= 100):
                valid_records += 1
            elif sensor_type == "soil_ph" and (3 <= value <= 9):
                valid_records += 1
            elif sensor_type == "humidity" and (0 <= value <= 100):
                valid_records += 1
            elif sensor_type in ["soil_temperature", "air_temperature"] and (-50 <= value <= 60):
                valid_records += 1
            else:
                # Unknown sensor type or value out of range
                pass
        
        return valid_records / len(records) if records else 0.0
    
    def _calculate_consistency(self, records: List[Dict[str, Any]]) -> float:
        """Calculate data consistency score."""
        if len(records) < 2:
            return 1.0  # Can't check consistency with single record
        
        inconsistencies = 0
        total_checks = 0
        
        for i in range(1, len(records)):
            prev = records[i-1]
            curr = records[i]
            
            # Check for large jumps in numerical values
            for key in prev.keys():
                if key in curr and isinstance(prev[key], (int, float)) and isinstance(curr[key], (int, float)):
                    if prev[key] is not None and curr[key] is not None:
                        total_checks += 1
                        
                        # Calculate relative change
                        if prev[key] != 0:
                            change = abs(curr[key] - prev[key]) / abs(prev[key])
                            # Flag changes > 50% as potentially inconsistent
                            if change > 0.5:
                                inconsistencies += 1
        
        if total_checks == 0:
            return 1.0
        
        return 1.0 - (inconsistencies / total_checks)
    
    def _calculate_timeliness(self, metadata: Dict[str, Any]) -> float:
        """Calculate data timeliness score."""
        extraction_time = metadata.get("extraction_time")
        if not extraction_time:
            return 0.5  # Unknown timeliness
        
        try:
            extraction_dt = datetime.fromisoformat(extraction_time.replace("Z", "+00:00"))
            now = datetime.utcnow()
            
            # Calculate hours since extraction
            hours_since = (now - extraction_dt).total_seconds() / 3600
            
            # Score decays with time
            if hours_since <= 1:
                return 1.0
            elif hours_since <= 6:
                return 0.9
            elif hours_since <= 24:
                return 0.7
            elif hours_since <= 72:
                return 0.5
            else:
                return 0.2
        except:
            return 0.5
    
    def _generate_feature_vectors(self, transformed_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Generate feature vectors for ML models."""
        feature_vectors = {}
        
        aligned_records = transformed_data.get("aligned_records", [])
        if not aligned_records:
            return feature_vectors
        
        # Create feature vectors for different use cases
        for i, record in enumerate(aligned_records):
            vector_id = f"vector_{i}"
            features = []
            
            # Weather features
            weather_prefixes = [s for s in self.data_sources.keys() if "weather" in s]
            for prefix in weather_prefixes:
                temp_key = f"{prefix}_temperature_avg"
                rain_key = f"{prefix}_rainfall_mm"
                
                if temp_key in record:
                    features.append(record[temp_key])
                if rain_key in record:
                    features.append(record[rain_key])
            
            # Market features
            market_prefixes = [s for s in self.data_sources.keys() if "market" in s]
            for prefix in market_prefixes:
                price_key = f"{prefix}_price_per_kg"
                if price_key in record:
                    features.append(record[price_key])
            
            # IoT features
            iot_prefixes = [s for s in self.data_sources.keys() if "iot" in s]
            for prefix in iot_prefixes:
                for key, value in record.items():
                    if key.startswith(prefix) and isinstance(value, (int, float)):
                        features.append(value)
            
            if features:
                feature_vectors[vector_id] = features
        
        return feature_vectors
    
    async def load_data(self, 
                       transformed_data: Dict[str, Any],
                       destination: str = "database") -> Dict[str, Any]:
        """
        Load transformed data to destination.
        
        Args:
            transformed_data: Transformed data from transform phase
            destination: Where to load data
        
        Returns:
            Load operation results
        """
        logger.info(f"💾 Loading data to {destination}")
        
        load_results = {
            "load_id": f"load_{datetime.utcnow().timestamp()}",
            "timestamp": datetime.utcnow().isoformat(),
            "destination": destination,
            "records_loaded": 0,
            "load_status": "pending",
            "errors": []
        }
        
        try:
            if destination == "database":
                result = await self._load_to_database(transformed_data)
            elif destination == "data_lake":
                result = await self._load_to_data_lake(transformed_data)
            elif destination == "cache":
                result = await self._load_to_cache(transformed_data)
            else:
                raise ValueError(f"Unknown destination: {destination}")
            
            load_results.update(result)
            load_results["load_status"] = "success"
            
            logger.info(f"✅ Successfully loaded {load_results['records_loaded']} records to {destination}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            load_results["load_status"] = "failed"
            load_results["errors"].append(str(e))
        
        return load_results
    
    async def _load_to_database(self, transformed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load data to database."""
        # In practice, this would use SQLAlchemy or similar
        await asyncio.sleep(0.1)  # Simulate database operations
        
        aligned_records = transformed_data.get("aligned_records", [])
        
        return {
            "records_loaded": len(aligned_records),
            "database_tables": ["weather", "market", "sensor"],
            "load_method": "batch_insert",
            "data_version": hashlib.md5(
                json.dumps(aligned_records, sort_keys=True).encode()
            ).hexdigest()[:8]
        }
    
    async def _load_to_data_lake(self, transformed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load data to data lake."""
        await asyncio.sleep(0.05)  # Simulate data lake operations
        
        # Store in compressed format
        import gzip
        import json
        
        data_str = json.dumps(transformed_data)
        compressed = gzip.compress(data_str.encode())
        
        return {
            "records_loaded": len(transformed_data.get("aligned_records", [])),
            "storage_size_bytes": len(compressed),
            "compression_ratio": round(len(data_str) / len(compressed), 1),
            "data_lake_path": f"s3://agrogram-data/{datetime.utcnow().date()}/etl_output.json.gz"
        }
    
    async def _load_to_cache(self, transformed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load data to cache."""
        # Store in instance cache
        cache_key = f"etl_result_{datetime.utcnow().timestamp()}"
        self.data_cache[cache_key] = transformed_data
        
        # Keep only last 10 results
        cache_keys = [k for k in self.data_cache.keys() if k.startswith("etl_result_")]
        if len(cache_keys) > 10:
            for key in sorted(cache_keys)[:-10]:
                del self.data_cache[key]
        
        return {
            "records_loaded": len(transformed_data.get("aligned_records", [])),
            "cache_key": cache_key,
            "cache_size": len(cache_keys),
            "expiry": "manual"
        }
    
    async def run_full_pipeline(self, 
                               sources: List[str] = None,
                               transformations: List[str] = None,
                               destination: str = "database") -> Dict[str, Any]:
        """
        Run complete ETL pipeline.
        
        Args:
            sources: Data sources to extract
            transformations: Transformations to apply
            destination: Where to load data
        
        Returns:
            Complete pipeline results
        """
        logger.info("🚀 Starting full ETL pipeline")
        
        start_time = datetime.utcnow()
        
        # Extract phase
        extract_result = await self.extract_all_sources(sources, parallel=True)
        
        # Transform phase
        transform_result = self.transform_data(extract_result, transformations)
        
        # Load phase
        load_result = await self.load_data(transform_result, destination)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate pipeline metrics
        total_records = len(transform_result.get("aligned_records", []))
        quality_scores = transform_result.get("quality_scores", {})
        avg_quality = np.mean([s.get("score", 0) for s in quality_scores.values()]) if quality_scores else 0
        
        pipeline_result = {
            "pipeline_id": f"pipeline_{start_time.timestamp()}",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "phases": {
                "extract": extract_result.get("summary", {}),
                "transform": {
                    "transformations_applied": transform_result.get("transformations_applied", []),
                    "aligned_records": total_records,
                    "quality_scores": quality_scores
                },
                "load": load_result
            },
            "overall_metrics": {
                "total_records_processed": total_records,
                "average_data_quality": round(avg_quality, 3),
                "pipeline_success": load_result.get("load_status") == "success",
                "data_freshness": "fresh" if duration < 300 else "stale"
            },
            "recommendations": self._generate_pipeline_recommendations(
                extract_result, transform_result, load_result
            )
        }
        
        logger.info(f"✅ ETL pipeline completed in {duration:.1f} seconds")
        
        return pipeline_result
    
    def _generate_pipeline_recommendations(self,
                                         extract_result: Dict[str, Any],
                                         transform_result: Dict[str, Any],
                                         load_result: Dict[str, Any]) -> List[str]:
        """Generate pipeline improvement recommendations."""
        recommendations = []
        
        # Check extraction success rate
        extract_summary = extract_result.get("summary", {})
        failed_sources = extract_summary.get("failed_sources", 0)
        if failed_sources > 0:
            recommendations.append(f"Fix {failed_sources} failing data source(s)")
        
        # Check data quality
        quality_scores = transform_result.get("quality_scores", {})
        poor_quality_sources = [
            source for source, score in quality_scores.items()
            if score.get("quality_level") in ["poor", "unusable"]
        ]
        
        if poor_quality_sources:
            recommendations.append(f"Improve data quality for: {', '.join(poor_quality_sources[:3])}")
        
        # Check load performance
        if load_result.get("load_status") != "success":
            recommendations.append("Investigate data loading issues")
        
        # Check pipeline duration
        pipeline_duration = load_result.get("timestamp")  # This would need actual duration tracking
        # In practice, would check if pipeline took too long
        
        if not recommendations:
            recommendations.append("Pipeline running optimally")
        
        return recommendations
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and health."""
        source_status = {}
        for name, source in self.data_sources.items():
            source_status[name] = {
                "type": source.source_type.value,
                "last_success": source.last_successful_fetch.isoformat() if source.last_successful_fetch else None,
                "success_rate": source.success_rate,
                "refresh_rate_minutes": source.refresh_rate_minutes,
                "status": "healthy" if source.success_rate > 0.8 else "degraded"
            }
        
        # Calculate overall health
        success_rates = [s.success_rate for s in self.data_sources.values()]
        avg_success_rate = np.mean(success_rates) if success_rates else 0
        
        return {
            "status_time": datetime.utcnow().isoformat(),
            "overall_health": "healthy" if avg_success_rate > 0.9 else 
                             "degraded" if avg_success_rate > 0.7 else "unhealthy",
            "avg_success_rate": round(avg_success_rate, 3),
            "data_sources": source_status,
            "cache_size": len(self.data_cache),
            "processing_history_count": len(self.processing_history),
            "recent_errors": [
                h for h in self.processing_history[-5:] 
                if any("failed" in v for v in h.get("results", {}).values())
            ],
            "recommendations": self._generate_status_recommendations(source_status)
        }
    
    def _generate_status_recommendations(self, source_status: Dict[str, Any]) -> List[str]:
        """Generate status-based recommendations."""
        recommendations = []
        
        # Check degraded sources
        degraded_sources = [
            name for name, status in source_status.items()
            if status["status"] == "degraded"
        ]
        
        if degraded_sources:
            recommendations.append(f"Check connectivity for: {', '.join(degraded_sources)}")
        
        # Check stale data
        now = datetime.utcnow()
        stale_sources = []
        for name, status in source_status.items():
            last_success = status.get("last_success")
            if last_success:
                try:
                    last_dt = datetime.fromisoformat(last_success.replace("Z", "+00:00"))
                    hours_since = (now - last_dt).total_seconds() / 3600
                    refresh_rate = status.get("refresh_rate_minutes", 60) / 60
                    
                    if hours_since > refresh_rate * 2:  # 2x refresh rate
                        stale_sources.append(name)
                except:
                    pass
        
        if stale_sources:
            recommendations.append(f"Data stale for: {', '.join(stale_sources)}")
        
        if not recommendations:
            recommendations.append("All systems operational")
        
        return recommendations


# Create global instance
etl_pipeline = ETLPipeline()