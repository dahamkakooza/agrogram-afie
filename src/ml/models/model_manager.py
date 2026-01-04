"""
Model Manager - Week 7-8 Implementation
Machine learning model management system
"""
import json
import pickle
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_manager")


class ModelType(Enum):
    """Types of ML models."""
    YIELD_PREDICTION = "yield_prediction"
    RISK_ASSESSMENT = "risk_assessment"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    DISEASE_DETECTION = "disease_detection"
    PRICE_FORECASTING = "price_forecasting"


class ModelStatus(Enum):
    """Model statuses."""
    TRAINING = "training"
    VALIDATING = "validating"
    PRODUCTION = "production"
    STAGING = "staging"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeploymentStage(Enum):
    """Deployment stages."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ModelMetadata:
    """Metadata for a machine learning model."""
    model_id: str
    model_type: ModelType
    version: str
    created_at: datetime
    updated_at: datetime
    status: ModelStatus
    deployment_stage: DeploymentStage
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_stats: Dict[str, Any]
    dependencies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert datetime to string
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        # Convert enums to strings
        data["model_type"] = self.model_type.value
        data["status"] = self.status.value
        data["deployment_stage"] = self.deployment_stage.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        # Convert string to datetime
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        # Convert strings to enums
        data["model_type"] = ModelType(data["model_type"])
        data["status"] = ModelStatus(data["status"])
        data["deployment_stage"] = DeploymentStage(data["deployment_stage"])
        return cls(**data)


@dataclass
class ExperimentResult:
    """Result of a model experiment."""
    experiment_id: str
    model_id: str
    timestamp: datetime
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    training_time_seconds: float
    validation_scores: Dict[str, float]
    artifacts: List[str]  # Paths to saved artifacts
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class ModelManager:
    """
    Machine learning model management system.
    
    Features:
    1. Model version control and registry
    2. Performance tracking and comparison
    3. Drift detection and monitoring
    4. Experiment tracking and A/B testing
    """
    
    def __init__(self, registry_path: str = "./models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.model_registry = self._load_registry()
        self.experiment_log = []
        self.drift_detectors = {}
        self.monitoring_alerts = []
        
        logger.info(f"✅ ModelManager initialized with registry at {registry_path}")
    
    def _load_registry(self) -> Dict[str, ModelMetadata]:
        """Load model registry from disk."""
        registry_file = self.registry_path / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                registry = {}
                for model_id, model_data in data.items():
                    registry[model_id] = ModelMetadata.from_dict(model_data)
                
                logger.info(f"📚 Loaded {len(registry)} models from registry")
                return registry
            except Exception as e:
                logger.error(f"❌ Failed to load registry: {e}")
        
        return {}
    
    def _save_registry(self):
        """Save model registry to disk."""
        registry_file = self.registry_path / "registry.json"
        
        try:
            registry_data = {
                model_id: metadata.to_dict()
                for model_id, metadata in self.model_registry.items()
            }
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info(f"💾 Saved {len(registry_data)} models to registry")
        except Exception as e:
            logger.error(f"❌ Failed to save registry: {e}")
    
    def register_model(self,
                      model_type: ModelType,
                      model_object: Any,
                      performance_metrics: Dict[str, float],
                      feature_importance: Dict[str, float],
                      hyperparameters: Dict[str, Any],
                      training_data_stats: Dict[str, Any],
                      version: str = None,
                      dependencies: List[str] = None) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model_type: Type of model
            model_object: The actual model object
            performance_metrics: Model performance metrics
            feature_importance: Feature importance scores
            hyperparameters: Model hyperparameters
            training_data_stats: Statistics about training data
            version: Model version (auto-generated if None)
            dependencies: List of dependencies
        
        Returns:
            Model ID
        """
        logger.info(f"📝 Registering new {model_type.value} model")
        
        # Generate model ID and version
        model_id = self._generate_model_id(model_type)
        if version is None:
            version = self._generate_version(model_type)
        
        # Create metadata
        now = datetime.utcnow()
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            version=version,
            created_at=now,
            updated_at=now,
            status=ModelStatus.VALIDATING,
            deployment_stage=DeploymentStage.DEVELOPMENT,
            performance_metrics=performance_metrics,
            feature_importance=feature_importance,
            hyperparameters=hyperparameters,
            training_data_stats=training_data_stats,
            dependencies=dependencies or []
        )
        
        # Save model object
        self._save_model_object(model_id, model_object)
        
        # Save metadata
        self.model_registry[model_id] = metadata
        self._save_registry()
        
        # Log registration
        logger.info(f"✅ Registered model {model_id} version {version}")
        
        return model_id
    
    def _generate_model_id(self, model_type: ModelType) -> str:
        """Generate unique model ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_id = f"{model_type.value}_{timestamp}"
        
        # Add random suffix to ensure uniqueness
        import random
        import string
        suffix = ''.join(random.choices(string.digits, k=4))
        
        return f"{base_id}_{suffix}"
    
    def _generate_version(self, model_type: ModelType) -> str:
        """Generate version number."""
        # Count existing models of this type
        same_type_models = [
            m for m in self.model_registry.values()
            if m.model_type == model_type
        ]
        
        major = 1
        minor = len(same_type_models) + 1
        patch = 0
        
        return f"{major}.{minor}.{patch}"
    
    def _save_model_object(self, model_id: str, model_object: Any):
        """Save model object to disk."""
        model_dir = self.registry_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        model_file = model_dir / "model.pkl"
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(model_object, f)
            
            logger.info(f"💾 Saved model object for {model_id}")
        except Exception as e:
            logger.error(f"❌ Failed to save model object for {model_id}: {e}")
            raise
    
    def get_model(self, model_id: str) -> Tuple[Optional[Any], Optional[ModelMetadata]]:
        """
        Get model object and metadata.
        
        Args:
            model_id: Model ID
        
        Returns:
            Tuple of (model_object, metadata)
        """
        if model_id not in self.model_registry:
            logger.warning(f"Model {model_id} not found in registry")
            return None, None
        
        metadata = self.model_registry[model_id]
        
        # Load model object
        model_dir = self.registry_path / model_id
        model_file = model_dir / "model.pkl"
        
        if not model_file.exists():
            logger.error(f"Model file not found for {model_id}")
            return None, metadata
        
        try:
            with open(model_file, 'rb') as f:
                model_object = pickle.load(f)
            
            logger.info(f"📂 Loaded model {model_id}")
            return model_object, metadata
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_id}: {e}")
            return None, metadata
    
    def update_model_status(self, 
                           model_id: str,
                           status: ModelStatus,
                           deployment_stage: DeploymentStage = None):
        """
        Update model status.
        
        Args:
            model_id: Model ID
            status: New status
            deployment_stage: New deployment stage (optional)
        """
        if model_id not in self.model_registry:
            logger.warning(f"Cannot update status: Model {model_id} not found")
            return
        
        metadata = self.model_registry[model_id]
        metadata.status = status
        metadata.updated_at = datetime.utcnow()
        
        if deployment_stage:
            metadata.deployment_stage = deployment_stage
        
        self._save_registry()
        logger.info(f"🔄 Updated model {model_id} status to {status.value}")
    
    def update_performance(self,
                          model_id: str,
                          new_metrics: Dict[str, float],
                          feature_importance: Dict[str, float] = None):
        """
        Update model performance metrics.
        
        Args:
            model_id: Model ID
            new_metrics: New performance metrics
            feature_importance: Updated feature importance (optional)
        """
        if model_id not in self.model_registry:
            logger.warning(f"Cannot update performance: Model {model_id} not found")
            return
        
        metadata = self.model_registry[model_id]
        metadata.performance_metrics.update(new_metrics)
        metadata.updated_at = datetime.utcnow()
        
        if feature_importance:
            metadata.feature_importance.update(feature_importance)
        
        self._save_registry()
        logger.info(f"📊 Updated performance metrics for model {model_id}")
    
    def compare_models(self, 
                      model_ids: List[str],
                      metric: str = "accuracy") -> List[Dict[str, Any]]:
        """
        Compare multiple models.
        
        Args:
            model_ids: List of model IDs to compare
            metric: Metric to compare
        
        Returns:
            Comparison results
        """
        comparison = []
        
        for model_id in model_ids:
            if model_id in self.model_registry:
                metadata = self.model_registry[model_id]
                score = metadata.performance_metrics.get(metric, 0)
                
                comparison.append({
                    "model_id": model_id,
                    "model_type": metadata.model_type.value,
                    "version": metadata.version,
                    "status": metadata.status.value,
                    "deployment_stage": metadata.deployment_stage.value,
                    "metric": metric,
                    "score": score,
                    "created_at": metadata.created_at.isoformat(),
                    "training_data_size": metadata.training_data_stats.get("samples", 0)
                })
        
        # Sort by metric score descending
        comparison.sort(key=lambda x: x["score"], reverse=True)
        
        return comparison
    
    def promote_model(self, 
                     model_id: str,
                     target_stage: DeploymentStage) -> bool:
        """
        Promote model to a new deployment stage.
        
        Args:
            model_id: Model ID
            target_stage: Target deployment stage
        
        Returns:
            Success status
        """
        if model_id not in self.model_registry:
            logger.warning(f"Cannot promote: Model {model_id} not found")
            return False
        
        metadata = self.model_registry[model_id]
        current_stage = metadata.deployment_stage
        
        # Check if promotion is valid
        valid_promotions = {
            DeploymentStage.DEVELOPMENT: [DeploymentStage.TESTING],
            DeploymentStage.TESTING: [DeploymentStage.STAGING],
            DeploymentStage.STAGING: [DeploymentStage.PRODUCTION],
            DeploymentStage.PRODUCTION: []  # Can't promote from production
        }
        
        if target_stage not in valid_promotions.get(current_stage, []):
            logger.warning(
                f"Cannot promote from {current_stage.value} to {target_stage.value}"
            )
            return False
        
        # Check performance requirements
        if target_stage == DeploymentStage.PRODUCTION:
            # Production models must have good performance
            accuracy = metadata.performance_metrics.get("accuracy", 0)
            if accuracy < 0.8:  # Example threshold
                logger.warning(
                    f"Model accuracy {accuracy} too low for production"
                )
                return False
        
        # Update deployment stage
        metadata.deployment_stage = target_stage
        metadata.updated_at = datetime.utcnow()
        
        # Update status based on stage
        if target_stage == DeploymentStage.PRODUCTION:
            metadata.status = ModelStatus.PRODUCTION
        elif target_stage == DeploymentStage.STAGING:
            metadata.status = ModelStatus.STAGING
        
        self._save_registry()
        
        logger.info(
            f"🚀 Promoted model {model_id} from {current_stage.value} to {target_stage.value}"
        )
        
        return True
    
    def log_experiment(self,
                      model_id: str,
                      hyperparameters: Dict[str, Any],
                      performance_metrics: Dict[str, float],
                      feature_importance: Dict[str, float],
                      training_time_seconds: float,
                      validation_scores: Dict[str, float],
                      artifacts: List[str],
                      notes: str = "") -> str:
        """
        Log an experiment result.
        
        Args:
            model_id: Model ID
            hyperparameters: Hyperparameters used
            performance_metrics: Performance metrics
            feature_importance: Feature importance
            training_time_seconds: Training time in seconds
            validation_scores: Validation scores
            artifacts: Paths to saved artifacts
            notes: Experiment notes
        
        Returns:
            Experiment ID
        """
        experiment_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        experiment = ExperimentResult(
            experiment_id=experiment_id,
            model_id=model_id,
            timestamp=datetime.utcnow(),
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            feature_importance=feature_importance,
            training_time_seconds=training_time_seconds,
            validation_scores=validation_scores,
            artifacts=artifacts,
            notes=notes
        )
        
        self.experiment_log.append(experiment)
        
        # Save experiment log
        self._save_experiment_log()
        
        logger.info(f"📝 Logged experiment {experiment_id} for model {model_id}")
        
        return experiment_id
    
    def _save_experiment_log(self):
        """Save experiment log to disk."""
        log_file = self.registry_path / "experiments.json"
        
        try:
            log_data = [exp.to_dict() for exp in self.experiment_log]
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save experiment log: {e}")
    
    def get_experiment_history(self,
                              model_id: str = None,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get experiment history.
        
        Args:
            model_id: Filter by model ID (optional)
            limit: Maximum number of experiments to return
        
        Returns:
            List of experiment results
        """
        if model_id:
            experiments = [
                exp for exp in self.experiment_log
                if exp.model_id == model_id
            ]
        else:
            experiments = self.experiment_log
        
        # Sort by timestamp descending
        experiments.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Convert to dict and limit results
        results = []
        for exp in experiments[:limit]:
            exp_dict = exp.to_dict()
            results.append(exp_dict)
        
        return results
    
    def detect_drift(self,
                    model_id: str,
                    current_data: pd.DataFrame,
                    reference_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Detect data drift for a model.
        
        Args:
            model_id: Model ID
            current_data: Current production data
            reference_data: Reference/training data (optional)
        
        Returns:
            Drift detection results
        """
        logger.info(f"🔍 Detecting drift for model {model_id}")
        
        if model_id not in self.model_registry:
            logger.warning(f"Cannot detect drift: Model {model_id} not found")
            return {"error": "Model not found"}
        
        metadata = self.model_registry[model_id]
        
        # Get reference data from training stats if not provided
        if reference_data is None:
            reference_stats = metadata.training_data_stats
            # In practice, would load actual training data
            reference_data = self._simulate_reference_data(reference_stats)
        
        # Calculate drift metrics
        drift_metrics = self._calculate_drift_metrics(
            reference_data, current_data
        )
        
        # Check for significant drift
        significant_drift = self._check_significant_drift(drift_metrics)
        
        # Store drift detection
        drift_id = f"drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        drift_record = {
            "drift_id": drift_id,
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat(),
            "drift_metrics": drift_metrics,
            "significant_drift": significant_drift,
            "drift_alert": significant_drift["alert_triggered"]
        }
        
        # Update model metadata with drift info
        if "drift_history" not in metadata.training_data_stats:
            metadata.training_data_stats["drift_history"] = []
        
        metadata.training_data_stats["drift_history"].append(drift_record)
        metadata.updated_at = datetime.utcnow()
        
        self._save_registry()
        
        # Trigger alert if needed
        if significant_drift["alert_triggered"]:
            self._trigger_drift_alert(model_id, drift_metrics, significant_drift)
        
        return drift_record
    
    def _simulate_reference_data(self, training_stats: Dict[str, Any]) -> pd.DataFrame:
        """Simulate reference data from training statistics."""
        # This is a simplified simulation
        # In practice, would load actual training data
        
        n_samples = training_stats.get("samples", 1000)
        n_features = training_stats.get("features", 10)
        
        # Generate synthetic data with known distribution
        np.random.seed(42)
        data = np.random.normal(
            loc=0,
            scale=1,
            size=(n_samples, n_features)
        )
        
        columns = [f"feature_{i}" for i in range(n_features)]
        return pd.DataFrame(data, columns=columns)
    
    def _calculate_drift_metrics(self,
                                reference_data: pd.DataFrame,
                                current_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate drift metrics between reference and current data."""
        metrics = {}
        
        # For each numeric column, calculate distribution differences
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in current_data.columns:
                ref_mean = reference_data[col].mean()
                ref_std = reference_data[col].std()
                curr_mean = current_data[col].mean()
                curr_std = current_data[col].std()
                
                # Calculate various drift metrics
                metrics[f"{col}_mean_drift"] = abs(curr_mean - ref_mean) / (ref_std + 1e-10)
                metrics[f"{col}_std_drift"] = abs(curr_std - ref_std) / (ref_std + 1e-10)
                
                # Kolmogorov-Smirnov test statistic (simplified)
                # In practice, would use scipy.stats.ks_2samp
                ref_sorted = np.sort(reference_data[col].dropna())
                curr_sorted = np.sort(current_data[col].dropna())
                
                # Simplified KS statistic
                n_ref = len(ref_sorted)
                n_curr = len(curr_sorted)
                all_values = np.concatenate([ref_sorted, curr_sorted])
                cdf_ref = np.searchsorted(ref_sorted, all_values, side='right') / n_ref
                cdf_curr = np.searchsorted(curr_sorted, all_values, side='right') / n_curr
                ks_stat = np.max(np.abs(cdf_ref - cdf_curr))
                
                metrics[f"{col}_ks_statistic"] = ks_stat
        
        # Overall drift scores
        if metrics:
            metrics["mean_drift_score"] = np.mean([
                v for k, v in metrics.items() if "mean_drift" in k
            ])
            metrics["ks_drift_score"] = np.mean([
                v for k, v in metrics.items() if "ks_statistic" in k
            ])
            metrics["overall_drift_score"] = (
                metrics["mean_drift_score"] + metrics["ks_drift_score"]
            ) / 2
        
        return metrics
    
    def _check_significant_drift(self, drift_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if drift is significant."""
        overall_score = drift_metrics.get("overall_drift_score", 0)
        mean_drift = drift_metrics.get("mean_drift_score", 0)
        ks_drift = drift_metrics.get("ks_drift_score", 0)
        
        # Define thresholds
        alert_threshold = 0.3
        warning_threshold = 0.2
        
        if overall_score > alert_threshold:
            severity = "high"
            alert_triggered = True
            message = f"Significant data drift detected (score: {overall_score:.3f})"
        elif overall_score > warning_threshold:
            severity = "medium"
            alert_triggered = True
            message = f"Moderate data drift detected (score: {overall_score:.3f})"
        else:
            severity = "low"
            alert_triggered = False
            message = "No significant data drift"
        
        return {
            "severity": severity,
            "alert_triggered": alert_triggered,
            "message": message,
            "overall_score": overall_score,
            "mean_drift_score": mean_drift,
            "ks_drift_score": ks_drift,
            "thresholds": {
                "alert": alert_threshold,
                "warning": warning_threshold
            }
        }
    
    def _trigger_drift_alert(self,
                            model_id: str,
                            drift_metrics: Dict[str, float],
                            significance: Dict[str, Any]):
        """Trigger drift alert."""
        alert = {
            "alert_id": f"drift_alert_{datetime.utcnow().timestamp()}",
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": significance["severity"],
            "message": significance["message"],
            "drift_score": significance["overall_score"],
            "drift_metrics": {k: v for k, v in drift_metrics.items() 
                             if not k.endswith("_score")},
            "recommended_actions": [
                "Review recent data quality",
                "Check for data pipeline issues",
                "Consider model retraining"
            ]
        }
        
        self.monitoring_alerts.append(alert)
        logger.warning(f"🚨 Drift alert for model {model_id}: {significance['message']}")
    
    def monitor_performance(self,
                          model_id: str,
                          new_predictions: List[float],
                          ground_truth: List[float]) -> Dict[str, Any]:
        """
        Monitor model performance degradation.
        
        Args:
            model_id: Model ID
            new_predictions: New predictions
            ground_truth: Ground truth labels
        
        Returns:
            Performance monitoring results
        """
        logger.info(f"📊 Monitoring performance for model {model_id}")
        
        if model_id not in self.model_registry:
            logger.warning(f"Cannot monitor: Model {model_id} not found")
            return {"error": "Model not found"}
        
        metadata = self.model_registry[model_id]
        baseline_metrics = metadata.performance_metrics
        
        # Calculate new performance metrics
        if len(new_predictions) != len(ground_truth):
            logger.error("Predictions and ground truth length mismatch")
            return {"error": "Length mismatch"}
        
        # Simplified metric calculation
        # In practice, would use proper metric functions
        accuracy = np.mean(np.array(new_predictions) == np.array(ground_truth))
        
        # Compare with baseline
        baseline_accuracy = baseline_metrics.get("accuracy", 0)
        accuracy_degradation = baseline_accuracy - accuracy
        
        # Check for significant degradation
        degradation_threshold = 0.05  # 5% degradation
        
        if accuracy_degradation > degradation_threshold:
            severity = "high"
            alert_triggered = True
            message = f"Significant performance degradation: {accuracy_degradation:.3f}"
        elif accuracy_degradation > degradation_threshold / 2:
            severity = "medium"
            alert_triggered = True
            message = f"Moderate performance degradation: {accuracy_degradation:.3f}"
        else:
            severity = "low"
            alert_triggered = False
            message = "Performance stable"
        
        # Create monitoring record
        monitor_id = f"monitor_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        monitor_record = {
            "monitor_id": monitor_id,
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat(),
            "current_accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_degradation": accuracy_degradation,
            "samples_processed": len(new_predictions),
            "performance_alert": {
                "severity": severity,
                "alert_triggered": alert_triggered,
                "message": message,
                "threshold": degradation_threshold
            }
        }
        
        # Update model metadata
        if "performance_history" not in metadata.training_data_stats:
            metadata.training_data_stats["performance_history"] = []
        
        metadata.training_data_stats["performance_history"].append(monitor_record)
        metadata.updated_at = datetime.utcnow()
        
        # Update performance metrics if degradation is small
        if not alert_triggered:
            metadata.performance_metrics["accuracy"] = (
                baseline_accuracy * 0.7 + accuracy * 0.3  # Exponential moving average
            )
        
        self._save_registry()
        
        # Trigger alert if needed
        if alert_triggered:
            self._trigger_performance_alert(model_id, monitor_record)
        
        return monitor_record
    
    def _trigger_performance_alert(self,
                                 model_id: str,
                                 monitor_record: Dict[str, Any]):
        """Trigger performance degradation alert."""
        alert = monitor_record["performance_alert"]
        
        performance_alert = {
            "alert_id": f"perf_alert_{datetime.utcnow().timestamp()}",
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": alert["severity"],
            "message": alert["message"],
            "current_accuracy": monitor_record["current_accuracy"],
            "baseline_accuracy": monitor_record["baseline_accuracy"],
            "degradation": monitor_record["accuracy_degradation"],
            "recommended_actions": [
                "Verify ground truth data quality",
                "Check for concept drift",
                "Schedule model retraining"
            ]
        }
        
        self.monitoring_alerts.append(performance_alert)
        logger.warning(f"⚠️ Performance alert for model {model_id}: {alert['message']}")
    
    def get_retraining_recommendations(self, model_id: str) -> Dict[str, Any]:
        """
        Get retraining recommendations for a model.
        
        Args:
            model_id: Model ID
        
        Returns:
            Retraining recommendations
        """
        if model_id not in self.model_registry:
            logger.warning(f"Cannot get recommendations: Model {model_id} not found")
            return {"error": "Model not found"}
        
        metadata = self.model_registry[model_id]
        training_stats = metadata.training_data_stats
        
        recommendations = {
            "model_id": model_id,
            "model_type": metadata.model_type.value,
            "current_version": metadata.version,
            "recommendations": [],
            "urgency": "low",
            "estimated_retraining_time": "1-2 hours"
        }
        
        # Check drift history
        drift_history = training_stats.get("drift_history", [])
        if drift_history:
            recent_drift = drift_history[-1]
            if recent_drift.get("significant_drift", {}).get("alert_triggered", False):
                recommendations["recommendations"].append(
                    "Data drift detected - retraining recommended"
                )
                recommendations["urgency"] = "high"
        
        # Check performance history
        perf_history = training_stats.get("performance_history", [])
        if perf_history:
            recent_perf = perf_history[-1]
            if recent_perf.get("performance_alert", {}).get("alert_triggered", False):
                recommendations["recommendations"].append(
                    "Performance degradation detected - retraining recommended"
                )
                recommendations["urgency"] = "high"
        
        # Check model age
        model_age_days = (datetime.utcnow() - metadata.created_at).days
        if model_age_days > 90:  # 3 months
            recommendations["recommendations"].append(
                f"Model is {model_age_days} days old - consider retraining with new data"
            )
            if recommendations["urgency"] == "low":
                recommendations["urgency"] = "medium"
        
        # Check data size increase
        original_samples = training_stats.get("samples", 0)
        # In practice, would compare with current data volume
        if original_samples > 0:
            # Simplified: assume data has doubled
            if False:  # Placeholder for actual check
                recommendations["recommendations"].append(
                    "Training data has significantly increased - retraining recommended"
                )
        
        # If no specific issues, suggest periodic retraining
        if not recommendations["recommendations"]:
            recommendations["recommendations"].append(
                "No urgent issues detected. Schedule periodic retraining every 3 months."
            )
        
        return recommendations
    
    def setup_ab_test(self,
                     model_a_id: str,
                     model_b_id: str,
                     test_parameters: Dict[str, Any]) -> str:
        """
        Set up A/B test between two models.
        
        Args:
            model_a_id: First model ID
            model_b_id: Second model ID
            test_parameters: Test configuration
        
        Returns:
            Test ID
        """
        logger.info(f"🧪 Setting up A/B test between {model_a_id} and {model_b_id}")
        
        # Validate models exist
        if model_a_id not in self.model_registry:
            raise ValueError(f"Model A {model_a_id} not found")
        if model_b_id not in self.model_registry:
            raise ValueError(f"Model B {model_b_id} not found")
        
        # Generate test ID
        test_id = f"ab_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create test configuration
        test_config = {
            "test_id": test_id,
            "model_a": model_a_id,
            "model_b": model_b_id,
            "parameters": test_parameters,
            "start_time": datetime.utcnow().isoformat(),
            "status": "active",
            "traffic_split": test_parameters.get("traffic_split", {"a": 0.5, "b": 0.5}),
            "success_metric": test_parameters.get("success_metric", "accuracy"),
            "minimum_sample_size": test_parameters.get("minimum_sample_size", 1000),
            "significance_level": test_parameters.get("significance_level", 0.05),
            "results": {
                "samples_processed": {"a": 0, "b": 0},
                "success_counts": {"a": 0, "b": 0},
                "confidence_intervals": {"a": None, "b": None},
                "p_value": None,
                "winner": None
            }
        }
        
        # Save test configuration
        tests_dir = self.registry_path / "ab_tests"
        tests_dir.mkdir(exist_ok=True)
        
        test_file = tests_dir / f"{test_id}.json"
        with open(test_file, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        logger.info(f"✅ A/B test {test_id} setup complete")
        
        return test_id
    
    def update_ab_test(self,
                      test_id: str,
                      model_variant: str,
                      success_count: int,
                      total_count: int) -> Dict[str, Any]:
        """
        Update A/B test with new results.
        
        Args:
            test_id: Test ID
            model_variant: 'a' or 'b'
            success_count: Number of successes
            total_count: Total samples processed
        
        Returns:
            Updated test results
        """
        # Load test configuration
        tests_dir = self.registry_path / "ab_tests"
        test_file = tests_dir / f"{test_id}.json"
        
        if not test_file.exists():
            raise ValueError(f"A/B test {test_id} not found")
        
        with open(test_file, 'r') as f:
            test_config = json.load(f)
        
        # Update results
        test_config["results"]["samples_processed"][model_variant] += total_count
        test_config["results"]["success_counts"][model_variant] += success_count
        
        # Calculate current success rates
        success_a = test_config["results"]["success_counts"]["a"]
        total_a = test_config["results"]["samples_processed"]["a"]
        success_b = test_config["results"]["success_counts"]["b"]
        total_b = test_config["results"]["samples_processed"]["b"]
        
        rate_a = success_a / total_a if total_a > 0 else 0
        rate_b = success_b / total_b if total_b > 0 else 0
        
        # Calculate confidence intervals (simplified)
        # In practice, would use proper statistical methods
        ci_a = self._calculate_confidence_interval(success_a, total_a)
        ci_b = self._calculate_confidence_interval(success_b, total_b)
        
        test_config["results"]["confidence_intervals"]["a"] = ci_a
        test_config["results"]["confidence_intervals"]["b"] = ci_b
        
        # Check if test is complete
        min_samples = test_config["minimum_sample_size"]
        if total_a >= min_samples and total_b >= min_samples:
            # Calculate p-value (simplified)
            # In practice, would use chi-squared or t-test
            p_value = self._calculate_ab_p_value(success_a, total_a, success_b, total_b)
            test_config["results"]["p_value"] = p_value
            
            # Determine winner
            significance_level = test_config["significance_level"]
            if p_value < significance_level:
                if rate_a > rate_b:
                    winner = "a"
                else:
                    winner = "b"
            else:
                winner = "inconclusive"
            
            test_config["results"]["winner"] = winner
            test_config["status"] = "completed"
            
            logger.info(f"🏆 A/B test {test_id} completed. Winner: {winner}")
        
        # Save updated configuration
        with open(test_file, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        return test_config
    
    def _calculate_confidence_interval(self, success: int, total: int) -> Dict[str, float]:
        """Calculate confidence interval for success rate."""
        if total == 0:
            return {"lower": 0, "upper": 0, "point": 0}
        
        p = success / total
        z = 1.96  # 95% confidence
        
        margin = z * np.sqrt(p * (1 - p) / total)
        
        return {
            "lower": max(0, p - margin),
            "upper": min(1, p + margin),
            "point": p
        }
    
    def _calculate_ab_p_value(self, 
                             success_a: int, total_a: int,
                             success_b: int, total_b: int) -> float:
        """Calculate p-value for A/B test (simplified)."""
        if total_a == 0 or total_b == 0:
            return 1.0
        
        p_a = success_a / total_a
        p_b = success_b / total_b
        
        # Simplified p-value calculation
        # In practice, would use proper statistical test
        diff = abs(p_a - p_b)
        pooled_se = np.sqrt(p_a * (1 - p_a) / total_a + p_b * (1 - p_b) / total_b)
        
        if pooled_se == 0:
            return 1.0
        
        z_score = diff / pooled_se
        # Simplified: convert z-score to p-value
        p_value = max(0.001, min(0.5, 0.5 - abs(z_score) * 0.1))
        
        return p_value
    
    def get_model_insights(self, model_id: str) -> Dict[str, Any]:
        """
        Get comprehensive insights for a model.
        
        Args:
            model_id: Model ID
        
        Returns:
            Model insights
        """
        if model_id not in self.model_registry:
            logger.warning(f"Cannot get insights: Model {model_id} not found")
            return {"error": "Model not found"}
        
        metadata = self.model_registry[model_id]
        training_stats = metadata.training_data_stats
        
        insights = {
            "model_id": model_id,
            "model_type": metadata.model_type.value,
            "version": metadata.version,
            "status": metadata.status.value,
            "deployment_stage": metadata.deployment_stage.value,
            "created": metadata.created_at.isoformat(),
            "last_updated": metadata.updated_at.isoformat(),
            "performance": metadata.performance_metrics,
            "training_data": {
                "samples": training_stats.get("samples", 0),
                "features": training_stats.get("features", 0),
                "data_quality": training_stats.get("data_quality", "unknown")
            },
            "feature_importance": metadata.feature_importance,
            "hyperparameters": metadata.hyperparameters,
            "dependencies": metadata.dependencies,
            "health_indicators": self._calculate_model_health(metadata, training_stats),
            "recommendations": self.get_retraining_recommendations(model_id)["recommendations"]
        }
        
        return insights
    
    def _calculate_model_health(self, 
                              metadata: ModelMetadata,
                              training_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate model health indicators."""
        health = {}
        
        # Performance health
        accuracy = metadata.performance_metrics.get("accuracy", 0)
        if accuracy >= 0.9:
            health["performance"] = "excellent"
        elif accuracy >= 0.8:
            health["performance"] = "good"
        elif accuracy >= 0.7:
            health["performance"] = "fair"
        else:
            health["performance"] = "poor"
        
        # Data health
        drift_history = training_stats.get("drift_history", [])
        if drift_history:
            recent_drift = drift_history[-1]
            if recent_drift.get("significant_drift", {}).get("alert_triggered", False):
                health["data_stability"] = "unstable"
            else:
                health["data_stability"] = "stable"
        else:
            health["data_stability"] = "unknown"
        
        # Model age health
        model_age_days = (datetime.utcnow() - metadata.created_at).days
        if model_age_days <= 30:
            health["freshness"] = "fresh"
        elif model_age_days <= 90:
            health["freshness"] = "aging"
        else:
            health["freshness"] = "stale"
        
        # Overall health
        if (health.get("performance") == "excellent" and 
            health.get("data_stability") == "stable" and
            health.get("freshness") in ["fresh", "aging"]):
            health["overall"] = "healthy"
        elif (health.get("performance") in ["good", "fair"] and
              health.get("data_stability") in ["stable", "unknown"]):
            health["overall"] = "degraded"
        else:
            health["overall"] = "unhealthy"
        
        return health


# Create global instance
model_manager = ModelManager()