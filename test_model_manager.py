# Save as /d/agrogram-afie/test_model_manager_fixed.py
import sys
import os

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath('.'))

print("Testing Model Manager with fixed pickling...")

# Define DummyModel at module level (not inside function)
class DummyModel:
    def predict(self, X):
        import numpy as np
        return np.random.rand(len(X))

def test_model_manager():
    """Fixed test with properly defined DummyModel."""
    
    try:
        from src.ml.model_manager import ModelManager, ModelType, ModelStatus, DeploymentStage
        
        print("[OK] Imports successful")
        
        # Initialize model manager with test path
        manager = ModelManager(registry_path="./test_models_fixed")
        
        # Register a model
        print("\n1. Testing model registration...")
        model_id = manager.register_model(
            model_type=ModelType.YIELD_PREDICTION,
            model_object=DummyModel(),  # Now this is picklable
            performance_metrics={
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.87,
                "f1_score": 0.84
            },
            feature_importance={
                "soil_ph": 0.15,
                "rainfall": 0.25,
                "temperature": 0.20
            },
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1
            },
            training_data_stats={
                "samples": 1000,
                "features": 15,
                "data_quality": "good"
            },
            dependencies=["scikit-learn==1.3.0", "xgboost==2.0.0"]
        )
        
        print(f"[OK] Model registered: {model_id}")
        
        # Test retrieval
        print("\n2. Testing model retrieval...")
        model_object, metadata = manager.get_model(model_id)
        
        if model_object and metadata:
            print(f"[OK] Model retrieved successfully")
            print(f"[INFO] Model type: {metadata.model_type.value}")
            print(f"[INFO] Accuracy: {metadata.performance_metrics.get('accuracy', 0):.3f}")
        else:
            print(f"[ERROR] Failed to retrieve model")
            return False
        
        # Test basic functionality
        print("\n3. Testing basic operations...")
        
        # Update status
        manager.update_model_status(
            model_id=model_id,
            status=ModelStatus.PRODUCTION,
            deployment_stage=DeploymentStage.STAGING
        )
        print("[OK] Status updated")
        
        # Compare models (register another one)
        model_id_2 = manager.register_model(
            model_type=ModelType.YIELD_PREDICTION,
            model_object=DummyModel(),
            performance_metrics={
                "accuracy": 0.82,
                "precision": 0.80,
                "recall": 0.83
            },
            feature_importance={
                "soil_ph": 0.14,
                "rainfall": 0.24
            },
            hyperparameters={
                "n_estimators": 150,
                "max_depth": 7
            },
            training_data_stats={
                "samples": 1200,
                "features": 15
            }
        )
        
        comparison = manager.compare_models(
            model_ids=[model_id, model_id_2],
            metric="accuracy"
        )
        
        print(f"[OK] Models compared: {len(comparison)}")
        
        # Clean up test directory
        import shutil
        if os.path.exists("./test_models_fixed"):
            shutil.rmtree("./test_models_fixed")
            print("[INFO] Cleaned up test directory")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_manager()
    if success:
        print("\n✅ Model Manager test passed!")
    else:
        print("\n❌ Model Manager test failed.")
        sys.exit(1)