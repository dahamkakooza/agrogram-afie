import ast
import os

files = {
    'yield_predictor': 'src/prediction/yield_predictor.py',
    'resource_orchestrator': 'src/optimization/resource_orchestrator.py',
    'risk_intelligence': 'src/prediction/risk_intelligence.py',
    'model_manager': 'src/ml/model_manager.py',
    'etl_pipeline': 'src/data/pipelines/etl_pipeline.py'
}

for name, path in files.items():
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                tree = ast.parse(f.read())
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                print(f"{name}.py: Classes found = {classes}")
        except Exception as e:
            print(f"{name}.py: Error reading - {e}")
    else:
        print(f"{name}.py: File not found")