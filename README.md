# aws-mlflow-mlops

End-to-end MLOps project with deployment with AWS, MLflow, and GitHub Actions.

## Workflow

1. Update config/config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update entity at `src/aws_mlflow_mlops/entity/config_entity.py`
5. Update configuration manager at `src/aws_ml_flow_mlops/config/configuration.py`
6. Update components at `src/aws_mlflow_mlops/components/__init__.py`
7. Update pipeline at `src/aws_mlflow_mlops/pipleine/__init__.py`
8. Update `main.py`
9. Update `app.py`
