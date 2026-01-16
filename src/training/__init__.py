"""
Training Module

This module contains training orchestration, hyperparameter tuning, and MLflow utilities
for all machine learning models in the IDC Databricks Challenge.
"""

from .trainer import ModelTrainer
from .hyperparameter_tuning import HyperparameterTuner
from .mlflow_utils import (
    log_model_metrics,
    log_feature_importance,
    log_confusion_matrix,
    register_model,
    transition_model_stage,
    get_latest_model_version,
    download_model_artifacts
)

__all__ = [
    'ModelTrainer',
    'HyperparameterTuner',
    'log_model_metrics',
    'log_feature_importance',
    'log_confusion_matrix',
    'register_model',
    'transition_model_stage',
    'get_latest_model_version',
    'download_model_artifacts',
]

__version__ = '1.0.0'
