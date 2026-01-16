"""
MLflow Utilities Module

This module provides utility functions for MLflow operations including:
- Model metrics logging
- Feature importance logging
- Confusion matrix visualization
- Model registration and versioning
- Model artifact management
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import os
import tempfile
from pathlib import Path
import mlflow
import mlflow.spark
import mlflow.sklearn
import mlflow.keras
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_model_metrics(metrics: Dict[str, float],
                     prefix: Optional[str] = None,
                     step: Optional[int] = None) -> None:
    """
    Log multiple metrics to MLflow.
    
    Args:
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for metric names (e.g., 'train', 'val', 'test')
        step: Optional step number for tracking metrics over time
        
    Example:
        log_model_metrics({'accuracy': 0.95, 'f1': 0.92}, prefix='val')
        # Logs as 'val_accuracy' and 'val_f1'
    """
    if not metrics:
        logger.warning("No metrics provided to log")
        return
    
    try:
        for metric_name, metric_value in metrics.items():
            full_name = f"{prefix}_{metric_name}" if prefix else metric_name
            
            if step is not None:
                mlflow.log_metric(full_name, metric_value, step=step)
            else:
                mlflow.log_metric(full_name, metric_value)
        
        logger.info(f"Logged {len(metrics)} metrics with prefix '{prefix}'")
        
    except Exception as e:
        logger.error(f"Failed to log metrics: {str(e)}")
        raise


def log_feature_importance(feature_importance: Dict[str, float],
                          top_n: int = 20,
                          create_plot: bool = True,
                          plot_path: Optional[str] = None) -> None:
    """
    Log feature importance to MLflow with optional visualization.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        top_n: Number of top features to log and plot
        create_plot: Whether to create a bar plot
        plot_path: Path to save plot (temporary file if None)
        
    Example:
        importance = {'age': 0.35, 'income': 0.25, 'tenure': 0.20}
        log_feature_importance(importance, top_n=10)
    """
    if not feature_importance:
        logger.warning("No feature importance data provided")
        return
    
    try:
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        # Log as metrics
        for idx, (feature, importance) in enumerate(sorted_features):
            mlflow.log_metric(f"importance_{idx+1}", importance)
            mlflow.log_param(f"top_feature_{idx+1}", feature)
        
        # Log as dictionary artifact
        import json
        importance_dict = dict(sorted_features)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(importance_dict, f, indent=2)
            temp_json = f.name
        
        mlflow.log_artifact(temp_json, "feature_importance")
        os.remove(temp_json)
        
        # Create plot if requested
        if create_plot:
            try:
                import matplotlib.pyplot as plt
                
                features = [f[0] for f in sorted_features]
                importances = [f[1] for f in sorted_features]
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(features)), importances, align='center')
                plt.yticks(range(len(features)), features)
                plt.xlabel('Importance Score')
                plt.ylabel('Feature')
                plt.title(f'Top {top_n} Feature Importances')
                plt.tight_layout()
                
                # Save plot
                if plot_path is None:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.png', delete=False) as f:
                        plot_path = f.name
                    should_remove = True
                else:
                    should_remove = False
                
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(plot_path, "plots")
                
                if should_remove:
                    os.remove(plot_path)
                
                plt.close()
                logger.info("Feature importance plot created and logged")
                
            except ImportError:
                logger.warning("Matplotlib not available for plotting")
            except Exception as e:
                logger.error(f"Failed to create plot: {str(e)}")
        
        logger.info(f"Logged top {top_n} feature importances")
        
    except Exception as e:
        logger.error(f"Failed to log feature importance: {str(e)}")
        raise


def log_confusion_matrix(y_true: Union[np.ndarray, List],
                        y_pred: Union[np.ndarray, List],
                        labels: Optional[List[str]] = None,
                        normalize: bool = False,
                        plot_path: Optional[str] = None) -> None:
    """
    Log confusion matrix to MLflow with visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names for display
        normalize: Whether to normalize the confusion matrix
        plot_path: Path to save plot (temporary file if None)
        
    Example:
        log_confusion_matrix(y_true, y_pred, labels=['No Churn', 'Churn'])
    """
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        if normalize:
            fmt = '.2%'
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
        else:
            fmt = 'd'
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.tight_layout()
        
        # Save and log
        if plot_path is None:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.png', delete=False) as f:
                plot_path = f.name
            should_remove = True
        else:
            should_remove = False
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(plot_path, "plots")
        
        if should_remove:
            os.remove(plot_path)
        
        plt.close()
        
        # Log confusion matrix values as metrics
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                metric_name = f"cm_{i}_{j}"
                mlflow.log_metric(metric_name, float(cm[i, j]))
        
        logger.info("Confusion matrix logged to MLflow")
        
    except ImportError as e:
        logger.warning(f"Required library not available: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to log confusion matrix: {str(e)}")
        raise


def register_model(model_uri: str,
                  model_name: str,
                  tags: Optional[Dict[str, str]] = None,
                  description: Optional[str] = None,
                  await_registration_for: int = 300) -> ModelVersion:
    """
    Register a model in MLflow Model Registry.
    
    Args:
        model_uri: URI of the model to register (e.g., 'runs:/<run_id>/model')
        model_name: Name for the registered model
        tags: Optional tags for the model version
        description: Optional description for the model version
        await_registration_for: Seconds to wait for registration (0 for async)
        
    Returns:
        ModelVersion object
        
    Example:
        model_version = register_model(
            model_uri=f"runs:/{run_id}/model",
            model_name="churn_prediction_prod",
            tags={'environment': 'production'}
        )
    """
    try:
        logger.info(f"Registering model: {model_name}")
        
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            await_registration_for=await_registration_for
        )
        
        # Add tags if provided
        if tags:
            client = MlflowClient()
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=value
                )
        
        # Add description if provided
        if description:
            client = MlflowClient()
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
        
        logger.info(f"Model registered: {model_name} version {model_version.version}")
        
        return model_version
        
    except Exception as e:
        logger.error(f"Failed to register model: {str(e)}")
        raise


def transition_model_stage(model_name: str,
                          version: Union[int, str],
                          stage: str,
                          archive_existing_versions: bool = False) -> ModelVersion:
    """
    Transition a model version to a new stage.
    
    Args:
        model_name: Registered model name
        version: Model version number
        stage: Target stage ('Staging', 'Production', 'Archived')
        archive_existing_versions: Whether to archive existing versions in target stage
        
    Returns:
        Updated ModelVersion object
        
    Example:
        transition_model_stage(
            model_name="churn_prediction_prod",
            version=3,
            stage="Production",
            archive_existing_versions=True
        )
    """
    valid_stages = ['None', 'Staging', 'Production', 'Archived']
    
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")
    
    try:
        client = MlflowClient()
        
        logger.info(f"Transitioning {model_name} v{version} to {stage}")
        
        # Transition to new stage
        model_version = client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions
        )
        
        logger.info(f"Successfully transitioned to {stage}")
        
        return model_version
        
    except Exception as e:
        logger.error(f"Failed to transition model stage: {str(e)}")
        raise


def get_latest_model_version(model_name: str,
                            stage: Optional[str] = None) -> Optional[ModelVersion]:
    """
    Get the latest version of a registered model.
    
    Args:
        model_name: Registered model name
        stage: Optional stage filter ('Staging', 'Production', 'Archived')
        
    Returns:
        Latest ModelVersion object or None if not found
        
    Example:
        latest = get_latest_model_version("churn_prediction_prod", stage="Production")
        if latest:
            print(f"Latest production version: {latest.version}")
    """
    try:
        client = MlflowClient()
        
        if stage:
            # Get versions in specific stage
            versions = client.get_latest_versions(model_name, stages=[stage])
        else:
            # Get all versions and find the latest
            versions = client.search_model_versions(f"name='{model_name}'")
            versions = sorted(versions, key=lambda x: int(x.version), reverse=True)
        
        if versions:
            latest = versions[0]
            logger.info(f"Latest version: {latest.version} (stage: {latest.current_stage})")
            return latest
        else:
            logger.warning(f"No versions found for {model_name}" + 
                         (f" in stage {stage}" if stage else ""))
            return None
        
    except Exception as e:
        logger.error(f"Failed to get latest model version: {str(e)}")
        return None


def download_model_artifacts(model_name: str,
                            version: Union[int, str],
                            download_path: Optional[str] = None) -> str:
    """
    Download artifacts for a specific model version.
    
    Args:
        model_name: Registered model name
        version: Model version number
        download_path: Local path to download artifacts (temp dir if None)
        
    Returns:
        Path to downloaded artifacts
        
    Example:
        artifact_path = download_model_artifacts("churn_prediction_prod", version=3)
        print(f"Artifacts downloaded to: {artifact_path}")
    """
    try:
        client = MlflowClient()
        
        # Get model version details
        model_version = client.get_model_version(model_name, version)
        
        if download_path is None:
            download_path = tempfile.mkdtemp(prefix=f"{model_name}_v{version}_")
        
        logger.info(f"Downloading artifacts for {model_name} v{version}")
        
        # Download artifacts
        artifact_uri = model_version.source
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            dst_path=download_path
        )
        
        logger.info(f"Artifacts downloaded to: {local_path}")
        
        return local_path
        
    except Exception as e:
        logger.error(f"Failed to download model artifacts: {str(e)}")
        raise


def load_model_from_registry(model_name: str,
                            version: Optional[Union[int, str]] = None,
                            stage: Optional[str] = None,
                            model_type: str = 'spark') -> Any:
    """
    Load a model from the MLflow Model Registry.
    
    Args:
        model_name: Registered model name
        version: Model version number (mutually exclusive with stage)
        stage: Model stage ('Staging', 'Production') (mutually exclusive with version)
        model_type: Type of model ('spark', 'sklearn', 'keras')
        
    Returns:
        Loaded model object
        
    Example:
        model = load_model_from_registry(
            model_name="churn_prediction_prod",
            stage="Production",
            model_type='spark'
        )
    """
    if version and stage:
        raise ValueError("Cannot specify both version and stage")
    
    if not version and not stage:
        raise ValueError("Must specify either version or stage")
    
    try:
        # Construct model URI
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage}"
        
        logger.info(f"Loading model from: {model_uri}")
        
        # Load model based on type
        if model_type == 'spark':
            model = mlflow.spark.load_model(model_uri)
        elif model_type == 'sklearn':
            model = mlflow.sklearn.load_model(model_uri)
        elif model_type == 'keras':
            model = mlflow.keras.load_model(model_uri)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info("Model loaded successfully")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def compare_model_versions(model_name: str,
                          versions: List[Union[int, str]],
                          metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Compare metrics across multiple model versions.
    
    Args:
        model_name: Registered model name
        versions: List of version numbers to compare
        metrics: Optional list of specific metrics to compare
        
    Returns:
        Dictionary mapping versions to their metrics
        
    Example:
        comparison = compare_model_versions(
            model_name="churn_prediction_prod",
            versions=[1, 2, 3],
            metrics=['accuracy', 'f1_score']
        )
    """
    try:
        client = MlflowClient()
        comparison = {}
        
        for version in versions:
            try:
                model_version = client.get_model_version(model_name, version)
                run_id = model_version.run_id
                
                # Get run metrics
                run = client.get_run(run_id)
                version_metrics = run.data.metrics
                
                # Filter metrics if specified
                if metrics:
                    version_metrics = {
                        k: v for k, v in version_metrics.items() 
                        if k in metrics
                    }
                
                comparison[str(version)] = {
                    'metrics': version_metrics,
                    'stage': model_version.current_stage,
                    'run_id': run_id,
                    'creation_timestamp': model_version.creation_timestamp
                }
            except Exception as e:
                logger.warning(f"Failed to get data for version {version}: {str(e)}")
                comparison[str(version)] = {
                    'error': str(e)
                }
        
        logger.info(f"Compared {len(versions)} versions of {model_name}")
        
        return comparison
        
    except Exception as e:
        logger.error(f"Failed to compare model versions: {str(e)}")
        raise


def delete_model_version(model_name: str,
                        version: Union[int, str]) -> None:
    """
    Delete a specific model version from the registry.
    
    Args:
        model_name: Registered model name
        version: Model version number to delete
        
    Warning:
        This operation cannot be undone. Use with caution.
        
    Example:
        delete_model_version("churn_prediction_prod", version=1)
    """
    try:
        client = MlflowClient()
        
        logger.warning(f"Deleting {model_name} version {version}")
        
        client.delete_model_version(
            name=model_name,
            version=version
        )
        
        logger.info(f"Successfully deleted version {version}")
        
    except Exception as e:
        logger.error(f"Failed to delete model version: {str(e)}")
        raise


def set_model_version_tag(model_name: str,
                         version: Union[int, str],
                         key: str,
                         value: str) -> None:
    """
    Set a tag on a model version.
    
    Args:
        model_name: Registered model name
        version: Model version number
        key: Tag key
        value: Tag value
        
    Example:
        set_model_version_tag(
            model_name="churn_prediction_prod",
            version=3,
            key="validation_status",
            value="approved"
        )
    """
    try:
        client = MlflowClient()
        
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key=key,
            value=value
        )
        
        logger.info(f"Set tag {key}={value} on {model_name} v{version}")
        
    except Exception as e:
        logger.error(f"Failed to set model version tag: {str(e)}")
        raise


def get_model_version_details(model_name: str,
                             version: Union[int, str]) -> Dict[str, Any]:
    """
    Get detailed information about a model version.
    
    Args:
        model_name: Registered model name
        version: Model version number
        
    Returns:
        Dictionary with model version details
        
    Example:
        details = get_model_version_details("churn_prediction_prod", version=3)
        print(f"Stage: {details['current_stage']}")
        print(f"Run ID: {details['run_id']}")
    """
    try:
        client = MlflowClient()
        
        model_version = client.get_model_version(model_name, version)
        
        details = {
            'name': model_version.name,
            'version': model_version.version,
            'creation_timestamp': model_version.creation_timestamp,
            'last_updated_timestamp': model_version.last_updated_timestamp,
            'current_stage': model_version.current_stage,
            'description': model_version.description,
            'run_id': model_version.run_id,
            'run_link': model_version.run_link,
            'source': model_version.source,
            'status': model_version.status,
            'tags': model_version.tags
        }
        
        logger.info(f"Retrieved details for {model_name} v{version}")
        
        return details
        
    except Exception as e:
        logger.error(f"Failed to get model version details: {str(e)}")
        raise
