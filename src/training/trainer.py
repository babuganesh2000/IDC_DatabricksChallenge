"""
Model Training Orchestrator

This module provides the ModelTrainer class for orchestrating model training,
evaluation, and experiment tracking across all model types in the project.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from datetime import datetime
import mlflow
import mlflow.spark
import mlflow.sklearn
from pyspark.sql import DataFrame
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)
import numpy as np

from ..models.base_model import BaseModel
from ..models.churn_prediction import ChurnPredictionModel
from ..models.clv_prediction import CLVPredictionModel
from ..models.purchase_prediction import PurchasePredictionModel
from ..models.recommendation import RecommendationModel
from ..models.segmentation import SegmentationModel
from .mlflow_utils import log_model_metrics, log_feature_importance, register_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrates model training, evaluation, and experiment tracking.
    
    Supports:
    - All 5 model types (Churn, CLV, Purchase, Recommendation, Segmentation)
    - Cross-validation
    - MLflow experiment tracking
    - Model registry integration
    - Comprehensive metrics logging
    """
    
    MODEL_CLASSES = {
        'churn_prediction': ChurnPredictionModel,
        'clv_prediction': CLVPredictionModel,
        'purchase_prediction': PurchasePredictionModel,
        'recommendation': RecommendationModel,
        'segmentation': SegmentationModel,
    }
    
    def __init__(self, 
                 model_type: str,
                 experiment_name: Optional[str] = None,
                 tracking_uri: Optional[str] = None):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_type: Type of model to train (one of MODEL_CLASSES keys)
            experiment_name: MLflow experiment name (auto-generated if None)
            tracking_uri: MLflow tracking URI (uses default if None)
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in self.MODEL_CLASSES:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(self.MODEL_CLASSES.keys())}"
            )
        
        self.model_type = model_type
        self.experiment_name = experiment_name or f"{model_type}_training"
        
        # Set MLflow tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        
        # Set up experiment
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set experiment: {str(e)}")
            raise
        
        self.model = None
        self.run_id = None
        self.metrics = {}
    
    def train_model(self,
                   train_data: DataFrame,
                   validation_data: Optional[DataFrame] = None,
                   params: Optional[Dict[str, Any]] = None,
                   feature_cols: Optional[List[str]] = None,
                   label_col: str = "label",
                   tags: Optional[Dict[str, str]] = None,
                   register_model_name: Optional[str] = None) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Train a model with MLflow tracking.
        
        Args:
            train_data: Training dataset
            validation_data: Optional validation dataset for evaluation
            params: Model hyperparameters
            feature_cols: List of feature column names
            label_col: Name of the label column
            tags: Optional tags for the MLflow run
            register_model_name: If provided, register model with this name
            
        Returns:
            Tuple of (trained_model, metrics_dict)
            
        Raises:
            ValueError: If required parameters are missing
        """
        if train_data is None or train_data.rdd.isEmpty():
            raise ValueError("Training data cannot be empty")
        
        # Initialize model
        model_class = self.MODEL_CLASSES[self.model_type]
        model = model_class(
            name=f"{self.model_type}_model",
            experiment_name=self.experiment_name
        )
        
        with mlflow.start_run(run_name=f"{self.model_type}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            self.run_id = run.info.run_id
            
            try:
                # Log tags
                default_tags = {
                    'model_type': self.model_type,
                    'training_timestamp': datetime.now().isoformat(),
                    'framework': 'pyspark'
                }
                if tags:
                    default_tags.update(tags)
                mlflow.set_tags(default_tags)
                
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                # Log dataset statistics
                train_count = train_data.count()
                mlflow.log_param('train_size', train_count)
                mlflow.log_param('label_col', label_col)
                
                if feature_cols:
                    mlflow.log_param('num_features', len(feature_cols))
                    mlflow.log_param('feature_cols', ','.join(feature_cols[:10]))  # Log first 10
                
                if validation_data:
                    val_count = validation_data.count()
                    mlflow.log_param('validation_size', val_count)
                
                logger.info(f"Starting training for {self.model_type} model")
                logger.info(f"Training samples: {train_count}")
                
                # Train model
                start_time = time.time()
                trained_model = model.train(
                    train_data=train_data,
                    validation_data=validation_data,
                    params=params
                )
                training_time = time.time() - start_time
                
                mlflow.log_metric('training_time_seconds', training_time)
                logger.info(f"Training completed in {training_time:.2f} seconds")
                
                # Evaluate model
                metrics = {}
                if validation_data:
                    logger.info("Evaluating model on validation data")
                    metrics = self.evaluate_model(model, validation_data)
                    log_model_metrics(metrics, prefix='val')
                    self.metrics = metrics
                
                # Log feature importance if available
                feature_importance = model.get_feature_importance()
                if feature_importance:
                    log_feature_importance(feature_importance, top_n=20)
                
                # Save model
                model.save_model('model')
                logger.info("Model saved to MLflow")
                
                # Register model if requested
                if register_model_name:
                    model_uri = f"runs:/{self.run_id}/model"
                    register_model(
                        model_uri=model_uri,
                        model_name=register_model_name,
                        tags=default_tags
                    )
                    logger.info(f"Model registered as: {register_model_name}")
                
                self.model = model
                mlflow.log_param('status', 'success')
                
                return model, metrics
                
            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                mlflow.log_param('status', 'failed')
                mlflow.log_param('error_message', str(e))
                raise
    
    def evaluate_model(self,
                      model: BaseModel,
                      test_data: DataFrame,
                      metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model instance
            test_data: Test dataset
            metrics: List of metric names to compute (uses defaults if None)
            
        Returns:
            Dictionary of metric names and values
            
        Raises:
            ValueError: If model is not trained or test_data is empty
        """
        if model is None or model.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        if test_data is None or test_data.rdd.isEmpty():
            raise ValueError("Test data cannot be empty")
        
        logger.info(f"Evaluating {self.model_type} model")
        
        try:
            # Use model's evaluate method
            eval_metrics = model.evaluate(test_data, metrics=metrics)
            
            logger.info(f"Evaluation metrics: {eval_metrics}")
            return eval_metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def log_metrics(self,
                   metrics: Dict[str, float],
                   prefix: Optional[str] = None,
                   step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric names and values
            prefix: Optional prefix for metric names (e.g., 'train', 'val')
            step: Optional step number for tracking
        """
        try:
            for metric_name, metric_value in metrics.items():
                full_name = f"{prefix}_{metric_name}" if prefix else metric_name
                
                if step is not None:
                    mlflow.log_metric(full_name, metric_value, step=step)
                else:
                    mlflow.log_metric(full_name, metric_value)
            
            logger.info(f"Logged metrics with prefix '{prefix}': {metrics}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
    
    def cross_validate(self,
                      data: DataFrame,
                      estimator: Any,
                      param_grid: Dict[str, List[Any]],
                      num_folds: int = 3,
                      evaluator: Optional[Any] = None,
                      metric: str = 'areaUnderROC',
                      parallelism: int = 4) -> Tuple[Any, Dict[str, float]]:
        """
        Perform cross-validation for hyperparameter tuning.
        
        Args:
            data: Input dataset
            estimator: Model estimator (e.g., RandomForestClassifier)
            param_grid: Dictionary of parameter names to lists of values
            num_folds: Number of cross-validation folds
            evaluator: Evaluator for model assessment
            metric: Metric name for evaluation
            parallelism: Number of parallel tasks
            
        Returns:
            Tuple of (best_model, best_metrics)
        """
        logger.info(f"Starting {num_folds}-fold cross-validation")
        logger.info(f"Parameter grid: {param_grid}")
        
        try:
            # Build parameter grid
            param_builder = ParamGridBuilder()
            for param_name, param_values in param_grid.items():
                param = getattr(estimator, param_name)
                param_builder.addGrid(param, param_values)
            
            params = param_builder.build()
            logger.info(f"Total parameter combinations: {len(params)}")
            
            # Set up evaluator
            if evaluator is None:
                if self.model_type in ['churn_prediction', 'purchase_prediction']:
                    evaluator = BinaryClassificationEvaluator(metricName=metric)
                elif self.model_type in ['segmentation']:
                    evaluator = MulticlassClassificationEvaluator(metricName='f1')
                elif self.model_type in ['clv_prediction']:
                    evaluator = RegressionEvaluator(metricName='rmse')
                else:
                    raise ValueError(f"Default evaluator not available for {self.model_type}")
            
            # Create cross-validator
            cv = CrossValidator(
                estimator=estimator,
                estimatorParamMaps=params,
                evaluator=evaluator,
                numFolds=num_folds,
                parallelism=parallelism,
                seed=42
            )
            
            # Run cross-validation
            start_time = time.time()
            cv_model = cv.fit(data)
            cv_time = time.time() - start_time
            
            logger.info(f"Cross-validation completed in {cv_time:.2f} seconds")
            
            # Log CV results
            if mlflow.active_run():
                mlflow.log_metric('cv_time_seconds', cv_time)
                mlflow.log_param('num_folds', num_folds)
                mlflow.log_param('cv_metric', metric)
                
                # Log fold metrics
                avg_metrics = np.mean(cv_model.avgMetrics)
                std_metrics = np.std(cv_model.avgMetrics)
                mlflow.log_metric('cv_avg_metric', avg_metrics)
                mlflow.log_metric('cv_std_metric', std_metrics)
            
            # Get best model and metrics
            best_model = cv_model.bestModel
            best_metrics = {
                'cv_best_metric': max(cv_model.avgMetrics),
                'cv_avg_metric': avg_metrics,
                'cv_std_metric': std_metrics
            }
            
            logger.info(f"Best CV metrics: {best_metrics}")
            
            return best_model, best_metrics
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            raise
    
    def train_with_cross_validation(self,
                                    data: DataFrame,
                                    estimator: Any,
                                    param_grid: Dict[str, List[Any]],
                                    num_folds: int = 3,
                                    test_data: Optional[DataFrame] = None,
                                    register_model_name: Optional[str] = None) -> Tuple[Any, Dict[str, float]]:
        """
        Train model using cross-validation with MLflow tracking.
        
        Args:
            data: Training dataset (will be split into folds)
            estimator: Model estimator
            param_grid: Parameter grid for tuning
            num_folds: Number of CV folds
            test_data: Optional held-out test set for final evaluation
            register_model_name: If provided, register best model
            
        Returns:
            Tuple of (best_model, metrics_dict)
        """
        with mlflow.start_run(run_name=f"{self.model_type}_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            self.run_id = run.info.run_id
            
            try:
                # Log configuration
                mlflow.set_tag('training_method', 'cross_validation')
                mlflow.set_tag('model_type', self.model_type)
                mlflow.log_param('num_folds', num_folds)
                
                # Perform cross-validation
                best_model, cv_metrics = self.cross_validate(
                    data=data,
                    estimator=estimator,
                    param_grid=param_grid,
                    num_folds=num_folds
                )
                
                # Log CV metrics
                self.log_metrics(cv_metrics, prefix='cv')
                
                # Evaluate on test set if provided
                final_metrics = cv_metrics.copy()
                if test_data:
                    logger.info("Evaluating best model on test set")
                    
                    # Create temporary model wrapper for evaluation
                    model_class = self.MODEL_CLASSES[self.model_type]
                    temp_model = model_class(
                        name=f"{self.model_type}_model",
                        experiment_name=self.experiment_name
                    )
                    temp_model.model = best_model
                    
                    test_metrics = self.evaluate_model(temp_model, test_data)
                    self.log_metrics(test_metrics, prefix='test')
                    final_metrics.update({f'test_{k}': v for k, v in test_metrics.items()})
                
                # Save best model
                mlflow.spark.log_model(best_model, 'model')
                logger.info("Best model saved to MLflow")
                
                # Register model if requested
                if register_model_name:
                    model_uri = f"runs:/{self.run_id}/model"
                    register_model(
                        model_uri=model_uri,
                        model_name=register_model_name,
                        tags={'training_method': 'cross_validation'}
                    )
                    logger.info(f"Model registered as: {register_model_name}")
                
                mlflow.log_param('status', 'success')
                self.metrics = final_metrics
                
                return best_model, final_metrics
                
            except Exception as e:
                logger.error(f"CV training failed: {str(e)}")
                mlflow.log_param('status', 'failed')
                mlflow.log_param('error_message', str(e))
                raise
    
    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current MLflow run.
        
        Returns:
            Dictionary with run information or None if no active run
        """
        if self.run_id is None:
            logger.warning("No active run ID")
            return None
        
        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(self.run_id)
            
            return {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'artifact_uri': run.info.artifact_uri,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': run.data.tags
            }
            
        except Exception as e:
            logger.error(f"Failed to get run info: {str(e)}")
            return None
    
    def compare_runs(self, run_ids: List[str]) -> DataFrame:
        """
        Compare multiple training runs.
        
        Args:
            run_ids: List of MLflow run IDs to compare
            
        Returns:
            DataFrame with comparison results
        """
        try:
            from pyspark.sql import SparkSession
            
            client = mlflow.tracking.MlflowClient()
            runs_data = []
            
            for run_id in run_ids:
                run = client.get_run(run_id)
                run_dict = {
                    'run_id': run_id,
                    'start_time': run.info.start_time,
                    'status': run.info.status,
                }
                run_dict.update(run.data.params)
                run_dict.update(run.data.metrics)
                runs_data.append(run_dict)
            
            # Convert to DataFrame
            spark = SparkSession.builder.getOrCreate()
            comparison_df = spark.createDataFrame(runs_data)
            
            logger.info(f"Compared {len(run_ids)} runs")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Failed to compare runs: {str(e)}")
            raise
    
    def __repr__(self) -> str:
        """String representation of the trainer."""
        return f"ModelTrainer(model_type='{self.model_type}', experiment='{self.experiment_name}')"
