"""
Base Model Abstract Class

This module provides the abstract base class for all ML models in the project.
It defines the common interface and shared functionality including MLflow integration,
logging, and validation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging
import mlflow
import mlflow.spark
from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.evaluation import Evaluator
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    This class provides a common interface and shared functionality for:
    - Model training with MLflow tracking
    - Model evaluation and validation
    - Feature importance extraction
    - Model persistence and loading
    """
    
    def __init__(self, name: str, experiment_name: str):
        """
        Initialize the base model.
        
        Args:
            name: Unique identifier for the model
            experiment_name: MLflow experiment name for tracking
        """
        self.name = name
        self.experiment_name = experiment_name
        self.model = None
        self.feature_cols = []
        self.metrics = {}
        
        # Set up MLflow experiment
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment set to: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set MLflow experiment: {str(e)}")
            raise
    
    @abstractmethod
    def train(self, train_data: DataFrame, validation_data: Optional[DataFrame] = None,
              params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Train the model on the provided dataset.
        
        Args:
            train_data: Training dataset as Spark DataFrame
            validation_data: Optional validation dataset
            params: Model-specific hyperparameters
            
        Returns:
            Trained model object
        """
        pass
    
    @abstractmethod
    def predict(self, data: DataFrame) -> DataFrame:
        """
        Generate predictions on the provided dataset.
        
        Args:
            data: Input dataset as Spark DataFrame
            
        Returns:
            DataFrame with predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, data: DataFrame, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the model on the provided dataset.
        
        Args:
            data: Evaluation dataset as Spark DataFrame
            metrics: List of metric names to compute
            
        Returns:
            Dictionary of metric names and values
        """
        pass
    
    def validate_data(self, data: DataFrame, required_cols: List[str]) -> bool:
        """
        Validate that the input data contains required columns.
        
        Args:
            data: Input DataFrame to validate
            required_cols: List of required column names
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for null values in required columns
        null_counts = data.select([
            (data[col].isNull().cast("int")).alias(col) 
            for col in required_cols
        ]).agg(*[f"sum({col}) as {col}" for col in required_cols]).collect()[0]
        
        null_cols = [col for col in required_cols if null_counts[col] > 0]
        if null_cols:
            logger.warning(f"Columns with null values: {null_cols}")
        
        logger.info("Data validation passed")
        return True
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for tracking
        """
        try:
            for metric_name, metric_value in metrics.items():
                if step is not None:
                    mlflow.log_metric(metric_name, metric_value, step=step)
                else:
                    mlflow.log_metric(metric_name, metric_value)
            logger.info(f"Logged metrics: {metrics}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameter names and values
        """
        try:
            mlflow.log_params(params)
            logger.info(f"Logged parameters: {params}")
        except Exception as e:
            logger.error(f"Failed to log parameters: {str(e)}")
    
    def log_feature_importance(self, importance_dict: Dict[str, float], 
                              top_n: int = 20) -> None:
        """
        Log feature importance to MLflow.
        
        Args:
            importance_dict: Dictionary mapping feature names to importance scores
            top_n: Number of top features to log
        """
        try:
            # Sort by importance
            sorted_features = sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_n]
            
            # Log as parameters
            for idx, (feature, importance) in enumerate(sorted_features):
                mlflow.log_param(f"top_feature_{idx+1}", feature)
                mlflow.log_metric(f"importance_{idx+1}", importance)
            
            logger.info(f"Logged top {top_n} feature importances")
        except Exception as e:
            logger.error(f"Failed to log feature importance: {str(e)}")
    
    def save_model(self, path: str, model: Optional[Any] = None) -> None:
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
            model: Model object to save (defaults to self.model)
        """
        model_to_save = model if model is not None else self.model
        
        if model_to_save is None:
            raise ValueError("No model to save. Train a model first.")
        
        try:
            if isinstance(model_to_save, (PipelineModel, Transformer)):
                mlflow.spark.log_model(model_to_save, path)
            else:
                mlflow.sklearn.log_model(model_to_save, path)
            logger.info(f"Model saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self, path: str, model_type: str = "spark") -> Any:
        """
        Load a model from the specified path.
        
        Args:
            path: Path to load the model from
            model_type: Type of model ('spark' or 'sklearn')
            
        Returns:
            Loaded model object
        """
        try:
            if model_type == "spark":
                self.model = mlflow.spark.load_model(path)
            elif model_type == "sklearn":
                self.model = mlflow.sklearn.load_model(path)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"Model loaded from: {path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def train_with_mlflow(self, train_func, train_data: DataFrame, 
                         validation_data: Optional[DataFrame] = None,
                         params: Optional[Dict[str, Any]] = None,
                         tags: Optional[Dict[str, str]] = None) -> Tuple[Any, Dict[str, float]]:
        """
        Wrapper for training with automatic MLflow tracking.
        
        Args:
            train_func: Function to execute for training
            train_data: Training dataset
            validation_data: Optional validation dataset
            params: Model hyperparameters
            tags: Optional tags for the MLflow run
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        with mlflow.start_run(run_name=f"{self.name}_training"):
            try:
                # Log tags
                if tags:
                    mlflow.set_tags(tags)
                
                mlflow.set_tag("model_name", self.name)
                
                # Log parameters
                if params:
                    self.log_params(params)
                
                # Log dataset info
                mlflow.log_param("train_size", train_data.count())
                if validation_data:
                    mlflow.log_param("validation_size", validation_data.count())
                
                # Training
                start_time = time.time()
                logger.info(f"Starting training for {self.name}")
                
                model = train_func(train_data, validation_data, params)
                
                training_time = time.time() - start_time
                mlflow.log_metric("training_time_seconds", training_time)
                logger.info(f"Training completed in {training_time:.2f} seconds")
                
                # Evaluation
                if validation_data:
                    self.model = model
                    metrics = self.evaluate(validation_data)
                    self.log_metrics(metrics)
                else:
                    metrics = {}
                
                # Save model
                self.save_model("model", model)
                
                return model, metrics
                
            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not available
        """
        if self.model is None:
            logger.warning("No trained model available")
            return None
        
        try:
            # For tree-based models
            if hasattr(self.model, 'featureImportances'):
                importances = self.model.featureImportances.toArray()
                return dict(zip(self.feature_cols, importances))
            
            # For pipeline models
            if isinstance(self.model, PipelineModel):
                last_stage = self.model.stages[-1]
                if hasattr(last_stage, 'featureImportances'):
                    importances = last_stage.featureImportances.toArray()
                    return dict(zip(self.feature_cols, importances))
            
            logger.warning("Feature importance not available for this model type")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract feature importance: {str(e)}")
            return None
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(name='{self.name}')"
