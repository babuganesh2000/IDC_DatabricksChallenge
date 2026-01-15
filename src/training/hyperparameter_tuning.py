"""
Hyperparameter Tuning Module

This module provides hyperparameter optimization using Hyperopt for Bayesian optimization.
Supports all model types with MLflow tracking integration.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import time
from datetime import datetime
import mlflow
import mlflow.spark
import numpy as np
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)

# Hyperopt imports
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, space_eval
    from hyperopt.pyll import scope
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Hyperopt not available. Install with: pip install hyperopt")

from ..models.base_model import BaseModel
from ..models.churn_prediction import ChurnPredictionModel
from ..models.clv_prediction import CLVPredictionModel
from ..models.purchase_prediction import PurchasePredictionModel
from ..models.recommendation import RecommendationModel
from ..models.segmentation import SegmentationModel
from .mlflow_utils import log_model_metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter optimization using Bayesian optimization with Hyperopt.
    
    Features:
    - Bayesian optimization with Tree-structured Parzen Estimators (TPE)
    - MLflow integration for tracking trials
    - Support for all model types
    - Configurable search spaces
    - Early stopping support
    """
    
    MODEL_CLASSES = {
        'churn_prediction': ChurnPredictionModel,
        'clv_prediction': CLVPredictionModel,
        'purchase_prediction': PurchasePredictionModel,
        'recommendation': RecommendationModel,
        'segmentation': SegmentationModel,
    }
    
    DEFAULT_SEARCH_SPACES = {
        'random_forest': {
            'numTrees': hp.quniform('numTrees', 10, 200, 10),
            'maxDepth': hp.quniform('maxDepth', 3, 20, 1),
            'minInstancesPerNode': hp.quniform('minInstancesPerNode', 1, 10, 1),
            'maxBins': hp.quniform('maxBins', 16, 64, 8),
            'subsamplingRate': hp.uniform('subsamplingRate', 0.6, 1.0),
        },
        'gbt': {
            'maxIter': hp.quniform('maxIter', 10, 100, 10),
            'maxDepth': hp.quniform('maxDepth', 3, 15, 1),
            'stepSize': hp.uniform('stepSize', 0.01, 0.3),
            'subsamplingRate': hp.uniform('subsamplingRate', 0.6, 1.0),
            'minInstancesPerNode': hp.quniform('minInstancesPerNode', 1, 10, 1),
        },
        'linear_regression': {
            'regParam': hp.loguniform('regParam', np.log(0.0001), np.log(1.0)),
            'elasticNetParam': hp.uniform('elasticNetParam', 0.0, 1.0),
            'maxIter': hp.quniform('maxIter', 50, 200, 10),
        },
        'neural_network': {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
            'batch_size': hp.quniform('batch_size', 32, 256, 32),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
            'hidden_units_1': hp.quniform('hidden_units_1', 64, 512, 64),
            'hidden_units_2': hp.quniform('hidden_units_2', 32, 256, 32),
        },
        'kmeans': {
            'k': hp.quniform('k', 2, 20, 1),
            'maxIter': hp.quniform('maxIter', 50, 300, 50),
            'initMode': hp.choice('initMode', ['k-means||', 'random']),
        },
        'als': {
            'rank': hp.quniform('rank', 5, 50, 5),
            'maxIter': hp.quniform('maxIter', 5, 20, 5),
            'regParam': hp.loguniform('regParam', np.log(0.001), np.log(0.1)),
            'alpha': hp.uniform('alpha', 0.5, 2.0),
        }
    }
    
    def __init__(self,
                 model_type: str,
                 experiment_name: Optional[str] = None,
                 tracking_uri: Optional[str] = None):
        """
        Initialize the HyperparameterTuner.
        
        Args:
            model_type: Type of model to tune
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
            
        Raises:
            ValueError: If model_type is not supported
            ImportError: If hyperopt is not available
        """
        if not HYPEROPT_AVAILABLE:
            raise ImportError("Hyperopt is required for hyperparameter tuning. Install with: pip install hyperopt")
        
        if model_type not in self.MODEL_CLASSES:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(self.MODEL_CLASSES.keys())}"
            )
        
        self.model_type = model_type
        self.experiment_name = experiment_name or f"{model_type}_hyperopt"
        
        # Set MLflow tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set up experiment
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set experiment: {str(e)}")
            raise
        
        self.trials = None
        self.best_params = None
        self.best_model = None
        self.best_score = None
    
    def get_default_search_space(self, algorithm: str) -> Dict[str, Any]:
        """
        Get the default search space for an algorithm.
        
        Args:
            algorithm: Algorithm name (e.g., 'random_forest', 'gbt')
            
        Returns:
            Dictionary defining the search space
            
        Raises:
            ValueError: If algorithm is not supported
        """
        if algorithm not in self.DEFAULT_SEARCH_SPACES:
            raise ValueError(
                f"No default search space for: {algorithm}. "
                f"Available: {list(self.DEFAULT_SEARCH_SPACES.keys())}"
            )
        
        return self.DEFAULT_SEARCH_SPACES[algorithm].copy()
    
    def create_objective_function(self,
                                  train_data: DataFrame,
                                  validation_data: DataFrame,
                                  model_builder: Callable,
                                  evaluator: Any,
                                  maximize: bool = True) -> Callable:
        """
        Create an objective function for Hyperopt optimization.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset for evaluation
            model_builder: Function that builds model given hyperparameters
            evaluator: Spark ML evaluator for model assessment
            maximize: Whether to maximize (True) or minimize (False) the metric
            
        Returns:
            Objective function compatible with Hyperopt
        """
        def objective(params: Dict[str, Any]) -> Dict[str, Any]:
            """Objective function for a single trial."""
            with mlflow.start_run(nested=True):
                try:
                    # Convert hyperopt parameters to appropriate types
                    typed_params = self._convert_param_types(params)
                    
                    # Log parameters
                    mlflow.log_params(typed_params)
                    
                    # Build and train model
                    start_time = time.time()
                    model = model_builder(typed_params)
                    trained_model = model.fit(train_data)
                    training_time = time.time() - start_time
                    
                    mlflow.log_metric('training_time', training_time)
                    
                    # Evaluate model
                    predictions = trained_model.transform(validation_data)
                    score = evaluator.evaluate(predictions)
                    
                    mlflow.log_metric('validation_score', score)
                    
                    # Hyperopt minimizes by default, so negate if maximizing
                    loss = -score if maximize else score
                    
                    logger.info(f"Trial completed - Score: {score:.4f}, Params: {typed_params}")
                    
                    return {
                        'loss': loss,
                        'status': STATUS_OK,
                        'score': score,
                        'params': typed_params,
                        'training_time': training_time
                    }
                    
                except Exception as e:
                    logger.error(f"Trial failed: {str(e)}")
                    mlflow.log_param('error', str(e))
                    
                    return {
                        'loss': float('inf'),
                        'status': STATUS_FAIL,
                        'error': str(e)
                    }
        
        return objective
    
    def tune_hyperparameters(self,
                           train_data: DataFrame,
                           validation_data: DataFrame,
                           search_space: Union[Dict[str, Any], str],
                           model_builder: Callable,
                           evaluator: Optional[Any] = None,
                           max_evals: int = 50,
                           maximize: bool = True,
                           early_stop_rounds: Optional[int] = None,
                           algorithm: Optional[str] = None,
                           tags: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters using Bayesian optimization.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            search_space: Either a dict defining search space or algorithm name for defaults
            model_builder: Function to build model from parameters
            evaluator: Evaluator for model assessment (auto-created if None)
            max_evals: Maximum number of evaluations
            maximize: Whether to maximize the metric
            early_stop_rounds: Stop if no improvement for N rounds
            algorithm: Algorithm name for logging
            tags: Optional tags for the parent run
            
        Returns:
            Tuple of (best_params, best_score)
        """
        # Get search space
        if isinstance(search_space, str):
            search_space = self.get_default_search_space(search_space)
            algorithm = algorithm or search_space
        
        # Create evaluator if not provided
        if evaluator is None:
            evaluator = self._get_default_evaluator()
        
        logger.info(f"Starting hyperparameter tuning with {max_evals} evaluations")
        logger.info(f"Search space: {list(search_space.keys())}")
        
        with mlflow.start_run(run_name=f"{self.model_type}_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            try:
                # Log configuration
                if tags:
                    mlflow.set_tags(tags)
                
                mlflow.set_tag('optimization_method', 'bayesian_hyperopt')
                mlflow.set_tag('model_type', self.model_type)
                
                if algorithm:
                    mlflow.set_tag('algorithm', algorithm)
                
                mlflow.log_param('max_evals', max_evals)
                mlflow.log_param('maximize', maximize)
                
                # Create objective function
                objective = self.create_objective_function(
                    train_data=train_data,
                    validation_data=validation_data,
                    model_builder=model_builder,
                    evaluator=evaluator,
                    maximize=maximize
                )
                
                # Initialize trials
                self.trials = Trials()
                
                # Run optimization
                start_time = time.time()
                
                best = fmin(
                    fn=objective,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=self.trials,
                    verbose=True,
                    show_progressbar=True
                )
                
                optimization_time = time.time() - start_time
                
                # Get best parameters
                self.best_params = space_eval(search_space, best)
                self.best_params = self._convert_param_types(self.best_params)
                
                # Get best score
                best_trial = min(self.trials.trials, key=lambda x: x['result']['loss'])
                self.best_score = best_trial['result'].get('score', -best_trial['result']['loss'])
                
                # Log results
                mlflow.log_params(self.best_params)
                mlflow.log_metric('best_score', self.best_score)
                mlflow.log_metric('optimization_time', optimization_time)
                mlflow.log_metric('num_trials', len(self.trials.trials))
                
                logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
                logger.info(f"Best score: {self.best_score:.4f}")
                logger.info(f"Best parameters: {self.best_params}")
                
                # Save trials history
                self._log_trials_history()
                
                return self.best_params, self.best_score
                
            except Exception as e:
                logger.error(f"Hyperparameter tuning failed: {str(e)}")
                mlflow.log_param('status', 'failed')
                mlflow.log_param('error', str(e))
                raise
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the best parameters from the last tuning run.
        
        Returns:
            Dictionary of best parameters or None if no tuning has been done
        """
        if self.best_params is None:
            logger.warning("No tuning results available. Run tune_hyperparameters first.")
            return None
        
        return self.best_params.copy()
    
    def get_trials_dataframe(self) -> Optional[Any]:
        """
        Get trials history as a Spark DataFrame.
        
        Returns:
            DataFrame with trial results or None if no trials
        """
        if self.trials is None or len(self.trials.trials) == 0:
            logger.warning("No trials available")
            return None
        
        try:
            from pyspark.sql import SparkSession
            
            trials_data = []
            for trial in self.trials.trials:
                trial_dict = {
                    'trial_id': trial['tid'],
                    'loss': trial['result']['loss'],
                }
                
                # Add parameters
                if 'params' in trial['result']:
                    for key, value in trial['result']['params'].items():
                        trial_dict[f'param_{key}'] = value
                
                # Add metrics
                if 'score' in trial['result']:
                    trial_dict['score'] = trial['result']['score']
                
                if 'training_time' in trial['result']:
                    trial_dict['training_time'] = trial['result']['training_time']
                
                trials_data.append(trial_dict)
            
            spark = SparkSession.builder.getOrCreate()
            return spark.createDataFrame(trials_data)
            
        except Exception as e:
            logger.error(f"Failed to create trials DataFrame: {str(e)}")
            return None
    
    def _convert_param_types(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert parameter values to appropriate types.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            Dictionary with converted parameter types
        """
        converted = {}
        for key, value in params.items():
            # Convert to int for parameters that should be integers
            if key in ['numTrees', 'maxDepth', 'maxIter', 'minInstancesPerNode', 
                      'maxBins', 'k', 'rank', 'batch_size', 'hidden_units_1', 'hidden_units_2']:
                converted[key] = int(value)
            else:
                converted[key] = value
        
        return converted
    
    def _get_default_evaluator(self) -> Any:
        """
        Get the default evaluator for the model type.
        
        Returns:
            Appropriate Spark ML evaluator
        """
        if self.model_type in ['churn_prediction', 'purchase_prediction']:
            return BinaryClassificationEvaluator(metricName='areaUnderROC')
        elif self.model_type == 'segmentation':
            return MulticlassClassificationEvaluator(metricName='f1')
        elif self.model_type == 'clv_prediction':
            return RegressionEvaluator(metricName='rmse')
        else:
            # For recommendation, use a custom evaluator or RMSE as fallback
            return RegressionEvaluator(metricName='rmse')
    
    def _log_trials_history(self) -> None:
        """Log trials history to MLflow."""
        if self.trials is None:
            return
        
        try:
            # Log best N trials
            sorted_trials = sorted(self.trials.trials, key=lambda x: x['result']['loss'])
            
            for idx, trial in enumerate(sorted_trials[:10]):
                result = trial['result']
                prefix = f"trial_{idx}"
                
                if 'score' in result:
                    mlflow.log_metric(f"{prefix}_score", result['score'])
                
                if 'params' in result:
                    for key, value in result['params'].items():
                        mlflow.log_param(f"{prefix}_{key}", value)
            
            logger.info("Logged top 10 trials to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log trials history: {str(e)}")
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot the optimization history.
        
        Args:
            save_path: Path to save the plot (displays if None)
        """
        if self.trials is None or len(self.trials.trials) == 0:
            logger.warning("No trials available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            losses = [trial['result']['loss'] for trial in self.trials.trials]
            
            # Plot loss over trials
            plt.figure(figsize=(10, 6))
            plt.plot(losses, marker='o', linestyle='-', alpha=0.7)
            plt.axhline(y=min(losses), color='r', linestyle='--', 
                       label=f'Best: {min(losses):.4f}')
            plt.xlabel('Trial')
            plt.ylabel('Loss')
            plt.title('Hyperparameter Optimization History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to: {save_path}")
                
                # Log to MLflow
                if mlflow.active_run():
                    mlflow.log_artifact(save_path)
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to plot optimization history: {str(e)}")
    
    def __repr__(self) -> str:
        """String representation of the tuner."""
        return f"HyperparameterTuner(model_type='{self.model_type}', experiment='{self.experiment_name}')"
