"""
Customer Lifetime Value (CLV) Prediction Model

Regression model to predict customer lifetime value.
Supports Linear Regression, Random Forest, and Gradient Boosted Trees.
"""

from typing import Any, Dict, List, Optional
import logging
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, log1p, exp
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor
)
from pyspark.ml.evaluation import RegressionEvaluator

from .base_model import BaseModel


logger = logging.getLogger(__name__)


class CLVPredictionModel(BaseModel):
    """
    Regression model for predicting Customer Lifetime Value.
    
    Features:
    - Multiple algorithm support (Linear, RF, GBT)
    - Log transformation for skewed CLV distributions
    - RFM-based feature engineering
    - Comprehensive regression metrics (RMSE, MAE, R²)
    """
    
    SUPPORTED_ALGORITHMS = ['linear_regression', 'random_forest', 'gbt']
    
    def __init__(self, name: str = "clv_prediction",
                 experiment_name: str = "clv_prediction_experiments"):
        """
        Initialize the CLV Prediction Model.
        
        Args:
            name: Model identifier
            experiment_name: MLflow experiment name
        """
        super().__init__(name, experiment_name)
        self.label_col = "clv"
        self.prediction_col = "predicted_clv"
        self.algorithm = None
        self.use_log_transform = False
    
    def engineer_features(self, data: DataFrame, feature_cols: List[str]) -> DataFrame:
        """
        Engineer features for CLV prediction.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info("Engineering features for CLV prediction")
            
            # Store original feature columns
            self.feature_cols = feature_cols.copy()
            
            # RFM-based features
            if all(col_name in feature_cols for col_name in ['recency', 'frequency', 'monetary']):
                # RFM score (weighted combination)
                data = data.withColumn(
                    'rfm_score',
                    (col('recency') * 0.2 + col('frequency') * 0.4 + col('monetary') * 0.4)
                )
                self.feature_cols.append('rfm_score')
                
                # Average order value
                data = data.withColumn(
                    'avg_order_value',
                    when(col('frequency') > 0, col('monetary') / col('frequency'))
                    .otherwise(0)
                )
                self.feature_cols.append('avg_order_value')
            
            # Customer tenure features
            if 'days_since_first_purchase' in feature_cols:
                data = data.withColumn(
                    'months_as_customer',
                    col('days_since_first_purchase') / 30.0
                )
                self.feature_cols.append('months_as_customer')
            
            # Purchase frequency features
            if 'total_purchases' in feature_cols and 'days_since_first_purchase' in feature_cols:
                data = data.withColumn(
                    'purchase_rate',
                    when(col('days_since_first_purchase') > 0, 
                         col('total_purchases') / (col('days_since_first_purchase') / 30.0))
                    .otherwise(0)
                )
                self.feature_cols.append('purchase_rate')
            
            # Engagement features
            if 'session_count' in feature_cols and 'purchase_count' in feature_cols:
                data = data.withColumn(
                    'conversion_rate',
                    when(col('session_count') > 0, col('purchase_count') / col('session_count'))
                    .otherwise(0)
                )
                self.feature_cols.append('conversion_rate')
            
            # Category diversity
            if 'unique_categories_purchased' in feature_cols:
                data = data.withColumn(
                    'category_diversity_score',
                    log1p(col('unique_categories_purchased'))
                )
                self.feature_cols.append('category_diversity_score')
            
            logger.info(f"Feature engineering complete. Total features: {len(self.feature_cols)}")
            return data
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def train(self, train_data: DataFrame, validation_data: Optional[DataFrame] = None,
              params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Train the CLV prediction model.
        
        Args:
            train_data: Training dataset with CLV label
            validation_data: Optional validation dataset
            params: Hyperparameters including:
                - algorithm: 'linear_regression', 'random_forest', or 'gbt'
                - feature_cols: List of feature column names
                - max_depth: Maximum tree depth (for tree models)
                - num_trees: Number of trees (for RF/GBT)
                - max_iter: Maximum iterations (for linear regression)
                - reg_param: Regularization parameter (for linear regression)
                - use_log_transform: Whether to log-transform CLV (default: True)
                
        Returns:
            Trained pipeline model
        """
        # Set default parameters
        default_params = {
            'algorithm': 'random_forest',
            'feature_cols': [],
            'max_depth': 10,
            'num_trees': 100,
            'max_iter': 100,
            'reg_param': 0.01,
            'use_log_transform': True,
        }
        params = {**default_params, **(params or {})}
        
        self.algorithm = params['algorithm']
        self.use_log_transform = params['use_log_transform']
        
        if self.algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {self.algorithm}. "
                f"Choose from {self.SUPPORTED_ALGORITHMS}"
            )
        
        # Validate data
        required_cols = params['feature_cols'] + [self.label_col]
        self.validate_data(train_data, required_cols)
        
        def _train_model(train_df: DataFrame, val_df: Optional[DataFrame],
                        hyperparams: Dict[str, Any]) -> Any:
            """Internal training function."""
            
            # Apply log transformation to target if specified
            if self.use_log_transform:
                train_df = train_df.withColumn(
                    f"{self.label_col}_log",
                    log1p(col(self.label_col))
                )
                target_col = f"{self.label_col}_log"
                logger.info("Applied log transformation to target variable")
            else:
                target_col = self.label_col
            
            # Feature engineering
            train_df = self.engineer_features(train_df, hyperparams['feature_cols'])
            
            # Create pipeline stages
            assembler = VectorAssembler(
                inputCols=self.feature_cols,
                outputCol="features",
                handleInvalid="skip"
            )
            
            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaled_features",
                withStd=True,
                withMean=True  # Mean centering for regression
            )
            
            # Select regressor
            if self.algorithm == 'linear_regression':
                regressor = LinearRegression(
                    featuresCol="scaled_features",
                    labelCol=target_col,
                    predictionCol=self.prediction_col,
                    maxIter=hyperparams['max_iter'],
                    regParam=hyperparams['reg_param'],
                    elasticNetParam=0.5,
                    standardization=False  # Already scaled
                )
            elif self.algorithm == 'random_forest':
                regressor = RandomForestRegressor(
                    featuresCol="scaled_features",
                    labelCol=target_col,
                    predictionCol=self.prediction_col,
                    maxDepth=hyperparams['max_depth'],
                    numTrees=hyperparams['num_trees'],
                    seed=42
                )
            else:  # gbt
                regressor = GBTRegressor(
                    featuresCol="scaled_features",
                    labelCol=target_col,
                    predictionCol=self.prediction_col,
                    maxDepth=hyperparams['max_depth'],
                    maxIter=hyperparams['num_trees'],
                    seed=42
                )
            
            # Create and fit pipeline
            pipeline = Pipeline(stages=[assembler, scaler, regressor])
            model = pipeline.fit(train_df)
            
            logger.info(f"Model training completed using {self.algorithm}")
            
            # Log feature importance
            feature_importance = self.get_feature_importance()
            if feature_importance:
                self.log_feature_importance(feature_importance)
            
            return model
        
        # Train with MLflow tracking
        model, metrics = self.train_with_mlflow(
            _train_model,
            train_data,
            validation_data,
            params,
            tags={'algorithm': self.algorithm, 'task': 'regression'}
        )
        
        self.model = model
        return model
    
    def predict(self, data: DataFrame) -> DataFrame:
        """
        Generate CLV predictions.
        
        Args:
            data: Input DataFrame with features
            
        Returns:
            DataFrame with CLV predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Feature engineering
            data = self.engineer_features(data,
                                         [col for col in self.feature_cols
                                          if col not in ['rfm_score', 'avg_order_value',
                                                        'months_as_customer', 'purchase_rate',
                                                        'conversion_rate', 'category_diversity_score']])
            
            # Make predictions
            predictions = self.model.transform(data)
            
            # Reverse log transformation if it was applied
            if self.use_log_transform:
                predictions = predictions.withColumn(
                    self.prediction_col,
                    exp(col(self.prediction_col)) - 1
                )
            
            # Ensure non-negative predictions
            predictions = predictions.withColumn(
                self.prediction_col,
                when(col(self.prediction_col) < 0, 0).otherwise(col(self.prediction_col))
            )
            
            logger.info("CLV predictions generated successfully")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def evaluate(self, data: DataFrame,
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            data: Test dataset with CLV labels
            metrics: List of metrics to compute (default: ['rmse', 'mae', 'r2'])
            
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2', 'mse']
        
        try:
            # Generate predictions
            predictions = self.predict(data)
            
            results = {}
            evaluator = RegressionEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col
            )
            
            # RMSE
            if 'rmse' in metrics:
                evaluator.setMetricName("rmse")
                results['rmse'] = evaluator.evaluate(predictions)
            
            # MAE
            if 'mae' in metrics:
                evaluator.setMetricName("mae")
                results['mae'] = evaluator.evaluate(predictions)
            
            # R²
            if 'r2' in metrics:
                evaluator.setMetricName("r2")
                results['r2'] = evaluator.evaluate(predictions)
            
            # MSE
            if 'mse' in metrics:
                evaluator.setMetricName("mse")
                results['mse'] = evaluator.evaluate(predictions)
            
            logger.info(f"Evaluation metrics: {results}")
            self.metrics = results
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from the trained model.
        
        Returns:
            Dictionary of feature importances
        """
        if self.model is None:
            return None
        
        try:
            # Get the regressor from the pipeline
            regressor = self.model.stages[-1]
            
            # Tree-based models have feature importance
            if self.algorithm in ['random_forest', 'gbt']:
                if hasattr(regressor, 'featureImportances'):
                    importances = regressor.featureImportances.toArray()
                    return dict(zip(self.feature_cols, importances))
            
            # Linear regression coefficients
            elif self.algorithm == 'linear_regression':
                if hasattr(regressor, 'coefficients'):
                    coefficients = regressor.coefficients.toArray()
                    # Use absolute values as importance
                    importances = [abs(coef) for coef in coefficients]
                    return dict(zip(self.feature_cols, importances))
            
            logger.warning(
                f"Feature importance not available for {self.algorithm}"
            )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}")
            return None
