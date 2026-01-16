"""
Purchase Prediction Model

Binary classification model to predict whether a customer will make a purchase.
Supports multiple algorithms including Random Forest, Gradient Boosted Trees,
and Logistic Regression.
"""

from typing import Any, Dict, List, Optional
import logging
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    RandomForestClassifier,
    GBTClassifier,
    LogisticRegression
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from .base_model import BaseModel


logger = logging.getLogger(__name__)


class PurchasePredictionModel(BaseModel):
    """
    Binary classification model for predicting customer purchases.
    
    Features:
    - Multiple algorithm support (RF, GBT, LogisticRegression)
    - Feature engineering for purchase behavior
    - Comprehensive metrics (AUC-ROC, Precision, Recall, F1)
    - Probability calibration
    """
    
    SUPPORTED_ALGORITHMS = ['random_forest', 'gbt', 'logistic_regression']
    
    def __init__(self, name: str = "purchase_prediction", 
                 experiment_name: str = "purchase_prediction_experiments"):
        """
        Initialize the Purchase Prediction Model.
        
        Args:
            name: Model identifier
            experiment_name: MLflow experiment name
        """
        super().__init__(name, experiment_name)
        self.label_col = "label"
        self.prediction_col = "prediction"
        self.probability_col = "probability"
        self.algorithm = None
    
    def engineer_features(self, data: DataFrame, feature_cols: List[str]) -> DataFrame:
        """
        Engineer features for purchase prediction.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info("Engineering features for purchase prediction")
            
            # Store original feature columns
            self.feature_cols = feature_cols.copy()
            
            # Add interaction features if behavioral data exists
            if 'view_count' in feature_cols and 'cart_count' in feature_cols:
                data = data.withColumn(
                    'view_to_cart_ratio',
                    when(col('view_count') > 0, col('cart_count') / col('view_count'))
                    .otherwise(0)
                )
                self.feature_cols.append('view_to_cart_ratio')
            
            if 'session_duration' in feature_cols and 'page_views' in feature_cols:
                data = data.withColumn(
                    'avg_time_per_page',
                    when(col('page_views') > 0, col('session_duration') / col('page_views'))
                    .otherwise(0)
                )
                self.feature_cols.append('avg_time_per_page')
            
            # Recency-based features
            if 'days_since_last_visit' in feature_cols:
                data = data.withColumn(
                    'is_recent_visitor',
                    when(col('days_since_last_visit') <= 7, 1.0).otherwise(0.0)
                )
                self.feature_cols.append('is_recent_visitor')
            
            logger.info(f"Feature engineering complete. Total features: {len(self.feature_cols)}")
            return data
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def train(self, train_data: DataFrame, validation_data: Optional[DataFrame] = None,
              params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Train the purchase prediction model.
        
        Args:
            train_data: Training dataset with label column
            validation_data: Optional validation dataset
            params: Hyperparameters including:
                - algorithm: 'random_forest', 'gbt', or 'logistic_regression'
                - feature_cols: List of feature column names
                - max_depth: Maximum tree depth (for tree models)
                - num_trees: Number of trees (for RF/GBT)
                - max_iter: Maximum iterations (for logistic regression)
                - reg_param: Regularization parameter (for logistic regression)
                
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
        }
        params = {**default_params, **(params or {})}
        
        self.algorithm = params['algorithm']
        
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
                withMean=False
            )
            
            # Select classifier
            if self.algorithm == 'random_forest':
                classifier = RandomForestClassifier(
                    featuresCol="scaled_features",
                    labelCol=self.label_col,
                    predictionCol=self.prediction_col,
                    probabilityCol=self.probability_col,
                    maxDepth=hyperparams['max_depth'],
                    numTrees=hyperparams['num_trees'],
                    seed=42
                )
            elif self.algorithm == 'gbt':
                classifier = GBTClassifier(
                    featuresCol="scaled_features",
                    labelCol=self.label_col,
                    predictionCol=self.prediction_col,
                    maxDepth=hyperparams['max_depth'],
                    maxIter=hyperparams['num_trees'],
                    seed=42
                )
            else:  # logistic_regression
                classifier = LogisticRegression(
                    featuresCol="scaled_features",
                    labelCol=self.label_col,
                    predictionCol=self.prediction_col,
                    probabilityCol=self.probability_col,
                    maxIter=hyperparams['max_iter'],
                    regParam=hyperparams['reg_param'],
                    elasticNetParam=0.5,
                    family="binomial"
                )
            
            # Create and fit pipeline
            pipeline = Pipeline(stages=[assembler, scaler, classifier])
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
            tags={'algorithm': self.algorithm, 'task': 'binary_classification'}
        )
        
        self.model = model
        return model
    
    def predict(self, data: DataFrame) -> DataFrame:
        """
        Generate purchase predictions.
        
        Args:
            data: Input DataFrame with features
            
        Returns:
            DataFrame with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Feature engineering
            data = self.engineer_features(data, 
                                         [col for col in self.feature_cols 
                                          if col not in ['view_to_cart_ratio', 
                                                        'avg_time_per_page',
                                                        'is_recent_visitor']])
            
            # Make predictions
            predictions = self.model.transform(data)
            
            logger.info("Predictions generated successfully")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def evaluate(self, data: DataFrame, 
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            data: Test dataset with labels
            metrics: List of metrics to compute 
                    (default: ['auc', 'precision', 'recall', 'f1'])
                    
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ['auc', 'precision', 'recall', 'f1', 'accuracy']
        
        try:
            # Generate predictions
            predictions = self.predict(data)
            
            results = {}
            
            # AUC-ROC
            if 'auc' in metrics:
                auc_evaluator = BinaryClassificationEvaluator(
                    labelCol=self.label_col,
                    rawPredictionCol="rawPrediction",
                    metricName="areaUnderROC"
                )
                results['auc_roc'] = auc_evaluator.evaluate(predictions)
            
            # Precision, Recall, F1, Accuracy
            multiclass_evaluator = MulticlassClassificationEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col
            )
            
            if 'precision' in metrics:
                multiclass_evaluator.setMetricName("weightedPrecision")
                results['precision'] = multiclass_evaluator.evaluate(predictions)
            
            if 'recall' in metrics:
                multiclass_evaluator.setMetricName("weightedRecall")
                results['recall'] = multiclass_evaluator.evaluate(predictions)
            
            if 'f1' in metrics:
                multiclass_evaluator.setMetricName("f1")
                results['f1_score'] = multiclass_evaluator.evaluate(predictions)
            
            if 'accuracy' in metrics:
                multiclass_evaluator.setMetricName("accuracy")
                results['accuracy'] = multiclass_evaluator.evaluate(predictions)
            
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
            # Get the classifier from the pipeline
            classifier = self.model.stages[-1]
            
            # Only tree-based models have feature importance
            if self.algorithm in ['random_forest', 'gbt']:
                if hasattr(classifier, 'featureImportances'):
                    importances = classifier.featureImportances.toArray()
                    return dict(zip(self.feature_cols, importances))
            
            logger.warning(
                f"Feature importance not available for {self.algorithm}"
            )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}")
            return None
