"""
Churn Prediction Model

Classification model with deep learning support to predict customer churn.
Supports Random Forest, Gradient Boosted Trees, and Neural Networks with
SMOTE for handling class imbalance.
"""

from typing import Any, Dict, List, Optional
import logging
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import mlflow
import mlflow.keras

# Conditional import for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    SMOTE = None

from .base_model import BaseModel


logger = logging.getLogger(__name__)


class ChurnPredictionModel(BaseModel):
    """
    Binary classification model for predicting customer churn.
    
    Features:
    - Multiple algorithm support (RF, GBT, Neural Network)
    - SMOTE for handling class imbalance
    - Deep learning with TensorFlow/Keras
    - Comprehensive metrics (AUC-ROC, Precision, Recall)
    """
    
    SUPPORTED_ALGORITHMS = ['random_forest', 'gbt', 'neural_network']
    
    def __init__(self, name: str = "churn_prediction",
                 experiment_name: str = "churn_prediction_experiments"):
        """
        Initialize the Churn Prediction Model.
        
        Args:
            name: Model identifier
            experiment_name: MLflow experiment name
        """
        super().__init__(name, experiment_name)
        self.label_col = "churn"
        self.prediction_col = "prediction"
        self.probability_col = "probability"
        self.algorithm = None
        self.use_smote = False
    
    def engineer_features(self, data: DataFrame, feature_cols: List[str]) -> DataFrame:
        """
        Engineer features for churn prediction.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info("Engineering features for churn prediction")
            
            # Store original feature columns
            self.feature_cols = feature_cols.copy()
            
            # Engagement decline features
            if 'sessions_last_30_days' in feature_cols and 'sessions_prev_30_days' in feature_cols:
                data = data.withColumn(
                    'session_decline',
                    when(col('sessions_prev_30_days') > 0,
                         (col('sessions_prev_30_days') - col('sessions_last_30_days')) / 
                         col('sessions_prev_30_days'))
                    .otherwise(0)
                )
                self.feature_cols.append('session_decline')
            
            # Purchase frequency decline
            if 'purchases_last_90_days' in feature_cols and 'purchases_prev_90_days' in feature_cols:
                data = data.withColumn(
                    'purchase_decline',
                    when(col('purchases_prev_90_days') > 0,
                         (col('purchases_prev_90_days') - col('purchases_last_90_days')) /
                         col('purchases_prev_90_days'))
                    .otherwise(0)
                )
                self.feature_cols.append('purchase_decline')
            
            # Recency features
            if 'days_since_last_purchase' in feature_cols:
                data = data.withColumn(
                    'is_dormant',
                    when(col('days_since_last_purchase') > 90, 1.0).otherwise(0.0)
                )
                self.feature_cols.append('is_dormant')
            
            # Customer value features
            if 'total_spent' in feature_cols and 'total_purchases' in feature_cols:
                data = data.withColumn(
                    'avg_purchase_value',
                    when(col('total_purchases') > 0, col('total_spent') / col('total_purchases'))
                    .otherwise(0)
                )
                self.feature_cols.append('avg_purchase_value')
            
            # Engagement rate
            if 'total_sessions' in feature_cols and 'days_as_customer' in feature_cols:
                data = data.withColumn(
                    'session_frequency',
                    when(col('days_as_customer') > 0, 
                         col('total_sessions') / (col('days_as_customer') / 30.0))
                    .otherwise(0)
                )
                self.feature_cols.append('session_frequency')
            
            # Support ticket features
            if 'support_tickets' in feature_cols:
                data = data.withColumn(
                    'has_support_issues',
                    when(col('support_tickets') > 0, 1.0).otherwise(0.0)
                )
                self.feature_cols.append('has_support_issues')
            
            logger.info(f"Feature engineering complete. Total features: {len(self.feature_cols)}")
            return data
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def apply_smote(self, train_data: DataFrame) -> DataFrame:
        """
        Apply SMOTE to handle class imbalance.
        
        Args:
            train_data: Training DataFrame with features and label
            
        Returns:
            Balanced DataFrame
        """
        if not SMOTE_AVAILABLE:
            raise ImportError(
                "SMOTE is not available. Install imbalanced-learn: "
                "pip install imbalanced-learn"
            )
        
        try:
            logger.info("Applying SMOTE for class imbalance")
            
            # Convert to pandas for SMOTE
            pandas_df = train_data.select(self.feature_cols + [self.label_col]).toPandas()
            
            X = pandas_df[self.feature_cols].values
            y = pandas_df[self.label_col].values
            
            # Apply SMOTE
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Convert back to Spark DataFrame
            import pandas as pd
            resampled_df = pd.DataFrame(X_resampled, columns=self.feature_cols)
            resampled_df[self.label_col] = y_resampled
            
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            balanced_data = spark.createDataFrame(resampled_df)
            
            logger.info(f"SMOTE applied. Original size: {len(pandas_df)}, New size: {len(resampled_df)}")
            
            return balanced_data
            
        except Exception as e:
            logger.error(f"SMOTE failed: {str(e)}")
            raise
    
    def build_neural_network(self, input_dim: int) -> Any:
        """
        Build a neural network model using Keras.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            model = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.BatchNormalization(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.BatchNormalization(),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc'),
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall')]
            )
            
            logger.info("Neural network built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to build neural network: {str(e)}")
            raise
    
    def train(self, train_data: DataFrame, validation_data: Optional[DataFrame] = None,
              params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Train the churn prediction model.
        
        Args:
            train_data: Training dataset with churn label
            validation_data: Optional validation dataset
            params: Hyperparameters including:
                - algorithm: 'random_forest', 'gbt', or 'neural_network'
                - feature_cols: List of feature column names
                - max_depth: Maximum tree depth (for tree models)
                - num_trees: Number of trees (for RF/GBT)
                - use_smote: Whether to apply SMOTE (default: True)
                - epochs: Number of epochs (for neural network)
                - batch_size: Batch size (for neural network)
                
        Returns:
            Trained model
        """
        # Set default parameters
        default_params = {
            'algorithm': 'random_forest',
            'feature_cols': [],
            'max_depth': 10,
            'num_trees': 100,
            'use_smote': True,
            'epochs': 50,
            'batch_size': 32,
        }
        params = {**default_params, **(params or {})}
        
        self.algorithm = params['algorithm']
        self.use_smote = params['use_smote']
        
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
            
            # Apply SMOTE if requested
            if self.use_smote and self.algorithm != 'neural_network':
                train_df = self.apply_smote(train_df)
            
            if self.algorithm == 'neural_network':
                # Train neural network
                return self._train_neural_network(train_df, val_df, hyperparams)
            else:
                # Train tree-based model
                return self._train_tree_model(train_df, hyperparams)
        
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
    
    def _train_tree_model(self, train_data: DataFrame, params: Dict[str, Any]) -> Any:
        """Train tree-based model (RF or GBT)."""
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
                maxDepth=params['max_depth'],
                numTrees=params['num_trees'],
                seed=42
            )
        else:  # gbt
            classifier = GBTClassifier(
                featuresCol="scaled_features",
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                maxDepth=params['max_depth'],
                maxIter=params['num_trees'],
                seed=42
            )
        
        # Create and fit pipeline
        pipeline = Pipeline(stages=[assembler, scaler, classifier])
        model = pipeline.fit(train_data)
        
        logger.info(f"Tree-based model training completed using {self.algorithm}")
        
        # Log feature importance
        feature_importance = self.get_feature_importance()
        if feature_importance:
            self.log_feature_importance(feature_importance)
        
        return model
    
    def _train_neural_network(self, train_data: DataFrame, 
                             validation_data: Optional[DataFrame],
                             params: Dict[str, Any]) -> Any:
        """Train neural network model."""
        # Convert to numpy arrays
        train_pandas = train_data.select(self.feature_cols + [self.label_col]).toPandas()
        X_train = train_pandas[self.feature_cols].values
        y_train = train_pandas[self.label_col].values
        
        # Apply SMOTE if requested
        if self.use_smote:
            if not SMOTE_AVAILABLE:
                raise ImportError(
                    "SMOTE is not available. Install imbalanced-learn: "
                    "pip install imbalanced-learn"
                )
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Prepare validation data
        X_val, y_val = None, None
        if validation_data:
            val_pandas = validation_data.select(self.feature_cols + [self.label_col]).toPandas()
            X_val = val_pandas[self.feature_cols].values
            y_val = val_pandas[self.label_col].values
        
        # Build and train model
        model = self.build_neural_network(len(self.feature_cols))
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        validation_data_tuple = (X_val, y_val) if X_val is not None else None
        
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=validation_data_tuple,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Neural network training completed")
        
        # Log training history
        for metric_name in history.history.keys():
            if not metric_name.startswith('val_'):
                for epoch, value in enumerate(history.history[metric_name]):
                    mlflow.log_metric(f"train_{metric_name}", value, step=epoch)
        
        return model
    
    def predict(self, data: DataFrame) -> DataFrame:
        """
        Generate churn predictions.
        
        Args:
            data: Input DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            if self.algorithm == 'neural_network':
                return self._predict_neural_network(data)
            else:
                return self._predict_tree_model(data)
                
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def _predict_tree_model(self, data: DataFrame) -> DataFrame:
        """Predict using tree-based model."""
        # Feature engineering
        data = self.engineer_features(data,
                                     [col for col in self.feature_cols
                                      if col not in ['session_decline', 'purchase_decline',
                                                    'is_dormant', 'avg_purchase_value',
                                                    'session_frequency', 'has_support_issues']])
        
        # Make predictions
        predictions = self.model.transform(data)
        logger.info("Tree-based predictions generated successfully")
        return predictions
    
    def _predict_neural_network(self, data: DataFrame) -> DataFrame:
        """Predict using neural network."""
        from pyspark.sql import SparkSession
        
        # Feature engineering
        data = self.engineer_features(data,
                                     [col for col in self.feature_cols
                                      if col not in ['session_decline', 'purchase_decline',
                                                    'is_dormant', 'avg_purchase_value',
                                                    'session_frequency', 'has_support_issues']])
        
        # Convert to numpy
        pandas_df = data.select(self.feature_cols).toPandas()
        X = pandas_df[self.feature_cols].values
        
        # Make predictions
        probabilities = self.model.predict(X)
        predictions_binary = (probabilities > 0.5).astype(int).flatten()
        
        # Add predictions back to DataFrame
        pandas_df[self.prediction_col] = predictions_binary
        pandas_df[self.probability_col] = probabilities.flatten()
        
        spark = SparkSession.builder.getOrCreate()
        result_df = spark.createDataFrame(pandas_df)
        
        logger.info("Neural network predictions generated successfully")
        return result_df
    
    def evaluate(self, data: DataFrame,
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            data: Test dataset with labels
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ['auc', 'precision', 'recall', 'f1', 'accuracy']
        
        try:
            # Generate predictions
            predictions = self.predict(data)
            
            results = {}
            
            # For neural network, compute metrics differently
            if self.algorithm == 'neural_network':
                # Convert to pandas for metrics
                pandas_predictions = predictions.select([self.label_col, self.prediction_col]).toPandas()
                
                from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
                
                if 'auc' in metrics:
                    results['auc_roc'] = roc_auc_score(
                        pandas_predictions[self.label_col],
                        pandas_predictions[self.prediction_col]
                    )
                if 'precision' in metrics:
                    results['precision'] = precision_score(
                        pandas_predictions[self.label_col],
                        pandas_predictions[self.prediction_col]
                    )
                if 'recall' in metrics:
                    results['recall'] = recall_score(
                        pandas_predictions[self.label_col],
                        pandas_predictions[self.prediction_col]
                    )
                if 'f1' in metrics:
                    results['f1_score'] = f1_score(
                        pandas_predictions[self.label_col],
                        pandas_predictions[self.prediction_col]
                    )
                if 'accuracy' in metrics:
                    results['accuracy'] = accuracy_score(
                        pandas_predictions[self.label_col],
                        pandas_predictions[self.prediction_col]
                    )
            else:
                # Tree-based model evaluation
                if 'auc' in metrics:
                    auc_evaluator = BinaryClassificationEvaluator(
                        labelCol=self.label_col,
                        rawPredictionCol="rawPrediction",
                        metricName="areaUnderROC"
                    )
                    results['auc_roc'] = auc_evaluator.evaluate(predictions)
                
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
            if self.algorithm == 'neural_network':
                logger.warning("Feature importance not available for neural networks")
                return None
            
            # Get the classifier from the pipeline
            classifier = self.model.stages[-1]
            
            if hasattr(classifier, 'featureImportances'):
                importances = classifier.featureImportances.toArray()
                return dict(zip(self.feature_cols, importances))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}")
            return None
