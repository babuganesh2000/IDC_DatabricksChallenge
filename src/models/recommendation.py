"""
Recommendation Model

Collaborative filtering model using Alternating Least Squares (ALS) algorithm
for product recommendations with weighted interactions.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, countDistinct, collect_list, size, array_distinct
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from .base_model import BaseModel


logger = logging.getLogger(__name__)


class RecommendationModel(BaseModel):
    """
    Collaborative filtering recommendation model using ALS.
    
    Features:
    - ALS (Alternating Least Squares) implementation
    - Weighted interactions (view=1, cart=3, purchase=5)
    - Hyperparameter tuning with cross-validation
    - Coverage and diversity metrics
    - Cold-start handling
    """
    
    # Interaction weights
    INTERACTION_WEIGHTS = {
        'view': 1.0,
        'cart': 3.0,
        'purchase': 5.0
    }
    
    def __init__(self, name: str = "recommendation",
                 experiment_name: str = "recommendation_experiments"):
        """
        Initialize the Recommendation Model.
        
        Args:
            name: Model identifier
            experiment_name: MLflow experiment name
        """
        super().__init__(name, experiment_name)
        self.user_col = "user_id"
        self.item_col = "item_id"
        self.rating_col = "rating"
        self.prediction_col = "prediction"
    
    def prepare_interactions(self, data: DataFrame, 
                           interaction_col: str = "event_type",
                           weights: Optional[Dict[str, float]] = None) -> DataFrame:
        """
        Prepare and weight user-item interactions.
        
        Args:
            data: DataFrame with user_id, item_id, and event_type columns
            interaction_col: Name of the interaction type column
            weights: Custom weights for interaction types
            
        Returns:
            DataFrame with weighted ratings
        """
        try:
            logger.info("Preparing interaction data")
            
            if weights is None:
                weights = self.INTERACTION_WEIGHTS
            
            # Map interaction types to weights
            from pyspark.sql.functions import when
            
            rating_expr = None
            for event_type, weight in weights.items():
                if rating_expr is None:
                    rating_expr = when(col(interaction_col) == event_type, weight)
                else:
                    rating_expr = rating_expr.when(col(interaction_col) == event_type, weight)
            
            # Default weight for unknown interaction types
            rating_expr = rating_expr.otherwise(1.0)
            
            # Create ratings DataFrame
            interactions = data.withColumn(self.rating_col, rating_expr)
            
            # Aggregate multiple interactions
            interactions = interactions.groupBy(self.user_col, self.item_col).agg(
                (sum(col(self.rating_col)) / count("*")).alias(self.rating_col)
            )
            
            logger.info(f"Prepared {interactions.count()} user-item interactions")
            
            return interactions
            
        except Exception as e:
            logger.error(f"Failed to prepare interactions: {str(e)}")
            raise
    
    def train(self, train_data: DataFrame, validation_data: Optional[DataFrame] = None,
              params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Train the recommendation model using ALS.
        
        Args:
            train_data: Training dataset with user_id, item_id, and rating
            validation_data: Optional validation dataset
            params: Hyperparameters including:
                - rank: Number of latent factors (default: 10)
                - max_iter: Maximum iterations (default: 10)
                - reg_param: Regularization parameter (default: 0.1)
                - alpha: Confidence parameter for implicit feedback (default: 1.0)
                - implicit_prefs: Whether to use implicit preferences (default: False)
                - cold_start_strategy: Strategy for cold start ('nan' or 'drop')
                - tune_hyperparameters: Whether to perform hyperparameter tuning
                
        Returns:
            Trained ALS model
        """
        # Set default parameters
        default_params = {
            'rank': 10,
            'max_iter': 10,
            'reg_param': 0.1,
            'alpha': 1.0,
            'implicit_prefs': False,
            'cold_start_strategy': 'drop',
            'tune_hyperparameters': False,
            'interaction_col': 'event_type',
            'weights': None,
        }
        params = {**default_params, **(params or {})}
        
        # Validate data
        required_cols = [self.user_col, self.item_col]
        
        # Check if we need to prepare interactions
        if self.rating_col not in train_data.columns:
            if params['interaction_col'] in train_data.columns:
                train_data = self.prepare_interactions(
                    train_data,
                    params['interaction_col'],
                    params['weights']
                )
            else:
                raise ValueError(
                    f"Data must have either '{self.rating_col}' or "
                    f"'{params['interaction_col']}' column"
                )
        
        required_cols.append(self.rating_col)
        self.validate_data(train_data, required_cols)
        
        def _train_model(train_df: DataFrame, val_df: Optional[DataFrame],
                        hyperparams: Dict[str, Any]) -> Any:
            """Internal training function."""
            
            if hyperparams['tune_hyperparameters'] and val_df is not None:
                logger.info("Performing hyperparameter tuning")
                model = self._train_with_tuning(train_df, val_df, hyperparams)
            else:
                logger.info("Training with fixed hyperparameters")
                model = self._train_fixed(train_df, hyperparams)
            
            return model
        
        # Train with MLflow tracking
        model, metrics = self.train_with_mlflow(
            _train_model,
            train_data,
            validation_data,
            params,
            tags={'algorithm': 'ALS', 'task': 'recommendation'}
        )
        
        self.model = model
        return model
    
    def _train_fixed(self, train_data: DataFrame, params: Dict[str, Any]) -> ALSModel:
        """Train ALS with fixed hyperparameters."""
        als = ALS(
            rank=params['rank'],
            maxIter=params['max_iter'],
            regParam=params['reg_param'],
            alpha=params['alpha'],
            userCol=self.user_col,
            itemCol=self.item_col,
            ratingCol=self.rating_col,
            coldStartStrategy=params['cold_start_strategy'],
            implicitPrefs=params['implicit_prefs'],
            seed=42
        )
        
        model = als.fit(train_data)
        logger.info("ALS model training completed")
        
        return model
    
    def _train_with_tuning(self, train_data: DataFrame, 
                          validation_data: DataFrame,
                          params: Dict[str, Any]) -> ALSModel:
        """Train ALS with hyperparameter tuning."""
        als = ALS(
            userCol=self.user_col,
            itemCol=self.item_col,
            ratingCol=self.rating_col,
            coldStartStrategy=params['cold_start_strategy'],
            implicitPrefs=params['implicit_prefs'],
            seed=42
        )
        
        # Create parameter grid
        param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [5, 10, 20]) \
            .addGrid(als.maxIter, [5, 10, 15]) \
            .addGrid(als.regParam, [0.01, 0.1, 1.0]) \
            .build()
        
        # Create evaluator
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol=self.rating_col,
            predictionCol=self.prediction_col
        )
        
        # Cross-validation
        cv = CrossValidator(
            estimator=als,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            seed=42
        )
        
        # Fit model
        cv_model = cv.fit(train_data)
        best_model = cv_model.bestModel
        
        # Log best parameters
        self.log_params({
            'best_rank': best_model.rank,
            'best_maxIter': best_model._java_obj.parent().getMaxIter(),
            'best_regParam': best_model._java_obj.parent().getRegParam(),
        })
        
        logger.info("Hyperparameter tuning completed")
        
        return best_model
    
    def predict(self, data: DataFrame, top_n: int = 10) -> DataFrame:
        """
        Generate top-N recommendations for users.
        
        Args:
            data: DataFrame with user_id column
            top_n: Number of recommendations per user
            
        Returns:
            DataFrame with recommendations
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Get unique users
            if self.user_col not in data.columns:
                raise ValueError(f"Data must contain '{self.user_col}' column")
            
            users = data.select(self.user_col).distinct()
            
            # Generate recommendations
            recommendations = self.model.recommendForUserSubset(users, top_n)
            
            logger.info(f"Generated top-{top_n} recommendations for {users.count()} users")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            raise
    
    def predict_for_all_users(self, top_n: int = 10) -> DataFrame:
        """
        Generate recommendations for all users in the training set.
        
        Args:
            top_n: Number of recommendations per user
            
        Returns:
            DataFrame with recommendations
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            recommendations = self.model.recommendForAllUsers(top_n)
            logger.info(f"Generated top-{top_n} recommendations for all users")
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            raise
    
    def predict_for_items(self, data: DataFrame, top_n: int = 10) -> DataFrame:
        """
        Generate top-N similar items for given items.
        
        Args:
            data: DataFrame with item_id column
            top_n: Number of similar items to return
            
        Returns:
            DataFrame with similar items
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            items = data.select(self.item_col).distinct()
            similar_items = self.model.recommendForItemSubset(items, top_n)
            
            logger.info(f"Generated top-{top_n} similar items")
            return similar_items
            
        except Exception as e:
            logger.error(f"Similar items generation failed: {str(e)}")
            raise
    
    def evaluate(self, data: DataFrame,
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the recommendation model.
        
        Args:
            data: Test dataset with user_id, item_id, and rating
            metrics: List of metrics to compute (default: ['rmse', 'coverage', 'diversity'])
            
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'coverage', 'diversity']
        
        try:
            results = {}
            
            # Prepare data if needed
            if self.rating_col not in data.columns:
                raise ValueError(f"Data must contain '{self.rating_col}' column for evaluation")
            
            # Generate predictions
            predictions = self.model.transform(data)
            
            # RMSE and MAE
            if 'rmse' in metrics or 'mae' in metrics:
                evaluator = RegressionEvaluator(
                    labelCol=self.rating_col,
                    predictionCol=self.prediction_col
                )
                
                if 'rmse' in metrics:
                    evaluator.setMetricName("rmse")
                    results['rmse'] = evaluator.evaluate(predictions)
                
                if 'mae' in metrics:
                    evaluator.setMetricName("mae")
                    results['mae'] = evaluator.evaluate(predictions)
            
            # Coverage: percentage of items that can be recommended
            if 'coverage' in metrics:
                total_items = data.select(self.item_col).distinct().count()
                recommended_items = self.model.recommendForAllUsers(10) \
                    .select("recommendations.item_id") \
                    .selectExpr("explode(item_id) as item_id") \
                    .distinct() \
                    .count()
                results['coverage'] = recommended_items / total_items if total_items > 0 else 0
            
            # Diversity: average number of unique items recommended
            if 'diversity' in metrics:
                user_recs = self.model.recommendForAllUsers(10)
                diversity_df = user_recs.select(
                    size(array_distinct(col("recommendations.item_id"))).alias("unique_items")
                )
                avg_diversity = diversity_df.agg({"unique_items": "avg"}).collect()[0][0]
                results['diversity'] = avg_diversity / 10.0 if avg_diversity else 0
            
            logger.info(f"Evaluation metrics: {results}")
            self.metrics = results
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def get_user_factors(self) -> DataFrame:
        """
        Get user latent factor matrix.
        
        Returns:
            DataFrame with user factors
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.userFactors
    
    def get_item_factors(self) -> DataFrame:
        """
        Get item latent factor matrix.
        
        Returns:
            DataFrame with item factors
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.itemFactors
    
    def get_similar_users(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar users based on latent factors.
        
        Args:
            user_id: Target user ID
            top_n: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            from pyspark.sql.functions import udf, array
            from pyspark.sql.types import DoubleType
            import numpy as np
            
            # Get user factors
            user_factors = self.model.userFactors
            
            # Get target user's factors
            target_factors = user_factors.filter(col("id") == user_id).collect()
            if not target_factors:
                logger.warning(f"User {user_id} not found in model")
                return []
            
            target_vector = np.array(target_factors[0]["features"])
            
            # Calculate cosine similarity
            def cosine_similarity(features):
                if features is None:
                    return 0.0
                vec = np.array(features)
                similarity = np.dot(target_vector, vec) / (
                    np.linalg.norm(target_vector) * np.linalg.norm(vec)
                )
                return float(similarity)
            
            similarity_udf = udf(cosine_similarity, DoubleType())
            
            # Compute similarities
            similar_users = user_factors \
                .filter(col("id") != user_id) \
                .withColumn("similarity", similarity_udf(col("features"))) \
                .orderBy(col("similarity").desc()) \
                .limit(top_n) \
                .select("id", "similarity") \
                .collect()
            
            result = [(row["id"], row["similarity"]) for row in similar_users]
            
            logger.info(f"Found {len(result)} similar users for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to find similar users: {str(e)}")
            raise
