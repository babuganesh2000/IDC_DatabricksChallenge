"""
Customer Segmentation Model

Clustering model for customer segmentation using K-Means and Bisecting K-Means.
Includes RFM-based segmentation and elbow method for optimal K selection.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit, udf
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from .base_model import BaseModel


logger = logging.getLogger(__name__)


class SegmentationModel(BaseModel):
    """
    Clustering model for customer segmentation.
    
    Features:
    - K-Means and Bisecting K-Means algorithms
    - RFM-based segmentation
    - Elbow method for optimal K selection
    - Silhouette score and within-cluster SSE metrics
    - Segment profiling and naming
    """
    
    SUPPORTED_ALGORITHMS = ['kmeans', 'bisecting_kmeans']
    
    # Default RFM segment names
    RFM_SEGMENTS = {
        (5, 5, 5): "Champions",
        (5, 4, 5): "Loyal Customers",
        (5, 5, 4): "Champions",
        (5, 4, 4): "Loyal Customers",
        (5, 3, 5): "Potential Loyalist",
        (4, 5, 5): "Champions",
        (4, 4, 5): "Loyal Customers",
        (4, 5, 4): "Champions",
        (4, 4, 4): "Loyal Customers",
        (3, 5, 5): "Potential Loyalist",
        (5, 2, 5): "At Risk",
        (5, 1, 5): "Can't Lose Them",
        (4, 2, 5): "At Risk",
        (4, 1, 5): "Can't Lose Them",
        (3, 3, 5): "Potential Loyalist",
        (5, 5, 2): "Big Spenders",
        (4, 5, 2): "Big Spenders",
        (3, 5, 3): "Promising",
        (2, 5, 5): "Promising",
        (2, 4, 5): "Promising",
        (1, 5, 5): "New Customers",
        (1, 4, 5): "New Customers",
        (1, 3, 5): "New Customers",
    }
    
    def __init__(self, name: str = "segmentation",
                 experiment_name: str = "segmentation_experiments"):
        """
        Initialize the Segmentation Model.
        
        Args:
            name: Model identifier
            experiment_name: MLflow experiment name
        """
        super().__init__(name, experiment_name)
        self.prediction_col = "segment"
        self.algorithm = None
        self.optimal_k = None
    
    def engineer_rfm_features(self, data: DataFrame,
                             recency_col: str = "recency",
                             frequency_col: str = "frequency", 
                             monetary_col: str = "monetary") -> DataFrame:
        """
        Engineer RFM features and create quintile scores.
        
        Args:
            data: Input DataFrame
            recency_col: Name of recency column
            frequency_col: Name of frequency column
            monetary_col: Name of monetary column
            
        Returns:
            DataFrame with RFM scores
        """
        try:
            logger.info("Engineering RFM features")
            
            from pyspark.sql.functions import percent_rank
            from pyspark.sql.window import Window
            
            # Create percentile ranks (inverted for recency - lower is better)
            recency_window = Window.orderBy(col(recency_col).desc())
            frequency_window = Window.orderBy(col(frequency_col))
            monetary_window = Window.orderBy(col(monetary_col))
            
            # Calculate percentile ranks
            data = data \
                .withColumn("r_rank", percent_rank().over(recency_window)) \
                .withColumn("f_rank", percent_rank().over(frequency_window)) \
                .withColumn("m_rank", percent_rank().over(monetary_window))
            
            # Convert to quintile scores (1-5)
            data = data \
                .withColumn("r_score",
                    when(col("r_rank") <= 0.2, 5)
                    .when(col("r_rank") <= 0.4, 4)
                    .when(col("r_rank") <= 0.6, 3)
                    .when(col("r_rank") <= 0.8, 2)
                    .otherwise(1)) \
                .withColumn("f_score",
                    when(col("f_rank") <= 0.2, 1)
                    .when(col("f_rank") <= 0.4, 2)
                    .when(col("f_rank") <= 0.6, 3)
                    .when(col("f_rank") <= 0.8, 4)
                    .otherwise(5)) \
                .withColumn("m_score",
                    when(col("m_rank") <= 0.2, 1)
                    .when(col("m_rank") <= 0.4, 2)
                    .when(col("m_rank") <= 0.6, 3)
                    .when(col("m_rank") <= 0.8, 4)
                    .otherwise(5))
            
            # Create combined RFM score
            data = data.withColumn(
                "rfm_score",
                col("r_score") * 100 + col("f_score") * 10 + col("m_score")
            )
            
            self.feature_cols = ["r_score", "f_score", "m_score"]
            
            logger.info("RFM features engineered successfully")
            return data
            
        except Exception as e:
            logger.error(f"RFM feature engineering failed: {str(e)}")
            raise
    
    def assign_rfm_segments(self, data: DataFrame) -> DataFrame:
        """
        Assign named segments based on RFM scores.
        
        Args:
            data: DataFrame with r_score, f_score, m_score columns
            
        Returns:
            DataFrame with segment names
        """
        try:
            logger.info("Assigning RFM segment names")
            
            def get_segment_name(r, f, m):
                """Map RFM scores to segment names."""
                key = (int(r), int(f), int(m))
                
                # Try exact match first
                if key in self.RFM_SEGMENTS:
                    return self.RFM_SEGMENTS[key]
                
                # Fallback to approximate matching
                if r >= 4 and f >= 4 and m >= 4:
                    return "Champions"
                elif r >= 4 and f >= 3:
                    return "Loyal Customers"
                elif r >= 4 and f <= 2:
                    return "At Risk"
                elif r <= 2 and f >= 4:
                    return "Promising"
                elif r <= 2 and f <= 2 and m >= 4:
                    return "Can't Lose Them"
                elif r <= 2 and f <= 2 and m <= 2:
                    return "Hibernating"
                elif f <= 2:
                    return "About to Sleep"
                else:
                    return "Need Attention"
            
            segment_udf = udf(get_segment_name, StringType())
            
            data = data.withColumn(
                "rfm_segment",
                segment_udf(col("r_score"), col("f_score"), col("m_score"))
            )
            
            logger.info("RFM segments assigned successfully")
            return data
            
        except Exception as e:
            logger.error(f"RFM segment assignment failed: {str(e)}")
            raise
    
    def find_optimal_k(self, data: DataFrame, 
                      feature_cols: List[str],
                      k_range: Tuple[int, int] = (2, 10)) -> int:
        """
        Use elbow method to find optimal number of clusters.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature columns
            k_range: Range of K values to test (min, max)
            
        Returns:
            Optimal K value
        """
        try:
            logger.info(f"Finding optimal K in range {k_range}")
            
            # Prepare features
            assembler = VectorAssembler(
                inputCols=feature_cols,
                outputCol="features",
                handleInvalid="skip"
            )
            
            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaled_features",
                withStd=True,
                withMean=True
            )
            
            prep_pipeline = Pipeline(stages=[assembler, scaler])
            prepared_data = prep_pipeline.fit(data).transform(data)
            
            # Test different K values
            costs = []
            silhouette_scores = []
            k_values = range(k_range[0], k_range[1] + 1)
            
            evaluator = ClusteringEvaluator(
                featuresCol="scaled_features",
                predictionCol=self.prediction_col,
                metricName="silhouette"
            )
            
            for k in k_values:
                logger.info(f"Testing K={k}")
                
                kmeans = KMeans(
                    featuresCol="scaled_features",
                    predictionCol=self.prediction_col,
                    k=k,
                    seed=42
                )
                
                model = kmeans.fit(prepared_data)
                predictions = model.transform(prepared_data)
                
                # Calculate metrics
                cost = model.summary.trainingCost
                silhouette = evaluator.evaluate(predictions)
                
                costs.append(cost)
                silhouette_scores.append(silhouette)
                
                logger.info(f"K={k}: Cost={cost:.2f}, Silhouette={silhouette:.4f}")
            
            # Find elbow using percentage change method
            cost_changes = []
            for i in range(1, len(costs)):
                pct_change = abs((costs[i] - costs[i-1]) / costs[i-1])
                cost_changes.append(pct_change)
            
            # Find point where improvement drops significantly
            if len(cost_changes) >= 2:
                change_diffs = [abs(cost_changes[i] - cost_changes[i-1]) 
                              for i in range(1, len(cost_changes))]
                elbow_idx = np.argmax(change_diffs) + 1
                optimal_k = k_values[elbow_idx + 1]
            else:
                # Fallback to highest silhouette score
                optimal_k = k_values[np.argmax(silhouette_scores)]
            
            logger.info(f"Optimal K determined: {optimal_k}")
            
            # Log K-selection metrics
            for i, k in enumerate(k_values):
                self.log_metrics({
                    f"k{k}_cost": costs[i],
                    f"k{k}_silhouette": silhouette_scores[i]
                })
            
            self.optimal_k = optimal_k
            return optimal_k
            
        except Exception as e:
            logger.error(f"Optimal K finding failed: {str(e)}")
            raise
    
    def train(self, train_data: DataFrame, validation_data: Optional[DataFrame] = None,
              params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Train the segmentation model.
        
        Args:
            train_data: Training dataset
            validation_data: Optional validation dataset (not used for clustering)
            params: Hyperparameters including:
                - algorithm: 'kmeans' or 'bisecting_kmeans'
                - feature_cols: List of feature column names
                - k: Number of clusters (or 'auto' to find optimal)
                - k_range: Range for auto K selection (default: (2, 10))
                - use_rfm: Whether to use RFM-based segmentation
                - max_iter: Maximum iterations
                
        Returns:
            Trained clustering model
        """
        # Set default parameters
        default_params = {
            'algorithm': 'kmeans',
            'feature_cols': [],
            'k': 5,
            'k_range': (2, 10),
            'use_rfm': False,
            'max_iter': 20,
        }
        params = {**default_params, **(params or {})}
        
        self.algorithm = params['algorithm']
        
        if self.algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {self.algorithm}. "
                f"Choose from {self.SUPPORTED_ALGORITHMS}"
            )
        
        # Validate data
        self.validate_data(train_data, params['feature_cols'])
        
        def _train_model(train_df: DataFrame, val_df: Optional[DataFrame],
                        hyperparams: Dict[str, Any]) -> Any:
            """Internal training function."""
            
            # RFM-based segmentation
            if hyperparams['use_rfm']:
                if all(col in train_df.columns for col in ['recency', 'frequency', 'monetary']):
                    train_df = self.engineer_rfm_features(train_df)
                    train_df = self.assign_rfm_segments(train_df)
                    hyperparams['feature_cols'] = self.feature_cols
                else:
                    logger.warning("RFM columns not found, using provided features")
            
            # Auto-select K if requested
            k = hyperparams['k']
            if k == 'auto':
                k = self.find_optimal_k(
                    train_df,
                    hyperparams['feature_cols'],
                    hyperparams['k_range']
                )
                logger.info(f"Using auto-selected K={k}")
            
            # Create pipeline stages
            assembler = VectorAssembler(
                inputCols=hyperparams['feature_cols'],
                outputCol="features",
                handleInvalid="skip"
            )
            
            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaled_features",
                withStd=True,
                withMean=True
            )
            
            # Select clustering algorithm
            if self.algorithm == 'kmeans':
                clusterer = KMeans(
                    featuresCol="scaled_features",
                    predictionCol=self.prediction_col,
                    k=k,
                    maxIter=hyperparams['max_iter'],
                    seed=42
                )
            else:  # bisecting_kmeans
                clusterer = BisectingKMeans(
                    featuresCol="scaled_features",
                    predictionCol=self.prediction_col,
                    k=k,
                    maxIter=hyperparams['max_iter'],
                    seed=42
                )
            
            # Create and fit pipeline
            pipeline = Pipeline(stages=[assembler, scaler, clusterer])
            model = pipeline.fit(train_df)
            
            logger.info(f"Clustering model training completed using {self.algorithm} with K={k}")
            
            # Log cluster centers if available
            clustering_model = model.stages[-1]
            if hasattr(clustering_model, 'clusterCenters'):
                centers = clustering_model.clusterCenters()
                self.log_params({f"cluster_{i}_center": str(center) 
                               for i, center in enumerate(centers)})
            
            return model
        
        # Train with MLflow tracking
        model, metrics = self.train_with_mlflow(
            _train_model,
            train_data,
            validation_data,
            params,
            tags={'algorithm': self.algorithm, 'task': 'clustering'}
        )
        
        self.model = model
        self.feature_cols = params['feature_cols']
        return model
    
    def predict(self, data: DataFrame) -> DataFrame:
        """
        Assign clusters to data.
        
        Args:
            data: Input DataFrame with features
            
        Returns:
            DataFrame with cluster assignments
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Make predictions
            predictions = self.model.transform(data)
            
            logger.info("Cluster assignments generated successfully")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def evaluate(self, data: DataFrame,
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the clustering model.
        
        Args:
            data: Dataset to evaluate
            metrics: List of metrics to compute (default: ['silhouette', 'sse'])
            
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ['silhouette', 'sse', 'cluster_sizes']
        
        try:
            # Generate predictions
            predictions = self.predict(data)
            
            results = {}
            
            # Silhouette score
            if 'silhouette' in metrics:
                evaluator = ClusteringEvaluator(
                    featuresCol="scaled_features",
                    predictionCol=self.prediction_col,
                    metricName="silhouette"
                )
                results['silhouette_score'] = evaluator.evaluate(predictions)
            
            # Within-cluster sum of squared errors
            if 'sse' in metrics:
                clustering_model = self.model.stages[-1]
                if hasattr(clustering_model, 'summary'):
                    if hasattr(clustering_model.summary, 'trainingCost'):
                        results['within_cluster_sse'] = clustering_model.summary.trainingCost
            
            # Cluster size distribution
            if 'cluster_sizes' in metrics:
                cluster_counts = predictions.groupBy(self.prediction_col).count().collect()
                for row in cluster_counts:
                    results[f'cluster_{row[self.prediction_col]}_size'] = row['count']
            
            logger.info(f"Evaluation metrics: {results}")
            self.metrics = results
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def get_cluster_profiles(self, data: DataFrame) -> Dict[int, Dict[str, float]]:
        """
        Get statistical profiles for each cluster.
        
        Args:
            data: DataFrame with predictions
            
        Returns:
            Dictionary mapping cluster IDs to feature statistics
        """
        try:
            logger.info("Generating cluster profiles")
            
            predictions = self.predict(data) if self.prediction_col not in data.columns else data
            
            profiles = {}
            
            # Get unique clusters
            clusters = predictions.select(self.prediction_col).distinct().collect()
            
            for cluster_row in clusters:
                cluster_id = cluster_row[self.prediction_col]
                cluster_data = predictions.filter(col(self.prediction_col) == cluster_id)
                
                # Calculate statistics for each feature
                profile = {}
                for feature in self.feature_cols:
                    if feature in cluster_data.columns:
                        stats = cluster_data.select(feature).summary("mean", "stddev", "min", "max").collect()
                        profile[feature] = {
                            'mean': float(stats[0][feature]) if stats[0][feature] else 0.0,
                            'stddev': float(stats[1][feature]) if stats[1][feature] else 0.0,
                            'min': float(stats[2][feature]) if stats[2][feature] else 0.0,
                            'max': float(stats[3][feature]) if stats[3][feature] else 0.0,
                        }
                
                profile['size'] = cluster_data.count()
                profiles[cluster_id] = profile
            
            logger.info(f"Generated profiles for {len(profiles)} clusters")
            return profiles
            
        except Exception as e:
            logger.error(f"Cluster profiling failed: {str(e)}")
            raise
    
    def get_cluster_centers(self) -> Optional[List[np.ndarray]]:
        """
        Get cluster centers.
        
        Returns:
            List of cluster center vectors
        """
        if self.model is None:
            return None
        
        try:
            clustering_model = self.model.stages[-1]
            if hasattr(clustering_model, 'clusterCenters'):
                return [np.array(center) for center in clustering_model.clusterCenters()]
            
            logger.warning("Cluster centers not available for this algorithm")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cluster centers: {str(e)}")
            return None
