"""Batch scoring pipeline for distributed inference.

Provides batch inference capabilities with Delta table integration and PySpark support.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType

import mlflow
from mlflow.pyfunc import PyFuncModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class BatchScoringPipeline:
    """Batch inference pipeline with distributed processing."""

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        checkpoint_location: Optional[str] = None,
    ):
        """Initialize batch scoring pipeline.

        Args:
            spark: SparkSession instance (creates new if None)
            checkpoint_location: Location for checkpoints (e.g., dbfs:/checkpoints/batch_scoring)
        """
        self.spark = spark or SparkSession.builder.appName("BatchScoringPipeline").getOrCreate()
        self.checkpoint_location = checkpoint_location or "/tmp/batch_scoring_checkpoints"
        self.model = None
        self.model_uri = None

        logger.info(
            "Initialized BatchScoringPipeline",
            app_name=self.spark.sparkContext.appName,
            checkpoint_location=self.checkpoint_location,
        )

    def load_model(self, model_uri: str, result_type: Optional[Any] = None) -> None:
        """Load model from MLflow.

        Args:
            model_uri: URI to model (e.g., models:/model_name/Production or runs:/<run_id>/model)
            result_type: Spark SQL DataType for model output (default: ArrayType(DoubleType()))

        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info("Loading model for batch scoring", model_uri=model_uri)
            
            # Default to array of doubles for classification probabilities
            if result_type is None:
                result_type = ArrayType(DoubleType())
            
            self.model = mlflow.pyfunc.spark_udf(
                self.spark,
                model_uri=model_uri,
                result_type=result_type,
            )
            self.model_uri = model_uri
            logger.info("Model loaded successfully", model_uri=model_uri)

        except Exception as e:
            logger.error("Failed to load model", model_uri=model_uri, error=str(e))
            raise

    def run_batch_inference(
        self,
        input_data: DataFrame,
        feature_columns: List[str],
        prediction_column: str = "prediction",
        probability_column: Optional[str] = "probability",
        batch_size: Optional[int] = None,
    ) -> DataFrame:
        """Run batch inference on input data.

        Args:
            input_data: Input DataFrame with features
            feature_columns: List of feature column names
            prediction_column: Name for prediction column
            probability_column: Name for probability column (None to skip)
            batch_size: Batch size for processing (None for auto)

        Returns:
            DataFrame with predictions

        Raises:
            ValueError: If model not loaded or invalid columns
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            batch_id = str(uuid.uuid4())
            logger.info(
                "Starting batch inference",
                batch_id=batch_id,
                input_rows=input_data.count(),
                feature_columns=feature_columns,
            )

            # Validate feature columns exist
            missing_cols = set(feature_columns) - set(input_data.columns)
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")

            # Create feature struct for model input
            feature_struct = F.struct(*[F.col(col) for col in feature_columns])

            # Apply model
            predictions_df = input_data.withColumn("_model_input", feature_struct).withColumn(
                "_predictions", self.model(F.col("_model_input"))
            )

            # Extract predictions (assuming binary classification)
            predictions_df = predictions_df.withColumn(
                prediction_column, F.element_at(F.col("_predictions"), 1).cast("double")
            )

            # Extract probabilities if requested
            if probability_column:
                predictions_df = predictions_df.withColumn(
                    probability_column, F.col("_predictions")
                )

            # Add metadata
            predictions_df = predictions_df.withColumn("batch_id", F.lit(batch_id)).withColumn(
                "prediction_timestamp", F.lit(datetime.utcnow().isoformat())
            ).withColumn("model_uri", F.lit(self.model_uri))

            # Drop temporary columns
            predictions_df = predictions_df.drop("_model_input", "_predictions")

            output_rows = predictions_df.count()
            logger.info(
                "Batch inference completed",
                batch_id=batch_id,
                output_rows=output_rows,
            )

            return predictions_df

        except Exception as e:
            logger.error("Batch inference failed", error=str(e))
            raise

    def save_predictions(
        self,
        predictions: DataFrame,
        output_path: str,
        mode: str = "append",
        partition_by: Optional[List[str]] = None,
        format: str = "delta",
        optimize: bool = True,
        z_order_by: Optional[List[str]] = None,
    ) -> None:
        """Save predictions to storage.

        Args:
            predictions: DataFrame with predictions
            output_path: Output path (e.g., dbfs:/mnt/predictions/model_v1)
            mode: Write mode (append, overwrite, error, ignore)
            partition_by: Columns to partition by
            format: Output format (delta, parquet, etc.)
            optimize: Run OPTIMIZE on Delta table
            z_order_by: Columns for Z-ordering (Delta only)

        Raises:
            Exception: If save fails
        """
        try:
            logger.info(
                "Saving predictions",
                output_path=output_path,
                mode=mode,
                format=format,
                partition_by=partition_by,
            )

            # Write predictions
            writer = predictions.write.format(format).mode(mode)

            if partition_by:
                writer = writer.partitionBy(*partition_by)

            writer.save(output_path)

            # Optimize Delta table if requested
            if format == "delta" and optimize:
                self._optimize_delta_table(output_path, z_order_by)

            logger.info("Predictions saved successfully", output_path=output_path)

        except Exception as e:
            logger.error("Failed to save predictions", output_path=output_path, error=str(e))
            raise

    def run_batch_with_checkpoint(
        self,
        input_path: str,
        output_path: str,
        feature_columns: List[str],
        model_uri: str,
        checkpoint_column: str = "id",
        **inference_kwargs,
    ) -> DataFrame:
        """Run batch inference with checkpoint support for fault tolerance.

        Args:
            input_path: Path to input data
            output_path: Path to save predictions
            feature_columns: List of feature column names
            model_uri: URI to model
            checkpoint_column: Column to use for checkpointing
            **inference_kwargs: Additional arguments for run_batch_inference

        Returns:
            DataFrame with predictions
        """
        try:
            batch_id = str(uuid.uuid4())
            checkpoint_path = f"{self.checkpoint_location}/{batch_id}"

            logger.info(
                "Starting batch inference with checkpointing",
                batch_id=batch_id,
                input_path=input_path,
                checkpoint_path=checkpoint_path,
            )

            # Load model
            self.load_model(model_uri)

            # Read input data
            input_data = self.spark.read.format("delta").load(input_path)

            # Check for existing predictions to resume from checkpoint
            try:
                existing_predictions = self.spark.read.format("delta").load(output_path)
                processed_ids = existing_predictions.select(checkpoint_column).distinct()

                # Filter out already processed records
                input_data = input_data.join(processed_ids, on=checkpoint_column, how="left_anti")

                logger.info(
                    "Resuming from checkpoint",
                    remaining_rows=input_data.count(),
                )
            except Exception:
                logger.info("No existing predictions found, processing all data")

            # Run inference
            predictions = self.run_batch_inference(input_data, feature_columns, **inference_kwargs)

            # Save predictions
            self.save_predictions(predictions, output_path, mode="append")

            return predictions

        except Exception as e:
            logger.error("Batch inference with checkpoint failed", error=str(e))
            raise

    def process_streaming_batch(
        self,
        input_path: str,
        output_path: str,
        feature_columns: List[str],
        model_uri: str,
        trigger_interval: str = "10 minutes",
        **inference_kwargs,
    ) -> None:
        """Process streaming batches from Delta table.

        Args:
            input_path: Path to input Delta table
            output_path: Path to output Delta table
            feature_columns: List of feature column names
            model_uri: URI to model
            trigger_interval: Trigger interval for streaming
            **inference_kwargs: Additional arguments for run_batch_inference
        """
        try:
            logger.info(
                "Starting streaming batch processing",
                input_path=input_path,
                output_path=output_path,
                trigger_interval=trigger_interval,
            )

            # Load model
            self.load_model(model_uri)

            # Read streaming data
            streaming_df = (
                self.spark.readStream.format("delta")
                .option("ignoreChanges", "true")
                .load(input_path)
            )

            # Define processing function
            def process_batch(batch_df: DataFrame, batch_id: int) -> None:
                if batch_df.count() > 0:
                    logger.info("Processing streaming batch", batch_id=batch_id, rows=batch_df.count())
                    predictions = self.run_batch_inference(batch_df, feature_columns, **inference_kwargs)
                    self.save_predictions(predictions, output_path, mode="append")

            # Start streaming query
            query = (
                streaming_df.writeStream.foreachBatch(process_batch)
                .trigger(processingTime=trigger_interval)
                .option("checkpointLocation", f"{self.checkpoint_location}/streaming")
                .start()
            )

            logger.info("Streaming batch processing started")
            query.awaitTermination()

        except Exception as e:
            logger.error("Streaming batch processing failed", error=str(e))
            raise

    def _optimize_delta_table(self, table_path: str, z_order_by: Optional[List[str]] = None) -> None:
        """Optimize Delta table.

        Args:
            table_path: Path to Delta table
            z_order_by: Columns for Z-ordering
        """
        try:
            logger.info("Optimizing Delta table", table_path=table_path, z_order_by=z_order_by)

            optimize_cmd = f"OPTIMIZE delta.`{table_path}`"
            if z_order_by:
                z_order_cols = ", ".join(z_order_by)
                optimize_cmd += f" ZORDER BY ({z_order_cols})"

            self.spark.sql(optimize_cmd)

            logger.info("Delta table optimization completed", table_path=table_path)

        except Exception as e:
            logger.warning("Failed to optimize Delta table", table_path=table_path, error=str(e))

    def get_prediction_statistics(self, predictions_path: str) -> Dict[str, Any]:
        """Get statistics about predictions.

        Args:
            predictions_path: Path to predictions Delta table

        Returns:
            Dictionary with statistics
        """
        try:
            logger.debug("Calculating prediction statistics", predictions_path=predictions_path)

            predictions_df = self.spark.read.format("delta").load(predictions_path)

            stats = {
                "total_predictions": predictions_df.count(),
                "unique_batch_ids": predictions_df.select("batch_id").distinct().count(),
                "prediction_summary": predictions_df.select("prediction").summary().collect(),
            }

            # Calculate prediction distribution
            if "prediction" in predictions_df.columns:
                prediction_dist = predictions_df.groupBy("prediction").count().collect()
                stats["prediction_distribution"] = {
                    row["prediction"]: row["count"] for row in prediction_dist
                }

            logger.info("Prediction statistics calculated", predictions_path=predictions_path)
            return stats

        except Exception as e:
            logger.error("Failed to calculate prediction statistics", error=str(e))
            raise
