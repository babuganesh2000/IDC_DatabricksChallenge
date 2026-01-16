"""Customer feature engineering module.

This module provides RFM (Recency, Frequency, Monetary) analysis and customer
behavioral features using PySpark DataFrames.
"""

import logging
from datetime import datetime
from typing import Optional

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

logger = logging.getLogger(__name__)


class CustomerFeatureEngineer:
    """Engineer customer-level features from e-commerce event data.

    This class provides methods to calculate RFM features, behavioral patterns,
    and purchase patterns for customer segmentation and predictive modeling.
    """

    def __init__(self, reference_date: Optional[datetime] = None):
        """Initialize CustomerFeatureEngineer.

        Args:
            reference_date: Reference date for recency calculation.
                          Defaults to current date if not provided.
        """
        self.reference_date = reference_date or datetime.now()
        logger.info(
            "Initialized CustomerFeatureEngineer with reference_date=%s",
            self.reference_date.strftime("%Y-%m-%d"),
        )

    def calculate_rfm_features(self, df: DataFrame) -> DataFrame:
        """Calculate RFM (Recency, Frequency, Monetary) features.

        Args:
            df: Input DataFrame with columns: user_id, event_time,
                event_type, price

        Returns:
            DataFrame with RFM features per user:
                - user_id
                - recency_days: Days since last purchase
                - frequency: Total number of purchases
                - monetary_value: Total purchase amount
                - avg_order_value: Average purchase amount

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"user_id", "event_time", "event_type", "price"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Calculating RFM features for customers")

        try:
            # Filter for purchase events only
            purchases_df = df.filter(F.col("event_type") == "purchase")

            # Calculate recency (days since last purchase)
            reference_ts = F.lit(self.reference_date.timestamp())
            recency_df = (
                purchases_df.groupBy("user_id")
                .agg(F.max("event_time").alias("last_purchase_time"))
                .withColumn(
                    "recency_days",
                    (reference_ts - F.unix_timestamp("last_purchase_time"))
                    / 86400.0,  # Convert seconds to days
                )
                .select("user_id", "recency_days")
            )

            # Calculate frequency (number of purchases)
            frequency_df = purchases_df.groupBy("user_id").agg(F.count("*").alias("frequency"))

            # Calculate monetary value (total and average purchase amount)
            monetary_df = purchases_df.groupBy("user_id").agg(
                F.sum("price").alias("monetary_value"),
                F.avg("price").alias("avg_order_value"),
            )

            # Combine all RFM features
            rfm_df = (
                recency_df.join(frequency_df, "user_id", "inner")
                .join(monetary_df, "user_id", "inner")
                .withColumn("recency_days", F.col("recency_days").cast(DoubleType()))
                .withColumn("frequency", F.col("frequency").cast(IntegerType()))
                .withColumn("monetary_value", F.col("monetary_value").cast(DoubleType()))
                .withColumn("avg_order_value", F.col("avg_order_value").cast(DoubleType()))
            )

            logger.info("Calculated RFM features for %d customers", rfm_df.count())
            return rfm_df

        except Exception as e:
            logger.error("Error calculating RFM features: %s", str(e))
            raise

    def calculate_behavioral_features(self, df: DataFrame) -> DataFrame:
        """Calculate behavioral features including session patterns and engagement.

        Args:
            df: Input DataFrame with columns: user_id, user_session,
                event_type, event_time

        Returns:
            DataFrame with behavioral features per user:
                - user_id
                - total_sessions: Total number of sessions
                - avg_session_length_minutes: Average session duration
                - total_events: Total number of events
                - events_per_session: Average events per session
                - unique_days_active: Number of unique days with activity

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"user_id", "user_session", "event_type", "event_time"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Calculating behavioral features for customers")

        try:
            # Calculate session-level metrics
            session_df = df.groupBy("user_id", "user_session").agg(
                F.min("event_time").alias("session_start"),
                F.max("event_time").alias("session_end"),
                F.count("*").alias("session_events"),
            )

            # Calculate session length in minutes
            session_df = session_df.withColumn(
                "session_length_minutes",
                (F.unix_timestamp("session_end") - F.unix_timestamp("session_start")) / 60.0,
            )

            # Aggregate to user level
            behavioral_df = session_df.groupBy("user_id").agg(
                F.count("user_session").alias("total_sessions"),
                F.avg("session_length_minutes").alias("avg_session_length_minutes"),
                F.sum("session_events").alias("total_events"),
            )

            # Calculate events per session
            behavioral_df = behavioral_df.withColumn(
                "events_per_session",
                F.col("total_events") / F.col("total_sessions"),
            )

            # Calculate unique days active
            days_active_df = df.groupBy("user_id").agg(
                F.countDistinct(F.to_date("event_time")).alias("unique_days_active")
            )

            # Combine all behavioral features
            behavioral_df = behavioral_df.join(days_active_df, "user_id", "inner")

            # Cast to appropriate types
            behavioral_df = (
                behavioral_df.withColumn(
                    "total_sessions", F.col("total_sessions").cast(IntegerType())
                )
                .withColumn(
                    "avg_session_length_minutes",
                    F.col("avg_session_length_minutes").cast(DoubleType()),
                )
                .withColumn("total_events", F.col("total_events").cast(IntegerType()))
                .withColumn(
                    "events_per_session",
                    F.col("events_per_session").cast(DoubleType()),
                )
                .withColumn(
                    "unique_days_active",
                    F.col("unique_days_active").cast(IntegerType()),
                )
            )

            logger.info(
                "Calculated behavioral features for %d customers",
                behavioral_df.count(),
            )
            return behavioral_df

        except Exception as e:
            logger.error("Error calculating behavioral features: %s", str(e))
            raise

    def calculate_purchase_patterns(self, df: DataFrame) -> DataFrame:
        """Calculate purchase patterns and preferences.

        Args:
            df: Input DataFrame with columns: user_id, event_type,
                product_id, category_code, brand, price

        Returns:
            DataFrame with purchase patterns per user:
                - user_id
                - total_products_viewed: Total products viewed
                - total_products_carted: Total products added to cart
                - total_products_purchased: Total products purchased
                - cart_to_purchase_rate: Conversion rate from cart to purchase
                - view_to_cart_rate: Conversion rate from view to cart
                - favorite_category: Most purchased category
                - favorite_brand: Most purchased brand
                - avg_product_price: Average price of viewed products
                - price_sensitivity: Std dev of product prices

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"user_id", "event_type", "product_id"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Calculating purchase patterns for customers")

        try:
            # Count events by type
            event_counts_df = (
                df.groupBy("user_id")
                .pivot("event_type", ["view", "cart", "purchase"])
                .agg(F.count("product_id"))
                .fillna(0)
                .withColumnRenamed("view", "total_products_viewed")
                .withColumnRenamed("cart", "total_products_carted")
                .withColumnRenamed("purchase", "total_products_purchased")
            )

            # Calculate conversion rates
            event_counts_df = event_counts_df.withColumn(
                "cart_to_purchase_rate",
                F.when(
                    F.col("total_products_carted") > 0,
                    F.col("total_products_purchased") / F.col("total_products_carted"),
                ).otherwise(0.0),
            ).withColumn(
                "view_to_cart_rate",
                F.when(
                    F.col("total_products_viewed") > 0,
                    F.col("total_products_carted") / F.col("total_products_viewed"),
                ).otherwise(0.0),
            )

            # Calculate favorite category (most purchased)
            purchases_df = df.filter(F.col("event_type") == "purchase")

            if "category_code" in df.columns:
                category_window = Window.partitionBy("user_id").orderBy(F.desc("category_count"))
                favorite_category_df = (
                    purchases_df.filter(F.col("category_code").isNotNull())
                    .groupBy("user_id", "category_code")
                    .agg(F.count("*").alias("category_count"))
                    .withColumn("rank", F.row_number().over(category_window))
                    .filter(F.col("rank") == 1)
                    .select("user_id", F.col("category_code").alias("favorite_category"))
                )
            else:
                favorite_category_df = (
                    df.select("user_id")
                    .distinct()
                    .withColumn("favorite_category", F.lit(None).cast("string"))
                )

            # Calculate favorite brand (most purchased)
            if "brand" in df.columns:
                brand_window = Window.partitionBy("user_id").orderBy(F.desc("brand_count"))
                favorite_brand_df = (
                    purchases_df.filter(F.col("brand").isNotNull())
                    .groupBy("user_id", "brand")
                    .agg(F.count("*").alias("brand_count"))
                    .withColumn("rank", F.row_number().over(brand_window))
                    .filter(F.col("rank") == 1)
                    .select("user_id", F.col("brand").alias("favorite_brand"))
                )
            else:
                favorite_brand_df = (
                    df.select("user_id")
                    .distinct()
                    .withColumn("favorite_brand", F.lit(None).cast("string"))
                )

            # Calculate price statistics
            if "price" in df.columns:
                price_stats_df = df.groupBy("user_id").agg(
                    F.avg("price").alias("avg_product_price"),
                    F.stddev("price").alias("price_sensitivity"),
                )
            else:
                price_stats_df = (
                    df.select("user_id")
                    .distinct()
                    .withColumn("avg_product_price", F.lit(0.0))
                    .withColumn("price_sensitivity", F.lit(0.0))
                )

            # Combine all purchase patterns
            purchase_patterns_df = (
                event_counts_df.join(favorite_category_df, "user_id", "left")
                .join(favorite_brand_df, "user_id", "left")
                .join(price_stats_df, "user_id", "left")
            )

            # Cast to appropriate types
            purchase_patterns_df = (
                purchase_patterns_df.withColumn(
                    "total_products_viewed",
                    F.col("total_products_viewed").cast(IntegerType()),
                )
                .withColumn(
                    "total_products_carted",
                    F.col("total_products_carted").cast(IntegerType()),
                )
                .withColumn(
                    "total_products_purchased",
                    F.col("total_products_purchased").cast(IntegerType()),
                )
                .withColumn(
                    "cart_to_purchase_rate",
                    F.col("cart_to_purchase_rate").cast(DoubleType()),
                )
                .withColumn(
                    "view_to_cart_rate",
                    F.col("view_to_cart_rate").cast(DoubleType()),
                )
                .withColumn(
                    "avg_product_price",
                    F.coalesce(F.col("avg_product_price"), F.lit(0.0)).cast(DoubleType()),
                )
                .withColumn(
                    "price_sensitivity",
                    F.coalesce(F.col("price_sensitivity"), F.lit(0.0)).cast(DoubleType()),
                )
            )

            logger.info(
                "Calculated purchase patterns for %d customers",
                purchase_patterns_df.count(),
            )
            return purchase_patterns_df

        except Exception as e:
            logger.error("Error calculating purchase patterns: %s", str(e))
            raise

    def calculate_all_features(self, df: DataFrame) -> DataFrame:
        """Calculate all customer features in one pass.

        Args:
            df: Input DataFrame with all required columns

        Returns:
            DataFrame with all customer features combined

        Raises:
            ValueError: If required columns are missing
        """
        logger.info("Calculating all customer features")

        try:
            rfm_features = self.calculate_rfm_features(df)
            behavioral_features = self.calculate_behavioral_features(df)
            purchase_patterns = self.calculate_purchase_patterns(df)

            # Combine all features
            all_features = (
                rfm_features.join(behavioral_features, "user_id", "left")
                .join(purchase_patterns, "user_id", "left")
                .fillna(0)
            )

            logger.info(
                "Successfully calculated all features for %d customers",
                all_features.count(),
            )
            return all_features

        except Exception as e:
            logger.error("Error calculating all customer features: %s", str(e))
            raise
