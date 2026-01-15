"""Behavioral feature engineering module.

This module provides user behavior pattern analysis including session features,
engagement metrics, and cart abandonment analysis using PySpark DataFrames.
"""

import logging

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

logger = logging.getLogger(__name__)


class BehavioralFeatureEngineer:
    """Engineer behavioral features from e-commerce event data.

    This class provides methods to calculate session-based features, engagement
    metrics, cart abandonment patterns, and time-based behavioral patterns for
    user segmentation and predictive modeling.
    """

    def __init__(self, session_timeout_minutes: int = 30):
        """Initialize BehavioralFeatureEngineer.

        Args:
            session_timeout_minutes: Minutes of inactivity to consider
                                    session ended
        """
        self.session_timeout_minutes = session_timeout_minutes
        logger.info(
            "Initialized BehavioralFeatureEngineer with session_timeout_minutes=%d",
            session_timeout_minutes,
        )

    def calculate_session_features(self, df: DataFrame) -> DataFrame:
        """Calculate session-based features.

        Args:
            df: Input DataFrame with columns: user_id, user_session,
                event_time, event_type

        Returns:
            DataFrame with session features per user:
                - user_id
                - total_sessions: Total number of sessions
                - avg_session_duration_minutes: Average session duration
                - max_session_duration_minutes: Maximum session duration
                - min_session_duration_minutes: Minimum session duration
                - avg_events_per_session: Average events per session
                - max_events_per_session: Maximum events per session
                - sessions_with_purchase: Number of sessions with purchase
                - session_conversion_rate: Rate of sessions with purchase

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"user_id", "user_session", "event_time", "event_type"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Calculating session features")

        try:
            # Calculate session-level metrics
            session_df = df.groupBy("user_id", "user_session").agg(
                F.min("event_time").alias("session_start"),
                F.max("event_time").alias("session_end"),
                F.count("*").alias("events_count"),
                F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias(
                    "purchase_count"
                ),
            )

            # Calculate session duration in minutes
            session_df = session_df.withColumn(
                "session_duration_minutes",
                (F.unix_timestamp("session_end") - F.unix_timestamp("session_start")) / 60.0,
            ).withColumn("has_purchase", F.when(F.col("purchase_count") > 0, 1).otherwise(0))

            # Aggregate to user level
            session_features_df = session_df.groupBy("user_id").agg(
                F.count("user_session").alias("total_sessions"),
                F.avg("session_duration_minutes").alias("avg_session_duration_minutes"),
                F.max("session_duration_minutes").alias("max_session_duration_minutes"),
                F.min("session_duration_minutes").alias("min_session_duration_minutes"),
                F.avg("events_count").alias("avg_events_per_session"),
                F.max("events_count").alias("max_events_per_session"),
                F.sum("has_purchase").alias("sessions_with_purchase"),
            )

            # Calculate session conversion rate
            session_features_df = session_features_df.withColumn(
                "session_conversion_rate",
                F.when(
                    F.col("total_sessions") > 0,
                    F.col("sessions_with_purchase") / F.col("total_sessions"),
                ).otherwise(0.0),
            )

            # Cast to appropriate types
            session_features_df = (
                session_features_df.withColumn(
                    "total_sessions", F.col("total_sessions").cast(IntegerType())
                )
                .withColumn(
                    "avg_session_duration_minutes",
                    F.col("avg_session_duration_minutes").cast(DoubleType()),
                )
                .withColumn(
                    "max_session_duration_minutes",
                    F.col("max_session_duration_minutes").cast(DoubleType()),
                )
                .withColumn(
                    "min_session_duration_minutes",
                    F.col("min_session_duration_minutes").cast(DoubleType()),
                )
                .withColumn(
                    "avg_events_per_session",
                    F.col("avg_events_per_session").cast(DoubleType()),
                )
                .withColumn(
                    "max_events_per_session",
                    F.col("max_events_per_session").cast(IntegerType()),
                )
                .withColumn(
                    "sessions_with_purchase",
                    F.col("sessions_with_purchase").cast(IntegerType()),
                )
                .withColumn(
                    "session_conversion_rate",
                    F.col("session_conversion_rate").cast(DoubleType()),
                )
            )

            logger.info("Calculated session features for %d users", session_features_df.count())
            return session_features_df

        except Exception as e:
            logger.error("Error calculating session features: %s", str(e))
            raise

    def calculate_engagement_metrics(self, df: DataFrame) -> DataFrame:
        """Calculate user engagement metrics.

        Args:
            df: Input DataFrame with columns: user_id, event_time,
                event_type, product_id

        Returns:
            DataFrame with engagement metrics per user:
                - user_id
                - total_events: Total number of events
                - view_events: Number of view events
                - cart_events: Number of cart events
                - purchase_events: Number of purchase events
                - unique_products_viewed: Number of unique products viewed
                - unique_products_purchased: Number of unique products purchased
                - engagement_score: Weighted engagement score
                - days_active: Number of days with activity
                - avg_events_per_day: Average events per active day

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"user_id", "event_time", "event_type", "product_id"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Calculating engagement metrics")

        try:
            # Count events by type
            event_counts_df = (
                df.groupBy("user_id")
                .pivot("event_type", ["view", "cart", "purchase"])
                .agg(F.count("*"))
                .fillna(0)
                .withColumnRenamed("view", "view_events")
                .withColumnRenamed("cart", "cart_events")
                .withColumnRenamed("purchase", "purchase_events")
            )

            # Calculate total events
            event_counts_df = event_counts_df.withColumn(
                "total_events",
                F.col("view_events") + F.col("cart_events") + F.col("purchase_events"),
            )

            # Count unique products
            unique_products_df = df.groupBy("user_id").agg(
                F.countDistinct(F.when(F.col("event_type") == "view", F.col("product_id"))).alias(
                    "unique_products_viewed"
                ),
                F.countDistinct(
                    F.when(F.col("event_type") == "purchase", F.col("product_id"))
                ).alias("unique_products_purchased"),
            )

            # Calculate days active
            days_active_df = df.groupBy("user_id").agg(
                F.countDistinct(F.to_date("event_time")).alias("days_active")
            )

            # Combine metrics
            engagement_df = event_counts_df.join(unique_products_df, "user_id", "inner").join(
                days_active_df, "user_id", "inner"
            )

            # Calculate engagement score
            # Formula: (views * 1) + (carts * 5) + (purchases * 20)
            engagement_df = engagement_df.withColumn(
                "engagement_score",
                (F.col("view_events") * 1)
                + (F.col("cart_events") * 5)
                + (F.col("purchase_events") * 20),
            )

            # Calculate average events per day
            engagement_df = engagement_df.withColumn(
                "avg_events_per_day",
                F.when(
                    F.col("days_active") > 0,
                    F.col("total_events") / F.col("days_active"),
                ).otherwise(0.0),
            )

            # Cast to appropriate types
            engagement_df = (
                engagement_df.withColumn("total_events", F.col("total_events").cast(IntegerType()))
                .withColumn("view_events", F.col("view_events").cast(IntegerType()))
                .withColumn("cart_events", F.col("cart_events").cast(IntegerType()))
                .withColumn("purchase_events", F.col("purchase_events").cast(IntegerType()))
                .withColumn(
                    "unique_products_viewed",
                    F.col("unique_products_viewed").cast(IntegerType()),
                )
                .withColumn(
                    "unique_products_purchased",
                    F.col("unique_products_purchased").cast(IntegerType()),
                )
                .withColumn("engagement_score", F.col("engagement_score").cast(DoubleType()))
                .withColumn("days_active", F.col("days_active").cast(IntegerType()))
                .withColumn(
                    "avg_events_per_day",
                    F.col("avg_events_per_day").cast(DoubleType()),
                )
            )

            logger.info("Calculated engagement metrics for %d users", engagement_df.count())
            return engagement_df

        except Exception as e:
            logger.error("Error calculating engagement metrics: %s", str(e))
            raise

    def calculate_cart_abandonment(self, df: DataFrame) -> DataFrame:
        """Calculate cart abandonment patterns.

        Args:
            df: Input DataFrame with columns: user_id, user_session,
                event_type, product_id

        Returns:
            DataFrame with cart abandonment metrics per user:
                - user_id
                - total_cart_additions: Total products added to cart
                - total_purchases: Total products purchased
                - total_cart_removals: Total products removed from cart
                - cart_abandonment_rate: Rate of cart abandonments
                - products_in_abandoned_carts: Average products in abandoned carts
                - sessions_with_abandoned_cart: Number of sessions with abandonment

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"user_id", "user_session", "event_type", "product_id"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Calculating cart abandonment patterns")

        try:
            # Count events by type
            event_counts_df = df.groupBy("user_id").agg(
                F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias(
                    "total_cart_additions"
                ),
                F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias(
                    "total_purchases"
                ),
                F.sum(F.when(F.col("event_type") == "remove_from_cart", 1).otherwise(0)).alias(
                    "total_cart_removals"
                ),
            )

            # Calculate abandonment rate
            event_counts_df = event_counts_df.withColumn(
                "cart_abandonment_rate",
                F.when(
                    F.col("total_cart_additions") > 0,
                    (F.col("total_cart_additions") - F.col("total_purchases"))
                    / F.col("total_cart_additions"),
                ).otherwise(0.0),
            )

            # Calculate session-level abandonment
            session_df = (
                df.groupBy("user_id", "user_session")
                .agg(
                    F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias(
                        "session_cart_additions"
                    ),
                    F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias(
                        "session_purchases"
                    ),
                )
                .withColumn(
                    "is_abandoned",
                    F.when(
                        (F.col("session_cart_additions") > 0) & (F.col("session_purchases") == 0),
                        1,
                    ).otherwise(0),
                )
                .withColumn(
                    "abandoned_cart_products",
                    F.when(F.col("is_abandoned") == 1, F.col("session_cart_additions")).otherwise(
                        0
                    ),
                )
            )

            # Aggregate session metrics to user level
            session_agg_df = session_df.groupBy("user_id").agg(
                F.sum("is_abandoned").alias("sessions_with_abandoned_cart"),
                F.avg("abandoned_cart_products").alias("products_in_abandoned_carts"),
            )

            # Combine all metrics
            abandonment_df = event_counts_df.join(session_agg_df, "user_id", "left")

            # Cast to appropriate types
            abandonment_df = (
                abandonment_df.withColumn(
                    "total_cart_additions",
                    F.col("total_cart_additions").cast(IntegerType()),
                )
                .withColumn("total_purchases", F.col("total_purchases").cast(IntegerType()))
                .withColumn(
                    "total_cart_removals",
                    F.col("total_cart_removals").cast(IntegerType()),
                )
                .withColumn(
                    "cart_abandonment_rate",
                    F.col("cart_abandonment_rate").cast(DoubleType()),
                )
                .withColumn(
                    "products_in_abandoned_carts",
                    F.coalesce(F.col("products_in_abandoned_carts"), F.lit(0.0)).cast(DoubleType()),
                )
                .withColumn(
                    "sessions_with_abandoned_cart",
                    F.coalesce(F.col("sessions_with_abandoned_cart"), F.lit(0)).cast(IntegerType()),
                )
            )

            logger.info(
                "Calculated cart abandonment patterns for %d users",
                abandonment_df.count(),
            )
            return abandonment_df

        except Exception as e:
            logger.error("Error calculating cart abandonment: %s", str(e))
            raise

    def calculate_time_patterns(self, df: DataFrame) -> DataFrame:
        """Calculate time-based behavioral patterns.

        Args:
            df: Input DataFrame with columns: user_id, event_time

        Returns:
            DataFrame with time pattern features per user:
                - user_id
                - most_active_hour: Hour of day with most activity (0-23)
                - most_active_day: Day of week with most activity (0-6)
                - weekend_activity_ratio: Ratio of weekend to weekday activity
                - morning_activity_ratio: Ratio of morning (6am-12pm) activity
                - afternoon_activity_ratio: Ratio of afternoon (12pm-6pm) activity
                - evening_activity_ratio: Ratio of evening (6pm-12am) activity
                - night_activity_ratio: Ratio of night (12am-6am) activity

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"user_id", "event_time"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Calculating time-based behavioral patterns")

        try:
            # Extract time features
            time_df = df.withColumn("hour", F.hour("event_time")).withColumn(
                "day_of_week", F.dayofweek("event_time")
            )

            # Calculate most active hour
            hour_counts_df = time_df.groupBy("user_id", "hour").agg(
                F.count("*").alias("hour_count")
            )
            hour_window = Window.partitionBy("user_id").orderBy(F.desc("hour_count"))
            most_active_hour_df = (
                hour_counts_df.withColumn("rank", F.row_number().over(hour_window))
                .filter(F.col("rank") == 1)
                .select("user_id", F.col("hour").alias("most_active_hour"))
            )

            # Calculate most active day
            day_counts_df = time_df.groupBy("user_id", "day_of_week").agg(
                F.count("*").alias("day_count")
            )
            day_window = Window.partitionBy("user_id").orderBy(F.desc("day_count"))
            most_active_day_df = (
                day_counts_df.withColumn("rank", F.row_number().over(day_window))
                .filter(F.col("rank") == 1)
                .select("user_id", F.col("day_of_week").alias("most_active_day"))
            )

            # Calculate weekend vs weekday activity
            weekend_df = time_df.groupBy("user_id").agg(
                F.sum(F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0)).alias(
                    "weekend_events"
                ),
                F.sum(F.when(~F.col("day_of_week").isin([1, 7]), 1).otherwise(0)).alias(
                    "weekday_events"
                ),
                F.count("*").alias("total_events"),
            )

            weekend_df = weekend_df.withColumn(
                "weekend_activity_ratio",
                F.when(
                    F.col("weekday_events") > 0,
                    F.col("weekend_events") / F.col("weekday_events"),
                ).otherwise(0.0),
            )

            # Calculate time of day activity ratios
            time_of_day_df = time_df.groupBy("user_id").agg(
                F.sum(F.when((F.col("hour") >= 6) & (F.col("hour") < 12), 1).otherwise(0)).alias(
                    "morning_events"
                ),
                F.sum(F.when((F.col("hour") >= 12) & (F.col("hour") < 18), 1).otherwise(0)).alias(
                    "afternoon_events"
                ),
                F.sum(F.when((F.col("hour") >= 18) & (F.col("hour") < 24), 1).otherwise(0)).alias(
                    "evening_events"
                ),
                F.sum(F.when(F.col("hour") < 6, 1).otherwise(0)).alias("night_events"),
                F.count("*").alias("total_events"),
            )

            time_of_day_df = (
                time_of_day_df.withColumn(
                    "morning_activity_ratio",
                    F.col("morning_events") / F.col("total_events"),
                )
                .withColumn(
                    "afternoon_activity_ratio",
                    F.col("afternoon_events") / F.col("total_events"),
                )
                .withColumn(
                    "evening_activity_ratio",
                    F.col("evening_events") / F.col("total_events"),
                )
                .withColumn(
                    "night_activity_ratio",
                    F.col("night_events") / F.col("total_events"),
                )
            )

            # Combine all time patterns
            time_patterns_df = (
                most_active_hour_df.join(most_active_day_df, "user_id", "inner")
                .join(weekend_df.select("user_id", "weekend_activity_ratio"), "user_id", "inner")
                .join(
                    time_of_day_df.select(
                        "user_id",
                        "morning_activity_ratio",
                        "afternoon_activity_ratio",
                        "evening_activity_ratio",
                        "night_activity_ratio",
                    ),
                    "user_id",
                    "inner",
                )
            )

            # Cast to appropriate types
            time_patterns_df = (
                time_patterns_df.withColumn(
                    "most_active_hour", F.col("most_active_hour").cast(IntegerType())
                )
                .withColumn("most_active_day", F.col("most_active_day").cast(IntegerType()))
                .withColumn(
                    "weekend_activity_ratio",
                    F.col("weekend_activity_ratio").cast(DoubleType()),
                )
                .withColumn(
                    "morning_activity_ratio",
                    F.col("morning_activity_ratio").cast(DoubleType()),
                )
                .withColumn(
                    "afternoon_activity_ratio",
                    F.col("afternoon_activity_ratio").cast(DoubleType()),
                )
                .withColumn(
                    "evening_activity_ratio",
                    F.col("evening_activity_ratio").cast(DoubleType()),
                )
                .withColumn(
                    "night_activity_ratio",
                    F.col("night_activity_ratio").cast(DoubleType()),
                )
            )

            logger.info("Calculated time patterns for %d users", time_patterns_df.count())
            return time_patterns_df

        except Exception as e:
            logger.error("Error calculating time patterns: %s", str(e))
            raise

    def calculate_all_features(self, df: DataFrame) -> DataFrame:
        """Calculate all behavioral features in one pass.

        Args:
            df: Input DataFrame with all required columns

        Returns:
            DataFrame with all behavioral features combined

        Raises:
            ValueError: If required columns are missing
        """
        logger.info("Calculating all behavioral features")

        try:
            session_features = self.calculate_session_features(df)
            engagement_metrics = self.calculate_engagement_metrics(df)
            cart_abandonment = self.calculate_cart_abandonment(df)
            time_patterns = self.calculate_time_patterns(df)

            # Combine all features
            all_features = (
                session_features.join(engagement_metrics, "user_id", "left")
                .join(cart_abandonment, "user_id", "left")
                .join(time_patterns, "user_id", "left")
                .fillna(0)
            )

            logger.info(
                "Successfully calculated all features for %d users",
                all_features.count(),
            )
            return all_features

        except Exception as e:
            logger.error("Error calculating all behavioral features: %s", str(e))
            raise
