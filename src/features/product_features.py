"""Product feature engineering module.

This module provides product-level features including popularity metrics,
conversion rates, and category-based features using PySpark DataFrames.
"""

import logging

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

logger = logging.getLogger(__name__)


class ProductFeatureEngineer:
    """Engineer product-level features from e-commerce event data.

    This class provides methods to calculate product popularity, conversion metrics,
    category features, and price-based features for recommendation systems and
    inventory management.
    """

    def __init__(self, min_interactions: int = 5):
        """Initialize ProductFeatureEngineer.

        Args:
            min_interactions: Minimum number of interactions required for
                            a product to be included in features
        """
        self.min_interactions = min_interactions
        logger.info(
            "Initialized ProductFeatureEngineer with min_interactions=%d",
            min_interactions,
        )

    def calculate_product_popularity(self, df: DataFrame) -> DataFrame:
        """Calculate product popularity metrics.

        Args:
            df: Input DataFrame with columns: product_id, event_type, user_id

        Returns:
            DataFrame with popularity features per product:
                - product_id
                - total_views: Total number of views
                - total_carts: Total number of cart additions
                - total_purchases: Total number of purchases
                - unique_viewers: Number of unique users who viewed
                - unique_purchasers: Number of unique users who purchased
                - popularity_score: Weighted popularity score

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"product_id", "event_type", "user_id"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Calculating product popularity metrics")

        try:
            # Count events by type for each product
            event_counts_df = (
                df.groupBy("product_id")
                .pivot("event_type", ["view", "cart", "purchase"])
                .agg(F.count("*"))
                .fillna(0)
                .withColumnRenamed("view", "total_views")
                .withColumnRenamed("cart", "total_carts")
                .withColumnRenamed("purchase", "total_purchases")
            )

            # Count unique users for each product
            unique_viewers_df = (
                df.filter(F.col("event_type") == "view")
                .groupBy("product_id")
                .agg(F.countDistinct("user_id").alias("unique_viewers"))
            )

            unique_purchasers_df = (
                df.filter(F.col("event_type") == "purchase")
                .groupBy("product_id")
                .agg(F.countDistinct("user_id").alias("unique_purchasers"))
            )

            # Combine metrics
            popularity_df = (
                event_counts_df.join(unique_viewers_df, "product_id", "left")
                .join(unique_purchasers_df, "product_id", "left")
                .fillna(0)
            )

            # Calculate weighted popularity score
            # Formula: (views * 1) + (carts * 3) + (purchases * 10)
            popularity_df = popularity_df.withColumn(
                "popularity_score",
                (F.col("total_views") * 1)
                + (F.col("total_carts") * 3)
                + (F.col("total_purchases") * 10),
            )

            # Filter by minimum interactions
            popularity_df = popularity_df.filter(
                (F.col("total_views") + F.col("total_carts") + F.col("total_purchases"))
                >= self.min_interactions
            )

            # Cast to appropriate types
            popularity_df = (
                popularity_df.withColumn("total_views", F.col("total_views").cast(IntegerType()))
                .withColumn("total_carts", F.col("total_carts").cast(IntegerType()))
                .withColumn("total_purchases", F.col("total_purchases").cast(IntegerType()))
                .withColumn("unique_viewers", F.col("unique_viewers").cast(IntegerType()))
                .withColumn("unique_purchasers", F.col("unique_purchasers").cast(IntegerType()))
                .withColumn("popularity_score", F.col("popularity_score").cast(DoubleType()))
            )

            logger.info("Calculated popularity metrics for %d products", popularity_df.count())
            return popularity_df

        except Exception as e:
            logger.error("Error calculating product popularity: %s", str(e))
            raise

    def calculate_conversion_metrics(self, df: DataFrame) -> DataFrame:
        """Calculate product conversion metrics.

        Args:
            df: Input DataFrame with columns: product_id, event_type

        Returns:
            DataFrame with conversion metrics per product:
                - product_id
                - view_to_cart_rate: Conversion rate from view to cart
                - cart_to_purchase_rate: Conversion rate from cart to purchase
                - view_to_purchase_rate: Overall conversion rate
                - cart_abandonment_rate: Rate of cart abandonments

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"product_id", "event_type"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Calculating product conversion metrics")

        try:
            # Count events by type
            event_counts_df = (
                df.groupBy("product_id")
                .pivot("event_type", ["view", "cart", "purchase"])
                .agg(F.count("*"))
                .fillna(0)
            )

            # Calculate conversion rates
            conversion_df = (
                event_counts_df.withColumn(
                    "view_to_cart_rate",
                    F.when(F.col("view") > 0, F.col("cart") / F.col("view")).otherwise(0.0),
                )
                .withColumn(
                    "cart_to_purchase_rate",
                    F.when(F.col("cart") > 0, F.col("purchase") / F.col("cart")).otherwise(0.0),
                )
                .withColumn(
                    "view_to_purchase_rate",
                    F.when(F.col("view") > 0, F.col("purchase") / F.col("view")).otherwise(0.0),
                )
                .withColumn(
                    "cart_abandonment_rate",
                    F.when(
                        F.col("cart") > 0,
                        1.0 - (F.col("purchase") / F.col("cart")),
                    ).otherwise(0.0),
                )
            )

            # Select only conversion metrics
            conversion_df = conversion_df.select(
                "product_id",
                F.col("view_to_cart_rate").cast(DoubleType()),
                F.col("cart_to_purchase_rate").cast(DoubleType()),
                F.col("view_to_purchase_rate").cast(DoubleType()),
                F.col("cart_abandonment_rate").cast(DoubleType()),
            )

            logger.info("Calculated conversion metrics for %d products", conversion_df.count())
            return conversion_df

        except Exception as e:
            logger.error("Error calculating conversion metrics: %s", str(e))
            raise

    def calculate_category_features(self, df: DataFrame) -> DataFrame:
        """Calculate category-level features for products.

        Args:
            df: Input DataFrame with columns: product_id, category_code,
                category_id, event_type

        Returns:
            DataFrame with category features per product:
                - product_id
                - category_code
                - category_id
                - category_popularity_rank: Rank within category by popularity
                - category_conversion_rank: Rank within category by conversion
                - products_in_category: Total products in same category

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"product_id", "event_type"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Calculating category features for products")

        try:
            # Check if category columns exist
            has_category_code = "category_code" in df.columns
            has_category_id = "category_id" in df.columns

            if not has_category_code and not has_category_id:
                logger.warning("No category columns found, returning minimal features")
                return (
                    df.select("product_id")
                    .distinct()
                    .withColumn("category_code", F.lit(None).cast("string"))
                    .withColumn("category_id", F.lit(None).cast("long"))
                    .withColumn("category_popularity_rank", F.lit(None).cast("int"))
                    .withColumn("category_conversion_rank", F.lit(None).cast("int"))
                    .withColumn("products_in_category", F.lit(0).cast("int"))
                )

            # Get product attributes
            product_attrs_df = df.select(
                "product_id",
                "category_code" if has_category_code else F.lit(None).alias("category_code"),
                "category_id" if has_category_id else F.lit(None).alias("category_id"),
            ).distinct()

            # Calculate product popularity scores
            popularity_df = (
                df.groupBy("product_id")
                .pivot("event_type", ["view", "cart", "purchase"])
                .agg(F.count("*"))
                .fillna(0)
                .withColumn(
                    "popularity_score",
                    (F.col("view") * 1) + (F.col("cart") * 3) + (F.col("purchase") * 10),
                )
                .withColumn(
                    "conversion_score",
                    F.when(F.col("view") > 0, F.col("purchase") / F.col("view")).otherwise(0.0),
                )
                .select("product_id", "popularity_score", "conversion_score")
            )

            # Join with product attributes
            product_df = product_attrs_df.join(popularity_df, "product_id", "left")

            # Create partition key based on available category columns
            if has_category_code:
                partition_col = "category_code"
            else:
                partition_col = "category_id"

            # Calculate ranks within category
            popularity_window = Window.partitionBy(partition_col).orderBy(
                F.desc("popularity_score")
            )
            conversion_window = Window.partitionBy(partition_col).orderBy(
                F.desc("conversion_score")
            )

            category_df = product_df.withColumn(
                "category_popularity_rank", F.row_number().over(popularity_window)
            ).withColumn("category_conversion_rank", F.row_number().over(conversion_window))

            # Count products in each category
            category_counts_df = product_df.groupBy(partition_col).agg(
                F.count("product_id").alias("products_in_category")
            )

            # Join with category counts
            category_df = category_df.join(category_counts_df, partition_col, "left")

            # Select final columns
            category_df = category_df.select(
                "product_id",
                "category_code",
                "category_id",
                F.col("category_popularity_rank").cast(IntegerType()),
                F.col("category_conversion_rank").cast(IntegerType()),
                F.col("products_in_category").cast(IntegerType()),
            )

            logger.info("Calculated category features for %d products", category_df.count())
            return category_df

        except Exception as e:
            logger.error("Error calculating category features: %s", str(e))
            raise

    def calculate_price_features(self, df: DataFrame) -> DataFrame:
        """Calculate price-based features for products.

        Args:
            df: Input DataFrame with columns: product_id, price, category_code

        Returns:
            DataFrame with price features per product:
                - product_id
                - avg_price: Average price of product
                - price_std: Price standard deviation
                - price_percentile_in_category: Price percentile within category
                - is_premium: Boolean indicating if product is premium priced

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"product_id", "price"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Calculating price features for products")

        try:
            # Calculate average price per product
            price_df = df.groupBy("product_id").agg(
                F.avg("price").alias("avg_price"),
                F.stddev("price").alias("price_std"),
            )

            # Get category information if available
            if "category_code" in df.columns:
                product_category_df = df.select("product_id", "category_code").distinct()
                price_df = price_df.join(product_category_df, "product_id", "left")

                # Calculate price percentile within category
                category_window = Window.partitionBy("category_code").orderBy("avg_price")
                price_df = price_df.withColumn(
                    "price_percentile_in_category",
                    F.percent_rank().over(category_window),
                )

                # Define premium products (top 25% in category)
                price_df = price_df.withColumn(
                    "is_premium",
                    F.when(F.col("price_percentile_in_category") >= 0.75, True).otherwise(False),
                )
            else:
                # Global price percentile if no category
                global_window = Window.orderBy("avg_price")
                price_df = price_df.withColumn(
                    "price_percentile_in_category",
                    F.percent_rank().over(global_window),
                )
                price_df = price_df.withColumn(
                    "is_premium",
                    F.when(F.col("price_percentile_in_category") >= 0.75, True).otherwise(False),
                )

            # Cast to appropriate types and handle nulls
            price_df = (
                price_df.withColumn("avg_price", F.col("avg_price").cast(DoubleType()))
                .withColumn(
                    "price_std",
                    F.coalesce(F.col("price_std"), F.lit(0.0)).cast(DoubleType()),
                )
                .withColumn(
                    "price_percentile_in_category",
                    F.col("price_percentile_in_category").cast(DoubleType()),
                )
            )

            # Select final columns
            if "category_code" in df.columns:
                price_df = price_df.select(
                    "product_id",
                    "avg_price",
                    "price_std",
                    "price_percentile_in_category",
                    "is_premium",
                )
            else:
                price_df = price_df.select(
                    "product_id",
                    "avg_price",
                    "price_std",
                    "price_percentile_in_category",
                    "is_premium",
                )

            logger.info("Calculated price features for %d products", price_df.count())
            return price_df

        except Exception as e:
            logger.error("Error calculating price features: %s", str(e))
            raise

    def calculate_all_features(self, df: DataFrame) -> DataFrame:
        """Calculate all product features in one pass.

        Args:
            df: Input DataFrame with all required columns

        Returns:
            DataFrame with all product features combined

        Raises:
            ValueError: If required columns are missing
        """
        logger.info("Calculating all product features")

        try:
            popularity_features = self.calculate_product_popularity(df)
            conversion_features = self.calculate_conversion_metrics(df)
            category_features = self.calculate_category_features(df)
            price_features = self.calculate_price_features(df)

            # Combine all features
            all_features = (
                popularity_features.join(conversion_features, "product_id", "left")
                .join(category_features, "product_id", "left")
                .join(price_features, "product_id", "left")
                .fillna(0)
            )

            logger.info(
                "Successfully calculated all features for %d products",
                all_features.count(),
            )
            return all_features

        except Exception as e:
            logger.error("Error calculating all product features: %s", str(e))
            raise
