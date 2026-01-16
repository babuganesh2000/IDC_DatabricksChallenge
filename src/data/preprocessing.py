"""Feature engineering and data transformation.

This module provides classes for preprocessing data, including cleaning,
handling missing values, encoding categorical variables, and normalizing features.
"""

from typing import List, Optional, Dict, Any, Union, Tuple

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    MinMaxScaler,
    StringIndexer,
    OneHotEncoder,
    Imputer,
)
from pyspark.ml import Pipeline

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocess and transform data using PySpark."""

    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize data preprocessor.

        Args:
            spark: SparkSession instance
        """
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.pipeline = None
        self.fitted_pipeline = None
        logger.info("Data preprocessor initialized")

    def clean_data(
        self,
        df: DataFrame,
        drop_duplicates: bool = True,
        subset: Optional[List[str]] = None,
    ) -> DataFrame:
        """Clean data by removing duplicates and invalid records.

        Args:
            df: Input DataFrame
            drop_duplicates: Whether to drop duplicate rows
            subset: Column subset for duplicate detection

        Returns:
            Cleaned DataFrame
        """
        logger.info(
            "Cleaning data",
            rows=df.count(),
            columns=len(df.columns),
            drop_duplicates=drop_duplicates,
        )

        original_count = df.count()
        duplicates_removed = 0

        # Drop duplicates if requested
        if drop_duplicates:
            df = df.dropDuplicates(subset=subset)
            duplicates_removed = original_count - df.count()
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate rows")

        # Remove rows where all values are null
        df = df.dropna(how="all")
        null_rows_removed = original_count - duplicates_removed - df.count()
        if null_rows_removed > 0:
            logger.info(f"Removed {null_rows_removed} rows with all null values")

        final_count = df.count()
        logger.info(
            "Data cleaning completed",
            original_rows=original_count,
            final_rows=final_count,
            rows_removed=original_count - final_count,
        )

        return df

    def handle_missing_values(
        self,
        df: DataFrame,
        strategy: str = "mean",
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        fill_values: Optional[Dict[str, Any]] = None,
    ) -> DataFrame:
        """Handle missing values in DataFrame.

        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'mode', 'drop', 'fill')
            numeric_columns: Numeric columns to impute (auto-detect if None)
            categorical_columns: Categorical columns to impute (auto-detect if None)
            fill_values: Custom fill values per column (for 'fill' strategy)

        Returns:
            DataFrame with missing values handled

        Raises:
            ValueError: If strategy is invalid
        """
        logger.info(f"Handling missing values", strategy=strategy)

        # Auto-detect column types if not provided
        if numeric_columns is None:
            numeric_columns = [
                f.name
                for f in df.schema.fields
                if f.dataType.typeName() in ["double", "float", "integer", "long"]
            ]

        if categorical_columns is None:
            categorical_columns = [
                f.name
                for f in df.schema.fields
                if f.dataType.typeName() == "string"
                and f.name not in numeric_columns
            ]

        logger.info(
            "Column types detected",
            numeric_columns=len(numeric_columns),
            categorical_columns=len(categorical_columns),
        )

        if strategy == "drop":
            # Drop rows with any null values
            original_count = df.count()
            df = df.dropna()
            logger.info(
                f"Dropped rows with missing values",
                rows_removed=original_count - df.count(),
            )

        elif strategy == "fill" and fill_values:
            # Fill with custom values
            df = df.fillna(fill_values)
            logger.info("Filled missing values with custom values")

        elif strategy in ["mean", "median", "mode"]:
            # Impute numeric columns
            if numeric_columns:
                imputer = Imputer(
                    strategy=strategy,
                    inputCols=numeric_columns,
                    outputCols=numeric_columns,
                )
                df = imputer.fit(df).transform(df)
                logger.info(
                    f"Imputed numeric columns using {strategy}",
                    columns=len(numeric_columns),
                )

            # Fill categorical columns with mode (most frequent value)
            if categorical_columns:
                for col in categorical_columns:
                    mode_value = (
                        df.groupBy(col)
                        .count()
                        .orderBy(F.desc("count"))
                        .first()
                    )
                    if mode_value:
                        df = df.fillna({col: mode_value[0]})
                logger.info(
                    "Filled categorical columns with mode",
                    columns=len(categorical_columns),
                )

        else:
            raise ValueError(
                f"Invalid strategy: {strategy}. "
                "Choose from: 'mean', 'median', 'mode', 'drop', 'fill'"
            )

        return df

    def encode_categories(
        self,
        df: DataFrame,
        columns: List[str],
        method: str = "onehot",
        handle_invalid: str = "keep",
    ) -> Tuple[DataFrame, Pipeline]:
        """Encode categorical variables.

        Args:
            df: Input DataFrame
            columns: Categorical columns to encode
            method: Encoding method ('onehot', 'label')
            handle_invalid: How to handle unseen categories ('keep', 'skip', 'error')

        Returns:
            Tuple of (encoded DataFrame, fitted pipeline)

        Raises:
            ValueError: If method is invalid
        """
        logger.info(
            f"Encoding categorical variables",
            columns=len(columns),
            method=method,
        )

        stages = []

        if method == "label":
            # Label encoding (string indexing)
            for col in columns:
                indexer = StringIndexer(
                    inputCol=col,
                    outputCol=f"{col}_indexed",
                    handleInvalid=handle_invalid,
                )
                stages.append(indexer)

            pipeline = Pipeline(stages=stages)
            fitted_pipeline = pipeline.fit(df)
            df = fitted_pipeline.transform(df)

            logger.info(f"Applied label encoding to {len(columns)} columns")

        elif method == "onehot":
            # One-hot encoding
            indexed_cols = []
            for col in columns:
                # First, string indexing
                indexer = StringIndexer(
                    inputCol=col,
                    outputCol=f"{col}_indexed",
                    handleInvalid=handle_invalid,
                )
                stages.append(indexer)
                indexed_cols.append(f"{col}_indexed")

            # Then, one-hot encoding
            encoder = OneHotEncoder(
                inputCols=indexed_cols,
                outputCols=[f"{col}_encoded" for col in columns],
                handleInvalid=handle_invalid,
            )
            stages.append(encoder)

            pipeline = Pipeline(stages=stages)
            fitted_pipeline = pipeline.fit(df)
            df = fitted_pipeline.transform(df)

            logger.info(f"Applied one-hot encoding to {len(columns)} columns")

        else:
            raise ValueError(
                f"Invalid encoding method: {method}. Choose from: 'onehot', 'label'"
            )

        self.fitted_pipeline = fitted_pipeline
        return df, fitted_pipeline

    def normalize_features(
        self,
        df: DataFrame,
        input_cols: List[str],
        method: str = "standard",
        output_col: str = "scaled_features",
    ) -> Tuple[DataFrame, Pipeline]:
        """Normalize/scale numeric features.

        Args:
            df: Input DataFrame
            input_cols: Numeric columns to scale
            method: Scaling method ('standard', 'minmax')
            output_col: Name for output vector column

        Returns:
            Tuple of (scaled DataFrame, fitted pipeline)

        Raises:
            ValueError: If method is invalid
        """
        logger.info(
            f"Normalizing features",
            columns=len(input_cols),
            method=method,
        )

        # Assemble features into a vector
        assembler = VectorAssembler(
            inputCols=input_cols,
            outputCol="features_vector",
        )

        # Apply scaling
        if method == "standard":
            scaler = StandardScaler(
                inputCol="features_vector",
                outputCol=output_col,
                withMean=True,
                withStd=True,
            )
        elif method == "minmax":
            scaler = MinMaxScaler(
                inputCol="features_vector",
                outputCol=output_col,
            )
        else:
            raise ValueError(
                f"Invalid scaling method: {method}. "
                "Choose from: 'standard', 'minmax'"
            )

        # Create and fit pipeline
        pipeline = Pipeline(stages=[assembler, scaler])
        fitted_pipeline = pipeline.fit(df)
        df = fitted_pipeline.transform(df)

        logger.info(
            f"Applied {method} scaling to {len(input_cols)} features",
            output_column=output_col,
        )

        self.fitted_pipeline = fitted_pipeline
        return df, fitted_pipeline

    def create_feature_interactions(
        self,
        df: DataFrame,
        column_pairs: List[Tuple[str, str]],
        interaction_type: str = "multiply",
    ) -> DataFrame:
        """Create interaction features between column pairs.

        Args:
            df: Input DataFrame
            column_pairs: List of column pairs to create interactions
            interaction_type: Type of interaction ('multiply', 'add', 'subtract')

        Returns:
            DataFrame with interaction features

        Raises:
            ValueError: If interaction type is invalid
        """
        logger.info(
            f"Creating feature interactions",
            pairs=len(column_pairs),
            type=interaction_type,
        )

        for col1, col2 in column_pairs:
            interaction_name = f"{col1}_{interaction_type}_{col2}"

            if interaction_type == "multiply":
                df = df.withColumn(interaction_name, F.col(col1) * F.col(col2))
            elif interaction_type == "add":
                df = df.withColumn(interaction_name, F.col(col1) + F.col(col2))
            elif interaction_type == "subtract":
                df = df.withColumn(interaction_name, F.col(col1) - F.col(col2))
            else:
                raise ValueError(
                    f"Invalid interaction type: {interaction_type}. "
                    "Choose from: 'multiply', 'add', 'subtract'"
                )

        logger.info(
            f"Created {len(column_pairs)} interaction features",
            interaction_type=interaction_type,
        )

        return df

    def create_polynomial_features(
        self,
        df: DataFrame,
        columns: List[str],
        degree: int = 2,
    ) -> DataFrame:
        """Create polynomial features.

        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features from
            degree: Polynomial degree

        Returns:
            DataFrame with polynomial features
        """
        logger.info(
            f"Creating polynomial features",
            columns=len(columns),
            degree=degree,
        )

        for col in columns:
            for d in range(2, degree + 1):
                poly_col_name = f"{col}_poly_{d}"
                df = df.withColumn(poly_col_name, F.pow(F.col(col), d))

        logger.info(
            f"Created polynomial features up to degree {degree}",
            original_columns=len(columns),
            new_columns=len(columns) * (degree - 1),
        )

        return df

    def bin_numeric_features(
        self,
        df: DataFrame,
        column: str,
        bins: Union[int, List[float]],
        labels: Optional[List[str]] = None,
    ) -> DataFrame:
        """Bin numeric features into discrete categories.

        Args:
            df: Input DataFrame
            column: Column to bin
            bins: Number of bins or list of bin edges
            labels: Labels for bins (auto-generated if None)

        Returns:
            DataFrame with binned feature
        """
        logger.info(f"Binning numeric feature", column=column, bins=bins)

        binned_col_name = f"{column}_binned"

        if isinstance(bins, int):
            # Equal-width binning
            min_val, max_val = df.agg(
                F.min(column), F.max(column)
            ).first()
            bin_edges = np.linspace(min_val, max_val, bins + 1)
        else:
            bin_edges = bins

        # Create bin labels
        if labels is None:
            labels = [f"bin_{i}" for i in range(len(bin_edges) - 1)]

        # Apply binning using bucketizer or case-when logic
        df = df.withColumn(
            binned_col_name,
            F.when(F.col(column).isNull(), None)
        )

        for i in range(len(bin_edges) - 1):
            condition = (F.col(column) >= bin_edges[i]) & (
                F.col(column) < bin_edges[i + 1]
            )
            if i == len(bin_edges) - 2:  # Last bin includes upper bound
                condition = (F.col(column) >= bin_edges[i]) & (
                    F.col(column) <= bin_edges[i + 1]
                )
            df = df.withColumn(
                binned_col_name,
                F.when(condition, labels[i]).otherwise(F.col(binned_col_name))
            )

        logger.info(
            f"Binned feature {column} into {len(labels)} categories",
            output_column=binned_col_name,
        )

        return df

    def extract_datetime_features(
        self,
        df: DataFrame,
        datetime_column: str,
        features: Optional[List[str]] = None,
    ) -> DataFrame:
        """Extract features from datetime columns.

        Args:
            df: Input DataFrame
            datetime_column: Datetime column name
            features: Features to extract (None = all)
                     Options: year, month, day, dayofweek, hour, minute, quarter

        Returns:
            DataFrame with extracted datetime features
        """
        if features is None:
            features = ["year", "month", "day", "dayofweek", "hour"]

        logger.info(
            f"Extracting datetime features",
            column=datetime_column,
            features=features,
        )

        # Ensure column is timestamp type
        df = df.withColumn(datetime_column, F.col(datetime_column).cast("timestamp"))

        feature_map = {
            "year": F.year,
            "month": F.month,
            "day": F.dayofmonth,
            "dayofweek": F.dayofweek,
            "hour": F.hour,
            "minute": F.minute,
            "quarter": F.quarter,
            "weekofyear": F.weekofyear,
        }

        for feature in features:
            if feature in feature_map:
                feature_col_name = f"{datetime_column}_{feature}"
                df = df.withColumn(
                    feature_col_name,
                    feature_map[feature](F.col(datetime_column))
                )
            else:
                logger.warning(f"Unknown datetime feature: {feature}")

        logger.info(
            f"Extracted {len(features)} datetime features",
            column=datetime_column,
        )

        return df

    def apply_pipeline(self, df: DataFrame) -> DataFrame:
        """Apply fitted pipeline to new data.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame

        Raises:
            RuntimeError: If pipeline not fitted
        """
        if self.fitted_pipeline is None:
            logger.error("Pipeline not fitted")
            raise RuntimeError(
                "Pipeline not fitted. Call encode_categories or normalize_features first."
            )

        logger.info("Applying fitted pipeline to data")
        df = self.fitted_pipeline.transform(df)
        logger.info("Pipeline applied successfully")

        return df

    def get_feature_statistics(self, df: DataFrame) -> Dict[str, Any]:
        """Calculate statistics for features.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with feature statistics
        """
        logger.info("Calculating feature statistics")

        stats = {}

        # Get numeric columns
        numeric_cols = [
            f.name
            for f in df.schema.fields
            if f.dataType.typeName() in ["double", "float", "integer", "long"]
        ]

        if numeric_cols:
            # Calculate summary statistics
            summary = df.select(numeric_cols).summary().collect()
            stats["numeric"] = {
                row["summary"]: {
                    col: row[col] for col in numeric_cols
                }
                for row in summary
            }

        # Get categorical columns
        categorical_cols = [
            f.name
            for f in df.schema.fields
            if f.dataType.typeName() == "string"
        ]

        if categorical_cols:
            stats["categorical"] = {}
            for col in categorical_cols:
                value_counts = (
                    df.groupBy(col)
                    .count()
                    .orderBy(F.desc("count"))
                    .limit(10)
                    .collect()
                )
                stats["categorical"][col] = [
                    {"value": row[col], "count": row["count"]}
                    for row in value_counts
                ]

        logger.info(
            "Feature statistics calculated",
            numeric_features=len(numeric_cols),
            categorical_features=len(categorical_cols),
        )

        return stats
