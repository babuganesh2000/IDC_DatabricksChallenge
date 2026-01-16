"""Data quality validation using Great Expectations.

This module provides classes for validating data quality, checking schemas,
and generating data quality reports.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate data quality using Great Expectations."""

    def __init__(
        self,
        context_root_dir: Optional[str] = None,
        spark: Optional[SparkSession] = None,
    ):
        """Initialize data validator.

        Args:
            context_root_dir: Root directory for Great Expectations context
            spark: SparkSession instance

        Raises:
            ImportError: If Great Expectations is not installed
        """
        try:
            import great_expectations as gx
            from great_expectations.core.batch import RuntimeBatchRequest
            self.gx = gx
            self.RuntimeBatchRequest = RuntimeBatchRequest
        except ImportError:
            logger.error("Great Expectations not installed")
            raise ImportError(
                "great_expectations package is required. "
                "Install with: pip install great-expectations"
            )

        self.spark = spark or SparkSession.builder.getOrCreate()
        self.context_root_dir = context_root_dir or "./gx"

        # Initialize Great Expectations DataContext
        try:
            self.context = self.gx.get_context(
                context_root_dir=self.context_root_dir
            )
            logger.info(
                "Great Expectations context initialized",
                root_dir=self.context_root_dir,
            )
        except Exception as e:
            logger.warning(
                f"Could not load existing context: {str(e)}. Creating new context."
            )
            Path(self.context_root_dir).mkdir(parents=True, exist_ok=True)
            self.context = self.gx.data_context.DataContext.create(
                self.context_root_dir
            )
            logger.info("New Great Expectations context created")

    def validate_schema(
        self,
        df: DataFrame,
        expected_schema: Union[StructType, Dict[str, str]],
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Validate DataFrame schema against expected schema.

        Args:
            df: Input DataFrame
            expected_schema: Expected schema (StructType or dict)
            strict: Whether to enforce strict schema matching

        Returns:
            Validation result dictionary
        """
        logger.info("Validating schema")

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "schema_match": False,
        }

        actual_schema = df.schema

        # Convert expected schema to dict if StructType
        if isinstance(expected_schema, StructType):
            expected_dict = {
                field.name: field.dataType.typeName()
                for field in expected_schema.fields
            }
        else:
            expected_dict = expected_schema

        actual_dict = {
            field.name: field.dataType.typeName()
            for field in actual_schema.fields
        }

        # Check for missing columns
        missing_cols = set(expected_dict.keys()) - set(actual_dict.keys())
        if missing_cols:
            error_msg = f"Missing columns: {', '.join(missing_cols)}"
            validation_result["errors"].append(error_msg)
            validation_result["valid"] = False
            logger.error(error_msg)

        # Check for extra columns
        extra_cols = set(actual_dict.keys()) - set(expected_dict.keys())
        if extra_cols:
            msg = f"Extra columns: {', '.join(extra_cols)}"
            if strict:
                validation_result["errors"].append(msg)
                validation_result["valid"] = False
                logger.error(msg)
            else:
                validation_result["warnings"].append(msg)
                logger.warning(msg)

        # Check data types for matching columns
        matching_cols = set(expected_dict.keys()) & set(actual_dict.keys())
        for col in matching_cols:
            if expected_dict[col] != actual_dict[col]:
                error_msg = (
                    f"Column '{col}' type mismatch: "
                    f"expected {expected_dict[col]}, got {actual_dict[col]}"
                )
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False
                logger.error(error_msg)

        # Overall schema match
        validation_result["schema_match"] = (
            validation_result["valid"] and not extra_cols
        )

        logger.info(
            "Schema validation completed",
            valid=validation_result["valid"],
            errors=len(validation_result["errors"]),
            warnings=len(validation_result["warnings"]),
        )

        return validation_result

    def check_data_quality(
        self,
        df: DataFrame,
        expectations: Optional[Dict[str, Any]] = None,
        expectation_suite_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check data quality against expectations.

        Args:
            df: Input DataFrame
            expectations: Dictionary of expectations to validate
            expectation_suite_name: Name of existing expectation suite

        Returns:
            Validation result dictionary
        """
        logger.info(
            "Checking data quality",
            expectation_suite=expectation_suite_name,
        )

        # Convert Spark DataFrame to Pandas for Great Expectations
        # Note: For large datasets, consider sampling
        try:
            pandas_df = df.limit(10000).toPandas()
            logger.info("Converted DataFrame to Pandas for validation")
        except Exception as e:
            logger.error(f"Failed to convert DataFrame: {str(e)}")
            raise RuntimeError(f"Failed to convert DataFrame: {str(e)}") from e

        # Create or get expectation suite
        if expectation_suite_name:
            try:
                suite = self.context.get_expectation_suite(expectation_suite_name)
                logger.info(f"Using existing expectation suite: {expectation_suite_name}")
            except Exception:
                suite = self.context.add_expectation_suite(expectation_suite_name)
                logger.info(f"Created new expectation suite: {expectation_suite_name}")
        else:
            expectation_suite_name = "default_suite"
            suite = self.context.add_or_update_expectation_suite(
                expectation_suite_name
            )

        # Add expectations if provided
        if expectations:
            self._add_expectations_to_suite(suite, expectations)

        # Create batch request
        try:
            batch_request = self.RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="runtime_data_connector",
                data_asset_name="validation_data",
                runtime_parameters={"batch_data": pandas_df},
                batch_identifiers={"default_identifier_name": "default_identifier"},
            )

            # Validate
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=expectation_suite_name,
            )

            results = validator.validate()

            validation_result = {
                "success": results.success,
                "statistics": results.statistics,
                "results": [
                    {
                        "expectation_type": result.expectation_config.expectation_type,
                        "success": result.success,
                        "result": result.result if hasattr(result, 'result') else None,
                    }
                    for result in results.results
                ],
            }

            logger.info(
                "Data quality check completed",
                success=results.success,
                expectations_evaluated=len(results.results),
            )

            return validation_result

        except Exception as e:
            logger.error(f"Data quality check failed: {str(e)}")
            raise RuntimeError(f"Data quality check failed: {str(e)}") from e

    def _add_expectations_to_suite(
        self,
        suite: Any,
        expectations: Dict[str, Any],
    ) -> None:
        """Add expectations to an expectation suite.

        Args:
            suite: Expectation suite
            expectations: Dictionary of expectations
        """
        logger.info("Adding expectations to suite")

        # Example expectations structure:
        # {
        #     "column_expectations": {
        #         "age": {
        #             "expect_column_values_to_be_between": {"min_value": 0, "max_value": 120},
        #             "expect_column_values_to_not_be_null": {}
        #         }
        #     },
        #     "table_expectations": {
        #         "expect_table_row_count_to_be_between": {"min_value": 100, "max_value": 10000}
        #     }
        # }

        column_expectations = expectations.get("column_expectations", {})
        for column, column_exps in column_expectations.items():
            for exp_type, exp_kwargs in column_exps.items():
                suite.add_expectation(
                    self.gx.core.ExpectationConfiguration(
                        expectation_type=exp_type,
                        kwargs={"column": column, **exp_kwargs},
                    )
                )

        table_expectations = expectations.get("table_expectations", {})
        for exp_type, exp_kwargs in table_expectations.items():
            suite.add_expectation(
                self.gx.core.ExpectationConfiguration(
                    expectation_type=exp_type,
                    kwargs=exp_kwargs,
                )
            )

        logger.info(f"Added expectations to suite: {suite.expectation_suite_name}")

    def generate_quality_report(
        self,
        df: DataFrame,
        report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive data quality report.

        Args:
            df: Input DataFrame
            report_path: Path to save report (optional)

        Returns:
            Data quality report dictionary
        """
        logger.info("Generating data quality report")

        report = {
            "summary": {},
            "column_statistics": {},
            "data_quality_checks": {},
        }

        # Summary statistics
        row_count = df.count()
        column_count = len(df.columns)

        report["summary"] = {
            "row_count": row_count,
            "column_count": column_count,
            "columns": df.columns,
        }

        logger.info(f"Dataset summary: {row_count} rows, {column_count} columns")

        # Column-level statistics
        for column in df.columns:
            col_stats = self._calculate_column_statistics(df, column)
            report["column_statistics"][column] = col_stats

        # Data quality checks
        report["data_quality_checks"] = self._perform_quality_checks(df)

        # Save report if path provided
        if report_path:
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Quality report saved to: {report_path}")

        logger.info("Data quality report generated")
        return report

    def _calculate_column_statistics(
        self,
        df: DataFrame,
        column: str,
    ) -> Dict[str, Any]:
        """Calculate statistics for a single column.

        Args:
            df: Input DataFrame
            column: Column name

        Returns:
            Column statistics dictionary
        """
        stats = {
            "data_type": str(df.schema[column].dataType),
            "null_count": df.filter(F.col(column).isNull()).count(),
            "null_percentage": 0.0,
            "distinct_count": df.select(column).distinct().count(),
        }

        total_count = df.count()
        if total_count > 0:
            stats["null_percentage"] = (stats["null_count"] / total_count) * 100

        # Numeric column statistics
        if df.schema[column].dataType.typeName() in [
            "double",
            "float",
            "integer",
            "long",
        ]:
            numeric_stats = df.select(
                F.min(column).alias("min"),
                F.max(column).alias("max"),
                F.mean(column).alias("mean"),
                F.stddev(column).alias("stddev"),
            ).first()

            stats.update(
                {
                    "min": numeric_stats["min"],
                    "max": numeric_stats["max"],
                    "mean": numeric_stats["mean"],
                    "stddev": numeric_stats["stddev"],
                }
            )

            # Percentiles
            percentiles = df.stat.approxQuantile(
                column, [0.25, 0.5, 0.75], 0.01
            )
            stats.update(
                {
                    "q1": percentiles[0] if len(percentiles) > 0 else None,
                    "median": percentiles[1] if len(percentiles) > 1 else None,
                    "q3": percentiles[2] if len(percentiles) > 2 else None,
                }
            )

        # String column statistics
        elif df.schema[column].dataType.typeName() == "string":
            value_counts = (
                df.groupBy(column)
                .count()
                .orderBy(F.desc("count"))
                .limit(5)
                .collect()
            )

            stats["top_values"] = [
                {"value": row[column], "count": row["count"]}
                for row in value_counts
            ]

            # Calculate average length
            avg_length = df.select(
                F.avg(F.length(F.col(column))).alias("avg_length")
            ).first()["avg_length"]
            stats["avg_length"] = avg_length

        return stats

    def _perform_quality_checks(self, df: DataFrame) -> Dict[str, Any]:
        """Perform basic data quality checks.

        Args:
            df: Input DataFrame

        Returns:
            Quality check results dictionary
        """
        checks = {}

        # Check for duplicate rows
        total_rows = df.count()
        distinct_rows = df.distinct().count()
        duplicate_rows = total_rows - distinct_rows

        checks["duplicate_rows"] = {
            "count": duplicate_rows,
            "percentage": (duplicate_rows / total_rows * 100) if total_rows > 0 else 0,
        }

        # Check for columns with high null percentage
        high_null_columns = []
        for column in df.columns:
            null_count = df.filter(F.col(column).isNull()).count()
            null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0
            if null_percentage > 50:
                high_null_columns.append(
                    {"column": column, "null_percentage": null_percentage}
                )

        checks["high_null_columns"] = high_null_columns

        # Check for constant columns (single unique value)
        constant_columns = []
        for column in df.columns:
            distinct_count = df.select(column).distinct().count()
            if distinct_count == 1:
                constant_columns.append(column)

        checks["constant_columns"] = constant_columns

        # Check for columns with low cardinality (potential categorical)
        low_cardinality_columns = []
        for column in df.columns:
            distinct_count = df.select(column).distinct().count()
            if 1 < distinct_count <= 10:
                low_cardinality_columns.append(
                    {"column": column, "distinct_count": distinct_count}
                )

        checks["low_cardinality_columns"] = low_cardinality_columns

        return checks

    def validate_data_freshness(
        self,
        df: DataFrame,
        timestamp_column: str,
        max_age_hours: int = 24,
    ) -> Dict[str, Any]:
        """Validate data freshness based on timestamp column.

        Args:
            df: Input DataFrame
            timestamp_column: Column containing timestamps
            max_age_hours: Maximum acceptable age in hours

        Returns:
            Freshness validation result
        """
        logger.info(
            "Validating data freshness",
            timestamp_column=timestamp_column,
            max_age_hours=max_age_hours,
        )

        try:
            # Get latest timestamp
            latest_timestamp = df.agg(
                F.max(timestamp_column).alias("latest")
            ).first()["latest"]

            # Calculate age
            current_time = datetime.now(timezone.utc)
            
            if latest_timestamp:
                age_seconds = (current_time - latest_timestamp).total_seconds()
                age_hours = age_seconds / 3600

                is_fresh = age_hours <= max_age_hours

                result = {
                    "is_fresh": is_fresh,
                    "latest_timestamp": latest_timestamp.isoformat(),
                    "age_hours": age_hours,
                    "max_age_hours": max_age_hours,
                }

                if not is_fresh:
                    logger.warning(
                        f"Data is stale",
                        age_hours=age_hours,
                        max_age_hours=max_age_hours,
                    )
                else:
                    logger.info(f"Data is fresh", age_hours=age_hours)

                return result
            else:
                logger.error("No timestamp data found")
                return {
                    "is_fresh": False,
                    "error": "No timestamp data found",
                }

        except Exception as e:
            logger.error(f"Freshness validation failed: {str(e)}")
            return {
                "is_fresh": False,
                "error": str(e),
            }

    def compare_datasets(
        self,
        df1: DataFrame,
        df2: DataFrame,
        key_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare two datasets and identify differences.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            key_columns: Columns to use as keys for comparison

        Returns:
            Comparison result dictionary
        """
        logger.info("Comparing datasets")

        comparison = {
            "row_counts": {
                "df1": df1.count(),
                "df2": df2.count(),
            },
            "column_differences": {},
            "schema_match": False,
        }

        # Schema comparison
        df1_cols = set(df1.columns)
        df2_cols = set(df2.columns)

        comparison["column_differences"] = {
            "only_in_df1": list(df1_cols - df2_cols),
            "only_in_df2": list(df2_cols - df1_cols),
            "common": list(df1_cols & df2_cols),
        }

        comparison["schema_match"] = (
            df1_cols == df2_cols and df1.schema == df2.schema
        )

        # Data comparison (if schemas match and key columns provided)
        if key_columns and comparison["schema_match"]:
            try:
                # Find rows in df1 but not in df2
                only_in_df1 = df1.join(df2, key_columns, "left_anti").count()

                # Find rows in df2 but not in df1
                only_in_df2 = df2.join(df1, key_columns, "left_anti").count()

                comparison["data_differences"] = {
                    "rows_only_in_df1": only_in_df1,
                    "rows_only_in_df2": only_in_df2,
                }

                logger.info(
                    "Dataset comparison completed",
                    rows_only_in_df1=only_in_df1,
                    rows_only_in_df2=only_in_df2,
                )
            except Exception as e:
                logger.warning(f"Could not perform data comparison: {str(e)}")

        return comparison
