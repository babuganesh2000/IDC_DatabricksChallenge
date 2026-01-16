"""Databricks Feature Store integration module.

This module provides integration with Databricks Feature Store for feature
management, versioning, and serving.
"""

import logging
from typing import Dict, List, Optional, Union

from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


class FeatureStoreManager:
    """Manage features in Databricks Feature Store.

    This class provides methods to create, write, read, and manage features
    in Databricks Feature Store for online and offline feature serving.
    """

    def __init__(self, database_name: str = "feature_store"):
        """Initialize FeatureStoreManager.

        Args:
            database_name: Database name for feature tables
        """
        self.database_name = database_name
        self._feature_store = None
        logger.info("Initialized FeatureStoreManager with database=%s", database_name)

    def _get_feature_store(self):
        """Get or create Feature Store client.

        Returns:
            Feature Store client instance

        Raises:
            ImportError: If Databricks Feature Store SDK is not available
        """
        if self._feature_store is None:
            try:
                from databricks.feature_store import FeatureStoreClient

                self._feature_store = FeatureStoreClient()
                logger.info("Initialized Databricks Feature Store client")
            except ImportError as e:
                logger.error("Databricks Feature Store SDK not available: %s", str(e))
                raise ImportError(
                    "Databricks Feature Store SDK is required. "
                    "Install with: pip install databricks-feature-store"
                ) from e
        return self._feature_store

    def create_feature_table(
        self,
        table_name: str,
        primary_keys: Union[str, List[str]],
        df: DataFrame,
        description: Optional[str] = None,
        partition_columns: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Create a new feature table in Feature Store.

        Args:
            table_name: Name of the feature table (without database prefix)
            primary_keys: Primary key column(s) for the feature table
            df: DataFrame with schema for the feature table
            description: Optional description of the feature table
            partition_columns: Optional columns to partition the table by
            tags: Optional tags for the feature table

        Raises:
            ValueError: If table name or primary keys are invalid
            RuntimeError: If table creation fails
        """
        if not table_name:
            raise ValueError("table_name cannot be empty")

        if isinstance(primary_keys, str):
            primary_keys = [primary_keys]

        if not primary_keys:
            raise ValueError("primary_keys cannot be empty")

        full_table_name = f"{self.database_name}.{table_name}"
        logger.info("Creating feature table: %s", full_table_name)

        try:
            fs = self._get_feature_store()

            # Create feature table
            fs.create_table(
                name=full_table_name,
                primary_keys=primary_keys,
                df=df,
                description=description or f"Feature table: {table_name}",
                partition_columns=partition_columns,
                tags=tags or {},
            )

            logger.info(
                "Successfully created feature table: %s with primary keys: %s",
                full_table_name,
                primary_keys,
            )

        except Exception as e:
            logger.error("Error creating feature table %s: %s", full_table_name, str(e))
            raise RuntimeError(f"Failed to create feature table {full_table_name}: {str(e)}") from e

    def write_features(
        self,
        table_name: str,
        df: DataFrame,
        mode: str = "merge",
    ) -> None:
        """Write features to an existing feature table.

        Args:
            table_name: Name of the feature table (without database prefix)
            df: DataFrame containing features to write
            mode: Write mode - 'merge' or 'overwrite'

        Raises:
            ValueError: If table name is invalid or mode is unsupported
            RuntimeError: If write operation fails
        """
        if not table_name:
            raise ValueError("table_name cannot be empty")

        if mode not in ["merge", "overwrite"]:
            raise ValueError(f"Unsupported mode: {mode}. Use 'merge' or 'overwrite'")

        full_table_name = f"{self.database_name}.{table_name}"
        logger.info("Writing features to table: %s with mode: %s", full_table_name, mode)

        try:
            fs = self._get_feature_store()

            if mode == "merge":
                fs.write_table(
                    name=full_table_name,
                    df=df,
                    mode="merge",
                )
            else:
                fs.write_table(
                    name=full_table_name,
                    df=df,
                    mode="overwrite",
                )

            logger.info(
                "Successfully wrote %d rows to feature table: %s",
                df.count(),
                full_table_name,
            )

        except Exception as e:
            logger.error("Error writing to feature table %s: %s", full_table_name, str(e))
            raise RuntimeError(
                f"Failed to write to feature table {full_table_name}: {str(e)}"
            ) from e

    def read_features(
        self,
        table_name: str,
        feature_names: Optional[List[str]] = None,
    ) -> DataFrame:
        """Read features from a feature table.

        Args:
            table_name: Name of the feature table (without database prefix)
            feature_names: Optional list of specific feature names to read.
                          If None, reads all features.

        Returns:
            DataFrame containing the requested features

        Raises:
            ValueError: If table name is invalid
            RuntimeError: If read operation fails
        """
        if not table_name:
            raise ValueError("table_name cannot be empty")

        full_table_name = f"{self.database_name}.{table_name}"
        logger.info("Reading features from table: %s", full_table_name)

        try:
            fs = self._get_feature_store()

            # Read features
            df = fs.read_table(name=full_table_name)

            # Filter specific features if requested
            if feature_names:
                # Get primary key columns
                table_metadata = fs.get_table(full_table_name)
                primary_keys = table_metadata.primary_keys

                # Select primary keys and requested features
                columns_to_select = list(set(primary_keys + feature_names))
                df = df.select(*columns_to_select)

                logger.info(
                    "Read %d features from table: %s",
                    len(feature_names),
                    full_table_name,
                )
            else:
                logger.info("Read all features from table: %s", full_table_name)

            return df

        except Exception as e:
            logger.error("Error reading from feature table %s: %s", full_table_name, str(e))
            raise RuntimeError(
                f"Failed to read from feature table {full_table_name}: {str(e)}"
            ) from e

    def get_feature_table_metadata(self, table_name: str) -> Dict:
        """Get metadata for a feature table.

        Args:
            table_name: Name of the feature table (without database prefix)

        Returns:
            Dictionary containing feature table metadata

        Raises:
            ValueError: If table name is invalid
            RuntimeError: If operation fails
        """
        if not table_name:
            raise ValueError("table_name cannot be empty")

        full_table_name = f"{self.database_name}.{table_name}"
        logger.info("Getting metadata for feature table: %s", full_table_name)

        try:
            fs = self._get_feature_store()
            table = fs.get_table(full_table_name)

            metadata = {
                "name": table.name,
                "primary_keys": table.primary_keys,
                "description": table.description,
                "partition_columns": getattr(table, "partition_columns", []),
                "tags": getattr(table, "tags", {}),
            }

            logger.info("Retrieved metadata for table: %s", full_table_name)
            return metadata

        except Exception as e:
            logger.error(
                "Error getting metadata for feature table %s: %s",
                full_table_name,
                str(e),
            )
            raise RuntimeError(
                f"Failed to get metadata for feature table {full_table_name}: {str(e)}"
            ) from e

    def delete_feature_table(self, table_name: str) -> None:
        """Delete a feature table.

        Args:
            table_name: Name of the feature table (without database prefix)

        Raises:
            ValueError: If table name is invalid
            RuntimeError: If deletion fails
        """
        if not table_name:
            raise ValueError("table_name cannot be empty")

        full_table_name = f"{self.database_name}.{table_name}"
        logger.warning("Deleting feature table: %s", full_table_name)

        try:
            fs = self._get_feature_store()
            fs.drop_table(name=full_table_name)

            logger.info("Successfully deleted feature table: %s", full_table_name)

        except Exception as e:
            logger.error("Error deleting feature table %s: %s", full_table_name, str(e))
            raise RuntimeError(f"Failed to delete feature table {full_table_name}: {str(e)}") from e

    def create_training_set(
        self,
        df: DataFrame,
        feature_lookups: List[Dict],
        label: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None,
    ) -> DataFrame:
        """Create a training dataset with features from Feature Store.

        Args:
            df: DataFrame containing keys to lookup features
            feature_lookups: List of feature lookup specifications, each containing:
                - table_name: Feature table name
                - lookup_key: Column(s) to join on
                - feature_names: Optional list of features to include
            label: Optional label column name
            exclude_columns: Optional columns to exclude from training set

        Returns:
            DataFrame with features joined for training

        Raises:
            ValueError: If feature_lookups is invalid
            RuntimeError: If training set creation fails
        """
        if not feature_lookups:
            raise ValueError("feature_lookups cannot be empty")

        logger.info("Creating training set with %d feature lookups", len(feature_lookups))

        try:
            fs = self._get_feature_store()

            # Convert feature lookups to Feature Store format
            from databricks.feature_store import FeatureLookup

            lookups = []
            for lookup_spec in feature_lookups:
                table_name = lookup_spec.get("table_name")
                if not table_name:
                    raise ValueError("table_name is required in feature_lookups")

                full_table_name = f"{self.database_name}.{table_name}"
                lookup_key = lookup_spec.get("lookup_key")
                feature_names = lookup_spec.get("feature_names")

                lookups.append(
                    FeatureLookup(
                        table_name=full_table_name,
                        lookup_key=lookup_key,
                        feature_names=feature_names,
                    )
                )

            # Create training set
            training_set = fs.create_training_set(
                df=df,
                feature_lookups=lookups,
                label=label,
                exclude_columns=exclude_columns or [],
            )

            # Load as DataFrame
            training_df = training_set.load_df()

            logger.info(
                "Successfully created training set with %d rows",
                training_df.count(),
            )
            return training_df

        except Exception as e:
            logger.error("Error creating training set: %s", str(e))
            raise RuntimeError(f"Failed to create training set: {str(e)}") from e

    def publish_features(
        self,
        table_name: str,
        online_store: Optional[Dict] = None,
    ) -> None:
        """Publish features to online store for real-time serving.

        Args:
            table_name: Name of the feature table (without database prefix)
            online_store: Optional online store configuration

        Raises:
            ValueError: If table name is invalid
            RuntimeError: If publishing fails
        """
        if not table_name:
            raise ValueError("table_name cannot be empty")

        full_table_name = f"{self.database_name}.{table_name}"
        logger.info("Publishing features to online store: %s", full_table_name)

        try:
            fs = self._get_feature_store()

            # Publish to online store
            fs.publish_table(
                name=full_table_name,
                online_store=online_store,
            )

            logger.info(
                "Successfully published features to online store: %s",
                full_table_name,
            )

        except Exception as e:
            logger.error(
                "Error publishing features to online store %s: %s",
                full_table_name,
                str(e),
            )
            raise RuntimeError(
                f"Failed to publish features to online store {full_table_name}: {str(e)}"
            ) from e

    def search_features(
        self,
        name_pattern: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """Search for feature tables matching criteria.

        Args:
            name_pattern: Optional pattern to match table names
            tags: Optional tags to filter by

        Returns:
            List of feature table metadata matching criteria

        Raises:
            RuntimeError: If search operation fails
        """
        logger.info("Searching for feature tables")

        try:
            fs = self._get_feature_store()

            # Search for tables
            if name_pattern:
                search_pattern = f"{self.database_name}.{name_pattern}"
            else:
                search_pattern = f"{self.database_name}.*"

            tables = fs.search_tables(name=search_pattern, tags=tags or {})

            results = []
            for table in tables:
                results.append(
                    {
                        "name": table.name,
                        "primary_keys": table.primary_keys,
                        "description": table.description,
                        "tags": getattr(table, "tags", {}),
                    }
                )

            logger.info("Found %d feature tables matching criteria", len(results))
            return results

        except Exception as e:
            logger.error("Error searching for feature tables: %s", str(e))
            raise RuntimeError(f"Failed to search feature tables: {str(e)}") from e
