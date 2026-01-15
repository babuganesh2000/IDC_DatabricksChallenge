"""Data ingestion from Kaggle and cloud storage.

This module provides classes for ingesting data from various sources including
Kaggle datasets and cloud storage services (Azure Blob Storage and AWS S3).
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class KaggleDataIngestion:
    """Ingest data from Kaggle datasets."""

    def __init__(
        self,
        api_username: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize Kaggle data ingestion.

        Args:
            api_username: Kaggle API username (defaults to env var)
            api_key: Kaggle API key (defaults to env var)

        Raises:
            ImportError: If kaggle package is not installed
            ValueError: If credentials are not provided
        """
        try:
            import kaggle
            self.kaggle = kaggle
        except ImportError:
            logger.error("Kaggle package not installed")
            raise ImportError(
                "kaggle package is required. Install with: pip install kaggle"
            )

        # Set credentials if provided
        if api_username and api_key:
            os.environ["KAGGLE_USERNAME"] = api_username
            os.environ["KAGGLE_KEY"] = api_key

        # Verify credentials
        if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
            logger.error("Kaggle credentials not found")
            raise ValueError(
                "Kaggle credentials required. Set KAGGLE_USERNAME and KAGGLE_KEY "
                "environment variables or pass them to constructor."
            )

        logger.info("Kaggle data ingestion initialized")

    def download_dataset(
        self,
        dataset_name: str,
        download_path: str = "./data/raw",
        unzip: bool = True,
        force: bool = False,
    ) -> str:
        """Download dataset from Kaggle.

        Args:
            dataset_name: Kaggle dataset name (format: owner/dataset-name)
            download_path: Local path to download dataset
            unzip: Whether to unzip downloaded files
            force: Force download even if files exist

        Returns:
            Path to downloaded dataset

        Raises:
            ValueError: If dataset name is invalid
            RuntimeError: If download fails
        """
        logger.info(
            f"Downloading Kaggle dataset",
            dataset_name=dataset_name,
            download_path=download_path,
        )

        # Validate dataset name
        if "/" not in dataset_name:
            logger.error(f"Invalid dataset name: {dataset_name}")
            raise ValueError(
                f"Invalid dataset name: {dataset_name}. "
                "Format should be: owner/dataset-name"
            )

        # Create download directory
        Path(download_path).mkdir(parents=True, exist_ok=True)

        try:
            # Download dataset
            self.kaggle.api.dataset_download_files(
                dataset_name,
                path=download_path,
                unzip=unzip,
                force=force,
                quiet=False,
            )

            logger.info(
                f"Successfully downloaded dataset",
                dataset_name=dataset_name,
                path=download_path,
            )
            return download_path

        except Exception as e:
            logger.error(
                f"Failed to download dataset",
                dataset_name=dataset_name,
                error=str(e),
            )
            raise RuntimeError(f"Failed to download dataset: {str(e)}") from e

    def list_dataset_files(self, dataset_name: str) -> List[Dict[str, Any]]:
        """List files in a Kaggle dataset.

        Args:
            dataset_name: Kaggle dataset name (format: owner/dataset-name)

        Returns:
            List of file metadata dictionaries

        Raises:
            ValueError: If dataset name is invalid
            RuntimeError: If listing fails
        """
        logger.info(f"Listing files for dataset", dataset_name=dataset_name)

        if "/" not in dataset_name:
            raise ValueError(
                f"Invalid dataset name: {dataset_name}. "
                "Format should be: owner/dataset-name"
            )

        try:
            files = self.kaggle.api.dataset_list_files(dataset_name).files
            file_list = [
                {"name": f.name, "size": f.size, "creation_date": f.creationDate}
                for f in files
            ]

            logger.info(
                f"Found {len(file_list)} files in dataset",
                dataset_name=dataset_name,
            )
            return file_list

        except Exception as e:
            logger.error(
                f"Failed to list dataset files",
                dataset_name=dataset_name,
                error=str(e),
            )
            raise RuntimeError(f"Failed to list dataset files: {str(e)}") from e


class CloudStorageIngestion:
    """Ingest data from cloud storage (Azure Blob Storage or AWS S3)."""

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        storage_account: Optional[str] = None,
        access_key: Optional[str] = None,
        container: Optional[str] = None,
    ):
        """Initialize cloud storage ingestion.

        Args:
            spark: SparkSession instance
            storage_account: Azure storage account name
            access_key: Azure storage account access key or AWS access key
            container: Azure container name or S3 bucket name
        """
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.storage_account = storage_account
        self.access_key = access_key
        self.container = container

        logger.info(
            "Cloud storage ingestion initialized",
            storage_account=storage_account,
            container=container,
        )

    def _configure_azure_storage(self) -> None:
        """Configure Azure Blob Storage access."""
        if not self.storage_account or not self.access_key:
            logger.error("Azure storage credentials not configured")
            raise ValueError(
                "Azure storage account and access key are required for Azure Blob Storage"
            )

        storage_key = f"fs.azure.account.key.{self.storage_account}.blob.core.windows.net"
        self.spark.conf.set(storage_key, self.access_key)
        logger.info("Azure Blob Storage configured")

    def _configure_s3_storage(self, aws_secret_key: Optional[str] = None) -> None:
        """Configure AWS S3 access.

        Args:
            aws_secret_key: AWS secret access key
        """
        if not self.access_key or not aws_secret_key:
            logger.error("AWS S3 credentials not configured")
            raise ValueError("AWS access key and secret key are required for S3")

        self.spark.conf.set("fs.s3a.access.key", self.access_key)
        self.spark.conf.set("fs.s3a.secret.key", aws_secret_key)
        logger.info("AWS S3 configured")

    def read_csv(
        self,
        path: str,
        header: bool = True,
        infer_schema: bool = True,
        **options: Any,
    ) -> DataFrame:
        """Read CSV file from cloud storage.

        Args:
            path: Path to CSV file (Azure: wasbs:// or S3: s3a://)
            header: Whether CSV has header row
            infer_schema: Whether to infer schema
            **options: Additional Spark CSV options

        Returns:
            Spark DataFrame

        Raises:
            ValueError: If path is invalid
            RuntimeError: If read fails
        """
        logger.info(f"Reading CSV from cloud storage", path=path)

        # Configure storage based on path
        if path.startswith("wasbs://") or path.startswith("abfss://"):
            self._configure_azure_storage()
        elif path.startswith("s3://") or path.startswith("s3a://"):
            # Note: aws_secret_key should be configured separately
            logger.warning("Ensure AWS S3 credentials are configured")

        try:
            df = (
                self.spark.read.format("csv")
                .option("header", header)
                .option("inferSchema", infer_schema)
                .options(**options)
                .load(path)
            )

            row_count = df.count()
            logger.info(
                f"Successfully read CSV",
                path=path,
                rows=row_count,
                columns=len(df.columns),
            )
            return df

        except Exception as e:
            logger.error(f"Failed to read CSV", path=path, error=str(e))
            raise RuntimeError(f"Failed to read CSV from {path}: {str(e)}") from e

    def read_parquet(self, path: str, **options: Any) -> DataFrame:
        """Read Parquet file from cloud storage.

        Args:
            path: Path to Parquet file (Azure: wasbs:// or S3: s3a://)
            **options: Additional Spark Parquet options

        Returns:
            Spark DataFrame

        Raises:
            ValueError: If path is invalid
            RuntimeError: If read fails
        """
        logger.info(f"Reading Parquet from cloud storage", path=path)

        # Configure storage based on path
        if path.startswith("wasbs://") or path.startswith("abfss://"):
            self._configure_azure_storage()
        elif path.startswith("s3://") or path.startswith("s3a://"):
            logger.warning("Ensure AWS S3 credentials are configured")

        try:
            df = self.spark.read.format("parquet").options(**options).load(path)

            row_count = df.count()
            logger.info(
                f"Successfully read Parquet",
                path=path,
                rows=row_count,
                columns=len(df.columns),
            )
            return df

        except Exception as e:
            logger.error(f"Failed to read Parquet", path=path, error=str(e))
            raise RuntimeError(f"Failed to read Parquet from {path}: {str(e)}") from e

    def write_csv(
        self,
        df: DataFrame,
        path: str,
        mode: str = "overwrite",
        header: bool = True,
        **options: Any,
    ) -> None:
        """Write DataFrame to CSV in cloud storage.

        Args:
            df: Spark DataFrame to write
            path: Destination path (Azure: wasbs:// or S3: s3a://)
            mode: Write mode (overwrite, append, etc.)
            header: Whether to write header row
            **options: Additional Spark CSV options

        Raises:
            RuntimeError: If write fails
        """
        logger.info(f"Writing CSV to cloud storage", path=path, mode=mode)

        # Configure storage based on path
        if path.startswith("wasbs://") or path.startswith("abfss://"):
            self._configure_azure_storage()
        elif path.startswith("s3://") or path.startswith("s3a://"):
            logger.warning("Ensure AWS S3 credentials are configured")

        try:
            (
                df.write.format("csv")
                .mode(mode)
                .option("header", header)
                .options(**options)
                .save(path)
            )

            logger.info(f"Successfully wrote CSV", path=path)

        except Exception as e:
            logger.error(f"Failed to write CSV", path=path, error=str(e))
            raise RuntimeError(f"Failed to write CSV to {path}: {str(e)}") from e

    def write_parquet(
        self,
        df: DataFrame,
        path: str,
        mode: str = "overwrite",
        partition_by: Optional[List[str]] = None,
        **options: Any,
    ) -> None:
        """Write DataFrame to Parquet in cloud storage.

        Args:
            df: Spark DataFrame to write
            path: Destination path (Azure: wasbs:// or S3: s3a://)
            mode: Write mode (overwrite, append, etc.)
            partition_by: Columns to partition by
            **options: Additional Spark Parquet options

        Raises:
            RuntimeError: If write fails
        """
        logger.info(
            f"Writing Parquet to cloud storage",
            path=path,
            mode=mode,
            partition_by=partition_by,
        )

        # Configure storage based on path
        if path.startswith("wasbs://") or path.startswith("abfss://"):
            self._configure_azure_storage()
        elif path.startswith("s3://") or path.startswith("s3a://"):
            logger.warning("Ensure AWS S3 credentials are configured")

        try:
            writer = df.write.format("parquet").mode(mode).options(**options)

            if partition_by:
                writer = writer.partitionBy(*partition_by)

            writer.save(path)

            logger.info(f"Successfully wrote Parquet", path=path)

        except Exception as e:
            logger.error(f"Failed to write Parquet", path=path, error=str(e))
            raise RuntimeError(f"Failed to write Parquet to {path}: {str(e)}") from e
