"""Production backup and rollback capabilities.

Handles model and data snapshots for disaster recovery and rollback scenarios.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from ..utils.logging_utils import get_logger
from .model_registry import ModelRegistryManager

logger = get_logger(__name__)


class ProductionBackupManager:
    """Manages production snapshots and rollback capabilities."""

    def __init__(
        self,
        backup_base_path: str = "/dbfs/mnt/backups",
        spark: Optional[SparkSession] = None,
        registry_manager: Optional[ModelRegistryManager] = None,
    ):
        """Initialize production backup manager.

        Args:
            backup_base_path: Base path for storing backups
            spark: SparkSession instance (creates new if None)
            registry_manager: ModelRegistryManager instance
        """
        self.backup_base_path = backup_base_path
        self.spark = spark or SparkSession.builder.appName("ProductionBackupManager").getOrCreate()
        self.registry = registry_manager or ModelRegistryManager()
        self.client = MlflowClient()

        # Ensure backup directory exists
        Path(backup_base_path).mkdir(parents=True, exist_ok=True)

        logger.info("Initialized ProductionBackupManager", backup_path=backup_base_path)

    def create_backup(
        self,
        model_name: str,
        backup_name: Optional[str] = None,
        include_data_snapshot: bool = False,
        data_path: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a backup of production model and optionally data.

        Args:
            model_name: Registered model name
            backup_name: Custom backup name (auto-generated if None)
            include_data_snapshot: Include data snapshot in backup
            data_path: Path to data table (required if include_data_snapshot=True)
            description: Optional backup description

        Returns:
            Backup information

        Raises:
            ValueError: If no production model found
            Exception: If backup creation fails
        """
        try:
            # Generate backup name if not provided
            if not backup_name:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{model_name}_production_backup_{timestamp}"

            logger.info(
                "Creating production backup",
                model_name=model_name,
                backup_name=backup_name,
                include_data=include_data_snapshot,
            )

            # Get production model
            try:
                production_model = self.registry.get_model_version(model_name, stage="Production")
            except Exception as e:
                raise ValueError(f"No production model found for '{model_name}': {str(e)}")

            # Create backup directory
            backup_path = f"{self.backup_base_path}/{backup_name}"
            Path(backup_path).mkdir(parents=True, exist_ok=True)

            backup_info = {
                "backup_name": backup_name,
                "backup_path": backup_path,
                "model_name": model_name,
                "model_version": production_model.version,
                "model_run_id": production_model.run_id,
                "created_at": datetime.utcnow().isoformat(),
                "description": description or f"Production backup of {model_name}",
                "components": {},
            }

            # Backup model artifacts
            logger.info("Backing up model artifacts", run_id=production_model.run_id)
            model_backup_path = self._backup_model_artifacts(
                production_model.run_id,
                f"{backup_path}/model",
            )
            backup_info["components"]["model"] = {
                "path": model_backup_path,
                "run_id": production_model.run_id,
                "version": production_model.version,
            }

            # Backup model metadata
            logger.info("Backing up model metadata")
            metadata = self._get_model_metadata(production_model)
            metadata_path = f"{backup_path}/metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            backup_info["components"]["metadata"] = metadata_path

            # Backup data snapshot if requested
            if include_data_snapshot:
                if not data_path:
                    raise ValueError("data_path required when include_data_snapshot=True")

                logger.info("Creating data snapshot", data_path=data_path)
                data_backup_path = self._backup_data(data_path, f"{backup_path}/data")
                backup_info["components"]["data"] = {
                    "source_path": data_path,
                    "backup_path": data_backup_path,
                }

            # Save backup manifest
            manifest_path = f"{backup_path}/backup_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(backup_info, f, indent=2)

            logger.info(
                "Production backup created successfully",
                backup_name=backup_name,
                backup_path=backup_path,
            )

            return backup_info

        except Exception as e:
            logger.error("Failed to create production backup", model_name=model_name, error=str(e))
            raise

    def restore_backup(
        self,
        backup_name: str,
        restore_model: bool = True,
        restore_data: bool = False,
        target_data_path: Optional[str] = None,
        new_model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Restore from a backup.

        Args:
            backup_name: Name of the backup to restore
            restore_model: Restore the model
            restore_data: Restore the data snapshot
            target_data_path: Target path for data restoration
            new_model_name: Register restored model with new name (uses original if None)

        Returns:
            Restoration information

        Raises:
            ValueError: If backup not found
            Exception: If restoration fails
        """
        try:
            backup_path = f"{self.backup_base_path}/{backup_name}"
            manifest_path = f"{backup_path}/backup_manifest.json"

            logger.info("Restoring from backup", backup_name=backup_name, backup_path=backup_path)

            # Load backup manifest
            if not Path(manifest_path).exists():
                raise ValueError(f"Backup not found: {backup_name}")

            with open(manifest_path, "r") as f:
                backup_info = json.load(f)

            restore_info = {
                "backup_name": backup_name,
                "restored_at": datetime.utcnow().isoformat(),
                "restored_components": {},
            }

            # Restore model if requested
            if restore_model:
                logger.info("Restoring model from backup")

                model_component = backup_info["components"]["model"]
                original_model_name = backup_info["model_name"]
                target_model_name = new_model_name or original_model_name

                # Re-register the model
                model_uri = f"file://{model_component['path']}"
                restored_version = self.registry.register_model(
                    model_uri=model_uri,
                    model_name=target_model_name,
                    description=f"Restored from backup: {backup_name}",
                    tags={
                        "restored_from_backup": backup_name,
                        "original_version": model_component["version"],
                        "restored_at": datetime.utcnow().isoformat(),
                    },
                )

                restore_info["restored_components"]["model"] = {
                    "model_name": target_model_name,
                    "version": restored_version.version,
                    "original_version": model_component["version"],
                }

                logger.info(
                    "Model restored successfully",
                    model_name=target_model_name,
                    version=restored_version.version,
                )

            # Restore data if requested
            if restore_data:
                if "data" not in backup_info["components"]:
                    logger.warning("No data snapshot found in backup")
                else:
                    if not target_data_path:
                        raise ValueError("target_data_path required when restore_data=True")

                    logger.info("Restoring data from backup", target_path=target_data_path)

                    data_component = backup_info["components"]["data"]
                    self._restore_data(data_component["backup_path"], target_data_path)

                    restore_info["restored_components"]["data"] = {
                        "source_backup": data_component["backup_path"],
                        "target_path": target_data_path,
                    }

                    logger.info("Data restored successfully", target_path=target_data_path)

            logger.info("Backup restoration completed", backup_name=backup_name)

            return restore_info

        except Exception as e:
            logger.error("Failed to restore backup", backup_name=backup_name, error=str(e))
            raise

    def list_backups(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backups.

        Args:
            model_name: Filter by model name (lists all if None)

        Returns:
            List of backup information
        """
        try:
            logger.debug("Listing backups", model_name=model_name)

            backups = []
            backup_base = Path(self.backup_base_path)

            if not backup_base.exists():
                logger.warning("Backup directory does not exist", path=self.backup_base_path)
                return []

            # Iterate through backup directories
            for backup_dir in backup_base.iterdir():
                if not backup_dir.is_dir():
                    continue

                manifest_path = backup_dir / "backup_manifest.json"
                if not manifest_path.exists():
                    continue

                try:
                    with open(manifest_path, "r") as f:
                        backup_info = json.load(f)

                    # Apply model name filter
                    if model_name and backup_info.get("model_name") != model_name:
                        continue

                    # Add size information
                    backup_size = sum(
                        f.stat().st_size for f in backup_dir.rglob("*") if f.is_file()
                    )
                    backup_info["size_bytes"] = backup_size
                    backup_info["size_mb"] = round(backup_size / (1024 * 1024), 2)

                    backups.append(backup_info)

                except Exception as e:
                    logger.warning(f"Failed to read backup manifest", path=manifest_path, error=str(e))
                    continue

            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)

            logger.info("Retrieved backups", count=len(backups), model_name=model_name)

            return backups

        except Exception as e:
            logger.error("Failed to list backups", error=str(e))
            raise

    def delete_backup(self, backup_name: str) -> None:
        """Delete a backup.

        Args:
            backup_name: Name of the backup to delete

        Raises:
            ValueError: If backup not found
        """
        try:
            backup_path = f"{self.backup_base_path}/{backup_name}"

            logger.warning("Deleting backup", backup_name=backup_name, backup_path=backup_path)

            if not Path(backup_path).exists():
                raise ValueError(f"Backup not found: {backup_name}")

            # Remove backup directory
            shutil.rmtree(backup_path)

            logger.info("Backup deleted successfully", backup_name=backup_name)

        except Exception as e:
            logger.error("Failed to delete backup", backup_name=backup_name, error=str(e))
            raise

    def rollback_to_backup(
        self,
        backup_name: str,
        promote_to_production: bool = True,
    ) -> Dict[str, Any]:
        """Rollback to a previous backup by restoring and promoting to production.

        Args:
            backup_name: Name of the backup to rollback to
            promote_to_production: Automatically promote to production

        Returns:
            Rollback information
        """
        try:
            logger.warning("Rolling back to backup", backup_name=backup_name)

            # Restore the backup
            restore_info = self.restore_backup(backup_name, restore_model=True, restore_data=False)

            rollback_info = {
                "backup_name": backup_name,
                "rolled_back_at": datetime.utcnow().isoformat(),
                "restore_info": restore_info,
            }

            # Promote to production if requested
            if promote_to_production and "model" in restore_info["restored_components"]:
                model_component = restore_info["restored_components"]["model"]
                model_name = model_component["model_name"]
                version = model_component["version"]

                logger.info("Promoting restored model to production", model_name=model_name, version=version)

                self.registry.transition_stage(
                    model_name,
                    version,
                    "Production",
                    archive_existing=True,
                    description=f"Rollback from backup: {backup_name}",
                )

                rollback_info["promoted_to_production"] = True

            logger.info("Rollback completed successfully", backup_name=backup_name)

            return rollback_info

        except Exception as e:
            logger.error("Failed to rollback to backup", backup_name=backup_name, error=str(e))
            raise

    def _backup_model_artifacts(self, run_id: str, backup_path: str) -> str:
        """Backup model artifacts from MLflow run.

        Args:
            run_id: MLflow run ID
            backup_path: Path to store backup

        Returns:
            Backup path
        """
        try:
            # Download artifacts from MLflow
            artifact_path = self.client.download_artifacts(run_id, "", backup_path)

            logger.debug("Model artifacts backed up", run_id=run_id, backup_path=artifact_path)

            return artifact_path

        except Exception as e:
            logger.error("Failed to backup model artifacts", run_id=run_id, error=str(e))
            raise

    def _get_model_metadata(self, model_version) -> Dict[str, Any]:
        """Get model metadata.

        Args:
            model_version: ModelVersion object

        Returns:
            Metadata dictionary
        """
        run = self.client.get_run(model_version.run_id)

        metadata = {
            "model_name": model_version.name,
            "version": model_version.version,
            "run_id": model_version.run_id,
            "stage": model_version.current_stage,
            "description": model_version.description,
            "tags": model_version.tags,
            "run_name": run.info.run_name,
            "experiment_id": run.info.experiment_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }

        return metadata

    def _backup_data(self, source_path: str, backup_path: str) -> str:
        """Backup data table.

        Args:
            source_path: Source Delta table path
            backup_path: Backup destination path

        Returns:
            Backup path
        """
        try:
            logger.debug("Creating data snapshot", source=source_path, target=backup_path)

            # Read source table
            df = self.spark.read.format("delta").load(source_path)

            # Write to backup location
            df.write.format("delta").mode("overwrite").save(backup_path)

            row_count = df.count()
            logger.info("Data snapshot created", backup_path=backup_path, rows=row_count)

            return backup_path

        except Exception as e:
            logger.error("Failed to backup data", source_path=source_path, error=str(e))
            raise

    def _restore_data(self, backup_path: str, target_path: str) -> None:
        """Restore data from backup.

        Args:
            backup_path: Backup source path
            target_path: Restoration target path
        """
        try:
            logger.debug("Restoring data", source=backup_path, target=target_path)

            # Read backup
            df = self.spark.read.format("delta").load(backup_path)

            # Write to target location
            df.write.format("delta").mode("overwrite").save(target_path)

            row_count = df.count()
            logger.info("Data restored", target_path=target_path, rows=row_count)

        except Exception as e:
            logger.error("Failed to restore data", backup_path=backup_path, error=str(e))
            raise
