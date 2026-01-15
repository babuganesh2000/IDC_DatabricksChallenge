"""Model registry management for MLOps lifecycle.

Handles model registration, versioning, stage transitions, and metadata management.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelRegistryManager:
    """Manages model lifecycle in MLflow Model Registry."""

    VALID_STAGES = ["None", "Staging", "Production", "Archived"]
    STAGE_TRANSITIONS = {
        "None": ["Staging"],
        "Staging": ["Production", "Archived"],
        "Production": ["Archived"],
        "Archived": [],
    }

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize model registry manager.

        Args:
            tracking_uri: MLflow tracking URI (uses default if None)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()
        logger.info("Initialized ModelRegistryManager", tracking_uri=tracking_uri or mlflow.get_tracking_uri())

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        await_registration: bool = True,
    ) -> ModelVersion:
        """Register a model in the model registry.

        Args:
            model_uri: URI to the model (e.g., runs:/<run_id>/model)
            model_name: Name for registered model
            tags: Optional tags for the model version
            description: Optional description
            await_registration: Wait for registration to complete

        Returns:
            ModelVersion object

        Raises:
            Exception: If registration fails
        """
        try:
            logger.info("Registering model", model_name=model_name, model_uri=model_uri)

            # Register the model
            model_version = mlflow.register_model(model_uri, model_name, await_registration_for=300 if await_registration else 0)

            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(model_name, model_version.version, key, value)
                logger.debug("Added tags to model version", model_name=model_name, version=model_version.version, tags=tags)

            # Add description if provided
            if description:
                self.client.update_model_version(
                    name=model_name, version=model_version.version, description=description
                )

            # Add registration metadata
            self.client.set_model_version_tag(
                model_name, model_version.version, "registered_at", datetime.utcnow().isoformat()
            )

            logger.info(
                "Model registered successfully",
                model_name=model_name,
                version=model_version.version,
                stage=model_version.current_stage,
            )

            return model_version

        except Exception as e:
            logger.error("Failed to register model", model_name=model_name, error=str(e))
            raise

    def transition_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
        description: Optional[str] = None,
    ) -> ModelVersion:
        """Transition model version to a new stage.

        Args:
            model_name: Name of registered model
            version: Model version
            stage: Target stage (None, Staging, Production, Archived)
            archive_existing: Archive existing versions in target stage
            description: Optional transition description

        Returns:
            Updated ModelVersion object

        Raises:
            ValueError: If invalid stage transition
            Exception: If transition fails
        """
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}")

        try:
            # Get current stage
            model_version = self.client.get_model_version(model_name, version)
            current_stage = model_version.current_stage

            # Validate transition
            if stage not in self.STAGE_TRANSITIONS.get(current_stage, []):
                if current_stage == stage:
                    logger.warning(
                        "Model already in target stage", model_name=model_name, version=version, stage=stage
                    )
                    return model_version
                else:
                    raise ValueError(
                        f"Invalid transition from '{current_stage}' to '{stage}'. "
                        f"Allowed transitions: {self.STAGE_TRANSITIONS.get(current_stage, [])}"
                    )

            logger.info(
                "Transitioning model stage",
                model_name=model_name,
                version=version,
                from_stage=current_stage,
                to_stage=stage,
            )

            # Archive existing versions in target stage if requested
            if archive_existing and stage in ["Staging", "Production"]:
                self._archive_existing_versions(model_name, stage)

            # Transition to new stage
            updated_version = self.client.transition_model_version_stage(
                name=model_name, version=version, stage=stage, archive_existing_versions=archive_existing
            )

            # Add transition metadata
            transition_info = f"{current_stage}->{stage}"
            self.client.set_model_version_tag(model_name, version, "last_transition", transition_info)
            self.client.set_model_version_tag(model_name, version, "transitioned_at", datetime.utcnow().isoformat())

            if description:
                self.client.update_model_version(name=model_name, version=version, description=description)

            logger.info("Model stage transition completed", model_name=model_name, version=version, new_stage=stage)

            return updated_version

        except Exception as e:
            logger.error("Failed to transition model stage", model_name=model_name, version=version, error=str(e))
            raise

    def archive_model(self, model_name: str, version: str) -> ModelVersion:
        """Archive a model version.

        Args:
            model_name: Name of registered model
            version: Model version

        Returns:
            Archived ModelVersion object
        """
        logger.info("Archiving model", model_name=model_name, version=version)
        return self.transition_stage(model_name, version, "Archived", description="Model archived")

    def list_models(
        self, stage: Optional[str] = None, name_filter: Optional[str] = None, max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """List registered models and their versions.

        Args:
            stage: Filter by stage (None, Staging, Production, Archived)
            name_filter: Filter by model name (substring match)
            max_results: Maximum number of results

        Returns:
            List of model information dictionaries
        """
        if stage and stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}")

        try:
            logger.debug("Listing models", stage=stage, name_filter=name_filter, max_results=max_results)

            # Get all registered models
            registered_models = self.client.search_registered_models(max_results=max_results)

            models_info = []
            for rm in registered_models:
                # Apply name filter
                if name_filter and name_filter.lower() not in rm.name.lower():
                    continue

                # Get latest versions for each stage
                versions_by_stage = {}
                for stage_name in self.VALID_STAGES:
                    versions = self.client.get_latest_versions(rm.name, stages=[stage_name])
                    if versions:
                        versions_by_stage[stage_name] = versions[0]

                # Apply stage filter
                if stage:
                    if stage not in versions_by_stage:
                        continue
                    versions_to_include = {stage: versions_by_stage[stage]}
                else:
                    versions_to_include = versions_by_stage

                model_info = {
                    "name": rm.name,
                    "description": rm.description,
                    "creation_timestamp": rm.creation_timestamp,
                    "last_updated_timestamp": rm.last_updated_timestamp,
                    "tags": rm.tags,
                    "versions": {},
                }

                for stage_name, version in versions_to_include.items():
                    model_info["versions"][stage_name] = {
                        "version": version.version,
                        "run_id": version.run_id,
                        "creation_timestamp": version.creation_timestamp,
                        "last_updated_timestamp": version.last_updated_timestamp,
                        "description": version.description,
                        "tags": version.tags,
                        "status": version.status,
                    }

                models_info.append(model_info)

            logger.info("Retrieved models", count=len(models_info), stage=stage)
            return models_info

        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            raise

    def get_model_version(self, model_name: str, version: Optional[str] = None, stage: Optional[str] = None) -> ModelVersion:
        """Get specific model version.

        Args:
            model_name: Name of registered model
            version: Specific version number (or use stage)
            stage: Stage to get latest version from (if version not specified)

        Returns:
            ModelVersion object

        Raises:
            ValueError: If neither version nor stage specified
        """
        if not version and not stage:
            raise ValueError("Must specify either version or stage")

        try:
            if version:
                logger.debug("Getting model version", model_name=model_name, version=version)
                return self.client.get_model_version(model_name, version)
            else:
                logger.debug("Getting latest model version", model_name=model_name, stage=stage)
                versions = self.client.get_latest_versions(model_name, stages=[stage])
                if not versions:
                    raise ValueError(f"No version found for model '{model_name}' in stage '{stage}'")
                return versions[0]

        except Exception as e:
            logger.error("Failed to get model version", model_name=model_name, error=str(e))
            raise

    def update_model_metadata(
        self, model_name: str, version: str, tags: Optional[Dict[str, str]] = None, description: Optional[str] = None
    ) -> None:
        """Update model version metadata.

        Args:
            model_name: Name of registered model
            version: Model version
            tags: Tags to add/update
            description: New description
        """
        try:
            logger.debug("Updating model metadata", model_name=model_name, version=version)

            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(model_name, version, key, value)

            if description:
                self.client.update_model_version(name=model_name, version=version, description=description)

            logger.info("Model metadata updated", model_name=model_name, version=version)

        except Exception as e:
            logger.error("Failed to update model metadata", model_name=model_name, version=version, error=str(e))
            raise

    def delete_model_version(self, model_name: str, version: str) -> None:
        """Delete a model version.

        Args:
            model_name: Name of registered model
            version: Model version to delete
        """
        try:
            logger.warning("Deleting model version", model_name=model_name, version=version)
            self.client.delete_model_version(model_name, version)
            logger.info("Model version deleted", model_name=model_name, version=version)

        except Exception as e:
            logger.error("Failed to delete model version", model_name=model_name, version=version, error=str(e))
            raise

    def _archive_existing_versions(self, model_name: str, stage: str) -> None:
        """Archive existing versions in a stage.

        Args:
            model_name: Name of registered model
            stage: Stage to archive versions from
        """
        try:
            existing_versions = self.client.get_latest_versions(model_name, stages=[stage])
            for version in existing_versions:
                logger.info(
                    "Archiving existing version",
                    model_name=model_name,
                    version=version.version,
                    from_stage=stage,
                )
                self.client.transition_model_version_stage(
                    name=model_name, version=version.version, stage="Archived"
                )

        except Exception as e:
            logger.warning("Failed to archive existing versions", model_name=model_name, stage=stage, error=str(e))
