"""Realtime model serving endpoint management.

Handles creation, configuration, and management of Databricks Model Serving endpoints.
"""

import time
from typing import Any, Dict, List, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ServingEndpointManager:
    """Manages Databricks Model Serving endpoints."""

    def __init__(self, workspace_client: Optional[WorkspaceClient] = None):
        """Initialize serving endpoint manager.

        Args:
            workspace_client: Databricks workspace client (creates new if None)
        """
        self.client = workspace_client or WorkspaceClient()
        logger.info("Initialized ServingEndpointManager")

    def create_endpoint(
        self,
        endpoint_name: str,
        model_name: str,
        model_version: str,
        workload_size: str = "Small",
        scale_to_zero: bool = True,
        min_instances: int = 0,
        max_instances: int = 10,
        tags: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Create a new model serving endpoint.

        Args:
            endpoint_name: Name for the endpoint
            model_name: Registered model name
            model_version: Model version to serve
            workload_size: Workload size (Small, Medium, Large)
            scale_to_zero: Enable scale-to-zero
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            tags: Optional tags for the endpoint

        Returns:
            Endpoint configuration

        Raises:
            Exception: If endpoint creation fails
        """
        try:
            logger.info(
                "Creating serving endpoint",
                endpoint_name=endpoint_name,
                model_name=model_name,
                model_version=model_version,
                workload_size=workload_size,
            )

            # Check if endpoint already exists
            try:
                existing = self.client.serving_endpoints.get(endpoint_name)
                logger.warning(
                    "Endpoint already exists",
                    endpoint_name=endpoint_name,
                    state=existing.state.config_update if existing.state else "unknown",
                )
                return self._endpoint_to_dict(existing)
            except Exception as e:
                # Endpoint doesn't exist, proceed with creation
                if "does not exist" not in str(e).lower() and "not found" not in str(e).lower():
                    logger.warning("Unexpected error checking endpoint existence", error=str(e))

            # Prepare served entity configuration
            served_entities = [
                ServedEntityInput(
                    entity_name=model_name,
                    entity_version=model_version,
                    workload_size=workload_size,
                    scale_to_zero_enabled=scale_to_zero,
                    min_provisioned_throughput=min_instances if not scale_to_zero else None,
                    max_provisioned_throughput=max_instances if not scale_to_zero else None,
                )
            ]

            # Create endpoint configuration
            config = EndpointCoreConfigInput(
                name=endpoint_name,
                served_entities=served_entities,
            )

            # Create the endpoint
            endpoint = self.client.serving_endpoints.create(
                name=endpoint_name,
                config=config,
                tags=tags or [],
            )

            # Wait for endpoint to be ready
            self._wait_for_endpoint_ready(endpoint_name, timeout=1800)

            logger.info("Serving endpoint created successfully", endpoint_name=endpoint_name)

            return self._endpoint_to_dict(endpoint)

        except Exception as e:
            logger.error("Failed to create serving endpoint", endpoint_name=endpoint_name, error=str(e))
            raise

    def update_endpoint(
        self,
        endpoint_name: str,
        model_name: str,
        model_version: str,
        workload_size: Optional[str] = None,
        scale_to_zero: Optional[bool] = None,
        min_instances: Optional[int] = None,
        max_instances: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update an existing serving endpoint.

        Args:
            endpoint_name: Name of the endpoint
            model_name: Registered model name
            model_version: New model version
            workload_size: New workload size
            scale_to_zero: Enable/disable scale-to-zero
            min_instances: New minimum instances
            max_instances: New maximum instances

        Returns:
            Updated endpoint configuration

        Raises:
            Exception: If update fails
        """
        try:
            logger.info(
                "Updating serving endpoint",
                endpoint_name=endpoint_name,
                model_name=model_name,
                model_version=model_version,
            )

            # Get current configuration
            current_endpoint = self.client.serving_endpoints.get(endpoint_name)

            # Prepare updated served entity
            served_entities = [
                ServedEntityInput(
                    entity_name=model_name,
                    entity_version=model_version,
                    workload_size=workload_size or "Small",
                    scale_to_zero_enabled=scale_to_zero if scale_to_zero is not None else True,
                    min_provisioned_throughput=min_instances if min_instances is not None else None,
                    max_provisioned_throughput=max_instances if max_instances is not None else None,
                )
            ]

            # Update endpoint configuration
            config = EndpointCoreConfigInput(
                served_entities=served_entities,
            )

            updated_endpoint = self.client.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=config.served_entities,
            )

            # Wait for update to complete
            self._wait_for_endpoint_ready(endpoint_name, timeout=1800)

            logger.info("Serving endpoint updated successfully", endpoint_name=endpoint_name)

            return self._endpoint_to_dict(updated_endpoint)

        except Exception as e:
            logger.error("Failed to update serving endpoint", endpoint_name=endpoint_name, error=str(e))
            raise

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete a serving endpoint.

        Args:
            endpoint_name: Name of the endpoint to delete

        Raises:
            Exception: If deletion fails
        """
        try:
            logger.warning("Deleting serving endpoint", endpoint_name=endpoint_name)

            self.client.serving_endpoints.delete(endpoint_name)

            logger.info("Serving endpoint deleted successfully", endpoint_name=endpoint_name)

        except Exception as e:
            logger.error("Failed to delete serving endpoint", endpoint_name=endpoint_name, error=str(e))
            raise

    def get_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Get serving endpoint details.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            Endpoint configuration

        Raises:
            Exception: If endpoint not found
        """
        try:
            logger.debug("Getting serving endpoint", endpoint_name=endpoint_name)

            endpoint = self.client.serving_endpoints.get(endpoint_name)

            return self._endpoint_to_dict(endpoint)

        except Exception as e:
            logger.error("Failed to get serving endpoint", endpoint_name=endpoint_name, error=str(e))
            raise

    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all serving endpoints.

        Returns:
            List of endpoint configurations
        """
        try:
            logger.debug("Listing serving endpoints")

            endpoints = self.client.serving_endpoints.list()

            endpoint_list = [self._endpoint_to_dict(ep) for ep in endpoints]

            logger.info("Retrieved serving endpoints", count=len(endpoint_list))

            return endpoint_list

        except Exception as e:
            logger.error("Failed to list serving endpoints", error=str(e))
            raise

    def query_endpoint(
        self,
        endpoint_name: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Query a serving endpoint with input data.

        Args:
            endpoint_name: Name of the endpoint
            input_data: Input data for prediction

        Returns:
            Prediction response

        Raises:
            Exception: If query fails
        """
        try:
            logger.debug("Querying serving endpoint", endpoint_name=endpoint_name)

            response = self.client.serving_endpoints.query(
                name=endpoint_name,
                dataframe_records=[input_data] if isinstance(input_data, dict) else input_data,
            )

            logger.info("Endpoint query successful", endpoint_name=endpoint_name)

            return response.as_dict()

        except Exception as e:
            logger.error("Failed to query serving endpoint", endpoint_name=endpoint_name, error=str(e))
            raise

    def check_health(self, endpoint_name: str) -> Dict[str, Any]:
        """Check health status of serving endpoint.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            Health status information
        """
        try:
            logger.debug("Checking endpoint health", endpoint_name=endpoint_name)

            endpoint = self.client.serving_endpoints.get(endpoint_name)

            health_status = {
                "endpoint_name": endpoint_name,
                "state": endpoint.state.config_update if endpoint.state else "unknown",
                "ready": endpoint.state.ready if endpoint.state else "unknown",
            }

            if endpoint.state and endpoint.state.ready == "READY":
                health_status["healthy"] = True
            else:
                health_status["healthy"] = False

            logger.info(
                "Endpoint health checked",
                endpoint_name=endpoint_name,
                healthy=health_status["healthy"],
            )

            return health_status

        except Exception as e:
            logger.error("Failed to check endpoint health", endpoint_name=endpoint_name, error=str(e))
            return {
                "endpoint_name": endpoint_name,
                "healthy": False,
                "error": str(e),
            }

    def configure_autoscaling(
        self,
        endpoint_name: str,
        min_instances: int,
        max_instances: int,
        target_utilization: float = 0.7,
    ) -> Dict[str, Any]:
        """Configure autoscaling for serving endpoint.

        Args:
            endpoint_name: Name of the endpoint
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            target_utilization: Target utilization (0.0 to 1.0)

        Returns:
            Updated endpoint configuration
        """
        try:
            logger.info(
                "Configuring autoscaling",
                endpoint_name=endpoint_name,
                min_instances=min_instances,
                max_instances=max_instances,
                target_utilization=target_utilization,
            )

            # Get current endpoint
            endpoint = self.client.serving_endpoints.get(endpoint_name)

            # Extract current model configuration
            if not endpoint.config or not endpoint.config.served_entities:
                raise ValueError("Endpoint has no served entities")

            current_entity = endpoint.config.served_entities[0]

            # Update with autoscaling configuration
            updated_entities = [
                ServedEntityInput(
                    entity_name=current_entity.entity_name,
                    entity_version=current_entity.entity_version,
                    workload_size=current_entity.workload_size,
                    scale_to_zero_enabled=False,
                    min_provisioned_throughput=min_instances,
                    max_provisioned_throughput=max_instances,
                )
            ]

            updated_endpoint = self.client.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=updated_entities,
            )

            logger.info("Autoscaling configured successfully", endpoint_name=endpoint_name)

            return self._endpoint_to_dict(updated_endpoint)

        except Exception as e:
            logger.error("Failed to configure autoscaling", endpoint_name=endpoint_name, error=str(e))
            raise

    def _wait_for_endpoint_ready(self, endpoint_name: str, timeout: int = 1800) -> None:
        """Wait for endpoint to be ready.

        Args:
            endpoint_name: Name of the endpoint
            timeout: Timeout in seconds

        Raises:
            TimeoutError: If endpoint not ready within timeout
        """
        logger.info("Waiting for endpoint to be ready", endpoint_name=endpoint_name, timeout=timeout)

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                endpoint = self.client.serving_endpoints.get(endpoint_name)

                if endpoint.state and endpoint.state.ready == "READY":
                    logger.info("Endpoint is ready", endpoint_name=endpoint_name)
                    return

                if endpoint.state and endpoint.state.config_update == "UPDATE_FAILED":
                    raise Exception(f"Endpoint update failed: {endpoint_name}")

                time.sleep(30)

            except Exception as e:
                if "does not exist" in str(e).lower():
                    raise
                logger.debug("Endpoint not ready yet", endpoint_name=endpoint_name)
                time.sleep(30)

        raise TimeoutError(f"Endpoint {endpoint_name} not ready within {timeout} seconds")

    def _endpoint_to_dict(self, endpoint: Any) -> Dict[str, Any]:
        """Convert endpoint object to dictionary.

        Args:
            endpoint: Endpoint object

        Returns:
            Dictionary representation
        """
        try:
            return {
                "name": endpoint.name,
                "id": endpoint.id,
                "state": endpoint.state.ready if endpoint.state else "unknown",
                "config_update": endpoint.state.config_update if endpoint.state else "unknown",
                "creation_timestamp": endpoint.creation_timestamp,
                "creator": endpoint.creator,
                "last_updated_timestamp": endpoint.last_updated_timestamp,
            }
        except Exception as e:
            logger.warning("Failed to convert endpoint to dict", error=str(e))
            return {"name": getattr(endpoint, "name", "unknown"), "error": str(e)}
