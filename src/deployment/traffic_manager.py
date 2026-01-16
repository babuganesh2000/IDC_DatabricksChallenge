"""Traffic management for blue-green deployments and A/B testing.

Handles gradual traffic shifting and rollback capabilities for model serving.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, TrafficConfig, Route

from ..utils.logging_utils import get_logger
from .realtime_serving import ServingEndpointManager

logger = get_logger(__name__)


class TrafficManager:
    """Manages traffic routing for blue-green deployments and A/B testing."""

    def __init__(
        self,
        serving_manager: Optional[ServingEndpointManager] = None,
        workspace_client: Optional[WorkspaceClient] = None,
    ):
        """Initialize traffic manager.

        Args:
            serving_manager: ServingEndpointManager instance
            workspace_client: Databricks workspace client
        """
        self.client = workspace_client or WorkspaceClient()
        self.serving_manager = serving_manager or ServingEndpointManager(self.client)

        logger.info("Initialized TrafficManager")

    def create_blue_green_deployment(
        self,
        endpoint_name: str,
        blue_model_name: str,
        blue_model_version: str,
        green_model_name: str,
        green_model_version: str,
        initial_green_traffic: int = 0,
        workload_size: str = "Small",
    ) -> Dict[str, Any]:
        """Create blue-green deployment with two model versions.

        Args:
            endpoint_name: Name for the endpoint
            blue_model_name: Blue (current) model name
            blue_model_version: Blue model version
            green_model_name: Green (new) model name
            green_model_version: Green model version
            initial_green_traffic: Initial traffic percentage to green (0-100)
            workload_size: Workload size for instances

        Returns:
            Deployment configuration

        Raises:
            Exception: If deployment creation fails
        """
        try:
            logger.info(
                "Creating blue-green deployment",
                endpoint_name=endpoint_name,
                blue_model=f"{blue_model_name}:{blue_model_version}",
                green_model=f"{green_model_name}:{green_model_version}",
                initial_green_traffic=initial_green_traffic,
            )

            # Validate traffic percentage
            if not 0 <= initial_green_traffic <= 100:
                raise ValueError("initial_green_traffic must be between 0 and 100")

            initial_blue_traffic = 100 - initial_green_traffic

            # Create served entities for both versions
            served_entities = [
                ServedEntityInput(
                    entity_name=blue_model_name,
                    entity_version=blue_model_version,
                    workload_size=workload_size,
                    scale_to_zero_enabled=False,
                    name="blue",
                ),
                ServedEntityInput(
                    entity_name=green_model_name,
                    entity_version=green_model_version,
                    workload_size=workload_size,
                    scale_to_zero_enabled=False,
                    name="green",
                ),
            ]

            # Create traffic configuration
            traffic_config = TrafficConfig(
                routes=[
                    Route(served_model_name="blue", traffic_percentage=initial_blue_traffic),
                    Route(served_model_name="green", traffic_percentage=initial_green_traffic),
                ]
            )

            # Check if endpoint exists
            try:
                existing = self.client.serving_endpoints.get(endpoint_name)
                logger.info("Updating existing endpoint for blue-green deployment")

                # Update endpoint with both versions
                self.client.serving_endpoints.update_config(
                    name=endpoint_name,
                    served_entities=served_entities,
                    traffic_config=traffic_config,
                )

            except Exception:
                logger.info("Creating new endpoint for blue-green deployment")

                # Create new endpoint
                from databricks.sdk.service.serving import EndpointCoreConfigInput

                config = EndpointCoreConfigInput(
                    name=endpoint_name,
                    served_entities=served_entities,
                    traffic_config=traffic_config,
                )

                self.client.serving_endpoints.create(
                    name=endpoint_name,
                    config=config,
                )

            # Wait for endpoint to be ready
            self.serving_manager._wait_for_endpoint_ready(endpoint_name)

            deployment_config = {
                "endpoint_name": endpoint_name,
                "deployment_type": "blue_green",
                "blue_model": f"{blue_model_name}:{blue_model_version}",
                "green_model": f"{green_model_name}:{green_model_version}",
                "blue_traffic": initial_blue_traffic,
                "green_traffic": initial_green_traffic,
                "created_at": datetime.utcnow().isoformat(),
            }

            logger.info("Blue-green deployment created successfully", endpoint_name=endpoint_name)

            return deployment_config

        except Exception as e:
            logger.error("Failed to create blue-green deployment", endpoint_name=endpoint_name, error=str(e))
            raise

    def shift_traffic(
        self,
        endpoint_name: str,
        target_traffic_config: Dict[str, int],
        gradual: bool = True,
        step_percentage: int = 10,
        step_duration_minutes: int = 5,
    ) -> Dict[str, Any]:
        """Shift traffic between model versions.

        Args:
            endpoint_name: Name of the endpoint
            target_traffic_config: Target traffic percentages (e.g., {"blue": 50, "green": 50})
            gradual: Perform gradual traffic shift
            step_percentage: Percentage to shift per step
            step_duration_minutes: Duration between steps

        Returns:
            Traffic shift results

        Raises:
            ValueError: If traffic percentages invalid
        """
        # Validate traffic config
        total_traffic = sum(target_traffic_config.values())
        if total_traffic != 100:
            raise ValueError(f"Traffic percentages must sum to 100, got {total_traffic}")

        try:
            logger.info(
                "Shifting traffic",
                endpoint_name=endpoint_name,
                target_config=target_traffic_config,
                gradual=gradual,
            )

            # Get current endpoint configuration
            endpoint = self.client.serving_endpoints.get(endpoint_name)

            # Get current traffic distribution
            current_traffic = {}
            if endpoint.config and endpoint.config.traffic_config:
                for route in endpoint.config.traffic_config.routes:
                    current_traffic[route.served_model_name] = route.traffic_percentage
            else:
                raise ValueError(f"Endpoint {endpoint_name} does not have traffic configuration")

            logger.debug("Current traffic distribution", traffic=current_traffic)

            if not gradual:
                # Direct traffic shift
                logger.info("Performing direct traffic shift")
                self._apply_traffic_config(endpoint_name, target_traffic_config)

                shift_result = {
                    "endpoint_name": endpoint_name,
                    "shift_type": "direct",
                    "current_traffic": current_traffic,
                    "target_traffic": target_traffic_config,
                    "completed_at": datetime.utcnow().isoformat(),
                }

            else:
                # Gradual traffic shift
                logger.info(
                    "Performing gradual traffic shift",
                    step_percentage=step_percentage,
                    step_duration=step_duration_minutes,
                )

                shift_steps = self._calculate_traffic_shift_steps(
                    current_traffic,
                    target_traffic_config,
                    step_percentage,
                )

                shift_result = {
                    "endpoint_name": endpoint_name,
                    "shift_type": "gradual",
                    "initial_traffic": current_traffic,
                    "target_traffic": target_traffic_config,
                    "steps": [],
                }

                for step_num, step_config in enumerate(shift_steps, 1):
                    logger.info(f"Executing traffic shift step {step_num}/{len(shift_steps)}", traffic=step_config)

                    self._apply_traffic_config(endpoint_name, step_config)

                    step_result = {
                        "step": step_num,
                        "traffic": step_config,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    shift_result["steps"].append(step_result)

                    # Wait before next step (except for last step)
                    if step_num < len(shift_steps):
                        wait_seconds = step_duration_minutes * 60
                        logger.debug(f"Waiting {step_duration_minutes} minutes before next step")
                        # Sleep in smaller increments to allow for interruption
                        for _ in range(wait_seconds // 10):
                            time.sleep(10)

                shift_result["completed_at"] = datetime.utcnow().isoformat()

            logger.info("Traffic shift completed successfully", endpoint_name=endpoint_name)

            return shift_result

        except Exception as e:
            logger.error("Failed to shift traffic", endpoint_name=endpoint_name, error=str(e))
            raise

    def rollback(
        self,
        endpoint_name: str,
        to_version: str = "blue",
        immediate: bool = True,
    ) -> Dict[str, Any]:
        """Rollback traffic to a specific version.

        Args:
            endpoint_name: Name of the endpoint
            to_version: Version name to rollback to (e.g., "blue")
            immediate: Perform immediate rollback (vs gradual)

        Returns:
            Rollback results
        """
        try:
            logger.warning(
                "Rolling back traffic",
                endpoint_name=endpoint_name,
                to_version=to_version,
                immediate=immediate,
            )

            # Get current traffic distribution
            endpoint = self.client.serving_endpoints.get(endpoint_name)
            current_traffic = {}

            if endpoint.config and endpoint.config.traffic_config:
                for route in endpoint.config.traffic_config.routes:
                    current_traffic[route.served_model_name] = route.traffic_percentage

            # Create rollback configuration (send all traffic to specified version)
            rollback_config = {version_name: 0 for version_name in current_traffic.keys()}
            if to_version not in rollback_config:
                raise ValueError(f"Version '{to_version}' not found in endpoint configuration")

            rollback_config[to_version] = 100

            # Perform rollback
            if immediate:
                logger.info("Performing immediate rollback")
                self._apply_traffic_config(endpoint_name, rollback_config)

                rollback_result = {
                    "endpoint_name": endpoint_name,
                    "rollback_type": "immediate",
                    "from_traffic": current_traffic,
                    "to_traffic": rollback_config,
                    "rolled_back_to": to_version,
                    "completed_at": datetime.utcnow().isoformat(),
                }

            else:
                logger.info("Performing gradual rollback")
                shift_result = self.shift_traffic(
                    endpoint_name,
                    rollback_config,
                    gradual=True,
                    step_percentage=20,
                    step_duration_minutes=2,
                )

                rollback_result = {
                    "endpoint_name": endpoint_name,
                    "rollback_type": "gradual",
                    "rolled_back_to": to_version,
                    "shift_details": shift_result,
                    "completed_at": datetime.utcnow().isoformat(),
                }

            logger.info("Rollback completed successfully", endpoint_name=endpoint_name, to_version=to_version)

            return rollback_result

        except Exception as e:
            logger.error("Failed to rollback traffic", endpoint_name=endpoint_name, error=str(e))
            raise

    def setup_ab_test(
        self,
        endpoint_name: str,
        variant_a_model: str,
        variant_a_version: str,
        variant_b_model: str,
        variant_b_version: str,
        variant_a_traffic: int = 50,
        workload_size: str = "Small",
    ) -> Dict[str, Any]:
        """Setup A/B test with two model variants.

        Args:
            endpoint_name: Name for the endpoint
            variant_a_model: Variant A model name
            variant_a_version: Variant A version
            variant_b_model: Variant B model name
            variant_b_version: Variant B version
            variant_a_traffic: Traffic percentage to variant A (0-100)
            workload_size: Workload size

        Returns:
            A/B test configuration
        """
        logger.info(
            "Setting up A/B test",
            endpoint_name=endpoint_name,
            variant_a=f"{variant_a_model}:{variant_a_version}",
            variant_b=f"{variant_b_model}:{variant_b_version}",
            split=f"{variant_a_traffic}/{100 - variant_a_traffic}",
        )

        # Use blue-green deployment with custom names
        return self.create_blue_green_deployment(
            endpoint_name=endpoint_name,
            blue_model_name=variant_a_model,
            blue_model_version=variant_a_version,
            green_model_name=variant_b_model,
            green_model_version=variant_b_version,
            initial_green_traffic=100 - variant_a_traffic,
            workload_size=workload_size,
        )

    def get_traffic_distribution(self, endpoint_name: str) -> Dict[str, Any]:
        """Get current traffic distribution for endpoint.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            Traffic distribution information
        """
        try:
            logger.debug("Getting traffic distribution", endpoint_name=endpoint_name)

            endpoint = self.client.serving_endpoints.get(endpoint_name)

            traffic_info = {
                "endpoint_name": endpoint_name,
                "distribution": {},
                "served_entities": [],
            }

            if endpoint.config:
                # Get traffic configuration
                if endpoint.config.traffic_config:
                    for route in endpoint.config.traffic_config.routes:
                        traffic_info["distribution"][route.served_model_name] = route.traffic_percentage

                # Get served entities
                if endpoint.config.served_entities:
                    for entity in endpoint.config.served_entities:
                        traffic_info["served_entities"].append({
                            "name": entity.name,
                            "entity_name": entity.entity_name,
                            "entity_version": entity.entity_version,
                            "workload_size": entity.workload_size,
                        })

            logger.info("Retrieved traffic distribution", endpoint_name=endpoint_name, distribution=traffic_info["distribution"])

            return traffic_info

        except Exception as e:
            logger.error("Failed to get traffic distribution", endpoint_name=endpoint_name, error=str(e))
            raise

    def _apply_traffic_config(self, endpoint_name: str, traffic_config: Dict[str, int]) -> None:
        """Apply traffic configuration to endpoint.

        Args:
            endpoint_name: Name of the endpoint
            traffic_config: Traffic configuration (version_name -> percentage)
        """
        try:
            routes = [
                Route(served_model_name=version_name, traffic_percentage=percentage)
                for version_name, percentage in traffic_config.items()
            ]

            traffic_config_obj = TrafficConfig(routes=routes)

            self.client.serving_endpoints.update_config(
                name=endpoint_name,
                traffic_config=traffic_config_obj,
            )

            # Wait for update to complete
            self.serving_manager._wait_for_endpoint_ready(endpoint_name, timeout=300)

            logger.debug("Traffic configuration applied", endpoint_name=endpoint_name, config=traffic_config)

        except Exception as e:
            logger.error("Failed to apply traffic config", endpoint_name=endpoint_name, error=str(e))
            raise

    def _calculate_traffic_shift_steps(
        self,
        current_traffic: Dict[str, int],
        target_traffic: Dict[str, int],
        step_percentage: int,
    ) -> List[Dict[str, int]]:
        """Calculate gradual traffic shift steps.

        Args:
            current_traffic: Current traffic distribution
            target_traffic: Target traffic distribution
            step_percentage: Percentage to shift per step

        Returns:
            List of traffic configurations for each step
        """
        steps = []

        # Determine primary transition (largest change)
        traffic_deltas = {
            version: target_traffic[version] - current_traffic.get(version, 0)
            for version in target_traffic.keys()
        }

        # Create intermediate steps
        num_steps = max(1, max(abs(delta) for delta in traffic_deltas.values()) // step_percentage)

        for step in range(1, num_steps + 1):
            step_config = {}
            for version in target_traffic.keys():
                current_val = current_traffic.get(version, 0)
                target_val = target_traffic[version]
                delta = target_val - current_val

                if step == num_steps:
                    # Last step: use exact target
                    step_config[version] = target_val
                else:
                    # Intermediate step: proportional shift
                    step_config[version] = int(current_val + (delta * step / num_steps))

            # Ensure percentages sum to 100
            total = sum(step_config.values())
            if total != 100:
                # Adjust largest value to make sum = 100
                max_version = max(step_config, key=step_config.get)
                step_config[max_version] += (100 - total)

            steps.append(step_config)

        return steps
