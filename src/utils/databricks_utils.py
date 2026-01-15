"""Databricks utilities for REST API interactions.

Provides wrappers for Databricks REST API operations.
"""

import os
from typing import Any, Dict, List, Optional

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import JobSettings

from .logging_utils import get_logger

logger = get_logger(__name__)


class DatabricksClient:
    """Client for Databricks REST API operations."""

    def __init__(self, host: Optional[str] = None, token: Optional[str] = None):
        """Initialize Databricks client.

        Args:
            host: Databricks workspace URL
            token: Databricks access token
        """
        self.host = host or os.getenv("DATABRICKS_HOST")
        self.token = token or os.getenv("DATABRICKS_TOKEN")

        if not self.host or not self.token:
            raise ValueError("Databricks host and token must be provided")

        self.host = self.host.rstrip("/")
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

        # Initialize SDK client
        self.client = WorkspaceClient(host=self.host, token=self.token)

    def _make_request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to Databricks API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.host}/api/2.0/{endpoint}"
        logger.debug(f"Making {method} request to {url}")

        response = requests.request(method=method, url=url, headers=self.headers, json=data, params=params)
        response.raise_for_status()

        return response.json() if response.content else {}

    def run_job(self, job_id: str, parameters: Optional[Dict[str, str]] = None) -> str:
        """Trigger a Databricks job run.

        Args:
            job_id: Job ID
            parameters: Job parameters

        Returns:
            Run ID
        """
        data = {"job_id": job_id}
        if parameters:
            data["notebook_params"] = parameters

        response = self._make_request("POST", "jobs/run-now", data=data)
        run_id = response.get("run_id")

        logger.info(f"Started job run", job_id=job_id, run_id=run_id)
        return str(run_id)

    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get status of a job run.

        Args:
            run_id: Run ID

        Returns:
            Run status information
        """
        response = self._make_request("GET", "jobs/runs/get", params={"run_id": run_id})
        return response

    def wait_for_run(self, run_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Wait for job run to complete.

        Args:
            run_id: Run ID
            timeout: Timeout in seconds

        Returns:
            Final run status
        """
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_run_status(run_id)
            state = status.get("state", {})
            life_cycle_state = state.get("life_cycle_state")

            if life_cycle_state in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
                result_state = state.get("result_state")
                logger.info(f"Job run completed", run_id=run_id, result_state=result_state)
                return status

            time.sleep(30)

        raise TimeoutError(f"Job run {run_id} did not complete within {timeout} seconds")

    def list_clusters(self) -> List[Dict[str, Any]]:
        """List all clusters in workspace.

        Returns:
            List of cluster information
        """
        response = self._make_request("GET", "clusters/list")
        return response.get("clusters", [])

    def get_cluster_status(self, cluster_id: str) -> Dict[str, Any]:
        """Get cluster status.

        Args:
            cluster_id: Cluster ID

        Returns:
            Cluster status information
        """
        response = self._make_request("GET", "clusters/get", params={"cluster_id": cluster_id})
        return response

    def create_serving_endpoint(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update model serving endpoint.

        Args:
            config: Endpoint configuration

        Returns:
            Endpoint information
        """
        endpoint_name = config.get("name")
        logger.info(f"Creating serving endpoint", endpoint_name=endpoint_name)

        response = self._make_request("POST", "serving-endpoints", data=config)
        return response

    def get_serving_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Get serving endpoint information.

        Args:
            endpoint_name: Endpoint name

        Returns:
            Endpoint information
        """
        response = self._make_request("GET", f"serving-endpoints/{endpoint_name}")
        return response

    def update_serving_endpoint(self, endpoint_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update serving endpoint configuration.

        Args:
            endpoint_name: Endpoint name
            config: New configuration

        Returns:
            Updated endpoint information
        """
        logger.info(f"Updating serving endpoint", endpoint_name=endpoint_name)
        response = self._make_request("PUT", f"serving-endpoints/{endpoint_name}/config", data=config)
        return response


def get_databricks_client(host: Optional[str] = None, token: Optional[str] = None) -> DatabricksClient:
    """Get Databricks client instance.

    Args:
        host: Databricks workspace URL
        token: Databricks access token

    Returns:
        Databricks client
    """
    return DatabricksClient(host, token)
