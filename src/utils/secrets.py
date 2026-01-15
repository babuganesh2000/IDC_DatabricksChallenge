"""Secret management module.

Handles retrieval of secrets from various sources.
"""

import os
from typing import Optional

from .logging_utils import get_logger

logger = get_logger(__name__)


class SecretManager:
    """Manager for secrets from various sources."""

    def __init__(self, backend: str = "env"):
        """Initialize secret manager.

        Args:
            backend: Secret backend (env, databricks, azure-kv)
        """
        self.backend = backend
        self._databricks_client = None

    def get_secret(self, key: str, scope: Optional[str] = None) -> Optional[str]:
        """Get secret value.

        Args:
            key: Secret key
            scope: Secret scope (for Databricks)

        Returns:
            Secret value or None if not found
        """
        if self.backend == "env":
            return self._get_from_env(key)
        elif self.backend == "databricks":
            return self._get_from_databricks(key, scope)
        elif self.backend == "azure-kv":
            return self._get_from_azure_kv(key)
        else:
            raise ValueError(f"Unsupported secret backend: {self.backend}")

    def _get_from_env(self, key: str) -> Optional[str]:
        """Get secret from environment variable.

        Args:
            key: Environment variable name

        Returns:
            Secret value or None
        """
        value = os.getenv(key)
        if value:
            logger.debug(f"Retrieved secret from environment", key=key)
        return value

    def _get_from_databricks(self, key: str, scope: Optional[str] = None) -> Optional[str]:
        """Get secret from Databricks Secret Scope.

        Args:
            key: Secret key
            scope: Secret scope

        Returns:
            Secret value or None
        """
        try:
            from databricks.sdk import WorkspaceClient

            if self._databricks_client is None:
                self._databricks_client = WorkspaceClient()

            if scope is None:
                raise ValueError("Scope is required for Databricks secrets")

            secret = self._databricks_client.secrets.get_secret(scope=scope, key=key)
            logger.debug(f"Retrieved secret from Databricks", key=key, scope=scope)
            return secret.value

        except Exception as e:
            logger.error(f"Failed to retrieve Databricks secret", key=key, scope=scope, error=str(e))
            return None

    def _get_from_azure_kv(self, key: str) -> Optional[str]:
        """Get secret from Azure Key Vault.

        Args:
            key: Secret name

        Returns:
            Secret value or None
        """
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient

            vault_url = os.getenv("AZURE_KEY_VAULT_URL")
            if not vault_url:
                raise ValueError("AZURE_KEY_VAULT_URL not set")

            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=vault_url, credential=credential)

            secret = client.get_secret(key)
            logger.debug(f"Retrieved secret from Azure Key Vault", key=key)
            return secret.value

        except Exception as e:
            logger.error(f"Failed to retrieve Azure Key Vault secret", key=key, error=str(e))
            return None


# Global secret manager instance
_secret_manager: Optional[SecretManager] = None


def get_secret_manager(backend: str = "env") -> SecretManager:
    """Get or create secret manager instance.

    Args:
        backend: Secret backend

    Returns:
        Secret manager instance
    """
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager(backend)
    return _secret_manager


def get_secret(key: str, scope: Optional[str] = None, backend: str = "env") -> Optional[str]:
    """Get secret value.

    Args:
        key: Secret key
        scope: Secret scope (for Databricks)
        backend: Secret backend

    Returns:
        Secret value or None
    """
    manager = get_secret_manager(backend)
    return manager.get_secret(key, scope)
