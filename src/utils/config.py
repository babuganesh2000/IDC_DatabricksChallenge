"""Configuration management module.

Handles loading and validating environment-specific configurations.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, validator


class DatabaseConfig(BaseModel):
    """Database configuration."""

    host: str
    catalog: str = "main"
    schema: str = "default"


class MLflowConfig(BaseModel):
    """MLflow configuration."""

    tracking_uri: str
    experiment_name: str
    registry_uri: Optional[str] = None


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str
    version: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics_threshold: Dict[str, float] = Field(default_factory=dict)


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    enabled: bool = True
    drift_threshold: float = 0.1
    performance_threshold: float = 0.85
    check_interval_hours: int = 1


class Config(BaseModel):
    """Main configuration class."""

    environment: str
    database: DatabaseConfig
    mlflow: MLflowConfig
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment value."""
        valid_envs = ["dev", "staging", "prod"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v


class ConfigManager:
    """Configuration manager for loading environment-specific configs."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "config"
        self.config_dir = config_dir
        self._config: Optional[Config] = None

    def load_config(self, environment: Optional[str] = None) -> Config:
        """Load configuration for specified environment.

        Args:
            environment: Environment name (dev, staging, prod)

        Returns:
            Configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "dev")

        config_file = self.config_dir / f"{environment}.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        self._config = Config(**config_data)
        return self._config

    @property
    def config(self) -> Config:
        """Get current configuration.

        Returns:
            Configuration object

        Raises:
            RuntimeError: If config hasn't been loaded
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model configuration

        Raises:
            KeyError: If model config doesn't exist
        """
        if model_name not in self.config.models:
            raise KeyError(f"Model configuration not found: {model_name}")
        return self.config.models[model_name]


# Global config manager instance
config_manager = ConfigManager()


def get_config(environment: Optional[str] = None) -> Config:
    """Get configuration for specified environment.

    Args:
        environment: Environment name

    Returns:
        Configuration object
    """
    return config_manager.load_config(environment)
