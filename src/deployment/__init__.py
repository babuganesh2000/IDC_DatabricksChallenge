"""Deployment module for MLOps pipelines.

Provides model registry, batch scoring, realtime serving, and deployment automation.
"""

from .backup_production import ProductionBackupManager
from .batch_scoring import BatchScoringPipeline
from .model_promoter import ModelPromoter
from .model_registry import ModelRegistryManager
from .realtime_serving import ServingEndpointManager
from .traffic_manager import TrafficManager

__all__ = [
    "ModelRegistryManager",
    "BatchScoringPipeline",
    "ServingEndpointManager",
    "ModelPromoter",
    "TrafficManager",
    "ProductionBackupManager",
]
