"""
Monitoring module for ML model observability and performance tracking.

This module provides comprehensive monitoring capabilities including:
- Data drift detection
- Model performance tracking
- Prediction drift monitoring
- Model staleness checking
- Multi-channel alerting
- Dashboard management
- Fairness and bias validation
- A/B testing framework
"""

from .data_drift import DataDriftDetector
from .model_performance import ModelPerformanceMonitor
from .prediction_drift import PredictionDriftMonitor
from .staleness_check import ModelnessChecker
from .alerting import AlertManager
from .dashboards import DashboardManager
from .fairness_check import FairnessChecker
from .ab_testing import ABTestManager

__all__ = [
    "DataDriftDetector",
    "ModelPerformanceMonitor",
    "PredictionDriftMonitor",
    "ModelnessChecker",
    "AlertManager",
    "DashboardManager",
    "FairnessChecker",
    "ABTestManager",
]

__version__ = "1.0.0"
