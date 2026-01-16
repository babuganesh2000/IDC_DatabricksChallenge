"""Feature engineering module.

This module provides feature engineering capabilities for customer analytics,
including RFM analysis, behavioral features, product features, and Databricks
Feature Store integration.
"""

from src.features.behavioral_features import BehavioralFeatureEngineer
from src.features.customer_features import CustomerFeatureEngineer
from src.features.feature_store import FeatureStoreManager
from src.features.product_features import ProductFeatureEngineer

__all__ = [
    "CustomerFeatureEngineer",
    "ProductFeatureEngineer",
    "BehavioralFeatureEngineer",
    "FeatureStoreManager",
]
