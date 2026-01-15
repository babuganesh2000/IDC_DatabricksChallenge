"""
Models Module

This module contains all machine learning models for the IDC Databricks Challenge.
Each model inherits from BaseModel and provides specific implementations for
different business use cases.
"""

from .base_model import BaseModel
from .purchase_prediction import PurchasePredictionModel
from .clv_prediction import CLVPredictionModel
from .churn_prediction import ChurnPredictionModel
from .recommendation import RecommendationModel
from .segmentation import SegmentationModel

__all__ = [
    'BaseModel',
    'PurchasePredictionModel',
    'CLVPredictionModel',
    'ChurnPredictionModel',
    'RecommendationModel',
    'SegmentationModel',
]

__version__ = '1.0.0'
