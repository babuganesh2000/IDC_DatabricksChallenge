"""Pydantic schemas for data validation."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, validator


class EcommerceEvent(BaseModel):
    """Schema for e-commerce event data."""

    event_time: datetime
    event_type: str = Field(..., description="Event type: view, cart, purchase")
    product_id: int
    category_id: Optional[int] = None
    category_code: Optional[str] = None
    brand: Optional[str] = None
    price: float = Field(..., gt=0)
    user_id: int
    user_session: str

    @validator("event_type")
    def validate_event_type(cls, v):
        """Validate event type."""
        valid_types = ["view", "cart", "purchase", "remove_from_cart"]
        if v not in valid_types:
            raise ValueError(f"event_type must be one of {valid_types}")
        return v

    @validator("price")
    def validate_price(cls, v):
        """Validate price is positive."""
        if v <= 0:
            raise ValueError("price must be positive")
        return v


class CustomerFeatures(BaseModel):
    """Schema for customer features."""

    user_id: int
    recency_days: float = Field(..., ge=0)
    frequency: int = Field(..., ge=0)
    monetary_value: float = Field(..., ge=0)
    avg_order_value: float = Field(..., ge=0)
    total_sessions: int = Field(..., ge=0)
    total_products_viewed: int = Field(..., ge=0)
    cart_abandonment_rate: float = Field(..., ge=0, le=1)
    favorite_category: Optional[str] = None
    favorite_brand: Optional[str] = None


class ProductFeatures(BaseModel):
    """Schema for product features."""

    product_id: int
    total_views: int = Field(..., ge=0)
    total_carts: int = Field(..., ge=0)
    total_purchases: int = Field(..., ge=0)
    conversion_rate: float = Field(..., ge=0, le=1)
    avg_price: float = Field(..., ge=0)
    category_popularity: float = Field(..., ge=0)
    brand_popularity: Optional[float] = None


class ModelPrediction(BaseModel):
    """Schema for model predictions."""

    user_id: int
    model_name: str
    prediction: float
    prediction_timestamp: datetime
    model_version: str
    confidence: Optional[float] = Field(None, ge=0, le=1)


class DataQualityReport(BaseModel):
    """Schema for data quality reports."""

    timestamp: datetime
    total_records: int
    null_counts: dict
    duplicate_count: int
    schema_violations: int
    quality_score: float = Field(..., ge=0, le=1)
    passed: bool
