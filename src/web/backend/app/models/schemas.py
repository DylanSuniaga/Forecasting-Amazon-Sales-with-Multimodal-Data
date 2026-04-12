"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class RankBand(str, Enum):
    """BSR rank band classification."""
    TOP_1 = "Top 1%"
    TOP_5 = "Top 5%"
    TOP_10 = "Top 10%"
    TOP_25 = "Top 25%"
    TOP_50 = "Top 50%"
    BOTTOM_50 = "Bottom 50%"


class SignalType(str, Enum):
    """Type of signal (strength or risk)."""
    STRENGTH = "strength"
    RISK = "risk"


class Signal(BaseModel):
    """A single signal (strength or risk) from the model."""
    type: SignalType
    label: str
    description: str
    impact: float = Field(..., ge=0, le=1, description="Impact score 0-1")


class ImageQualityMetrics(BaseModel):
    """Image quality metrics extracted from uploaded image."""
    width: int
    height: int
    aspect_ratio: float
    brightness_mean: float
    brightness_std: float
    contrast: float
    saturation_mean: float
    colorfulness: float
    sharpness: float
    blur_score: float
    white_bg_pct: float
    edge_density: float
    
    # Comparison to category median (computed if category provided)
    vs_category_median: Optional[Dict[str, float]] = None


class EvaluationRequest(BaseModel):
    """Request body for product evaluation."""
    title: str = Field(..., min_length=1, max_length=500, description="Product title")
    description: str = Field(default="", max_length=5000, description="Product description/bullets")
    category: str = Field(..., description="Main category")
    subcategory: Optional[str] = Field(default=None, description="Subcategory")
    # Images are uploaded separately via multipart form


class KeywordSuggestion(BaseModel):
    """A keyword suggestion for product text optimization."""
    keyword: str
    prevalence: float = Field(..., ge=0, le=1, description="Fraction of top sellers using this keyword")
    source: str = Field(default="title", description="Where the keyword applies: title or bullets")
    suggestion: str


class NLPFeedback(BaseModel):
    """NLP analysis feedback for product text."""
    title_score: float = Field(..., ge=0, le=100, description="How well title matches top sellers 0-100")
    bullets_score: float = Field(..., ge=0, le=100, description="How well bullets match top sellers 0-100")
    missing_keywords: List[KeywordSuggestion] = Field(default_factory=list, description="Keywords to add")
    weak_keywords: List[KeywordSuggestion] = Field(default_factory=list, description="Keywords to reconsider")
    title_stats: Dict[str, Any] = Field(default_factory=dict)
    bullets_stats: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResponse(BaseModel):
    """Response body for product evaluation."""
    # Primary scores
    launch_viability_score: float = Field(..., ge=0, le=100, description="Overall viability 0-100")
    bsr_entry_probability: float = Field(..., ge=0, le=1, description="Probability of getting BSR rank")
    expected_rank_band: RankBand
    competitive_intensity: float = Field(..., ge=0, le=100, description="Category competition 0-100")

    # Confidence
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")

    # Signal decomposition
    strengths: List[Signal]
    risks: List[Signal]

    # Category context
    category_notes: str
    category_baseline_viability: float
    percentile_in_category: float

    # Image quality summary
    image_quality: ImageQualityMetrics

    # NLP feedback (keyword suggestions and text analysis)
    nlp_feedback: Optional[NLPFeedback] = None

    # Feature importance (top features driving this prediction)
    top_feature_importances: Optional[List[Dict[str, Any]]] = None

    # Feature summary (for transparency)
    feature_summary: Dict[str, Any]

    # Metadata
    product_hash: str
    model_version: str


class WhatIfRequest(BaseModel):
    """Request for what-if analysis (recompute with changes)."""
    original_hash: str
    new_title: Optional[str] = None
    # New image uploaded separately


class WhatIfResponse(BaseModel):
    """Response for what-if analysis showing deltas."""
    original: EvaluationResponse
    modified: EvaluationResponse
    deltas: Dict[str, float]  # Key differences


class CategoryInfo(BaseModel):
    """Information about a category."""
    name: str
    subcategories: List[str]
    sample_count: int
    avg_bsr_rate: float
    model_auc: float
    difficulty_score: float  # Higher = more competitive


class CategoryListResponse(BaseModel):
    """Response with all available categories."""
    categories: List[CategoryInfo]


class ValidationMetrics(BaseModel):
    """Validation metrics for a model."""
    category: str
    model_type: str
    roc_auc: float
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    sample_size: int
    # Regression metrics
    reg_r2: Optional[float] = None
    reg_mae: Optional[float] = None


class ValidationSummaryResponse(BaseModel):
    """Summary of model validation metrics."""
    overall_auc: float
    overall_f1: float
    overall_r2: Optional[float] = None
    overall_mae: Optional[float] = None
    by_category: List[ValidationMetrics]
    calibration_summary: Dict[str, float]


class ModelInfo(BaseModel):
    """Information about a loaded model."""
    name: str
    category: str
    model_type: str
    loaded: bool
    metrics: Optional[Dict[str, float]]


class ModelRegistryResponse(BaseModel):
    """Response listing all models in registry."""
    embedding_model_loaded: bool
    category_models: List[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: int
    gpu_available: bool


# API Payload Examples for documentation
EXAMPLE_EVALUATE_REQUEST = {
    "title": "Premium Stainless Steel Water Bottle 32oz Vacuum Insulated",
    "description": "Keep your drinks cold for 24 hours or hot for 12 hours. BPA-free, leak-proof lid. Perfect for gym, office, or outdoor activities.",
    "category": "Home & Kitchen",
    "subcategory": "Kitchen & Dining"
}

EXAMPLE_EVALUATE_RESPONSE = {
    "launch_viability_score": 72.5,
    "bsr_entry_probability": 0.85,
    "expected_rank_band": "Top 25%",
    "competitive_intensity": 68.0,
    "confidence": 0.82,
    "strengths": [
        {"type": "strength", "label": "Image Quality", "description": "High resolution with clean white background", "impact": 0.85},
        {"type": "strength", "label": "Product Visibility", "description": "Good object occupancy and centering", "impact": 0.72},
    ],
    "risks": [
        {"type": "risk", "label": "High Competition", "description": "Category has many similar products", "impact": 0.65},
    ],
    "category_notes": "Home & Kitchen is highly competitive but image quality matters significantly.",
    "category_baseline_viability": 65.0,
    "percentile_in_category": 72.0,
    "image_quality": {
        "width": 1500,
        "height": 1500,
        "aspect_ratio": 1.0,
        "brightness_mean": 220.5,
        "brightness_std": 45.2,
        "contrast": 45.2,
        "saturation_mean": 85.3,
        "colorfulness": 42.1,
        "sharpness": 850.0,
        "blur_score": 0.001,
        "white_bg_pct": 65.0,
        "edge_density": 0.08,
    },
    "feature_summary": {
        "top_features_used": 20,
        "embedding_dimension": 128,
    },
    "product_hash": "abc123def456",
    "model_version": "1.0.0"
}
