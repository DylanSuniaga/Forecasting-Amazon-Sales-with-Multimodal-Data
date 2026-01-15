"""
Application configuration settings.
"""
from pathlib import Path
from typing import List

# Handle pydantic v1 vs v2 compatibility
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # type: ignore


class Settings(BaseSettings):
    """Application settings loaded from environment or defaults."""
    
    # API Settings
    API_TITLE: str = "Amazon Product Launch Viability API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = True
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "src" / "web" / "backend" / "models"
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Model Settings
    CNN_BACKBONE: str = "efficientnet_b0"
    CLIP_MODEL: str = "ViT-B-32"
    DEVICE: str = "cpu"  # Change to "cuda" if GPU available
    
    # Categories available (from notebook analysis)
    CATEGORIES: List[str] = [
        "Home & Kitchen",
        "Health & Household", 
        "Office Products",
        "Baby",
        "Clothing, Shoes & Jewelry",
        "Kitchen & Dining",
        "Electronics",
        "Cell Phones & Accessories",
        "Tools & Home Improvement",
        "Video Games",
        "Pet Supplies",
        "Sports & Outdoors",
        "Industrial & Scientific",
        "Musical Instruments",
    ]
    
    # Subcategories mapping (simplified - expand as needed)
    SUBCATEGORIES: dict = {
        "Home & Kitchen": ["Kitchen & Dining", "Bedding", "Bath", "Furniture", "Home Decor"],
        "Health & Household": ["Vitamins", "Personal Care", "Household Supplies", "Medical Supplies"],
        "Office Products": ["Office Electronics", "Office Furniture", "Writing Supplies", "Filing"],
        "Baby": ["Feeding", "Diapering", "Nursery", "Baby Care", "Strollers"],
        "Clothing, Shoes & Jewelry": ["Women", "Men", "Kids", "Shoes", "Jewelry"],
        "Kitchen & Dining": ["Cookware", "Dinnerware", "Kitchen Utensils", "Storage"],
        "Electronics": ["Computers", "TV & Video", "Audio", "Cameras", "Wearables"],
        "Cell Phones & Accessories": ["Cell Phones", "Cases", "Chargers", "Screen Protectors"],
        "Tools & Home Improvement": ["Power Tools", "Hand Tools", "Hardware", "Electrical"],
        "Video Games": ["PlayStation", "Xbox", "Nintendo", "PC Gaming", "Accessories"],
        "Pet Supplies": ["Dogs", "Cats", "Fish", "Birds", "Small Animals"],
        "Sports & Outdoors": ["Exercise", "Outdoor Recreation", "Team Sports", "Camping"],
        "Industrial & Scientific": ["Lab Equipment", "Safety", "Industrial Hardware", "Measuring"],
        "Musical Instruments": ["Guitars", "Keyboards", "Drums", "Recording", "Accessories"],
    }
    
    # Top features from EDA (used for model training)
    TOP_FEATURES: List[str] = [
        "cnn_pca_0003",
        "clip_pca_0005",
        "clip_pca_0002",
        "clip_pca_0000",
        "cnn_pca_0000",
        "clip_pca_0003",
        "clip_pca_0001",
        "det_conf_mean_mean",
        "det_conf_max_mean",
        "det_has_person_mean",
        "brightness_std_mean",
        "contrast_mean",
        "aspect_ratio_mean",
        "det_main_box_center_dist_mean",
        "cnn_pca_0009",
        "cnn_pca_0001",
        "clip_pca_0009",
        "cnn_pca_0002",
        "exposure_clipped_high_pct_mean",
        "border_clutter_score_mean",
    ]
    
    class Config:
        env_file = ".env"


settings = Settings()

# Ensure directories exist
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
