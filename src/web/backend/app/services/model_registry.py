"""
Model Registry for loading and managing ML models.

Handles TWO-STAGE prediction:
1. Classification: Predict if product will have BSR (has_main_bsr)
2. Regression: If BSR predicted, estimate the rank (log scale)

Also manages:
- PCA transformers for embedding dimensionality reduction
- Feature configurations for each model type

Models saved per category:
- {category}_clf.pkl - XGBoost classifier (Top Features)
- {category}_reg.pkl - LightGBM/XGBoost regressor (All Features)

Supporting files:
- pca_transformers.pkl - Fitted PCA for CNN and CLIP embeddings
- feature_config.pkl - Feature lists for clf and reg models
"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..core.config import settings


class FeaturePipeline:
    """
    Handles feature transformation pipeline including PCA.
    Must match image_feature_extraction.ipynb exactly.
    """
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.pca_cnn = None
        self.pca_clip = None
        self.cnn_cols = []
        self.clip_cols = []
        self.all_features_clf = []  # Changed from top_features_clf
        self.all_features_reg = []
        self.category_medians = {}  # Category medians for image quality metrics
        self.loaded = False  # Initialize loaded flag
        
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load PCA transformers and feature config."""
        self.loaded = False
        
        # Load PCA transformers
        pca_path = self.models_dir / "pca_transformers.pkl"
        if pca_path.exists():
            try:
                with open(pca_path, 'rb') as f:
                    data = pickle.load(f)
                self.pca_cnn = data.get('pca_cnn')
                self.pca_clip = data.get('pca_clip')
                self.cnn_cols = data.get('cnn_cols', [])
                self.clip_cols = data.get('clip_cols', [])
                print(f"  Loaded PCA transformers: CNN={self.pca_cnn is not None}, CLIP={self.pca_clip is not None}")
            except Exception as e:
                print(f"  ✗ Failed to load PCA transformers: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ⚠️  PCA transformers file not found: {pca_path}")
        
        # Load feature config
        config_path = self.models_dir / "feature_config.pkl"
        if config_path.exists():
            try:
                with open(config_path, 'rb') as f:
                    data = pickle.load(f)
                # Support both old (top_features_clf) and new (all_features_clf) format
                self.all_features_clf = data.get('all_features_clf', data.get('top_features_clf', []))
                self.all_features_reg = data.get('all_features_reg', [])
                self.category_medians = data.get('category_medians', {})
                
                if len(self.all_features_clf) > 0 and len(self.all_features_reg) > 0:
                    self.loaded = True
                    print(f"  ✓ Loaded feature config: CLF features={len(self.all_features_clf)}, REG features={len(self.all_features_reg)}")
                    if self.category_medians:
                        print(f"  ✓ Loaded category medians for {len(self.category_medians)} categories")
                else:
                    print(f"  ⚠️  Feature config loaded but empty: CLF={len(self.all_features_clf)}, REG={len(self.all_features_reg)}")
            except Exception as e:
                print(f"  ✗ Failed to load feature config: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ⚠️  Feature config file not found: {config_path}")
    
    def apply_pca(self, cnn_embedding: Optional[np.ndarray], clip_embedding: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Apply PCA to raw embeddings to get PCA features.
        
        Args:
            cnn_embedding: Raw CNN embedding (1280 dims for EfficientNet-B0)
            clip_embedding: Raw CLIP embedding (512 dims for ViT-B-32)
        
        Returns:
            Dict of PCA features: cnn_pca_0000, cnn_pca_0001, ..., clip_pca_0000, ...
        """
        pca_features = {}
        
        # Apply CNN PCA
        if self.pca_cnn is not None and cnn_embedding is not None:
            try:
                cnn_emb = np.array(cnn_embedding).reshape(1, -1)
                cnn_emb = np.nan_to_num(cnn_emb, nan=0.0, posinf=0.0, neginf=0.0)
                cnn_pca = self.pca_cnn.transform(cnn_emb)[0]
                for j, val in enumerate(cnn_pca):
                    pca_features[f'cnn_pca_{j:04d}'] = float(val)
            except Exception as e:
                print(f"CNN PCA transform error: {e}")
        
        # Apply CLIP PCA
        if self.pca_clip is not None and clip_embedding is not None:
            try:
                clip_emb = np.array(clip_embedding).reshape(1, -1)
                clip_emb = np.nan_to_num(clip_emb, nan=0.0, posinf=0.0, neginf=0.0)
                clip_pca = self.pca_clip.transform(clip_emb)[0]
                for j, val in enumerate(clip_pca):
                    pca_features[f'clip_pca_{j:04d}'] = float(val)
            except Exception as e:
                print(f"CLIP PCA transform error: {e}")
        
        return pca_features
    
    def build_clf_features(self, all_features: Dict[str, float]) -> np.ndarray:
        """Build feature vector for classification model (All features - matches notebooks)."""
        feature_vector = []
        for feat_name in self.all_features_clf:
            feature_vector.append(all_features.get(feat_name, 0.0))
        return np.array(feature_vector, dtype=np.float32)
    
    def build_reg_features(self, all_features: Dict[str, float]) -> np.ndarray:
        """Build feature vector for regression model (All features)."""
        feature_vector = []
        for feat_name in self.all_features_reg:
            feature_vector.append(all_features.get(feat_name, 0.0))
        return np.array(feature_vector, dtype=np.float32)


class CategoryModel:
    """Wrapper for category-specific models (clf + reg)."""
    
    def __init__(self, category: str, models_dir: Path):
        self.category = category
        self.models_dir = models_dir
        
        # Classification model (predicts has_bsr)
        self.clf_model = None
        self.clf_scaler = None
        self.clf_features = None
        self.clf_loaded = False
        
        # Regression model (predicts log_bsr_rank)
        self.reg_model = None
        self.reg_scaler = None
        self.reg_features = None
        self.reg_loaded = False
        
        # Model type names (for display)
        self.clf_model_type = 'XGBoost'
        self.reg_model_type = 'LightGBM'

        # Metrics from training (for display)
        self._clf_metrics = self._get_default_clf_metrics()
        self._reg_metrics = self._get_default_reg_metrics()
        
        # Try to load models
        self._load_models()
    
    def _get_category_key(self) -> str:
        """Convert category name to filename key."""
        return self.category.lower().replace(" ", "_").replace("&", "and").replace(",", "")
    
    def _get_default_clf_metrics(self) -> Dict[str, float]:
        """Default classification metrics from notebook experiments."""
        metrics_by_category = {
            "Home & Kitchen": {"roc_auc": 0.957, "f1": 0.907, "precision": 0.880, "recall": 0.936},
            "Health & Household": {"roc_auc": 0.968, "f1": 0.935, "precision": 0.921, "recall": 0.950},
            "Office Products": {"roc_auc": 0.982, "f1": 0.955, "precision": 0.943, "recall": 0.966},
            "Baby": {"roc_auc": 0.957, "f1": 0.894, "precision": 0.868, "recall": 0.921},
            "Clothing, Shoes & Jewelry": {"roc_auc": 0.868, "f1": 0.825, "precision": 0.775, "recall": 0.883},
            "Kitchen & Dining": {"roc_auc": 0.965, "f1": 0.919, "precision": 0.903, "recall": 0.936},
            "Electronics": {"roc_auc": 0.987, "f1": 0.952, "precision": 0.930, "recall": 0.976},
            "Cell Phones & Accessories": {"roc_auc": 0.988, "f1": 0.967, "precision": 0.958, "recall": 0.975},
            "Tools & Home Improvement": {"roc_auc": 0.963, "f1": 0.878, "precision": 0.882, "recall": 0.874},
            "Video Games": {"roc_auc": 0.980, "f1": 0.953, "precision": 0.944, "recall": 0.962},
            "Pet Supplies": {"roc_auc": 0.875, "f1": 0.782, "precision": 0.760, "recall": 0.806},
            "Sports & Outdoors": {"roc_auc": 0.913, "f1": 0.859, "precision": 0.844, "recall": 0.874},
            "Industrial & Scientific": {"roc_auc": 0.941, "f1": 0.861, "precision": 0.861, "recall": 0.861},
            "Musical Instruments": {"roc_auc": 0.986, "f1": 0.953, "precision": 0.938, "recall": 0.968},
        }
        return metrics_by_category.get(self.category, {"roc_auc": 0.85, "f1": 0.80, "precision": 0.78, "recall": 0.82})
    
    def _get_default_reg_metrics(self) -> Dict[str, float]:
        """Default regression metrics from notebook experiments."""
        metrics_by_category = {
            "Home & Kitchen": {"r2": 0.158, "mae": 1.54},
            "Health & Household": {"r2": 0.268, "mae": 1.32},
            "Office Products": {"r2": 0.121, "mae": 1.17},
            "Baby": {"r2": 0.101, "mae": 1.23},
            "Clothing, Shoes & Jewelry": {"r2": 0.283, "mae": 1.36},
            "Kitchen & Dining": {"r2": 0.185, "mae": 1.43},
            "Electronics": {"r2": 0.175, "mae": 1.26},
            "Cell Phones & Accessories": {"r2": 0.079, "mae": 1.13},
            "Tools & Home Improvement": {"r2": 0.040, "mae": 1.51},
            "Video Games": {"r2": 0.107, "mae": 1.16},
            "Pet Supplies": {"r2": 0.065, "mae": 1.00},
            "Sports & Outdoors": {"r2": 0.315, "mae": 1.54},
            "Industrial & Scientific": {"r2": 0.166, "mae": 1.58},
            "Musical Instruments": {"r2": 0.020, "mae": 1.73},
        }
        return metrics_by_category.get(self.category, {"r2": 0.10, "mae": 1.50})
    
    def _load_models(self):
        """Load classification and regression models from disk."""
        cat_key = self._get_category_key()
        
        # Load classification model
        clf_path = self.models_dir / f"{cat_key}_clf.pkl"
        print(f"  Checking for CLF model at: {clf_path}")
        if clf_path.exists():
            try:
                with open(clf_path, 'rb') as f:
                    data = pickle.load(f)
                self.clf_model = data.get('model')
                self.clf_scaler = data.get('scaler')
                self.clf_features = data.get('features', [])
                self._clf_metrics = data.get('metrics', self._clf_metrics)
                self.clf_model_type = data.get('model_type', 'XGBoost')

                if self.clf_model is not None:
                    self.clf_loaded = True
                    print(f"  ✓ Loaded CLF model for {self.category} ({self.clf_model_type}, {len(self.clf_features)} features)")
                else:
                    print(f"  ✗ CLF model is None in pickle file for {self.category}")
            except Exception as e:
                print(f"  ✗ Failed to load CLF model for {self.category}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ✗ CLF model not found at {clf_path}")
            # List available models for debugging
            available = list(self.models_dir.glob("*_clf.pkl"))
            if available:
                print(f"    Available CLF models: {[f.name for f in available[:5]]}")
        
        # Load regression model
        reg_path = self.models_dir / f"{cat_key}_reg.pkl"
        print(f"  Checking for REG model at: {reg_path}")
        if reg_path.exists():
            try:
                with open(reg_path, 'rb') as f:
                    data = pickle.load(f)
                self.reg_model = data.get('model')
                self.reg_scaler = data.get('scaler')
                self.reg_features = data.get('features', [])
                self._reg_metrics = data.get('metrics', self._reg_metrics)
                self.reg_model_type = data.get('model_type', 'LightGBM')

                if self.reg_model is not None:
                    self.reg_loaded = True
                    print(f"  ✓ Loaded REG model for {self.category} ({self.reg_model_type}, {len(self.reg_features)} features)")
                else:
                    print(f"  ✗ REG model is None in pickle file for {self.category}")
            except Exception as e:
                print(f"  ✗ Failed to load REG model for {self.category}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ✗ REG model not found at {reg_path}")
            # List available models for debugging
            available = list(self.models_dir.glob("*_reg.pkl"))
            if available:
                print(f"    Available REG models: {[f.name for f in available[:5]]}")
    
    def predict_has_bsr(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict if product will have BSR rank (classification).
        Returns probability and predicted class.
        
        Args:
            features: Feature vector - must match the exact features the model was trained with
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if self.clf_loaded and self.clf_model is not None:
            try:
                # Verify feature count matches
                expected_features = len(self.clf_features) if self.clf_features else None
                actual_features = features.shape[1]
                
                if expected_features and actual_features != expected_features:
                    print(f"CLF feature mismatch: model expects {expected_features}, got {actual_features}")
                    # Try to fix by using only the features the model was trained with
                    # This shouldn't happen if feature pipeline is working correctly
                    raise ValueError(f"Feature count mismatch: expected {expected_features}, got {actual_features}")
                
                if self.clf_scaler is not None:
                    features = self.clf_scaler.transform(features)
                
                probability = self.clf_model.predict_proba(features)[0, 1]
                prediction = int(probability > 0.5)
                
                return {
                    'probability': float(probability),
                    'prediction': prediction,
                    'confidence': abs(probability - 0.5) * 2,
                    'is_placeholder': False
                }
            except Exception as e:
                print(f"CLF prediction error: {e}")
                import traceback
                traceback.print_exc()
        
        # Placeholder prediction
        # Use roc_auc if available, otherwise use default
        base_metric = self._clf_metrics.get('roc_auc', 0.75)
        base_prob = base_metric * 0.7 + 0.15  # Scale ROC-AUC to probability range
        noise = np.random.normal(0, 0.08)
        probability = np.clip(base_prob + noise, 0.1, 0.95)
        
        return {
            'probability': float(probability),
            'prediction': int(probability > 0.5),
            'confidence': abs(probability - 0.5) * 2,
            'is_placeholder': True
        }
    
    def predict_bsr_rank(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict BSR rank (regression). Returns log-scale prediction.
        Only call this if predict_has_bsr returned prediction=1.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if self.reg_loaded and self.reg_model is not None:
            try:
                if self.reg_scaler is not None:
                    features = self.reg_scaler.transform(features)
                
                log_rank = self.reg_model.predict(features)[0]
                rank = np.expm1(log_rank)  # Convert back from log scale
                
                return {
                    'log_rank': float(log_rank),
                    'estimated_rank': float(rank),
                    'is_placeholder': False
                }
            except Exception as e:
                print(f"REG prediction error: {e}")
        
        # Placeholder prediction
        log_rank = np.random.uniform(8, 12)  # ~3000 to ~160000
        rank = np.expm1(log_rank)
        
        return {
            'log_rank': float(log_rank),
            'estimated_rank': float(rank),
            'is_placeholder': True
        }
    
    @property
    def clf_metrics(self) -> Dict[str, float]:
        return self._clf_metrics
    
    @property
    def reg_metrics(self) -> Dict[str, float]:
        return self._reg_metrics
    
    @property
    def is_loaded(self) -> bool:
        """Check if at least classification model is loaded."""
        return self.clf_loaded


class ModelRegistry:
    """Registry for managing all ML models and feature pipeline."""
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or settings.MODELS_DIR
        self._category_models: Dict[str, CategoryModel] = {}
        self._embedding_model_loaded = False
        self._feature_pipeline: Optional[FeaturePipeline] = None
        
        print(f"ModelRegistry initialized. Models dir: {self.models_dir}")
        print(f"  Models dir exists: {self.models_dir.exists()}")
        if self.models_dir.exists():
            model_files = list(self.models_dir.glob("*_clf.pkl"))
            print(f"  Found {len(model_files)} CLF model files")
            if len(model_files) > 0:
                print(f"  Example: {model_files[0].name}")
        self._load_feature_pipeline()
        self._scan_models()
    
    def _load_feature_pipeline(self):
        """Load the feature transformation pipeline (PCA, feature lists)."""
        if self.models_dir.exists():
            self._feature_pipeline = FeaturePipeline(self.models_dir)
    
    def _scan_models(self):
        """Scan and pre-load all models."""
        if not self.models_dir.exists():
            print(f"Models directory does not exist: {self.models_dir}")
            return
        
        # Pre-load models for all categories
        for category in settings.CATEGORIES:
            self._category_models[category] = CategoryModel(category, self.models_dir)
    
    @property
    def feature_pipeline(self) -> Optional[FeaturePipeline]:
        """Get the feature transformation pipeline."""
        return self._feature_pipeline
    
    def get_category_model(self, category: str) -> CategoryModel:
        """Get model for a specific category."""
        if category not in self._category_models:
            self._category_models[category] = CategoryModel(category, self.models_dir)
        return self._category_models[category]
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models and their status."""
        models = []
        
        for category in settings.CATEGORIES:
            model = self.get_category_model(category)
            models.append({
                'name': f"{category}",
                'category': category,
                'clf_loaded': model.clf_loaded,
                'reg_loaded': model.reg_loaded,
                'clf_metrics': model.clf_metrics,
                'reg_metrics': model.reg_metrics,
                'clf_model_type': model.clf_model_type,
                'reg_model_type': model.reg_model_type,
            })
        
        return models
    
    @property
    def embedding_model_loaded(self) -> bool:
        return self._embedding_model_loaded
    
    def mark_embedding_loaded(self):
        self._embedding_model_loaded = True


# Singleton instance
_registry_instance = None

def get_model_registry() -> ModelRegistry:
    """Get or create the singleton model registry."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
