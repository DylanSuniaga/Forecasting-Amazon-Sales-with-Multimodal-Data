"""
Inference service for product evaluation.

TWO-STAGE PREDICTION:
1. Classification: Does product have BSR? (has_main_bsr)
2. Regression: If yes, what's the expected rank? (log_main_bsr_rank)

This matches the notebook approach where we first predict if a product
will get a BSR rank, then estimate the rank if positive.
"""
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ..core.config import settings
from ..models.schemas import (
    EvaluationResponse, Signal, SignalType, RankBand,
    ImageQualityMetrics
)
from .image_features import get_feature_extractor
from .model_registry import get_model_registry


class InferenceService:
    """Main inference service for product evaluation."""
    
    def __init__(self):
        self.feature_extractor = get_feature_extractor(device=settings.DEVICE)
        self.model_registry = get_model_registry()
        
        # Category baseline statistics
        self._category_baselines = {
            "Home & Kitchen": {"bsr_rate": 0.70, "difficulty": 72, "image_sensitive": True},
            "Health & Household": {"bsr_rate": 0.70, "difficulty": 68, "image_sensitive": True},
            "Office Products": {"bsr_rate": 0.70, "difficulty": 58, "image_sensitive": False},
            "Baby": {"bsr_rate": 0.70, "difficulty": 65, "image_sensitive": True},
            "Clothing, Shoes & Jewelry": {"bsr_rate": 0.70, "difficulty": 78, "image_sensitive": True},
            "Kitchen & Dining": {"bsr_rate": 0.70, "difficulty": 70, "image_sensitive": True},
            "Electronics": {"bsr_rate": 0.70, "difficulty": 75, "image_sensitive": False},
            "Cell Phones & Accessories": {"bsr_rate": 0.70, "difficulty": 65, "image_sensitive": False},
            "Tools & Home Improvement": {"bsr_rate": 0.70, "difficulty": 55, "image_sensitive": False},
            "Video Games": {"bsr_rate": 0.70, "difficulty": 60, "image_sensitive": False},
            "Pet Supplies": {"bsr_rate": 0.70, "difficulty": 62, "image_sensitive": True},
            "Sports & Outdoors": {"bsr_rate": 0.70, "difficulty": 68, "image_sensitive": True},
            "Industrial & Scientific": {"bsr_rate": 0.70, "difficulty": 45, "image_sensitive": False},
            "Musical Instruments": {"bsr_rate": 0.70, "difficulty": 50, "image_sensitive": False},
        }
    
    def evaluate_product(
        self,
        title: str,
        description: str,
        category: str,
        subcategory: Optional[str],
        image_paths: List[str],
        precomputed_cnn_embedding: Optional[List[float]] = None,
        precomputed_clip_embedding: Optional[List[float]] = None,
        precomputed_cnn_pca: Optional[Dict[str, float]] = None,
        precomputed_clip_pca: Optional[Dict[str, float]] = None
    ) -> EvaluationResponse:
        """
        Evaluate a product's launch viability using two-stage prediction.
        
        Stage 1: Classification - Will this product get a BSR rank?
        Stage 2: Regression - If yes, what rank band?
        
        Feature Pipeline:
        1. Extract image features (quality, composition, raw embeddings)
        2. Apply PCA to raw embeddings
        3. Build feature vectors (top features for clf, all features for reg)
        4. Run model predictions
        """
        # Generate product hash
        product_hash = self._compute_product_hash(title, description, image_paths)
        
        # Extract image features (single image) OR use pre-computed features
        if precomputed_cnn_pca is not None and precomputed_clip_pca is not None:
            # Use pre-computed PCA features directly (from suggestions with existing dataset)
            primary_features = {
                # Add placeholder quality/composition features (not critical for suggestions)
                'width': 1000,
                'height': 1000,
                'aspect_ratio': 1.0,
                'brightness_mean': 128,
                'brightness_std': 50,
                'contrast': 50,
                'saturation_mean': 100,
                'colorfulness': 40,
                'sharpness': 500,
                'blur_score': 0.01,
                'edge_density': 0.1,
                'white_bg_pct': 50,
                'object_occupancy_proxy': 0.3,
                'border_clutter_score': 0.1,
                'bg_uniformity': 20,
            }
            # Build features with pre-computed PCA (skip PCA transformation)
            all_features = self._build_all_features(primary_features, precomputed_cnn_pca=precomputed_cnn_pca, precomputed_clip_pca=precomputed_clip_pca)
        elif precomputed_cnn_embedding is not None and precomputed_clip_embedding is not None:
            # Use pre-computed raw embeddings (will apply PCA)
            primary_features = {
                'cnn_embedding': precomputed_cnn_embedding,
                'clip_embedding': precomputed_clip_embedding,
                # Add placeholder quality/composition features
                'width': 1000,
                'height': 1000,
                'aspect_ratio': 1.0,
                'brightness_mean': 128,
                'brightness_std': 50,
                'contrast': 50,
                'saturation_mean': 100,
                'colorfulness': 40,
                'sharpness': 500,
                'blur_score': 0.01,
                'edge_density': 0.1,
                'white_bg_pct': 50,
                'object_occupancy_proxy': 0.3,
                'border_clutter_score': 0.1,
                'bg_uniformity': 20,
            }
            all_features = self._build_all_features(primary_features)
        else:
            # Extract from uploaded/downloaded image
            image_features = self._extract_image_features(image_paths)
            if not image_features:
                image_features = [{}]
            primary_features = image_features[0]
            all_features = self._build_all_features(primary_features)
        
        # Get category model
        model = self.model_registry.get_category_model(category)
        pipeline = self.model_registry.feature_pipeline
        
        # Build feature vectors for clf and reg
        clf_features = self._build_clf_feature_vector(all_features, pipeline)
        reg_features = self._build_reg_feature_vector(all_features, pipeline)
        
        # Stage 1: Classification - Has BSR?
        clf_result = model.predict_has_bsr(clf_features)
        bsr_probability = clf_result['probability']
        has_bsr_prediction = clf_result['prediction']
        
        # Stage 2: Regression - If BSR probability > 50%, estimate rank
        estimated_rank = None
        if bsr_probability > 0.5:  # Use probability threshold, not just prediction
            reg_result = model.predict_bsr_rank(reg_features)
            estimated_rank = reg_result['estimated_rank']
        
        # Compute scores
        launch_score = self._compute_launch_score(bsr_probability, primary_features, category)
        rank_band = self._estimate_rank_band(bsr_probability, estimated_rank, category)
        competitive_intensity = self._get_competitive_intensity(category)
        
        # Generate signals
        strengths, risks = self._decompose_signals(primary_features, category, clf_result)
        
        # Category context
        baseline = self._category_baselines.get(category, {"bsr_rate": 0.7, "difficulty": 65, "image_sensitive": True})
        category_notes = self._generate_category_notes(category, baseline, has_bsr_prediction)
        
        # Image quality metrics
        image_quality = self._build_image_quality_metrics(primary_features, category)
        
        # Percentile position
        percentile = self._estimate_percentile(launch_score, category)
        
        return EvaluationResponse(
            launch_viability_score=launch_score,
            bsr_entry_probability=bsr_probability,
            expected_rank_band=rank_band,
            competitive_intensity=competitive_intensity,
            confidence=clf_result['confidence'],
            strengths=strengths,
            risks=risks,
            category_notes=category_notes,
            category_baseline_viability=baseline['bsr_rate'] * 100,
            percentile_in_category=percentile,
            image_quality=image_quality,
            feature_summary={
                'top_features_used': len(settings.TOP_FEATURES),
                'embedding_dimension': 128,
                'images_processed': len(image_paths),
                'clf_model_loaded': model.clf_loaded,
                'reg_model_loaded': model.reg_loaded,
                'is_placeholder': not (model.clf_loaded and (not (bsr_probability > 0.5) or model.reg_loaded)),  # Only placeholder if CLF not loaded, or REG needed but not loaded
                'has_bsr_prediction': has_bsr_prediction,
                'estimated_rank': estimated_rank
            },
            product_hash=product_hash,
            model_version=settings.API_VERSION
        )
    
    def _compute_product_hash(self, title: str, description: str, image_paths: List[str]) -> str:
        content = f"{title}|{description}|{','.join(sorted(image_paths))}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _extract_image_features(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        features_list = []
        for path in image_paths:
            features = self.feature_extractor.extract_all_features(path)
            if features.get('valid', False):
                features_list.append(features)
        return features_list
    
    def _build_all_features(
        self, 
        features: Dict[str, Any],
        precomputed_cnn_pca: Optional[Dict[str, float]] = None,
        precomputed_clip_pca: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Build complete feature dictionary including PCA features.
        This converts raw image features into the format expected by models.
        
        If precomputed_cnn_pca and precomputed_clip_pca are provided, use them directly
        (dataset already has PCA features, no need to apply PCA transformation).
        """
        all_features = {}
        
        # Copy quality and composition features with _mean suffix (matching training data)
        quality_features = [
            'brightness_mean', 'brightness_std', 'contrast', 'saturation_mean',
            'saturation_std', 'colorfulness', 'sharpness_laplacian_var', 'sharpness',
            'blur_score', 'aspect_ratio', 'entropy', 'exposure_clipped_high_pct',
            'exposure_clipped_low_pct', 'width', 'height', 'filesize_kb'
        ]
        for feat in quality_features:
            if feat in features:
                # Use _mean suffix for aggregated features (matching training data format)
                all_features[f'{feat}_mean'] = float(features[feat])
                all_features[feat] = float(features[feat])  # Also keep without suffix
        
        composition_features = [
            'edge_density', 'bg_uniformity', 'white_bg_pct', 'dominant_color_count',
            'saliency_peak_ratio', 'saliency_center_dist', 'object_occupancy_proxy',
            'border_clutter_score'
        ]
        for feat in composition_features:
            if feat in features:
                all_features[f'{feat}_mean'] = float(features[feat])
                all_features[feat] = float(features[feat])
        
        # Detection features (placeholder values if not available)
        det_features = {
            'det_num_objects_mean': 1.0,
            'det_main_box_area_ratio_mean': 0.4,
            'det_main_box_center_dist_mean': 50.0,
            'det_has_person_mean': 0.0,
            'det_secondary_objects_count_mean': 0.0,
            'det_conf_mean_mean': 0.5,
            'det_conf_max_mean': 0.7,
        }
        all_features.update(det_features)
        
        # Use pre-computed PCA features if available (dataset already has PCA features)
        if precomputed_cnn_pca is not None and precomputed_clip_pca is not None:
            # Use PCA features directly from dataset (no transformation needed)
            all_features.update(precomputed_cnn_pca)
            all_features.update(precomputed_clip_pca)
        else:
            # Apply PCA to raw embeddings (for new images)
            pipeline = self.model_registry.feature_pipeline
            if pipeline is not None and pipeline.loaded:
                cnn_emb = features.get('cnn_embedding')
                clip_emb = features.get('clip_embedding')
                if cnn_emb is not None and clip_emb is not None:
                    pca_features = pipeline.apply_pca(cnn_emb, clip_emb)
                    all_features.update(pca_features)
                else:
                    # Generate placeholder PCA features if embeddings not available
                    for i in range(128):
                        all_features[f'cnn_pca_{i:04d}'] = np.random.randn() * 0.1
                        all_features[f'clip_pca_{i:04d}'] = np.random.randn() * 0.1
            else:
                # Generate placeholder PCA features
                for i in range(128):
                    all_features[f'cnn_pca_{i:04d}'] = np.random.randn() * 0.1
                    all_features[f'clip_pca_{i:04d}'] = np.random.randn() * 0.1
        
        # Add num_images (always 1 for web)
        all_features['num_images'] = 1.0
        
        return all_features
    
    def _build_clf_feature_vector(self, all_features: Dict[str, float], pipeline) -> np.ndarray:
        """Build feature vector for classification model (Top Features)."""
        if pipeline is not None and pipeline.loaded:
            return pipeline.build_clf_features(all_features)
        
        # Fallback: use default top features
        top_features = settings.TOP_FEATURES
        feature_vector = []
        for feat_name in top_features:
            feature_vector.append(all_features.get(feat_name, 0.0))
        return np.array(feature_vector, dtype=np.float32)
    
    def _build_reg_feature_vector(self, all_features: Dict[str, float], pipeline) -> np.ndarray:
        """Build feature vector for regression model (All Features)."""
        if pipeline is not None and pipeline.loaded:
            return pipeline.build_reg_features(all_features)
        
        # Fallback: use same as classification
        return self._build_clf_feature_vector(all_features, pipeline)
    
    def _compute_launch_score(
        self,
        bsr_probability: float,
        features: Dict[str, Any],
        category: str
    ) -> float:
        """Compute overall launch viability score (0-100)."""
        # Base score from BSR probability (70% weight)
        base_score = bsr_probability * 70
        
        # Image quality bonus (up to 30 points)
        quality_score = 0
        
        # Resolution bonus
        width = features.get('width', 0)
        height = features.get('height', 0)
        if width >= 1000 and height >= 1000:
            quality_score += 8
        elif width >= 500 and height >= 500:
            quality_score += 4
        
        # White background bonus
        white_bg = features.get('white_bg_pct', 0)
        if white_bg > 50:
            quality_score += 6
        elif white_bg > 30:
            quality_score += 3
        
        # Sharpness bonus
        sharpness = features.get('sharpness', 0)
        if sharpness > 500:
            quality_score += 6
        elif sharpness > 200:
            quality_score += 3
        
        # Low clutter bonus
        clutter = features.get('border_clutter_score', 1)
        if clutter < 0.1:
            quality_score += 5
        elif clutter < 0.2:
            quality_score += 2
        
        # Good object occupancy
        occupancy = features.get('object_occupancy_proxy', 0)
        if 0.2 <= occupancy <= 0.6:
            quality_score += 5
        
        return min(100, base_score + quality_score)
    
    def _estimate_rank_band(
        self,
        probability: float,
        estimated_rank: Optional[float],
        category: str
    ) -> RankBand:
        """Estimate expected rank band."""
        # Use regression estimate if available
        if estimated_rank is not None:
            # Rough mapping based on typical BSR distributions
            if estimated_rank < 5000:
                return RankBand.TOP_5
            elif estimated_rank < 20000:
                return RankBand.TOP_10
            elif estimated_rank < 50000:
                return RankBand.TOP_25
            elif estimated_rank < 150000:
                return RankBand.TOP_50
            else:
                return RankBand.BOTTOM_50
        
        # Fall back to probability-based estimate
        if probability >= 0.90:
            return RankBand.TOP_5
        elif probability >= 0.80:
            return RankBand.TOP_10
        elif probability >= 0.70:
            return RankBand.TOP_25
        elif probability >= 0.55:
            return RankBand.TOP_50
        else:
            return RankBand.BOTTOM_50
    
    def _get_competitive_intensity(self, category: str) -> float:
        baseline = self._category_baselines.get(category, {"difficulty": 65})
        return baseline['difficulty']
    
    def _decompose_signals(
        self,
        features: Dict[str, Any],
        category: str,
        clf_result: Dict[str, Any]
    ) -> Tuple[List[Signal], List[Signal]]:
        """Decompose prediction into human-readable signals."""
        strengths = []
        risks = []
        
        baseline = self._category_baselines.get(category, {"image_sensitive": True})
        
        # BSR prediction signal
        if clf_result['prediction'] == 1:
            strengths.append(Signal(
                type=SignalType.STRENGTH,
                label="BSR Entry Likely",
                description=f"Model predicts this product will achieve BSR ranking ({clf_result['probability']*100:.0f}% confidence)",
                impact=clf_result['probability']
            ))
        else:
            risks.append(Signal(
                type=SignalType.RISK,
                label="BSR Entry Uncertain",
                description=f"Model uncertain about BSR entry ({clf_result['probability']*100:.0f}% probability)",
                impact=1 - clf_result['probability']
            ))
        
        # Image Resolution
        width = features.get('width', 0)
        height = features.get('height', 0)
        if width >= 1000 and height >= 1000:
            strengths.append(Signal(
                type=SignalType.STRENGTH,
                label="High Resolution",
                description=f"Image resolution ({width}x{height}) meets Amazon best practices",
                impact=0.85
            ))
        elif width < 500 or height < 500:
            risks.append(Signal(
                type=SignalType.RISK,
                label="Low Resolution",
                description=f"Image resolution ({width}x{height}) below recommended minimum",
                impact=0.75
            ))
        
        # White Background
        white_bg = features.get('white_bg_pct', 0)
        if white_bg > 60:
            strengths.append(Signal(
                type=SignalType.STRENGTH,
                label="Clean Background",
                description="White/clean background aids product visibility",
                impact=0.72
            ))
        elif white_bg < 20:
            risks.append(Signal(
                type=SignalType.RISK,
                label="Busy Background",
                description="Low white background percentage may reduce clarity",
                impact=0.55
            ))
        
        # Sharpness
        sharpness = features.get('sharpness', 0)
        if sharpness > 500:
            strengths.append(Signal(
                type=SignalType.STRENGTH,
                label="Sharp Image",
                description="High sharpness indicates professional quality",
                impact=0.68
            ))
        elif sharpness < 100:
            risks.append(Signal(
                type=SignalType.RISK,
                label="Image Blur",
                description="Low sharpness detected, may appear unfocused",
                impact=0.70
            ))
        
        # Border Clutter
        clutter = features.get('border_clutter_score', 0)
        if clutter < 0.1:
            strengths.append(Signal(
                type=SignalType.STRENGTH,
                label="Clean Borders",
                description="Minimal edge clutter improves focus on product",
                impact=0.55
            ))
        elif clutter > 0.25:
            risks.append(Signal(
                type=SignalType.RISK,
                label="Border Clutter",
                description="Detected visual noise near image borders",
                impact=0.45
            ))
        
        # Competitive intensity
        if baseline.get('difficulty', 0) > 70:
            risks.append(Signal(
                type=SignalType.RISK,
                label="High Competition",
                description=f"{category} is highly competitive",
                impact=0.60
            ))
        
        # Sort by impact and take top 3 each
        strengths.sort(key=lambda x: x.impact, reverse=True)
        risks.sort(key=lambda x: x.impact, reverse=True)
        
        return strengths[:3], risks[:3]
    
    def _generate_category_notes(
        self,
        category: str,
        baseline: Dict[str, Any],
        has_bsr_prediction: int
    ) -> str:
        notes = []
        
        if has_bsr_prediction == 1:
            notes.append("Model predicts this product will achieve a BSR ranking.")
        else:
            notes.append("Model is uncertain about BSR ranking for this product.")
        
        if baseline.get('image_sensitive'):
            notes.append(f"{category} is image-sensitive; visual quality significantly impacts ranking.")
        
        difficulty = baseline.get('difficulty', 65)
        if difficulty > 70:
            notes.append("High competition in this category.")
        elif difficulty < 50:
            notes.append("Lower competition relative to other categories.")
        
        return " ".join(notes)
    
    def _build_image_quality_metrics(
        self,
        features: Dict[str, Any],
        category: str
    ) -> ImageQualityMetrics:
        """Build image quality metrics with category median comparison."""
        # Get category medians from feature pipeline
        pipeline = self.model_registry.feature_pipeline
        category_medians = {}
        if pipeline and hasattr(pipeline, 'category_medians'):
            category_medians = pipeline.category_medians.get(category, {})
        
        # Map feature names to median keys (medians use _mean suffix)
        vs_median = {}
        if category_medians:
            # Map current product metrics to category medians
            metric_mapping = {
                'brightness_mean': 'brightness_mean_mean',
                'brightness_std': 'brightness_std_mean',
                'contrast': 'contrast_mean',
                'saturation_mean': 'saturation_mean_mean',
                'colorfulness': 'colorfulness_mean',
                'sharpness': 'sharpness_mean',
                'white_bg_pct': 'white_bg_pct_mean',
                'edge_density': 'edge_density_mean',
                'aspect_ratio': 'aspect_ratio_mean'
            }
            
            for metric_key, median_key in metric_mapping.items():
                if median_key in category_medians:
                    current_val = features.get(metric_key, 0)
                    median_val = category_medians[median_key]
                    if median_val > 0:
                        vs_median[metric_key] = float((current_val - median_val) / median_val * 100)
        
        return ImageQualityMetrics(
            width=int(features.get('width', 0)),
            height=int(features.get('height', 0)),
            aspect_ratio=float(features.get('aspect_ratio', 1.0)),
            brightness_mean=float(features.get('brightness_mean', 128)),
            brightness_std=float(features.get('brightness_std', 50)),
            contrast=float(features.get('contrast', 50)),
            saturation_mean=float(features.get('saturation_mean', 100)),
            colorfulness=float(features.get('colorfulness', 40)),
            sharpness=float(features.get('sharpness', 500)),
            blur_score=float(features.get('blur_score', 0.01)),
            white_bg_pct=float(features.get('white_bg_pct', 50)),
            edge_density=float(features.get('edge_density', 0.1)),
            vs_category_median=vs_median if vs_median else None
        )
    
    def _estimate_percentile(self, launch_score: float, category: str) -> float:
        return min(99, max(1, launch_score))


# Singleton instance
_inference_instance = None

def get_inference_service() -> InferenceService:
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = InferenceService()
    return _inference_instance
