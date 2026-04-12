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
    ImageQualityMetrics, NLPFeedback, KeywordSuggestion
)
from .image_features import get_feature_extractor
from .model_registry import get_model_registry
from .text_features import get_text_feature_extractor


class InferenceService:
    """Main inference service for product evaluation."""
    
    def __init__(self):
        self.feature_extractor = get_feature_extractor(device=settings.DEVICE)
        self.model_registry = get_model_registry()
        self.text_feature_extractor = get_text_feature_extractor()
        
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
        precomputed_clip_pca: Optional[Dict[str, float]] = None,
        price: Optional[float] = None
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
        
        # Extract text features and merge into all_features
        text_features = self.text_feature_extractor.extract_features(title, description, category)
        all_features.update(text_features)

        # Override price features if price provided by user
        if price is not None and price > 0:
            all_features['log_price'] = float(np.log1p(price))
            # Price bin logic (simplified - matches notebook 05)
            all_features['price_bin'] = 1.0  # mid default
            all_features['price_vs_category_median'] = 0.0

        # Generate NLP feedback for the response
        nlp_feedback_data = self.text_feature_extractor.generate_keyword_feedback(title, description, category)

        # Get category model
        model = self.model_registry.get_category_model(category)
        pipeline = self.model_registry.feature_pipeline

        # Build feature vectors for clf and reg
        # Use model's stored feature lists if available (exact features model was trained with)
        clf_features = self._build_clf_feature_vector(all_features, pipeline, model.clf_features)
        reg_features = self._build_reg_feature_vector(all_features, pipeline, model.reg_features)
        
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
        
        # Build NLP feedback model if data available
        nlp_feedback = None
        if nlp_feedback_data is not None:
            nlp_feedback = NLPFeedback(
                title_score=nlp_feedback_data['title_score'],
                bullets_score=nlp_feedback_data['bullets_score'],
                missing_keywords=[
                    KeywordSuggestion(**kw) for kw in nlp_feedback_data.get('missing_keywords', [])
                ],
                weak_keywords=[
                    KeywordSuggestion(**kw) for kw in nlp_feedback_data.get('weak_keywords', [])
                ],
                title_stats=nlp_feedback_data.get('title_stats', {}),
                bullets_stats=nlp_feedback_data.get('bullets_stats', {}),
            )

        # Add NLP-based signals
        if nlp_feedback_data:
            title_score = nlp_feedback_data.get('title_score', 0)
            if title_score > 60:
                strengths.append(Signal(
                    type=SignalType.STRENGTH,
                    label="Strong Title Keywords",
                    description=f"Title matches {title_score:.0f}% of top seller keyword patterns",
                    impact=min(title_score / 100, 0.9)
                ))
            elif title_score < 30 and title:
                risks.append(Signal(
                    type=SignalType.RISK,
                    label="Weak Title Keywords",
                    description=f"Title only matches {title_score:.0f}% of top seller patterns - see keyword suggestions",
                    impact=0.6
                ))

            missing_count = len(nlp_feedback_data.get('missing_keywords', []))
            if missing_count > 5:
                risks.append(Signal(
                    type=SignalType.RISK,
                    label="Missing Key Terms",
                    description=f"{missing_count} high-value keywords missing from your listing",
                    impact=0.55
                ))

        # Extract top feature importances from the classification model
        top_feature_importances = self._get_top_feature_importances(model, category)

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
            nlp_feedback=nlp_feedback,
            top_feature_importances=top_feature_importances,
            feature_summary={
                'top_features_used': len(settings.TOP_FEATURES),
                'embedding_dimension': 128,
                'images_processed': len(image_paths),
                'clf_model_loaded': model.clf_loaded,
                'reg_model_loaded': model.reg_loaded,
                'is_placeholder': not (model.clf_loaded and (not (bsr_probability > 0.5) or model.reg_loaded)),
                'has_bsr_prediction': has_bsr_prediction,
                'estimated_rank': estimated_rank,
                'text_features_loaded': self.text_feature_extractor.loaded,
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
    
    def _build_clf_feature_vector(self, all_features: Dict[str, float], pipeline, model_features: Optional[List[str]] = None) -> np.ndarray:
        """
        Build feature vector for classification model.
        
        Args:
            all_features: Dictionary of all available features
            pipeline: Feature pipeline (for fallback)
            model_features: Exact list of features the model was trained with (preferred)
        """
        # Priority 1: Use model's exact feature list (most accurate)
        if model_features and len(model_features) > 0:
            feature_vector = []
            for feat_name in model_features:
                feature_vector.append(all_features.get(feat_name, 0.0))
            return np.array(feature_vector, dtype=np.float32)
        
        # Priority 2: Use pipeline's feature list
        if pipeline is not None and pipeline.loaded:
            return pipeline.build_clf_features(all_features)
        
        # Fallback: use default top features
        top_features = settings.TOP_FEATURES
        feature_vector = []
        for feat_name in top_features:
            feature_vector.append(all_features.get(feat_name, 0.0))
        return np.array(feature_vector, dtype=np.float32)
    
    def _build_reg_feature_vector(self, all_features: Dict[str, float], pipeline, model_features: Optional[List[str]] = None) -> np.ndarray:
        """
        Build feature vector for regression model.
        
        Args:
            all_features: Dictionary of all available features
            pipeline: Feature pipeline (for fallback)
            model_features: Exact list of features the model was trained with (preferred)
        """
        # Priority 1: Use model's exact feature list (most accurate)
        if model_features and len(model_features) > 0:
            feature_vector = []
            for feat_name in model_features:
                feature_vector.append(all_features.get(feat_name, 0.0))
            return np.array(feature_vector, dtype=np.float32)
        
        # Priority 2: Use pipeline's feature list
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
    
    def _get_top_feature_importances(self, model, category: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract feature importances grouped into interpretable categories.

        Embedding dimensions (cnn_pca_*, clip_emb_*, title_tfidf_pca_*, etc.) are
        aggregated into groups like "Visual Patterns (CNN)" so users see meaningful
        explanations instead of opaque feature names.
        """
        try:
            if not model.clf_loaded or model.clf_model is None or not model.clf_features:
                return None

            importances = model.clf_model.feature_importances_
            feature_names = model.clf_features

            if len(importances) != len(feature_names):
                return None

            # Human-readable names for interpretable features
            readable = {
                'sharpness_laplacian_var_mean': 'Image Sharpness',
                'sharpness_laplacian_var_max': 'Image Sharpness (Best)',
                'white_bg_pct_mean': 'White Background %',
                'contrast_mean': 'Image Contrast',
                'brightness_mean_mean': 'Brightness',
                'brightness_std_mean': 'Brightness Variation',
                'edge_density_mean': 'Edge Density',
                'colorfulness_mean': 'Colorfulness',
                'aspect_ratio_mean': 'Aspect Ratio',
                'border_clutter_score_mean': 'Border Clutter',
                'det_conf_mean_mean': 'Object Detection Confidence',
                'det_conf_max_mean': 'Best Object Detection',
                'det_has_person_mean': 'Person in Image',
                'det_num_objects_mean': 'Number of Objects',
                'det_main_box_center_dist_mean': 'Subject Centering',
                'det_main_box_area_ratio_mean': 'Subject Size',
                'object_occupancy_proxy_mean': 'Product Coverage',
                'saturation_mean_mean': 'Color Saturation',
                'entropy_mean': 'Image Complexity',
                'blur_score_mean': 'Image Blur',
                'filesize_kb_mean': 'Image File Size',
                'bg_uniformity_mean': 'Background Uniformity',
                'saliency_peak_ratio_mean': 'Visual Attention Focus',
                'saliency_center_dist_mean': 'Subject Center Distance',
                'num_images': 'Number of Images',
                'has_multiple_images': 'Multiple Images',
                'num_colors': 'Color Variants',
                'num_colors_available': 'Color Options',
                'title_word_count': 'Title Word Count',
                'title_char_count': 'Title Length',
                'title_flesch_reading_ease': 'Title Readability',
                'title_flesch_kincaid_grade': 'Title Grade Level',
                'title_unique_word_ratio': 'Title Word Variety',
                'title_avg_word_length': 'Title Avg Word Length',
                'title_separator_count': 'Title Separators',
                'title_has_brand': 'Brand in Title',
                'title_has_size_spec': 'Size in Title',
                'title_has_color_spec': 'Color in Title',
                'bullets_total_word_count': 'Bullets Word Count',
                'bullets_keyword_density': 'Bullet Keyword Density',
                'bullets_count': 'Number of Bullets',
                'bullets_avg_length': 'Avg Bullet Length',
                'has_bullets': 'Has Bullet Points',
                'log_price': 'Product Price',
                'sd_price': 'Listed Price',
                'sd_list_price': 'List Price',
                'title_length_x_sharpness': 'Title Length x Sharpness',
                'price_x_image_count': 'Price x Image Count',
                'text_richness_x_image_count': 'Text Richness x Images',
                'clip_cnn_cos_sim': 'Visual Embedding Alignment',
                'exposure_clipped_high_pct_mean': 'Overexposure %',
                'exposure_clipped_low_pct_mean': 'Underexposure %',
                'dominant_color_count_mean': 'Dominant Colors',
                'ocr_word_count_mean': 'Text on Image (Words)',
                'ocr_char_count_mean': 'Text on Image (Chars)',
                'ocr_has_claims_mean': 'Claims on Image',
                'ocr_allcaps_ratio_mean': 'ALL-CAPS Text Ratio',
                'text_overlay_area_proxy_mean': 'Text Overlay Area',
            }

            # Group definitions: prefix -> (group_name, display_name, type)
            embedding_groups = {
                'cnn_pca_': ('cnn_pca', 'Visual Patterns (CNN)', 'image'),
                'cnn_emb_': ('cnn_emb', 'Visual Patterns (CNN)', 'image'),
                'clip_pca_': ('clip_pca', 'Visual-Semantic Similarity (CLIP)', 'image'),
                'clip_emb_': ('clip_emb', 'Visual-Semantic Similarity (CLIP)', 'image'),
                'hero_cnn_emb_': ('hero_cnn', 'Main Image Visual Pattern', 'image'),
                'hero_clip_emb_': ('hero_clip', 'Main Image Semantic Match', 'image'),
                'title_tfidf_pca_': ('title_tfidf', 'Title Keywords (TF-IDF)', 'text'),
                'bullets_tfidf_pca_': ('bullets_tfidf', 'Bullet Keywords (TF-IDF)', 'text'),
            }

            # Accumulate importances: grouped embeddings + individual interpretable
            grouped = {}  # group_key -> {importance, display_name, type}
            individual = []  # (name, importance, display_name, type)

            for name, imp in zip(feature_names, importances):
                if imp <= 0:
                    continue

                # Check if this feature belongs to an embedding group
                matched_group = None
                for prefix, (group_key, group_display, group_type) in embedding_groups.items():
                    if name.startswith(prefix):
                        matched_group = (group_key, group_display, group_type)
                        break

                if matched_group:
                    gk, gd, gt = matched_group
                    if gk not in grouped:
                        grouped[gk] = {'importance': 0.0, 'display_name': gd, 'type': gt}
                    grouped[gk]['importance'] += float(imp)
                elif name in readable:
                    # Interpretable feature with a human name
                    feat_type = self._classify_feature_type(name)
                    individual.append((name, float(imp), readable[name], feat_type))
                else:
                    # Hero image quality features or other unrecognized
                    if name.startswith('hero_'):
                        # Map hero_X to readable X equivalent
                        base = name[5:]  # strip 'hero_'
                        hero_display = readable.get(base + '_mean', readable.get(base, None))
                        if hero_display:
                            display = f"Main Image {hero_display}"
                        else:
                            display = f"Main Image: {base.replace('_', ' ').title()}"
                        feat_type = 'image'
                        individual.append((name, float(imp), display, feat_type))
                    else:
                        # Unknown feature — skip rather than show cryptic name
                        pass

            # Combine grouped and individual, sort by importance
            all_items = []
            for gk, ginfo in grouped.items():
                all_items.append({
                    'feature': gk,
                    'display_name': ginfo['display_name'],
                    'importance': round(ginfo['importance'], 4),
                    'type': ginfo['type'],
                })
            for name, imp, display, ftype in individual:
                all_items.append({
                    'feature': name,
                    'display_name': display,
                    'importance': round(imp, 4),
                    'type': ftype,
                })

            all_items.sort(key=lambda x: x['importance'], reverse=True)
            return all_items[:10]

        except Exception as e:
            print(f"Feature importance extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _classify_feature_type(name: str) -> str:
        """Classify a feature name into a display type."""
        if any(k in name for k in ['title_', 'bullets_', 'has_bullets']):
            return 'text'
        if any(k in name for k in ['price', 'log_price', 'sd_price', 'sd_list_price']):
            return 'price'
        return 'image'

    def _estimate_percentile(self, launch_score: float, category: str) -> float:
        return min(99, max(1, launch_score))


# Singleton instance
_inference_instance = None

def get_inference_service() -> InferenceService:
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = InferenceService()
    return _inference_instance
