"""
Text Feature Extraction and NLP Feedback Service.

Handles:
1. TF-IDF vectorization + SVD transformation for model input
2. Text statistics computation (readability, word counts, etc.)
3. Keyword feedback generation (compare user text against category top performers)
"""
import re
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False


class TextFeatureExtractor:
    """Extracts text features and generates keyword feedback for product evaluation."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.loaded = False

        # TF-IDF vectorizers
        self.tfidf_title = None
        self.tfidf_bullets = None

        # SVD transformers
        self.svd_title = None
        self.svd_bullets = None

        # Category profiles for keyword comparison
        self.category_profiles = {}

        # Feature name mappings
        self.tfidf_feature_names = {}

        self._load_artifacts()

    def _load_artifacts(self):
        """Load all NLP model artifacts from disk."""
        try:
            # TF-IDF vectorizers
            title_vec_path = self.models_dir / 'tfidf_title_vectorizer.pkl'
            bullets_vec_path = self.models_dir / 'tfidf_bullets_vectorizer.pkl'

            if title_vec_path.exists():
                with open(title_vec_path, 'rb') as f:
                    self.tfidf_title = pickle.load(f)
                print(f"  Loaded title TF-IDF vectorizer ({len(self.tfidf_title.vocabulary_)} terms)")

            if bullets_vec_path.exists():
                with open(bullets_vec_path, 'rb') as f:
                    obj = pickle.load(f)
                if obj is not None and hasattr(obj, 'vocabulary_'):
                    self.tfidf_bullets = obj
                    print(f"  Loaded bullets TF-IDF vectorizer ({len(self.tfidf_bullets.vocabulary_)} terms)")
                else:
                    print(f"  Bullets TF-IDF vectorizer is None (no bullet data available)")

            # SVD transformers
            title_svd_path = self.models_dir / 'tfidf_title_svd.pkl'
            bullets_svd_path = self.models_dir / 'tfidf_bullets_svd.pkl'

            if title_svd_path.exists():
                with open(title_svd_path, 'rb') as f:
                    self.svd_title = pickle.load(f)
                print(f"  Loaded title SVD ({self.svd_title.n_components} components)")

            if bullets_svd_path.exists():
                with open(bullets_svd_path, 'rb') as f:
                    obj = pickle.load(f)
                if obj is not None and hasattr(obj, 'n_components'):
                    self.svd_bullets = obj
                    print(f"  Loaded bullets SVD ({self.svd_bullets.n_components} components)")
                else:
                    print(f"  Bullets SVD is None (no bullet data available)")

            # Category profiles
            profiles_path = self.models_dir / 'category_tfidf_profiles.pkl'
            if profiles_path.exists():
                with open(profiles_path, 'rb') as f:
                    self.category_profiles = pickle.load(f)
                print(f"  Loaded category TF-IDF profiles ({len(self.category_profiles)} categories)")

            # Feature name mappings
            names_path = self.models_dir / 'tfidf_feature_names.pkl'
            if names_path.exists():
                with open(names_path, 'rb') as f:
                    self.tfidf_feature_names = pickle.load(f)
                print(f"  Loaded TF-IDF feature name mappings")

            self.loaded = (self.tfidf_title is not None and self.svd_title is not None)
            if self.loaded:
                print("  Text feature extractor: READY")
            else:
                print("  Text feature extractor: NOT LOADED (missing artifacts)")

        except Exception as e:
            print(f"  Text feature extractor load error: {e}")
            self.loaded = False

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for TF-IDF: lowercase, remove special chars, normalize whitespace."""
        if not text or not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_features(self, title: str, bullets: str, category: str) -> Dict[str, float]:
        """
        Extract all text features for model input.

        Returns dict of feature_name -> float value, including:
        - TF-IDF SVD features (title_tfidf_pca_0000..0049, bullets_tfidf_pca_0000..0049)
        - Text statistics (title_word_count, title_flesch_reading_ease, etc.)
        - Engineered features (log_price placeholder, has_bullets, etc.)
        """
        features = {}

        clean_title = self._clean_text(title)
        clean_bullets = self._clean_text(bullets)

        # TF-IDF + SVD features
        if self.loaded:
            features.update(self._compute_tfidf_svd_features(clean_title, clean_bullets))
        else:
            # Zero-fill if not loaded
            for i in range(50):
                features[f'title_tfidf_pca_{i:04d}'] = 0.0
                features[f'bullets_tfidf_pca_{i:04d}'] = 0.0

        # Text statistics
        features.update(self._compute_text_statistics(title, bullets, clean_title, clean_bullets))

        # Engineered features that depend on text
        features.update(self._compute_text_engineered_features(title, bullets, clean_title, clean_bullets))

        return features

    def _compute_tfidf_svd_features(self, clean_title: str, clean_bullets: str) -> Dict[str, float]:
        """Compute TF-IDF -> SVD features."""
        features = {}

        # Title TF-IDF -> SVD
        if self.tfidf_title is not None and self.svd_title is not None:
            title_tfidf = self.tfidf_title.transform([clean_title])
            title_svd = self.svd_title.transform(title_tfidf)[0]
            for i, val in enumerate(title_svd):
                features[f'title_tfidf_pca_{i:04d}'] = float(val)

        # Bullets TF-IDF -> SVD
        if self.tfidf_bullets is not None and self.svd_bullets is not None:
            bullets_tfidf = self.tfidf_bullets.transform([clean_bullets])
            bullets_svd = self.svd_bullets.transform(bullets_tfidf)[0]
            for i, val in enumerate(bullets_svd):
                features[f'bullets_tfidf_pca_{i:04d}'] = float(val)

        return features

    def _compute_text_statistics(
        self, raw_title: str, raw_bullets: str,
        clean_title: str, clean_bullets: str
    ) -> Dict[str, float]:
        """Compute text statistics features."""
        features = {}

        # Title features
        words = clean_title.split() if clean_title else []
        features['title_char_count'] = float(len(clean_title))
        features['title_word_count'] = float(len(words))
        features['title_avg_word_length'] = float(np.mean([len(w) for w in words])) if words else 0.0
        features['title_unique_word_ratio'] = float(len(set(words)) / max(len(words), 1)) if words else 0.0

        # Readability
        if HAS_TEXTSTAT and clean_title and len(words) >= 3:
            features['title_flesch_reading_ease'] = float(textstat.flesch_reading_ease(clean_title))
            features['title_flesch_kincaid_grade'] = float(textstat.flesch_kincaid_grade(clean_title))
        else:
            features['title_flesch_reading_ease'] = 0.0
            features['title_flesch_kincaid_grade'] = 0.0

        # Title structure
        if raw_title and isinstance(raw_title, str):
            features['title_separator_count'] = float(
                raw_title.count('|') + raw_title.count(' - ') + raw_title.count(',')
            )
            features['title_has_brand'] = 0.0  # Can't detect without brand column at inference

            size_pattern = re.compile(
                r'\b\d+\s*(?:oz|inch|in|cm|mm|ml|liter|litre|lb|lbs|kg|ft|feet|gallon|gal|qt|quart|pack|pc|pcs|count|ct)\b',
                re.IGNORECASE
            )
            features['title_has_size_spec'] = 1.0 if size_pattern.search(raw_title) else 0.0

            color_pattern = re.compile(
                r'\b(?:black|white|red|blue|green|yellow|pink|purple|orange|gold|silver|grey|gray|brown|navy|teal|beige|ivory)\b',
                re.IGNORECASE
            )
            features['title_has_color_spec'] = 1.0 if color_pattern.search(raw_title) else 0.0
        else:
            features['title_separator_count'] = 0.0
            features['title_has_brand'] = 0.0
            features['title_has_size_spec'] = 0.0
            features['title_has_color_spec'] = 0.0

        # Bullet features
        if raw_bullets and isinstance(raw_bullets, str):
            parts = re.split(r'[\n\r|]+', raw_bullets)
            bullet_parts = [p.strip() for p in parts if p.strip()]
            features['bullets_count'] = float(len(bullet_parts))
            features['bullets_avg_length'] = float(np.mean([len(p) for p in bullet_parts])) if bullet_parts else 0.0
        else:
            features['bullets_count'] = 0.0
            features['bullets_avg_length'] = 0.0

        bullet_words = clean_bullets.split() if clean_bullets else []
        features['bullets_total_word_count'] = float(len(bullet_words))
        features['has_bullets'] = 1.0 if features['bullets_count'] > 0 else 0.0

        # Keyword density
        filler = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                   'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
                   'this', 'that', 'it', 'its', 'our', 'your', 'you', 'we', 'they'}
        if bullet_words:
            meaningful = [w for w in bullet_words if w not in filler and len(w) > 2]
            features['bullets_keyword_density'] = float(len(meaningful) / len(bullet_words))
        else:
            features['bullets_keyword_density'] = 0.0

        return features

    def _compute_text_engineered_features(
        self, raw_title: str, raw_bullets: str,
        clean_title: str, clean_bullets: str
    ) -> Dict[str, float]:
        """Compute engineered features that depend on text."""
        features = {}

        # These features need values at inference that match training
        features['log_price'] = 0.0
        features['has_multiple_images'] = 0.0  # Will be set by caller if known
        features['num_colors_available'] = 0.0
        # NOTE: price_vs_category_median, price_bin, title_length_vs_category_median,
        # has_aplus_content excluded from models (BSR-group leakage)

        # Interaction features
        title_wc = len(clean_title.split()) if clean_title else 0
        bullet_wc = len(clean_bullets.split()) if clean_bullets else 0
        features['title_length_x_sharpness'] = 0.0  # Will be combined with image sharpness
        features['price_x_image_count'] = 0.0
        features['text_richness_x_image_count'] = float(title_wc + bullet_wc)

        return features

    def generate_keyword_feedback(
        self, title: str, bullets: str, category: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate keyword feedback by comparing user's text against category top performers.

        Returns NLP feedback dict with:
        - title_score, bullets_score (0-100)
        - missing_keywords: keywords used by top sellers but missing from user's text
        - weak_keywords: keywords in user's text rarely seen in top sellers
        """
        if not self.loaded or category not in self.category_profiles:
            return None

        profile = self.category_profiles[category]
        clean_title = self._clean_text(title)
        clean_bullets = self._clean_text(bullets)

        # Get title feature names
        title_feature_names = self.tfidf_feature_names.get('title_feature_names', [])
        bullets_feature_names = self.tfidf_feature_names.get('bullets_feature_names', [])

        # Transform user text to TF-IDF
        user_title_tfidf = self.tfidf_title.transform([clean_title])
        user_title_dense = np.asarray(user_title_tfidf.toarray()).flatten()

        has_bullets_model = self.tfidf_bullets is not None
        if has_bullets_model:
            user_bullets_tfidf = self.tfidf_bullets.transform([clean_bullets])
            user_bullets_dense = np.asarray(user_bullets_tfidf.toarray()).flatten()
        else:
            user_bullets_dense = np.array([])

        # Category average and prevalence
        cat_title_avg = profile['title_avg_tfidf']
        cat_title_prev = profile['title_prevalence']
        cat_bullets_avg = profile.get('bullets_avg_tfidf')
        cat_bullets_prev = profile.get('bullets_prevalence')

        # Find missing keywords (high in category avg, low/zero in user text)
        missing_keywords = []

        # Title missing keywords
        title_diff = cat_title_avg - user_title_dense
        title_missing_idx = np.argsort(title_diff)[-20:][::-1]
        for idx in title_missing_idx:
            if idx < len(title_feature_names) and title_diff[idx] > 0.01 and cat_title_prev[idx] > 0.15:
                keyword = title_feature_names[idx]
                prevalence = float(cat_title_prev[idx])
                # Only suggest single words and meaningful bigrams
                if len(keyword) > 2:
                    missing_keywords.append({
                        'keyword': keyword,
                        'prevalence': prevalence,
                        'source': 'title',
                        'suggestion': f"Add '{keyword}' to your title ({prevalence*100:.0f}% of top sellers use it)"
                    })

        # Bullets missing keywords
        if clean_bullets and has_bullets_model and cat_bullets_avg is not None:
            bullets_diff = cat_bullets_avg - user_bullets_dense
            bullets_missing_idx = np.argsort(bullets_diff)[-15:][::-1]
            for idx in bullets_missing_idx:
                if idx < len(bullets_feature_names) and bullets_diff[idx] > 0.01 and cat_bullets_prev[idx] > 0.15:
                    keyword = bullets_feature_names[idx]
                    prevalence = float(cat_bullets_prev[idx])
                    if len(keyword) > 2:
                        missing_keywords.append({
                            'keyword': keyword,
                            'prevalence': prevalence,
                            'source': 'bullets',
                            'suggestion': f"Consider adding '{keyword}' to bullet points ({prevalence*100:.0f}% of top sellers mention it)"
                        })

        # Deduplicate and limit
        seen = set()
        unique_missing = []
        for kw in missing_keywords:
            if kw['keyword'] not in seen:
                seen.add(kw['keyword'])
                unique_missing.append(kw)
        missing_keywords = unique_missing[:10]

        # Find weak keywords (in user text but rarely in top sellers)
        weak_keywords = []
        user_title_nonzero = np.nonzero(user_title_dense)[0]
        for idx in user_title_nonzero:
            if idx < len(title_feature_names) and cat_title_prev[idx] < 0.05 and user_title_dense[idx] > 0.1:
                keyword = title_feature_names[idx]
                if len(keyword) > 2:
                    weak_keywords.append({
                        'keyword': keyword,
                        'prevalence': float(cat_title_prev[idx]),
                        'source': 'title',
                        'suggestion': f"'{keyword}' appears rarely in top sellers - consider replacing"
                    })
        weak_keywords = weak_keywords[:5]

        # Compute title score (0-100): how well does user's title match top performers?
        if np.linalg.norm(cat_title_avg) > 0 and np.linalg.norm(user_title_dense) > 0:
            cosine_sim = np.dot(user_title_dense, cat_title_avg) / (
                np.linalg.norm(user_title_dense) * np.linalg.norm(cat_title_avg)
            )
            title_score = float(np.clip(cosine_sim * 100, 0, 100))
        else:
            title_score = 0.0

        # Compute bullets score
        if cat_bullets_avg is not None and len(user_bullets_dense) > 0 and np.linalg.norm(cat_bullets_avg) > 0 and np.linalg.norm(user_bullets_dense) > 0:
            cosine_sim = np.dot(user_bullets_dense, cat_bullets_avg) / (
                np.linalg.norm(user_bullets_dense) * np.linalg.norm(cat_bullets_avg)
            )
            bullets_score = float(np.clip(cosine_sim * 100, 0, 100))
        else:
            bullets_score = 0.0

        # Text statistics for display
        title_words = clean_title.split() if clean_title else []
        bullet_parts = re.split(r'[\n\r|]+', bullets) if bullets else []
        bullet_parts = [p.strip() for p in bullet_parts if p.strip()]

        title_stats = {
            'char_count': len(title) if title else 0,
            'word_count': len(title_words),
            'avg_word_length': float(np.mean([len(w) for w in title_words])) if title_words else 0,
            'unique_word_ratio': float(len(set(title_words)) / max(len(title_words), 1)),
            'separator_count': (title.count('|') + title.count(' - ') + title.count(',')) if title else 0,
            'has_size_spec': bool(re.search(r'\b\d+\s*(?:oz|inch|in|cm|mm|ml|lb|lbs|kg|ft|pack|pcs|count|ct)\b', title or '', re.I)),
            'has_color_spec': bool(re.search(r'\b(?:black|white|red|blue|green|yellow|pink|purple|orange|gold|silver|grey|gray|brown)\b', title or '', re.I)),
        }

        if HAS_TEXTSTAT and clean_title and len(title_words) >= 3:
            title_stats['reading_ease'] = float(textstat.flesch_reading_ease(clean_title))
            title_stats['grade_level'] = float(textstat.flesch_kincaid_grade(clean_title))

        bullets_stats = {
            'bullet_count': len(bullet_parts),
            'avg_bullet_length': float(np.mean([len(p) for p in bullet_parts])) if bullet_parts else 0,
            'total_word_count': len(clean_bullets.split()) if clean_bullets else 0,
        }

        return {
            'title_score': title_score,
            'bullets_score': bullets_score,
            'missing_keywords': missing_keywords,
            'weak_keywords': weak_keywords,
            'title_stats': title_stats,
            'bullets_stats': bullets_stats,
        }


# Singleton
_extractor_instance = None

def get_text_feature_extractor(models_dir: Path = None) -> TextFeatureExtractor:
    """Get or create singleton text feature extractor."""
    global _extractor_instance
    if _extractor_instance is None:
        from ..core.config import settings
        _extractor_instance = TextFeatureExtractor(models_dir or settings.MODELS_DIR)
    return _extractor_instance
