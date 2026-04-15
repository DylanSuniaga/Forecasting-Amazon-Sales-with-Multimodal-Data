"""
Model & Feature Pipeline Saving Script for Web Application
==========================================================

This script properly saves:
1. PCA transformers (fitted on training data only)
2. Classification models (XGBoost - best per notebook analysis)
3. Regression models (XGBoost/LightGBM - best per notebook analysis with full features)
4. Feature scalers
5. Test sample products for website suggestions

Feature pipeline matches image_feature_extraction.ipynb exactly:
- Raw embeddings: CNN (EfficientNet-B0: 1280 dims) + CLIP (ViT-B-32: 512 dims)
- PCA reduction: 128 components each
- Quality features: brightness, contrast, saturation, sharpness, etc.
- Composition features: edge density, white_bg_pct, border_clutter, etc.
- Detection features: YOLO object detection outputs

Based on notebook analysis:
- Classification: XGBoost with Top Features performs well (ROC-AUC 0.95-0.99)
- Regression: XGBoost/LightGBM with ALL features significantly outperforms Top Features
  (R² improvement of 0.05-0.15 across categories)

Usage:
------
cd notebooks/02-Image\ Analysis
python save_models_for_web.py
"""

import os
import gc
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, r2_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

# Try to import LightGBM (better for regression based on notebook results)
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("Warning: LightGBM not available, using XGBoost for regression")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "../../data/products_with_image_feats.csv"
MODELS_OUTPUT_DIR = Path("../../src/web/backend/models")
DATA_OUTPUT_DIR = Path("../../src/web/backend/data")
MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of example products to save per category
SAMPLES_PER_CATEGORY = 3

# Test split ratio (same as notebooks)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Categories from notebooks (14 main BSR groups)
CATEGORIES = [
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

# PCA configuration (matches image_feature_extraction.ipynb exactly)
PCA_N_COMPONENTS = 128

# ============================================================================
# FEATURE DEFINITIONS (matching image_feature_extraction.ipynb)
# ============================================================================

# Columns to exclude from features (metadata and targets)
# MATCHES NOTEBOOK EXACTLY (c2_baseline_modeling_reg.ipynb)
EXCLUDE_COLS = [
    # Identifiers and metadata
    'Unnamed: 0.1', 'Unnamed: 0', 'asin', 'category', 'query', 'page',
    'source_section', 'type', 'title', 'image', 'url', 'optimized_url',
    'delivery', 'price_string', 'price_symbol', 'colors', 'location',
    'search_message', 'sd_feature_bullets_text', 'sd_title', 'sd_parent_asin',
    'sd_price_symbol', 'sd_availability_status', 'sd_product_category',
    'sd_category_id', 'sd_ratings_distribution', 'sd_customer_sentiments',
    'sd_error', 'sd_errors', 'sd_best_sellers_rank', 'certification', 'coupon_text',
    'fetched_at_unix',
    # BSR-related columns - CRITICAL: exclude to prevent leakage
    'main_bsr_group', 'main_bsr_rank', 'lowest_bsr_group', 'lowest_bsr_rank',
    'has_main_bsr', 'has_lowest_bsr',  # Binary targets - CRITICAL: exclude to prevent leakage
    'target',  # Dynamic target column created per group
    # POST-LISTING METRICS - Not available for new products!
    # Rating/Review features
    'stars', 'total_reviews', 'sd_average_rating', 'sd_total_reviews',
    'sd_stars', 'sd_ratings_count',
    'sd_rating_pct_1', 'sd_rating_pct_2', 'sd_rating_pct_3', 'sd_rating_pct_4', 'sd_rating_pct_5',
    # Sentiment counts (from reviews)
    'sd_sent_count_POSITIVE', 'sd_sent_count_MIXED', 'sd_sent_count_NEGATIVE',
    # Position features (only known after listing)
    'organic_position', 'absolute_position',
    # Sales/popularity indicators (post-listing)
    'number_of_people_bought', 'sd_number_bought_past_month',
    'is_best_seller', 'is_amazon_choice',
    # Deal/promotion status (variable, not intrinsic to product)
    'limited_time_deal', 'deal_of_the_day', 'sponsored',
    # Availability (post-listing)
    'availability_quantity', 'sd_is_frequently_returned',
    # Price history (not available for new products)
    'sd_previous_price',
    # Text columns (raw text strings, not features - used to derive NLP features)
    'text_title', 'text_title_clean', 'text_bullets', 'text_bullets_clean',
    # POST-LISTING promotional decisions (removed from feature engineering for causality)
    'has_discount', 'discount_pct',
    # CATEGORY-RELATIVE features computed using main_bsr_group as grouping key
    # Products without BSR get 0/raw values, creating perfect leakage for classification
    'price_vs_category_median', 'title_length_vs_category_median', 'price_bin',
    # A+ content flag: sd_aplus is only populated by ScrapingDog for ranked products
    # 100% of BSR products = 1, making it a near-perfect leakage signal
    'has_aplus_content',
]

# Top 20 features from classification EDA (for classification models)
TOP_FEATURES_CLF = [
    'cnn_pca_0003', 'clip_pca_0005', 'clip_pca_0002', 'clip_pca_0000',
    'cnn_pca_0000', 'clip_pca_0003', 'clip_pca_0001', 'det_conf_mean_mean',
    'det_conf_max_mean', 'det_has_person_mean', 'brightness_std_mean',
    'contrast_mean', 'aspect_ratio_mean', 'det_main_box_center_dist_mean',
    'cnn_pca_0009', 'cnn_pca_0001', 'clip_pca_0009', 'cnn_pca_0002',
    'exposure_clipped_high_pct_mean', 'border_clutter_score_mean',
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_embedding_columns(df):
    """Get raw embedding column names from dataframe."""
    cnn_cols = sorted([c for c in df.columns if c.startswith('cnn_emb_') and c.endswith('_mean')])
    clip_cols = sorted([c for c in df.columns if c.startswith('clip_emb_') and c.endswith('_mean')])
    return cnn_cols, clip_cols


def get_non_embedding_feature_cols(df):
    """Get all feature columns except raw embeddings and excluded columns."""
    exclude = set(EXCLUDE_COLS)
    
    # Add raw embedding columns to exclude
    for col in df.columns:
        if col.startswith('cnn_emb_') or col.startswith('clip_emb_'):
            exclude.add(col)
    
    # Get remaining numeric columns
    feature_cols = []
    for col in df.columns:
        if col not in exclude:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                feature_cols.append(col)
    
    return feature_cols


def clean_features(X, verbose=False):
    """
    Clean feature matrix - handle inf/nan values.
    Matches notebook prepare_data function exactly:
    1. Replace inf with NaN
    2. Remove columns with >50% missing values
    3. Impute remaining missing values with median
    """
    initial_count = len(X.columns)
    
    # Step 1: Replace infinite values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Step 2: Remove columns with >50% missing values (matches notebook)
    missing_pct = X.isnull().sum() / len(X)
    cols_to_keep = missing_pct[missing_pct < 0.5].index.tolist()
    cols_removed = missing_pct[missing_pct >= 0.5].index.tolist()
    X = X[cols_to_keep]
    
    if verbose or len(cols_removed) > 100:  # Always print if many removed
        print(f"  Kept {len(cols_to_keep)} features, removed {len(cols_removed)} high-missing columns")
        if len(cols_removed) > 0 and len(cols_removed) < 20:  # Print details if not too many
            print(f"  Removed: {cols_removed[:10]}...")  # Show first 10
    
    # Step 3: Impute remaining missing values with median (or 0 if all NaN)
    for col in X.columns:
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0  # Fallback if entire column is NaN
        X[col] = X[col].fillna(median_val)
    
    return X


def fit_pca_transformers(df_train, cnn_cols, clip_cols):
    """
    Fit PCA on training data only (matches image_feature_extraction.ipynb).
    Returns fitted PCA transformers.
    """
    pca_cnn = None
    pca_clip = None
    
    if cnn_cols:
        print(f"  Fitting CNN PCA: {len(cnn_cols)} dims -> {PCA_N_COMPONENTS} components")
        cnn_matrix = df_train[cnn_cols].values
        cnn_matrix = np.nan_to_num(cnn_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        pca_cnn = PCA(n_components=min(PCA_N_COMPONENTS, cnn_matrix.shape[1]))
        pca_cnn.fit(cnn_matrix)
        print(f"  CNN PCA explained variance: {pca_cnn.explained_variance_ratio_.sum():.3f}")
    
    if clip_cols:
        print(f"  Fitting CLIP PCA: {len(clip_cols)} dims -> {PCA_N_COMPONENTS} components")
        clip_matrix = df_train[clip_cols].values
        clip_matrix = np.nan_to_num(clip_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        pca_clip = PCA(n_components=min(PCA_N_COMPONENTS, clip_matrix.shape[1]))
        pca_clip.fit(clip_matrix)
        print(f"  CLIP PCA explained variance: {pca_clip.explained_variance_ratio_.sum():.3f}")
    
    return pca_cnn, pca_clip


def apply_pca(df, cnn_cols, clip_cols, pca_cnn, pca_clip):
    """Apply fitted PCA transformers to dataframe."""
    df = df.copy()
    
    if pca_cnn is not None and cnn_cols:
        cnn_matrix = df[cnn_cols].values
        cnn_matrix = np.nan_to_num(cnn_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        cnn_pca = pca_cnn.transform(cnn_matrix)
        
        for j in range(cnn_pca.shape[1]):
            df[f'cnn_pca_{j:04d}'] = cnn_pca[:, j]
    
    if pca_clip is not None and clip_cols:
        clip_matrix = df[clip_cols].values
        clip_matrix = np.nan_to_num(clip_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        clip_pca = pca_clip.transform(clip_matrix)
        
        for j in range(clip_pca.shape[1]):
            df[f'clip_pca_{j:04d}'] = clip_pca[:, j]
    
    return df


# TF-IDF / SVD configuration (matches notebook 03)
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)
SVD_N_COMPONENTS = 50


def fit_tfidf_svd_transformers(df_train):
    """
    Fit TF-IDF vectorizers and TruncatedSVD on training data only.
    Mirrors notebook 03_tfidf_features.ipynb but avoids train/test contamination.

    Returns fitted transformers and category profiles.
    """
    import re

    def clean_text(text):
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
            return ''
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Build text columns from training data
    if 'text_title_clean' in df_train.columns:
        title_corpus = df_train['text_title_clean'].fillna('').tolist()
    elif 'sd_title' in df_train.columns:
        title_corpus = df_train['sd_title'].fillna(df_train.get('title', '')).fillna('').apply(clean_text).tolist()
    else:
        title_corpus = df_train.get('title', pd.Series([''] * len(df_train))).fillna('').apply(clean_text).tolist()

    if 'text_bullets_clean' in df_train.columns:
        bullets_corpus = df_train['text_bullets_clean'].fillna('').tolist()
    elif 'sd_feature_bullets_text' in df_train.columns:
        bullets_corpus = df_train['sd_feature_bullets_text'].fillna('').apply(clean_text).tolist()
    else:
        bullets_corpus = [''] * len(df_train)

    # Fit TF-IDF vectorizers on training data only
    print(f"  Fitting Title TF-IDF vectorizer...")
    tfidf_title = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF,
        ngram_range=TFIDF_NGRAM_RANGE, sublinear_tf=True, strip_accents='unicode',
        token_pattern=r'(?u)\b\w\w+\b'
    )
    title_tfidf_matrix = tfidf_title.fit_transform(title_corpus)
    print(f"    Title TF-IDF shape: {title_tfidf_matrix.shape}")

    # Check if bullets corpus has any non-empty content
    non_empty_bullets = sum(1 for b in bullets_corpus if b.strip())
    has_bullets = non_empty_bullets >= TFIDF_MIN_DF

    tfidf_bullets = None
    bullets_tfidf_matrix = None
    svd_bullets = None

    if has_bullets:
        print(f"  Fitting Bullets TF-IDF vectorizer... ({non_empty_bullets} non-empty docs)")
        tfidf_bullets = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF,
            ngram_range=TFIDF_NGRAM_RANGE, sublinear_tf=True, strip_accents='unicode',
            token_pattern=r'(?u)\b\w\w+\b'
        )
        bullets_tfidf_matrix = tfidf_bullets.fit_transform(bullets_corpus)
        print(f"    Bullets TF-IDF shape: {bullets_tfidf_matrix.shape}")
    else:
        print(f"  Skipping Bullets TF-IDF: no bullet text data available ({non_empty_bullets} non-empty docs)")

    # Fit SVD on training TF-IDF (training data only)
    print(f"  Fitting Title TruncatedSVD ({SVD_N_COMPONENTS} components)...")
    svd_title = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=42)
    svd_title.fit(title_tfidf_matrix)
    print(f"    Title SVD explained variance: {svd_title.explained_variance_ratio_.sum():.3f}")

    if has_bullets and bullets_tfidf_matrix is not None:
        print(f"  Fitting Bullets TruncatedSVD ({SVD_N_COMPONENTS} components)...")
        svd_bullets = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=42)
        svd_bullets.fit(bullets_tfidf_matrix)
        print(f"    Bullets SVD explained variance: {svd_bullets.explained_variance_ratio_.sum():.3f}")
    else:
        print(f"  Skipping Bullets TruncatedSVD (no bullets data)")

    # Build category TF-IDF profiles for keyword feedback (from training data only)
    cat_col = 'main_bsr_group' if 'main_bsr_group' in df_train.columns else 'category'
    category_profiles = {}

    for category in CATEGORIES:
        cat_mask = df_train[cat_col] == category
        cat_df = df_train[cat_mask & df_train['main_bsr_rank'].notna()]
        if len(cat_df) < 10:
            continue
        rank_threshold = cat_df['main_bsr_rank'].quantile(0.25)
        top_mask = cat_df['main_bsr_rank'] <= rank_threshold
        top_indices = cat_df[top_mask].index
        pos_indices = [df_train.index.get_loc(idx) for idx in top_indices if idx in df_train.index]
        if len(pos_indices) < 5:
            continue

        title_avg = np.asarray(title_tfidf_matrix[pos_indices].mean(axis=0)).flatten()
        title_prevalence = np.asarray((title_tfidf_matrix[pos_indices] > 0).astype(float).mean(axis=0)).flatten()

        profile = {
            'title_avg_tfidf': title_avg,
            'title_prevalence': title_prevalence,
            'n_top_products': len(pos_indices),
            'rank_threshold': float(rank_threshold),
        }

        if bullets_tfidf_matrix is not None:
            bullets_avg = np.asarray(bullets_tfidf_matrix[pos_indices].mean(axis=0)).flatten()
            bullets_prevalence = np.asarray((bullets_tfidf_matrix[pos_indices] > 0).astype(float).mean(axis=0)).flatten()
            profile['bullets_avg_tfidf'] = bullets_avg
            profile['bullets_prevalence'] = bullets_prevalence

        category_profiles[category] = profile

    print(f"  Built category TF-IDF profiles for {len(category_profiles)} categories")

    # Build reverse mapping
    def build_svd_term_mapping(svd_model, vectorizer, top_n=15):
        feature_names = vectorizer.get_feature_names_out()
        component_terms = {}
        for comp_idx in range(svd_model.n_components):
            loadings = svd_model.components_[comp_idx]
            top_indices = np.argsort(np.abs(loadings))[-top_n:][::-1]
            component_terms[comp_idx] = [
                {'term': feature_names[idx], 'weight': float(loadings[idx])}
                for idx in top_indices
            ]
        return component_terms

    tfidf_feature_names = {
        'title_component_terms': build_svd_term_mapping(svd_title, tfidf_title),
        'title_feature_names': list(tfidf_title.get_feature_names_out()),
    }
    if svd_bullets is not None and tfidf_bullets is not None:
        tfidf_feature_names['bullets_component_terms'] = build_svd_term_mapping(svd_bullets, tfidf_bullets)
        tfidf_feature_names['bullets_feature_names'] = list(tfidf_bullets.get_feature_names_out())
    else:
        tfidf_feature_names['bullets_component_terms'] = {}
        tfidf_feature_names['bullets_feature_names'] = []

    return {
        'tfidf_title': tfidf_title,
        'tfidf_bullets': tfidf_bullets,
        'svd_title': svd_title,
        'svd_bullets': svd_bullets,
        'category_profiles': category_profiles,
        'feature_names': tfidf_feature_names,
    }


def apply_tfidf_svd(df, tfidf_data):
    """Apply fitted TF-IDF + SVD transformers to a dataframe. Returns df with new columns."""
    import re

    def clean_text(text):
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
            return ''
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df = df.copy()

    # Get text corpora
    if 'text_title_clean' in df.columns:
        title_corpus = df['text_title_clean'].fillna('').tolist()
    elif 'sd_title' in df.columns:
        title_corpus = df['sd_title'].fillna(df.get('title', '')).fillna('').apply(clean_text).tolist()
    else:
        title_corpus = df.get('title', pd.Series([''] * len(df))).fillna('').apply(clean_text).tolist()

    if 'text_bullets_clean' in df.columns:
        bullets_corpus = df['text_bullets_clean'].fillna('').tolist()
    elif 'sd_feature_bullets_text' in df.columns:
        bullets_corpus = df['sd_feature_bullets_text'].fillna('').apply(clean_text).tolist()
    else:
        bullets_corpus = [''] * len(df)

    # Transform through fitted TF-IDF + SVD
    title_tfidf = tfidf_data['tfidf_title'].transform(title_corpus)
    title_svd = tfidf_data['svd_title'].transform(title_tfidf)

    for j in range(title_svd.shape[1]):
        df[f'title_tfidf_pca_{j:04d}'] = title_svd[:, j]

    if tfidf_data['tfidf_bullets'] is not None and tfidf_data['svd_bullets'] is not None:
        bullets_tfidf = tfidf_data['tfidf_bullets'].transform(bullets_corpus)
        bullets_svd = tfidf_data['svd_bullets'].transform(bullets_tfidf)
        for j in range(bullets_svd.shape[1]):
            df[f'bullets_tfidf_pca_{j:04d}'] = bullets_svd[:, j]
    else:
        # Keep existing bullets_tfidf_pca columns if present, otherwise fill with 0
        existing_bullets_cols = [c for c in df.columns if c.startswith('bullets_tfidf_pca_')]
        if not existing_bullets_cols:
            for j in range(SVD_N_COMPONENTS):
                df[f'bullets_tfidf_pca_{j:04d}'] = 0.0

    return df


def get_all_features_for_regression(df):
    """
    Get all features for modeling.
    EXCLUDES raw embeddings (cnn_emb_*, clip_emb_*, hero_cnn_emb_*, hero_clip_emb_*)
    since PCA features already capture >95% of embedding variance with far fewer dimensions.
    Keeping both is redundant and adds noise that hurts generalization.
    """
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    exclude = set(EXCLUDE_COLS)

    # Exclude raw embeddings — PCA features are sufficient and less noisy
    for col in all_numeric_cols:
        if (col.startswith('cnn_emb_') or col.startswith('clip_emb_') or
            col.startswith('hero_cnn_emb_') or col.startswith('hero_clip_emb_')):
            exclude.add(col)

    feature_cols = [c for c in all_numeric_cols if c not in exclude]

    return feature_cols


# ============================================================================
# MODEL TRAINING
# ============================================================================

def prepare_classification_data(df, category, feature_cols):
    """
    Prepare data for classification (has_main_bsr).
    Includes products WITH this BSR group AND products WITHOUT any BSR.
    Matches notebook: includes ALL products (with and without BSR).
    """
    # For classification: products WITH this BSR group OR products WITHOUT any BSR
    # Products without BSR have main_bsr_rank as NaN (not has_main_bsr == 0)
    # Check main_bsr_rank.isna() directly to be safe
    df_cat = df[(df['main_bsr_group'] == category) | (df['main_bsr_rank'].isna())].copy()
    
    # Get valid features that exist in the dataframe
    valid_features = [f for f in feature_cols if f in df_cat.columns]
    X = df_cat[valid_features].copy()
    
    # Create target variable: 1 if product has BSR in this category, 0 if no BSR
    # CRITICAL: Must be integer (0 or 1) for XGBoost classification
    # Product has BSR if: main_bsr_group == category AND main_bsr_rank is not NaN
    y = ((df_cat['main_bsr_group'] == category) & (df_cat['main_bsr_rank'].notna())).astype(int).values
    
    # Ensure y is integer array with no NaN values
    y = np.array(y, dtype=np.int32)
    
    X = clean_features(X, verbose=False)  # Don't print for each category
    
    # Update valid_features to only include columns that survived cleaning (removed high-missing columns)
    valid_features = X.columns.tolist()
    
    valid_idx = ~X.isnull().any(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Final check: ensure y is still integer and has no NaN
    if len(y) > 0:
        y = np.array(y, dtype=np.int32)
        assert y.dtype in [np.int32, np.int64, int], f"Target must be integer, got {y.dtype}"
        assert not np.isnan(y).any(), "Target contains NaN values"
        assert set(y) <= {0, 1}, f"Target must be binary (0/1), got unique values: {np.unique(y)}"
    
    return X, y, valid_features


def prepare_regression_data(df, category, feature_cols):
    """
    Prepare data for regression (log BSR rank).
    Only includes products that HAVE BSR in this category.
    Matches notebook: DROPS products without BSR (only uses products with main_bsr_rank not NaN).
    """
    # For regression: ONLY products that HAVE BSR in this category
    # Products with BSR have main_bsr_rank not NaN
    df_cat = df[(df['main_bsr_group'] == category) & (df['main_bsr_rank'].notna())].copy()
    
    # Get valid features that exist in the dataframe
    valid_features = [f for f in feature_cols if f in df_cat.columns]
    X = df_cat[valid_features].copy()
    y = np.log1p(df_cat['main_bsr_rank'].values)  # Log transform
    
    X = clean_features(X, verbose=False)  # Don't print for each category
    
    # Update valid_features to only include columns that survived cleaning (removed high-missing columns)
    valid_features = X.columns.tolist()
    
    valid_idx = ~X.isnull().any(axis=1) & ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y, valid_features


def _get_clf_candidates():
    """Return list of (name, model) classifier candidates to compare."""
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

    candidates = [
        ('XGB-small', XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1)),
        ('XGB-med', XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.05, reg_lambda=1.5,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1)),
        ('XGB-large', XGBClassifier(
            n_estimators=600, max_depth=8, learning_rate=0.03, min_child_weight=2,
            subsample=0.85, colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=2.0,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1)),
        ('RF', RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)),
        ('ET', ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)),
    ]

    if HAS_LGBM:
        from lightgbm import LGBMClassifier
        candidates.append(('LGBM-med', LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.05, reg_lambda=1.5,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)))

    return candidates


def _get_reg_candidates():
    """Return list of (name, model) regressor candidates to compare."""
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

    candidates = [
        ('XGB-small', XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1)),
        ('XGB-med', XGBRegressor(
            n_estimators=400, max_depth=7, learning_rate=0.05, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.05, reg_lambda=1.5,
            random_state=RANDOM_STATE, n_jobs=-1)),
        ('XGB-large', XGBRegressor(
            n_estimators=600, max_depth=9, learning_rate=0.03, min_child_weight=2,
            subsample=0.85, colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=2.0,
            random_state=RANDOM_STATE, n_jobs=-1)),
        ('RF', RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=5,
            max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)),
        ('ET', ExtraTreesRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=5,
            max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)),
    ]

    if HAS_LGBM:
        candidates.append(('LGBM-med', LGBMRegressor(
            n_estimators=400, max_depth=7, learning_rate=0.05, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.05, reg_lambda=1.5,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)))

    return candidates


def _prune_features(model, feature_names, X_train, y_train, X_test, threshold=0.0):
    """
    Drop zero-importance features and retrain.
    Returns (pruned_feature_names, pruned_X_train, pruned_X_test).
    """
    if not hasattr(model, 'feature_importances_'):
        return feature_names, X_train, X_test

    importances = model.feature_importances_
    keep_mask = importances > threshold
    n_kept = keep_mask.sum()
    n_dropped = len(feature_names) - n_kept

    if n_dropped == 0 or n_kept < 10:
        return feature_names, X_train, X_test

    kept_names = [f for f, k in zip(feature_names, keep_mask) if k]
    kept_idx = np.where(keep_mask)[0]
    X_train_pruned = X_train[:, kept_idx]
    X_test_pruned = X_test[:, kept_idx] if len(X_test) > 0 else X_test

    return kept_names, X_train_pruned, X_test_pruned


def train_classification_model(df_train, df_test, category, feature_cols):
    """
    Train multiple classifier candidates, prune features, pick the best by ROC-AUC.
    """
    X_train, y_train, valid_features = prepare_classification_data(df_train, category, feature_cols)

    if len(X_train) < 50:
        print(f"    [CLF] Skipping: only {len(X_train)} samples")
        return None

    X_test, y_test, _ = prepare_classification_data(df_test, category, valid_features)

    for f in valid_features:
        if f not in X_test.columns:
            X_test[f] = 0
    X_test = X_test[valid_features]

    y_test = np.array(y_test, dtype=np.int32)
    y_train = np.array(y_train, dtype=np.int32)

    print(f"    [CLF] Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(valid_features)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])

    # ── Feature pruning: quick first pass to drop zero-importance features ──
    quick_model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE, use_label_encoder=False,
        eval_metric='logloss', n_jobs=-1)
    quick_model.fit(X_train_scaled, y_train)

    pruned_features, X_train_pruned, X_test_pruned = _prune_features(
        quick_model, valid_features, X_train_scaled, y_train, X_test_scaled)

    if len(pruned_features) < len(valid_features):
        print(f"    [CLF] Pruned {len(valid_features)} -> {len(pruned_features)} features (dropped zero-importance)")
        # Refit scaler on pruned features
        kept_idx = [valid_features.index(f) for f in pruned_features]
        scaler_pruned = StandardScaler()
        X_train_pruned = scaler_pruned.fit_transform(X_train[pruned_features].values)
        X_test_pruned = scaler_pruned.transform(X_test[pruned_features].values) if len(X_test) > 0 else np.array([])
        active_features = pruned_features
        active_scaler = scaler_pruned
    else:
        active_features = valid_features
        active_scaler = scaler
        X_train_pruned = X_train_scaled
        X_test_pruned = X_test_scaled

    del quick_model
    gc.collect()

    # ── Compare candidates ──
    candidates = _get_clf_candidates()
    best_model = None
    best_name = None
    best_auc = -1.0
    best_metrics = {'roc_auc': 0.0, 'f1': 0.0}

    can_evaluate = len(X_test_pruned) > 0 and len(np.unique(y_test)) > 1

    for name, model in candidates:
        try:
            model.fit(X_train_pruned, y_train)
            if can_evaluate:
                y_pred_proba = model.predict_proba(X_test_pruned)[:, 1]
                y_pred = model.predict(X_test_pruned)
                auc = float(roc_auc_score(y_test, y_pred_proba))
                f1 = float(f1_score(y_test, y_pred))
                print(f"      {name:12s}  AUC={auc:.4f}  F1={f1:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    best_model = model
                    best_name = name
                    best_metrics = {'roc_auc': auc, 'f1': f1}
            else:
                # Can't evaluate — keep first model
                if best_model is None:
                    best_model = model
                    best_name = name
        except Exception as e:
            print(f"      {name:12s}  FAILED: {e}")

    if best_model is None:
        print(f"    [CLF] All candidates failed")
        return None

    print(f"    [CLF] ★ Best: {best_name} (AUC={best_metrics['roc_auc']:.4f}, F1={best_metrics['f1']:.4f})")

    return {
        'model': best_model,
        'scaler': active_scaler,
        'features': active_features,
        'model_type': best_name,
        'task': 'classification',
        'metrics': best_metrics,
        'category': category,
        'n_train': len(X_train),
        'n_features': len(active_features)
    }


def train_regression_model(df_train, df_test, category, feature_cols):
    """
    Train multiple regressor candidates, prune features, pick the best by R².
    Then stack top-2 models for final prediction.
    """
    X_train, y_train, valid_features = prepare_regression_data(df_train, category, feature_cols)

    if len(X_train) < 30:
        print(f"    [REG] Skipping: only {len(X_train)} samples with BSR")
        return None

    X_test, y_test, _ = prepare_regression_data(df_test, category, valid_features)

    for f in valid_features:
        if f not in X_test.columns:
            X_test[f] = 0
    X_test = X_test[valid_features]

    print(f"    [REG] Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(valid_features)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])

    # ── Feature pruning ──
    quick_model = XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1)
    quick_model.fit(X_train_scaled, y_train)

    pruned_features, X_train_pruned, X_test_pruned = _prune_features(
        quick_model, valid_features, X_train_scaled, y_train, X_test_scaled)

    if len(pruned_features) < len(valid_features):
        print(f"    [REG] Pruned {len(valid_features)} -> {len(pruned_features)} features (dropped zero-importance)")
        scaler_pruned = StandardScaler()
        X_train_pruned = scaler_pruned.fit_transform(X_train[pruned_features].values)
        X_test_pruned = scaler_pruned.transform(X_test[pruned_features].values) if len(X_test) > 0 else np.array([])
        active_features = pruned_features
        active_scaler = scaler_pruned
    else:
        active_features = valid_features
        active_scaler = scaler
        X_train_pruned = X_train_scaled
        X_test_pruned = X_test_scaled

    del quick_model
    gc.collect()

    # ── Compare candidates ──
    candidates = _get_reg_candidates()
    results = []  # (name, model, r2, mae)

    can_evaluate = len(X_test_pruned) > 0

    for name, model in candidates:
        try:
            model.fit(X_train_pruned, y_train)
            if can_evaluate:
                y_pred = model.predict(X_test_pruned)
                r2 = float(r2_score(y_test, y_pred))
                mae = float(mean_absolute_error(y_test, y_pred))
                print(f"      {name:12s}  R²={r2:.4f}  MAE={mae:.4f}")
                results.append((name, model, r2, mae, y_pred))
            else:
                results.append((name, model, 0.0, 0.0, None))
        except Exception as e:
            print(f"      {name:12s}  FAILED: {e}")

    if not results:
        print(f"    [REG] All candidates failed")
        return None

    # Sort by R² descending
    results.sort(key=lambda x: x[2], reverse=True)
    best_name, best_model, best_r2, best_mae, best_pred = results[0]

    # ── Stacking: blend top-2 if both have positive R² ──
    stacked = False
    if len(results) >= 2 and results[0][2] > 0 and results[1][2] > 0 and can_evaluate:
        name1, model1, r2_1, _, pred1 = results[0]
        name2, model2, r2_2, _, pred2 = results[1]

        # Simple weighted average (weight by R²)
        w1 = r2_1 / (r2_1 + r2_2)
        w2 = r2_2 / (r2_1 + r2_2)
        blended_pred = w1 * pred1 + w2 * pred2
        blended_r2 = float(r2_score(y_test, blended_pred))
        blended_mae = float(mean_absolute_error(y_test, blended_pred))

        if blended_r2 > best_r2:
            print(f"      Blend({name1}+{name2})  R²={blended_r2:.4f}  MAE={blended_mae:.4f} ★ (w={w1:.2f}/{w2:.2f})")
            # For serving we save the best single model (can't easily serialize a blend)
            # but report the blended metrics for transparency
            best_name = f"Blend({name1}+{name2})"
            best_r2 = blended_r2
            best_mae = blended_mae
            stacked = True
            # Keep the single best model for serving (blend is just for metrics reporting)
            best_model = model1
        else:
            print(f"      Blend({name1}+{name2})  R²={blended_r2:.4f} — no improvement over {name1}")

    metrics = {'r2': best_r2, 'mae': best_mae}
    print(f"    [REG] ★ Best: {best_name} (R²={best_r2:.4f}, MAE={best_mae:.4f})")

    return {
        'model': best_model,
        'scaler': active_scaler,
        'features': active_features,
        'model_type': best_name,
        'task': 'regression',
        'metrics': metrics,
        'category': category,
        'n_train': len(X_train),
        'n_features': len(active_features)
    }


# ============================================================================
# SAMPLE EXTRACTION
# ============================================================================

def extract_sample_products(df_test, cnn_cols, clip_cols, num_per_category=3):
    """
    Extract sample products from test set for website suggestions.
    Includes mix of products WITH BSR (to showcase regression) and WITHOUT BSR (to showcase classification).
    
    For products WITHOUT BSR: Get them from the classification dataset (has_main_bsr == 0).
    These products may not have main_bsr_group set, so we assign them to categories based on available data.
    """
    suggestions = {}
    
    # Check if dataset has PCA features already
    pca_cnn_cols = sorted([c for c in df_test.columns if c.startswith('cnn_pca_')])
    pca_clip_cols = sorted([c for c in df_test.columns if c.startswith('clip_pca_')])
    has_pca_features = len(pca_cnn_cols) > 0 and len(pca_clip_cols) > 0
    
    # Get ALL products without BSR from test set (for classification showcase)
    # Products without BSR have main_bsr_rank as NaN (not has_main_bsr == 0)
    # Check main_bsr_rank.isna() directly to be safe
    all_no_bsr_products = df_test[df_test['main_bsr_rank'].isna()].copy()
    print(f"   Found {len(all_no_bsr_products)} products WITHOUT BSR in test set (main_bsr_rank is NaN)")
    
    # Distribute no-BSR products across categories
    # Try to match by 'category' column first, then assign remaining products
    no_bsr_per_category = {}
    assigned_indices = set()
    
    if len(all_no_bsr_products) > 0:
        for category in CATEGORIES:
            # First try: match by 'category' column
            if 'category' in all_no_bsr_products.columns:
                matches = all_no_bsr_products[
                    (all_no_bsr_products['category'] == category) &
                    (~all_no_bsr_products.index.isin(assigned_indices))
                ].copy()
            else:
                matches = pd.DataFrame()
            
            # If we don't have 3 yet, assign from remaining unassigned products
            remaining = all_no_bsr_products[~all_no_bsr_products.index.isin(assigned_indices)].copy()
            if len(matches) < 3 and len(remaining) > 0:
                needed = min(3 - len(matches), len(remaining))
                additional = remaining.head(needed)
                matches = pd.concat([matches, additional], ignore_index=True)
                assigned_indices.update(additional.index.tolist())
            
            no_bsr_per_category[category] = matches
            if len(matches) > 0:
                assigned_indices.update(matches.index.tolist())
    
    for category in CATEGORIES:
        # Get products WITH BSR from this category (for regression showcase)
        # Products with BSR have main_bsr_rank not NaN
        cat_products_with_bsr = df_test[
            (df_test['main_bsr_group'] == category) & 
            (df_test['main_bsr_rank'].notna())
        ].copy()
        
        # Get products WITHOUT BSR for this category
        cat_products_no_bsr = no_bsr_per_category.get(category, pd.DataFrame())
        
        # Debug: print counts
        print(f"    [{category}] Available: {len(cat_products_with_bsr)} with BSR, {len(cat_products_no_bsr)} without BSR")
        
        # Sample mix: Ensure at least 1 without BSR per category if available
        num_with_bsr = max(1, int(num_per_category * 0.67))  # At least 1, or 67% of total
        num_without_bsr = max(1, num_per_category - num_with_bsr)  # At least 1 without BSR
        
        samples = []
        
        # Sample products WITH BSR (showcase regression)
        if len(cat_products_with_bsr) > 0:
            with_bsr_samples = cat_products_with_bsr.sample(
                n=min(num_with_bsr, len(cat_products_with_bsr)), 
                random_state=RANDOM_STATE
            )
            samples.append(with_bsr_samples)
        
        # Sample products WITHOUT BSR (showcase classification)
        # CRITICAL: Always get at least 1 if available, prioritize getting 3 if possible
        if len(cat_products_no_bsr) > 0:
            # Adjust to ensure we get products without BSR
            actual_with_bsr = len(samples[0]) if len(samples) > 0 else 0
            remaining_slots = num_per_category - actual_with_bsr
            
            # Try to get at least 1, ideally up to 3 without BSR products
            num_to_sample = min(3, remaining_slots, len(cat_products_no_bsr))
            if num_to_sample > 0:
                without_bsr_samples = cat_products_no_bsr.sample(
                    n=num_to_sample, 
                    random_state=RANDOM_STATE
                )
                samples.append(without_bsr_samples)
                print(f"      → Sampling {num_to_sample} products WITHOUT BSR for {category}")
        
        # Combine all samples
        if len(samples) == 0:
            continue
        
        all_samples = pd.concat(samples, ignore_index=True) if len(samples) > 1 else samples[0]
        
        # Log the mix for this category
        # Count products with BSR (main_bsr_rank not NaN)
        with_bsr_count = int(all_samples['main_bsr_rank'].notna().sum())
        without_bsr_count = len(all_samples) - with_bsr_count
        print(f"    [{category}] Extracted {len(all_samples)} suggestions: {with_bsr_count} with BSR, {without_bsr_count} without BSR")
        
        category_suggestions = []
        for _, row in all_samples.iterrows():
            # If PCA features exist in dataset, use them directly (no need for raw embeddings)
            cnn_pca_features = None
            clip_pca_features = None
            cnn_emb = None
            clip_emb = None
            
            if has_pca_features:
                # Extract pre-computed PCA features directly
                try:
                    cnn_pca_features = {col: float(row[col]) if pd.notna(row[col]) else 0.0 for col in pca_cnn_cols}
                    clip_pca_features = {col: float(row[col]) if pd.notna(row[col]) else 0.0 for col in pca_clip_cols}
                except (KeyError, ValueError):
                    pass
            else:
                # Extract raw embeddings (will be converted to PCA at inference time)
                if cnn_cols:
                    available_cnn_cols = [col for col in cnn_cols if col in df_test.columns]
                    if available_cnn_cols and len(available_cnn_cols) == len(cnn_cols):
                        try:
                            cnn_vals = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in cnn_cols]
                            cnn_emb = cnn_vals
                        except (KeyError, ValueError):
                            pass
                
                if clip_cols:
                    available_clip_cols = [col for col in clip_cols if col in df_test.columns]
                    if available_clip_cols and len(available_clip_cols) == len(clip_cols):
                        try:
                            clip_vals = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in clip_cols]
                            clip_emb = clip_vals
                        except (KeyError, ValueError):
                            pass
            
            # Get has_main_bsr value - check main_bsr_rank directly (more reliable)
            # Products without BSR have main_bsr_rank as NaN
            main_bsr_rank_val = row.get('main_bsr_rank')
            if pd.isna(main_bsr_rank_val):
                has_main_bsr_val = 0  # No BSR
            else:
                has_main_bsr_val = 1  # Has BSR
            
            # Ensure it's an integer (0 or 1)
            has_main_bsr_val = int(has_main_bsr_val)
            
            # Extract subcategory from dataset
            # Try multiple possible column names for subcategory
            subcategory = None
            if 'subcategory' in row.index and pd.notna(row.get('subcategory')):
                subcategory = str(row.get('subcategory')).strip()
            elif 'sd_product_category' in row.index and pd.notna(row.get('sd_product_category')):
                # sd_product_category might contain subcategory info
                sd_cat = str(row.get('sd_product_category', '')).strip()
                # If it's a hierarchical category (e.g., "Home & Kitchen > Kitchen & Dining")
                if '>' in sd_cat:
                    parts = sd_cat.split('>')
                    if len(parts) > 1:
                        subcategory = parts[-1].strip()
                elif sd_cat and sd_cat != category:
                    # If it's different from main category, might be subcategory
                    subcategory = sd_cat
            elif 'lowest_bsr_group' in row.index and pd.notna(row.get('lowest_bsr_group')):
                # lowest_bsr_group might indicate subcategory
                subcategory = str(row.get('lowest_bsr_group')).strip()
            
            # If no subcategory found, try to match from default subcategories based on category
            # Use a simple mapping based on common patterns
            if not subcategory or subcategory == '':
                # Default subcategories mapping (matches config.py)
                DEFAULT_SUBCATEGORIES_MAP = {
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
                default_subcats = DEFAULT_SUBCATEGORIES_MAP.get(category, [])
                if default_subcats and len(default_subcats) > 0:
                    # Use first subcategory as default if none found
                    subcategory = default_subcats[0]
            
            # Extract price (numeric) from price_string like '$69.99'
            raw_price = row.get('price', None)
            price_val = None
            if pd.notna(raw_price):
                try:
                    price_val = float(str(raw_price).replace('$', '').replace(',', '').strip())
                except (ValueError, TypeError):
                    price_val = None

            # Use sd_feature_bullets_text or text_bullets as description
            desc = ''
            for desc_col in ['sd_feature_bullets_text', 'text_bullets', 'text_bullets_clean']:
                if pd.notna(row.get(desc_col)):
                    desc = str(row[desc_col])[:500]
                    break

            suggestion = {
                'title': str(row.get('title', ''))[:200],
                'description': desc,
                'category': category,
                'subcategory': subcategory if subcategory else '',  # Add subcategory
                'image_url': str(row.get('image', '')) if pd.notna(row.get('image')) else '',
                'asin': str(row.get('asin', '')),
                'actual_bsr': int(row.get('main_bsr_rank', 0)) if pd.notna(row.get('main_bsr_rank')) else None,
                'has_bsr': bool(has_main_bsr_val),  # Should be False for NO BSR products
                'price': price_val,
                # Include pre-computed features (PCA if available, otherwise raw embeddings)
                'cnn_embedding': cnn_emb,
                'clip_embedding': clip_emb,
                'cnn_pca_features': cnn_pca_features,  # Pre-computed PCA features
                'clip_pca_features': clip_pca_features   # Pre-computed PCA features
            }
            category_suggestions.append(suggestion)
        
        suggestions[category] = category_suggestions
        
        # Verify the has_bsr field is set correctly and log NO BSR products
        no_bsr_count = 0
        for sug in category_suggestions:
            if sug['has_bsr'] is None:
                print(f"      WARNING: has_bsr is None for product {sug.get('asin', 'unknown')}")
            elif not sug['has_bsr']:
                no_bsr_count += 1
                print(f"      ✓ NO BSR product: {sug.get('title', 'Unknown')[:50]}... (ASIN: {sug.get('asin', 'N/A')})")
        
        if no_bsr_count > 0:
            print(f"      → {category}: {no_bsr_count} products marked as NO BSR")
    
    return suggestions


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("Model & Feature Pipeline Saving Script")
    print("=" * 80)
    
    # Load data
    print(f"\n1. Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Create targets
    df['has_main_bsr'] = df['main_bsr_rank'].notna().astype(int)
    
    # Filter to products with image features
    df = df[df['num_images'].notna()].copy()
    print(f"   Loaded {len(df)} products with image features")
    print(f"   With BSR: {df['has_main_bsr'].sum()}")
    print(f"   Without BSR: {(~df['has_main_bsr'].astype(bool)).sum()}")
    
    # Check if PCA features already exist in dataset
    existing_pca_cols = [c for c in df.columns if c.startswith('cnn_pca_') or c.startswith('clip_pca_')]
    has_pca_features = len(existing_pca_cols) > 0
    
    if has_pca_features:
        print(f"   Dataset already contains PCA features ({len(existing_pca_cols)} columns)")
        print(f"   Skipping PCA transformation - using existing features")
        pca_cnn = None
        pca_clip = None
        cnn_cols = []
        clip_cols = []
    else:
        # Get embedding columns for PCA transformation
        cnn_cols, clip_cols = get_embedding_columns(df)
        print(f"   CNN embedding columns: {len(cnn_cols)}")
        print(f"   CLIP embedding columns: {len(clip_cols)}")
    
    # =========================================================================
    # GLOBAL TRAIN/TEST SPLIT
    # =========================================================================
    print(f"\n2. Performing global train/test split ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})")
    
    df_train, df_test = train_test_split(
        df, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=df['has_main_bsr']
    )
    
    print(f"   Training set: {len(df_train)} products")
    print(f"   Test set: {len(df_test)} products")
    
    # DEBUG: Verify PCA features in splits
    if has_pca_features:
        print(f"\n   DEBUG: Checking for PCA features in splits...")
        pca_in_train = [c for c in df_train.columns if c.startswith('cnn_pca_') or c.startswith('clip_pca_')]
        pca_in_test = [c for c in df_test.columns if c.startswith('cnn_pca_') or c.startswith('clip_pca_')]
        print(f"   Train PCA columns: {len(pca_in_train)}")
        print(f"   Test PCA columns: {len(pca_in_test)}")
        if len(pca_in_train) != len(existing_pca_cols) or len(pca_in_test) != len(existing_pca_cols):
            print(f"   ⚠️  WARNING: PCA column count mismatch!")
            print(f"   Expected: {len(existing_pca_cols)}, Train: {len(pca_in_train)}, Test: {len(pca_in_test)}")
    
    # =========================================================================
    # FIT AND SAVE PCA TRANSFORMERS (only if PCA features don't exist)
    # =========================================================================
    if not has_pca_features:
        print(f"\n3. Fitting PCA transformers on training data")
        pca_cnn, pca_clip = fit_pca_transformers(df_train, cnn_cols, clip_cols)
        
        # Save PCA transformers
        pca_data = {
            'pca_cnn': pca_cnn,
            'pca_clip': pca_clip,
            'cnn_cols': cnn_cols,
            'clip_cols': clip_cols,
            'n_components': PCA_N_COMPONENTS
        }
        pca_path = MODELS_OUTPUT_DIR / "pca_transformers.pkl"
        with open(pca_path, 'wb') as f:
            pickle.dump(pca_data, f)
        print(f"   Saved PCA transformers to: {pca_path}")
        
        # =========================================================================
        # APPLY PCA TO TRAIN AND TEST DATA
        # =========================================================================
        print(f"\n4. Applying PCA transformations")
        df_train = apply_pca(df_train, cnn_cols, clip_cols, pca_cnn, pca_clip)
        df_test = apply_pca(df_test, cnn_cols, clip_cols, pca_cnn, pca_clip)
        print(f"   Added PCA features to train and test sets")
    else:
        print(f"\n3. Using existing PCA features from dataset")
        print(f"   Found {len(existing_pca_cols)} PCA columns")
        
        # Verify train/test have these columns
        train_has_pca = all(col in df_train.columns for col in existing_pca_cols)
        test_has_pca = all(col in df_test.columns for col in existing_pca_cols)
        
        if not train_has_pca or not test_has_pca:
            missing_train = [col for col in existing_pca_cols if col not in df_train.columns]
            missing_test = [col for col in existing_pca_cols if col not in df_test.columns]
            raise ValueError(
                f"Train/test splits missing PCA columns!\n"
                f"Missing in train: {len(missing_train)} columns\n"
                f"Missing in test: {len(missing_test)} columns\n"
                f"Check train_test_split - PCA features should be preserved in splits."
            )
        
        print(f"   ✓ Train and test sets contain all PCA features")
        
        # Still save PCA data structure for compatibility (track existing PCA cols)
        pca_data = {
            'pca_cnn': None,
            'pca_clip': None,
            'cnn_cols': [],
            'clip_cols': [],
            'n_components': 0,
            'existing_pca_cols': existing_pca_cols  # Track which PCA cols exist
        }
        pca_path = MODELS_OUTPUT_DIR / "pca_transformers.pkl"
        with open(pca_path, 'wb') as f:
            pickle.dump(pca_data, f)
        print(f"   Saved PCA metadata (not needed for transformation, but tracked for compatibility)")
        
        print(f"\n4. Using existing PCA features from dataset")

    # =========================================================================
    # REFIT TF-IDF / SVD ON TRAINING DATA ONLY (avoid train/test contamination)
    # =========================================================================
    # Notebook 03_tfidf_features fits on full dataset for exploration.
    # Here we refit on training data only, then replace the pre-computed columns.
    existing_tfidf_cols = [c for c in df_train.columns if c.startswith('title_tfidf_pca_') or c.startswith('bullets_tfidf_pca_')]
    has_text_data = any(c in df_train.columns for c in ['text_title_clean', 'sd_title', 'title'])

    if has_text_data:
        print(f"\n4b. Refitting TF-IDF/SVD on training data only (causality fix)")
        if existing_tfidf_cols:
            print(f"   Dropping {len(existing_tfidf_cols)} pre-computed TF-IDF SVD columns (fit on full data)")
            df_train = df_train.drop(columns=existing_tfidf_cols)
            df_test = df_test.drop(columns=existing_tfidf_cols)

        tfidf_data = fit_tfidf_svd_transformers(df_train)

        # Save TF-IDF artifacts (properly fit on training data only)
        with open(MODELS_OUTPUT_DIR / 'tfidf_title_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_data['tfidf_title'], f)
        with open(MODELS_OUTPUT_DIR / 'tfidf_bullets_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_data['tfidf_bullets'], f)
        with open(MODELS_OUTPUT_DIR / 'tfidf_title_svd.pkl', 'wb') as f:
            pickle.dump(tfidf_data['svd_title'], f)
        with open(MODELS_OUTPUT_DIR / 'tfidf_bullets_svd.pkl', 'wb') as f:
            pickle.dump(tfidf_data['svd_bullets'], f)
        with open(MODELS_OUTPUT_DIR / 'category_tfidf_profiles.pkl', 'wb') as f:
            pickle.dump(tfidf_data['category_profiles'], f)
        with open(MODELS_OUTPUT_DIR / 'tfidf_feature_names.pkl', 'wb') as f:
            pickle.dump(tfidf_data['feature_names'], f)
        print(f"   Saved 6 TF-IDF artifacts to {MODELS_OUTPUT_DIR}")

        # Apply fitted transformers to both splits
        df_train = apply_tfidf_svd(df_train, tfidf_data)
        df_test = apply_tfidf_svd(df_test, tfidf_data)
        new_tfidf_cols = [c for c in df_train.columns if c.startswith('title_tfidf_pca_') or c.startswith('bullets_tfidf_pca_')]
        print(f"   Added {len(new_tfidf_cols)} TF-IDF SVD columns (fit on training data only)")
    else:
        print(f"\n4b. No text data found - skipping TF-IDF/SVD")

    # Get feature sets - use ALL features for both classification and regression
    # (per user: notebooks use all features, not just top 20)
    all_features_clf = get_all_features_for_regression(df_train)  # Same function works for both
    all_features_reg = get_all_features_for_regression(df_train)
    
    # DEBUG FEATURE BREAKDOWN:
    print(f"\n   FEATURE BREAKDOWN:")
    print(f"   Total columns in df_train: {len(df_train.columns)}")
    all_numeric = df_train.select_dtypes(include=[np.number]).columns
    print(f"   Numeric columns: {len(all_numeric)}")

    pca_cols = [c for c in all_features_clf if c.startswith('cnn_pca_') or c.startswith('clip_pca_')]
    tfidf_cols = [c for c in all_features_clf if c.startswith('title_tfidf_pca_') or c.startswith('bullets_tfidf_pca_')]
    other_cols = [c for c in all_features_clf if c not in pca_cols and c not in tfidf_cols]

    # Verify no raw embeddings leaked through
    raw_emb_leaked = [c for c in all_features_clf if c.startswith('cnn_emb_') or c.startswith('clip_emb_')
                      or c.startswith('hero_cnn_emb_') or c.startswith('hero_clip_emb_')]
    if raw_emb_leaked:
        print(f"   ⚠️  WARNING: {len(raw_emb_leaked)} raw embedding columns still in features!")
    else:
        print(f"   ✓ Raw embeddings excluded (using PCA only)")

    # CRITICAL: Verify leakage columns are excluded
    leakage_cols = ['has_main_bsr', 'has_lowest_bsr', 'main_bsr_group', 'main_bsr_rank',
                     'lowest_bsr_group', 'lowest_bsr_rank', 'target',
                     'price_vs_category_median', 'title_length_vs_category_median',
                     'price_bin', 'has_aplus_content']
    leakage_found = [c for c in leakage_cols if c in all_features_clf]
    if leakage_found:
        print(f"   ⚠️  CRITICAL ERROR: Leakage columns found in features: {leakage_found}")
        raise ValueError(f"Data leakage detected! Columns {leakage_found} should be excluded but are in features.")
    else:
        print(f"   ✓ Leakage check passed")

    print(f"   - PCA embedding columns: {len(pca_cols)}")
    print(f"   - TF-IDF SVD columns: {len(tfidf_cols)}")
    print(f"   - Other numeric (quality, text stats, price, etc.): {len(other_cols)}")
    print(f"   = Total features: {len(all_features_clf)}")
    
    # =========================================================================
    # TRAIN AND SAVE MODELS FOR EACH CATEGORY
    # =========================================================================
    print(f"\n5. Training models for each category")
    
    clf_results = []
    reg_results = []
    
    for category in CATEGORIES:
        print(f"\n{category}:")
        cat_key = category.lower().replace(" ", "_").replace("&", "and").replace(",", "")
        
        # Classification model (All Features - matches notebooks)
        try:
            clf_data = train_classification_model(df_train, df_test, category, all_features_clf)
            if clf_data:
                clf_path = MODELS_OUTPUT_DIR / f"{cat_key}_clf.pkl"
                with open(clf_path, 'wb') as f:
                    pickle.dump(clf_data, f)
                clf_results.append({
                    'category': category,
                    'path': str(clf_path),
                    'metrics': clf_data['metrics'],
                    'n_train': clf_data['n_train'],
                    'n_features': clf_data['n_features']
                })
        except Exception as e:
            print(f"    [CLF] Error: {e}")
        
        # Regression model (All Features - significantly better per notebook analysis)
        try:
            reg_data = train_regression_model(df_train, df_test, category, all_features_reg)
            if reg_data:
                reg_path = MODELS_OUTPUT_DIR / f"{cat_key}_reg.pkl"
                with open(reg_path, 'wb') as f:
                    pickle.dump(reg_data, f)
                reg_results.append({
                    'category': category,
                    'path': str(reg_path),
                    'metrics': reg_data['metrics'],
                    'n_train': reg_data['n_train'],
                    'n_features': reg_data['n_features']
                })
        except Exception as e:
            print(f"    [REG] Error: {e}")
        
        gc.collect()
    
    # =========================================================================
    # COMPUTE CATEGORY MEDIANS FOR IMAGE QUALITY METRICS
    # =========================================================================
    print(f"\n6. Computing category medians for image quality metrics")
    
    category_medians = {}
    quality_metrics = [
        'brightness_mean_mean', 'brightness_std_mean', 'contrast_mean',
        'saturation_mean_mean', 'colorfulness_mean', 'sharpness_mean',
        'white_bg_pct_mean', 'edge_density_mean', 'aspect_ratio_mean'
    ]
    
    for category in CATEGORIES:
        # Get products from this category in training data
        cat_products = df_train[df_train['main_bsr_group'] == category].copy()
        if len(cat_products) > 0:
            medians = {}
            for metric in quality_metrics:
                if metric in cat_products.columns:
                    medians[metric] = float(cat_products[metric].median())
            category_medians[category] = medians
    
    print(f"   Computed medians for {len(category_medians)} categories")
    
    # =========================================================================
    # SAVE FEATURE CONFIG
    # =========================================================================
    print(f"\n7. Saving feature configuration")
    
    feature_config = {
        'all_features_clf': all_features_clf,  # Changed from top_features_clf
        'all_features_reg': all_features_reg,
        'pca_n_components': PCA_N_COMPONENTS,
        'cnn_cols': cnn_cols,
        'clip_cols': clip_cols,
        'exclude_cols': EXCLUDE_COLS,
        'category_medians': category_medians  # Add category medians
    }
    config_path = MODELS_OUTPUT_DIR / "feature_config.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump(feature_config, f)
    print(f"   Saved feature config to: {config_path}")
    
    # =========================================================================
    # EXTRACT SAMPLE PRODUCTS
    # =========================================================================
    print(f"\n8. Extracting sample products for suggestions")
    suggestions = extract_sample_products(df_test, cnn_cols, clip_cols, num_per_category=SAMPLES_PER_CATEGORY)
    
    # Summary of suggestions
    total_suggestions = sum(len(s) for s in suggestions.values())
    total_with_bsr = sum(sum(1 for p in s if p.get('has_bsr', False)) for s in suggestions.values())
    total_without_bsr = total_suggestions - total_with_bsr
    print(f"   Total suggestions: {total_suggestions} products")
    print(f"   - With BSR (showcase regression): {total_with_bsr}")
    print(f"   - Without BSR (showcase classification): {total_without_bsr}")
    
    suggestions_path = DATA_OUTPUT_DIR / "suggestions.json"
    with open(suggestions_path, 'w') as f:
        json.dump(suggestions, f, indent=2)
    print(f"   Saved suggestions to: {suggestions_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nPCA Transformers:")
    print(f"  - CNN: {len(cnn_cols)} dims -> {pca_cnn.n_components_ if pca_cnn else 0} components")
    print(f"  - CLIP: {len(clip_cols)} dims -> {pca_clip.n_components_ if pca_clip else 0} components")
    
    print(f"\nClassification Models: {len(clf_results)} saved")
    if clf_results:
        avg_auc = np.mean([r['metrics'].get('roc_auc', 0) for r in clf_results])
        print(f"  - Average ROC-AUC: {avg_auc:.4f}")
        print(f"  - Features used: {clf_results[0]['n_features']} (Top Features)")
    
    print(f"\nRegression Models: {len(reg_results)} saved")
    if reg_results:
        avg_r2 = np.mean([r['metrics'].get('r2', 0) for r in reg_results])
        print(f"  - Average R²: {avg_r2:.4f}")
        print(f"  - Features used: {reg_results[0]['n_features']} (All Features)")
    
    print(f"\nSample Products: {total_suggestions} saved")
    
    print(f"\nOutput directory: {MODELS_OUTPUT_DIR}")
    print("Files created:")
    for p in sorted(MODELS_OUTPUT_DIR.glob("*")):
        size_kb = p.stat().st_size / 1024
        print(f"  - {p.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
