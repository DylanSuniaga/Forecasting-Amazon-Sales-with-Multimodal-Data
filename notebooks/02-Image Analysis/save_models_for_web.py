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
from sklearn.decomposition import PCA
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
EXCLUDE_COLS = [
    'Unnamed: 0.1', 'Unnamed: 0', 'asin', 'category', 'query', 'page',
    'source_section', 'type', 'title', 'image', 'url', 'optimized_url',
    'delivery', 'price_string', 'price_symbol', 'colors', 'location',
    'search_message', 'sd_feature_bullets_text', 'sd_title', 'sd_parent_asin',
    'sd_price_symbol', 'sd_availability_status', 'sd_product_category',
    'sd_category_id', 'sd_ratings_distribution', 'sd_customer_sentiments',
    'sd_error', 'sd_errors', 'sd_best_sellers_rank', 'certification', 'coupon_text',
    'fetched_at_unix',
    # Targets
    'main_bsr_group', 'main_bsr_rank', 'lowest_bsr_rank', 'lowest_bsr_group',
    'has_main_bsr', 'has_lowest_bsr', 'log_main_bsr_rank',
    # Review/rating features (potential leakage)
    'stars', 'total_reviews', 'sd_average_rating', 'sd_total_reviews',
    'sd_stars', 'sd_ratings_count',
    'sd_rating_pct_1', 'sd_rating_pct_2', 'sd_rating_pct_3', 'sd_rating_pct_4', 'sd_rating_pct_5',
    'sd_sent_count_POSITIVE', 'sd_sent_count_MIXED', 'sd_sent_count_NEGATIVE',
    # Position features (would be leakage)
    'organic_position', 'absolute_position',
    # Sales features (leakage)
    'number_of_people_bought', 'sd_number_bought_past_month',
    'is_best_seller', 'is_amazon_choice',
    # Other leakage
    'limited_time_deal', 'deal_of_the_day', 'sponsored',
    'availability_quantity', 'sd_is_frequently_returned',
    'sd_previous_price'
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


def clean_features(X):
    """Clean feature matrix - handle inf/nan values."""
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0
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


def get_all_features_for_regression(df):
    """
    Get all features for regression (better performance per notebook analysis).
    Excludes raw embeddings but includes PCA features.
    """
    exclude = set(EXCLUDE_COLS)
    
    # Exclude raw embedding columns (but keep PCA columns)
    for col in df.columns:
        if col.startswith('cnn_emb_') or col.startswith('clip_emb_'):
            exclude.add(col)
    
    feature_cols = []
    for col in df.columns:
        if col not in exclude:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                feature_cols.append(col)
    
    return feature_cols


# ============================================================================
# MODEL TRAINING
# ============================================================================

def prepare_classification_data(df, category, feature_cols):
    """
    Prepare data for classification (has_main_bsr).
    Includes products WITH this BSR group AND products WITHOUT any BSR.
    """
    df_cat = df[(df['main_bsr_group'] == category) | (df['has_main_bsr'] == 0)].copy()
    
    valid_features = [f for f in feature_cols if f in df_cat.columns]
    X = df_cat[valid_features].copy()
    y = df_cat['has_main_bsr'].values
    
    X = clean_features(X)
    
    valid_idx = ~X.isnull().any(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y, valid_features


def prepare_regression_data(df, category, feature_cols):
    """
    Prepare data for regression (log BSR rank).
    Only includes products that HAVE BSR in this category.
    """
    df_cat = df[(df['main_bsr_group'] == category) & (df['has_main_bsr'] == 1)].copy()
    
    valid_features = [f for f in feature_cols if f in df_cat.columns]
    X = df_cat[valid_features].copy()
    y = np.log1p(df_cat['main_bsr_rank'].values)  # Log transform
    
    X = clean_features(X)
    
    valid_idx = ~X.isnull().any(axis=1) & ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y, valid_features


def train_classification_model(df_train, df_test, category, feature_cols):
    """Train XGBoost classifier (best per notebook analysis)."""
    X_train, y_train, valid_features = prepare_classification_data(df_train, category, feature_cols)
    
    if len(X_train) < 50:
        print(f"    [CLF] Skipping: only {len(X_train)} samples")
        return None
    
    X_test, y_test, _ = prepare_regression_data(df_test, category, valid_features)
    if len(X_test) == 0:
        X_test, y_test, _ = prepare_classification_data(df_test, category, valid_features)
    
    # Ensure same features
    for f in valid_features:
        if f not in X_test.columns:
            X_test[f] = 0
    X_test = X_test[valid_features]
    
    print(f"    [CLF] Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(valid_features)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])
    
    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    metrics = {}
    if len(X_test) > 0 and len(np.unique(y_test)) > 1:
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
        metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
        metrics['f1'] = float(f1_score(y_test, y_pred))
        print(f"    [CLF] Test ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
    else:
        metrics = {'roc_auc': 0.0, 'f1': 0.0}
        print(f"    [CLF] No valid test samples")
    
    return {
        'model': model,
        'scaler': scaler,
        'features': valid_features,
        'model_type': 'XGBoost',
        'task': 'classification',
        'metrics': metrics,
        'category': category,
        'n_train': len(X_train),
        'n_features': len(valid_features)
    }


def train_regression_model(df_train, df_test, category, feature_cols):
    """
    Train regression model (XGBoost or LightGBM - both perform well per notebook analysis).
    Uses ALL features for better performance.
    """
    X_train, y_train, valid_features = prepare_regression_data(df_train, category, feature_cols)
    
    if len(X_train) < 30:
        print(f"    [REG] Skipping: only {len(X_train)} samples with BSR")
        return None
    
    X_test, y_test, _ = prepare_regression_data(df_test, category, valid_features)
    
    # Ensure same features
    for f in valid_features:
        if f not in X_test.columns:
            X_test[f] = 0
    X_test = X_test[valid_features]
    
    print(f"    [REG] Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(valid_features)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])
    
    # Train model
    if HAS_LGBM:
        model = LGBMRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        model_type = 'LightGBM'
    else:
        model = XGBRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model_type = 'XGBoost'
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    metrics = {}
    if len(X_test) > 0:
        y_pred = model.predict(X_test_scaled)
        metrics['r2'] = float(r2_score(y_test, y_pred))
        metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
        print(f"    [REG] Test R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f} (log scale)")
    else:
        metrics = {'r2': 0.0, 'mae': 0.0}
        print(f"    [REG] No test samples")
    
    return {
        'model': model,
        'scaler': scaler,
        'features': valid_features,
        'model_type': model_type,
        'task': 'regression',
        'metrics': metrics,
        'category': category,
        'n_train': len(X_train),
        'n_features': len(valid_features)
    }


# ============================================================================
# SAMPLE EXTRACTION
# ============================================================================

def extract_sample_products(df_test, num_per_category=3):
    """Extract sample products from test set for website suggestions."""
    suggestions = {}
    
    for category in CATEGORIES:
        # Get products from this category that have BSR (successful products)
        cat_products = df_test[
            (df_test['main_bsr_group'] == category) & 
            (df_test['has_main_bsr'] == 1)
        ].copy()
        
        if len(cat_products) == 0:
            continue
        
        # Sample products
        samples = cat_products.sample(n=min(num_per_category, len(cat_products)), random_state=RANDOM_STATE)
        
        category_suggestions = []
        for _, row in samples.iterrows():
            suggestion = {
                'title': str(row.get('title', ''))[:200],
                'description': str(row.get('sd_feature_bullets_text', ''))[:500] if pd.notna(row.get('sd_feature_bullets_text')) else '',
                'category': category,
                'image_url': str(row.get('image', '')) if pd.notna(row.get('image')) else '',
                'asin': str(row.get('asin', '')),
                'actual_bsr': int(row.get('main_bsr_rank', 0)) if pd.notna(row.get('main_bsr_rank')) else None,
                'has_bsr': bool(row.get('has_main_bsr', 0))
            }
            category_suggestions.append(suggestion)
        
        suggestions[category] = category_suggestions
    
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
    
    # Get embedding columns
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
    
    # =========================================================================
    # FIT AND SAVE PCA TRANSFORMERS (on training data only!)
    # =========================================================================
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
    
    # Get feature sets
    top_features_clf = [f for f in TOP_FEATURES_CLF if f in df_train.columns]
    all_features_reg = get_all_features_for_regression(df_train)
    
    print(f"   Classification features (Top 20): {len(top_features_clf)}")
    print(f"   Regression features (All): {len(all_features_reg)}")
    
    # =========================================================================
    # TRAIN AND SAVE MODELS FOR EACH CATEGORY
    # =========================================================================
    print(f"\n5. Training models for each category")
    
    clf_results = []
    reg_results = []
    
    for category in CATEGORIES:
        print(f"\n{category}:")
        cat_key = category.lower().replace(" ", "_").replace("&", "and").replace(",", "")
        
        # Classification model (Top Features - sufficient per notebook analysis)
        try:
            clf_data = train_classification_model(df_train, df_test, category, top_features_clf)
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
    # SAVE FEATURE CONFIG
    # =========================================================================
    print(f"\n6. Saving feature configuration")
    
    feature_config = {
        'top_features_clf': top_features_clf,
        'all_features_reg': all_features_reg,
        'pca_n_components': PCA_N_COMPONENTS,
        'cnn_cols': cnn_cols,
        'clip_cols': clip_cols,
        'exclude_cols': EXCLUDE_COLS
    }
    config_path = MODELS_OUTPUT_DIR / "feature_config.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump(feature_config, f)
    print(f"   Saved feature config to: {config_path}")
    
    # =========================================================================
    # EXTRACT SAMPLE PRODUCTS
    # =========================================================================
    print(f"\n7. Extracting sample products from test set")
    suggestions = extract_sample_products(df_test, num_per_category=SAMPLES_PER_CATEGORY)
    
    suggestions_path = DATA_OUTPUT_DIR / "suggestions.json"
    with open(suggestions_path, 'w') as f:
        json.dump(suggestions, f, indent=2)
    
    total_suggestions = sum(len(v) for v in suggestions.values())
    print(f"   Saved {total_suggestions} sample products to: {suggestions_path}")
    
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
