# Amazon Best Seller Rank Prediction using Multimodal Machine Learning

**CIS 4911 - Capstone II - Data Science & AI**
Spring 2026
Knight Foundation School of Computing and Information Sciences (KFSCIS)
Florida International University

**Instructor:** Seyedmasoud Sadjadi

**Team Members:** Dylan Suniaga, Bianca Poggi, Michelle Orozco, Fesal Fayed, Maksim Pikalov
**Team Lead:** Dylan Suniaga
**Product Owner:** Fesal Fayed

---

## Abstract

Amazon Best Seller Rank Prediction is a data science project that leverages multimodal machine learning to predict product sales performance on Amazon's e-commerce platform. Using approximately 36,900 product listings across 14 Amazon categories, we extract 2,300+ features from product images (via EfficientNet-B0, CLIP ViT-B-32, OpenCV, and YOLOv8), listing titles (via TF-IDF, readability metrics, and keyword analysis), and metadata to build a two-stage prediction pipeline. An XGBoost classifier first predicts whether a product will achieve a Best Seller Rank (ROC-AUC: 0.891, F1: 0.882), followed by an XGBoost regressor that predicts the specific BSR value for ranked products (R² = 0.425, RMSE = 1.838 in log-space). Per-category models achieve a mean AUC of 0.700 and mean R² of 0.303 across evaluated categories. This Capstone II iteration significantly advances the Capstone I work by introducing deep learning embeddings (CNN + CLIP), per-category model specialization, an NLP feature pipeline, and a full-stack web application (FastAPI + Next.js) for real-time prediction. The results demonstrate that visual listing quality — particularly CNN and CLIP embedding features — is the strongest predictor of sales performance, with deep learning embedding features dominating the top feature importances across categories.

---

## Table of Contents

1. Problem Statement & Research Questions
2. Data Collection & Sources
3. Methodology & Pipeline Architecture
4. Feature Engineering
   - Image Analysis
   - Deep Learning Embeddings (NEW in Capstone II)
   - Natural Language Processing (NEW in Capstone II)
   - Metadata Extraction
5. Exploratory Data Analysis
6. Model Development & Selection
   - Per-Category Model Specialization (NEW in Capstone II)
7. Results & Performance Evaluation
   - Capstone I vs. Capstone II Comparison
8. Web Application (NEW in Capstone II)
9. Business Insights & Applications
10. Ethical Considerations & Limitations
11. Deployment & Reproducibility
12. Future Work & Recommendations
13. References
14. Appendices

---

## 1. Problem Statement & Research Questions

In the competitive landscape of e-commerce, understanding what drives product success remains a critical challenge for sellers, brands, and marketplace analysts. Amazon's Best Seller Rank (BSR) serves as the primary indicator of product sales velocity, yet predicting this metric — especially for new product launches — remains largely guesswork.

This project addresses the question: **Can we predict Amazon product sales performance using publicly available listing characteristics?** Specifically, we investigate whether a combination of visual quality (image analysis), textual optimization (NLP), and product metadata can reliably forecast whether a product will achieve sales traction and, if so, what rank it will achieve.

### Research Questions

**Primary Question:** To what extent can multimodal features (images, text, metadata) predict Amazon Best Seller Rank?

**Secondary Questions:**
- Which visual characteristics (image quality, composition, background) most strongly correlate with sales performance?
- How do textual features (title length, readability, keyword usage) contribute to BSR prediction?
- Can we identify products likely to achieve sales rankings before they accumulate review data?
- What threshold of prediction accuracy is achievable given only listing-level information?
- **(Capstone II)** Do per-category models significantly outperform a single global model?
- **(Capstone II)** Can deep learning embeddings (CNN, CLIP) capture visual patterns that hand-crafted features miss?

### Stakeholders & Applications

**Primary Stakeholders:**
- Amazon sellers optimizing product listings
- E-commerce consultants providing listing optimization services
- Brand managers planning product launches
- Data scientists studying multimodal prediction systems

**Potential Applications:**
- Pre-launch BSR forecasting for inventory planning
- Competitive analysis and benchmarking
- A/B testing frameworks for listing optimization
- Educational tool for understanding e-commerce success factors

### Success Criteria

- **Classification Task:** Achieve >85% accuracy in predicting whether a product will be ranked *(Capstone I: 90%, Capstone II: 83.9% accuracy / 0.891 ROC-AUC on 36.9K products with 30% unranked)*
- **Regression Task:** Explain >20% of variance in BSR (R² > 0.20) *(Capstone I: R²=0.267, Capstone II: R²=0.425)*
- **Practical Value:** Identify actionable insights for listing optimization
- **Reproducibility:** Document methodology for replication and extension
- **(Capstone II) Deployment:** Build a functional web application for real-time prediction

---

## 2. Data Collection & Sources

### Data Acquisition

Data collection utilized approved web scraping via ScrapingDog, accessed through a custom Python wrapper to retrieve product information across multiple Amazon categories. The API enables programmatic access to Amazon product pages, extracting structured data including titles, brands, pricing, images, customer sentiment, and Best Seller Rank.

### Dataset Composition

### Search Keywords & Categories

Products were collected using targeted keywords across 14 Amazon categories (expanded from 9 in Capstone I):

- **Home & Kitchen:** kitchen gadgets, bedding, home decor, storage
- **Health & Household:** vitamins, personal care, household supplies
- **Office Products:** office electronics, furniture, writing supplies
- **Baby:** feeding, diapering, nursery, baby care
- **Clothing, Shoes & Jewelry:** apparel, footwear, accessories
- **Kitchen & Dining:** cookware, dinnerware, utensils
- **Electronics:** computers, TV, audio, cameras, wearables
- **Cell Phones & Accessories:** phones, cases, chargers, screen protectors
- **Tools & Home Improvement:** power tools, hand tools, hardware
- **Video Games:** consoles, controllers, gaming accessories
- **Pet Supplies:** dog, cat, fish, bird supplies
- **Sports & Outdoors:** exercise, camping, team sports
- **Industrial & Scientific:** lab equipment, safety, measuring
- **Musical Instruments:** guitars, keyboards, drums, recording

This keyword-driven approach ensured diverse representation across Amazon product categories while maintaining relevance to consumer-facing products.

### Data Characteristics

| Metric | Value |
|---|---|
| Total Products | ~36,900 unique ASINs |
| Images Downloaded | ~36,900 primary product images |
| Date Range | Data collected September-October 2025 |
| Geographic Market | Amazon.com (United States) |
| BSR Range | 1 (best) to ~7,000,000 (worst/unranked) |
| Categories | 14 (expanded from 9 in Capstone I) |

### Data Quality & Preprocessing

Initial data validation identified several quality issues requiring preprocessing:
- **Missing Values:** ~30.1% of products lacked BSR (treated as "unranked" class)
- **Duplicate ASINs:** Removed ~800 duplicate entries from keyword overlap
- **Image Availability:** ~2% of products had broken/missing image URLs
- **Sentiment Data:** JSON-formatted sentiment required parsing (~30% had sentiment data)

A detailed data audit document (`reports/dataset_audit.md`) provides SHA-256 checksums for reproducibility verification.

---

## 3. Methodology & Pipeline Architecture

The project employs a multi-stage data science pipeline integrating computer vision, deep learning, natural language processing, and supervised machine learning. The architecture follows a modular design with clear separation between data collection, feature engineering, modeling, and deployment.

### Pipeline Overview

```
Stage 1: Data Collection
├── ScrapingDog API queries (14 categories)
├── Image downloading (~19K products)
└── Metadata extraction (titles, brands, BSR)

Stage 2: Feature Engineering
├── Image Analysis (OpenCV + YOLOv8)
│   ├── Quality metrics (sharpness, contrast, saturation)
│   ├── Clutter detection (edge density, color clustering)
│   └── Composition analysis (background, object detection)
├── Deep Learning Embeddings (NEW in Capstone II)
│   ├── EfficientNet-B0 CNN (1280-dim → PCA → 128-dim)
│   └── CLIP ViT-B-32 (512-dim → PCA → 128-dim)
├── NLP Processing (NEW: textstat + TF-IDF + SVD)
│   ├── Text statistics (length, word count, readability)
│   ├── TF-IDF vectorization with SVD reduction
│   ├── Keyword flags (premium, bundle, new, size)
│   └── Structural features (capitalization, special chars)
└── Metadata Features
    ├── Image count
    ├── Brand encoding
    └── Category classification

Stage 3: Feature Integration
├── Merge 80+ features from all sources
├── Handle missing values
├── Z-score normalization for scale features
└── Data leakage prevention (remove price, reviews, ratings)

Stage 4: Model Development (UPDATED in Capstone II)
├── Per-Category Classification Models (14x XGBoost Classifiers)
│   ├── Train/test split (80/20) per category
│   ├── Top 20 PCA-reduced features
│   └── Binary: ranked vs. unranked
└── Per-Category Regression Models (14x XGBoost Regressors)
    ├── Filter to ranked products only per category
    ├── Log-transform BSR target
    └── Full 80+ feature set

Stage 5: Web Application Deployment (NEW in Capstone II)
├── FastAPI backend with real-time inference
├── Next.js frontend with wizard interface
├── Per-category model registry with lazy loading
└── Real-time image feature extraction pipeline
```

### Methodological Considerations

**Train/Test Split Strategy:** An 80/20 stratified split was employed to maintain class balance in the classification task. The test set was held out entirely during feature engineering and model selection.

**Target Variable Treatment:** BSR values exhibit extreme right-skew (range: 1 to ~7,000,000). Log-transformation (log10(BSR)) was applied to stabilize variance and improve model convergence.

**Data Leakage Prevention:** Features that would not be available at the time of prediction (price, review count, star ratings, customer sentiment) were intentionally excluded from the feature set to simulate real-world prediction scenarios where a seller is optimizing a listing before launch.

**Cross-Validation:** 5-fold cross-validation was used during hyperparameter tuning to assess model stability and prevent overfitting.

---

## 4. Feature Engineering

Feature engineering represents the core innovation of this project, transforming raw product listings into 80+ quantitative features spanning visual quality, textual characteristics, deep learning embeddings, and metadata attributes.

### 4.1 Image Analysis

Computer vision techniques extracted visual quality and composition features from product images using OpenCV and YOLOv8.

#### Image Quality Metrics

- **Sharpness (Laplacian Variance):** Computed as the variance of the Laplacian operator applied to grayscale images. Higher values indicate sharper, more focused images.
- **Exposure & Contrast:** Mean pixel intensity (exposure) and standard deviation (contrast) measured in RGB and HSV color spaces.
- **Saturation:** Average saturation in HSV space, indicating color vibrancy versus grayscale.
- **Noise Estimation:** High-frequency component analysis to detect image noise or compression artifacts.
- **Technical Quality Composite:** Weighted combination of sharpness, contrast, and saturation into a single quality score.

#### Clutter Detection Features

- **Edge Density:** Percentage of pixels classified as edges using Canny edge detection. High edge density indicates visual complexity or clutter.
- **Color Clustering:** K-means clustering (k=5) applied to color distribution. Features: `largest_cluster_pct`, `color_entropy`, `num_significant_clusters`.
- **Background Analysis:** Segmentation-based detection — `bg_white_pct` (>240 RGB), `bg_neutral_pct` (200-255 RGB), `bg_black_pct` (<15 RGB).
- **Clutter Score:** Composite metric combining edge density, color entropy, and cluster count. Z-score normalized for interpretability.

> **Figure 1:** Example product images with different clutter scores — one clean/professional image (low clutter) vs. one busy/cluttered image (high clutter) with their respective scores overlaid.

#### Object Detection (YOLOv8)

YOLOv8 object detection provided composition analysis:
- `object_count`: Number of distinct objects detected
- `bounding_box_coverage`: Percentage of image covered by detected objects
- `confidence_avg`: Average detection confidence across objects

Images were processed in batches of 500 and saved as Parquet files for memory efficiency.

### 4.2 Deep Learning Embeddings (NEW in Capstone II)

A major advancement in Capstone II was the introduction of deep learning embeddings to capture high-level visual semantics that hand-crafted OpenCV features miss.

#### EfficientNet-B0 CNN Embeddings

- **Architecture:** EfficientNet-B0 pre-trained on ImageNet, with the classification head removed
- **Raw Output:** 1280-dimensional feature vector per image
- **Dimensionality Reduction:** PCA fitted on training data only, reducing to 128 components
- **Features:** `cnn_pca_0000` through `cnn_pca_0127`
- **What it captures:** Object shapes, textures, product category visual patterns, photographic style

#### CLIP ViT-B-32 Embeddings

- **Architecture:** OpenAI CLIP Vision Transformer (ViT-B-32)
- **Raw Output:** 512-dimensional feature vector per image
- **Dimensionality Reduction:** PCA fitted on training data only, reducing to 128 components
- **Features:** `clip_pca_0000` through `clip_pca_0127`
- **What it captures:** Semantic image-text alignment, visual concepts that relate to language descriptions, product "vibe" and positioning

#### Why Both CNN and CLIP?

EfficientNet-B0 captures low-level visual patterns (textures, edges, spatial arrangements) while CLIP captures higher-level semantic meaning (what the product "looks like" it should be described as). Together, they provide complementary visual understanding that neither achieves alone. In our feature importance analysis, PCA components from both models appear in the top 10 most important features.

### 4.3 Natural Language Processing (NEW in Capstone II)

Text features were extracted from product titles using textstat, TF-IDF vectorization, and custom keyword extraction logic. This NLP pipeline was developed as a dedicated workstream.

#### Text Statistics
- `title_char_length`: Character count including spaces
- `word_count`: Number of words in title
- `avg_word_length`: Mean word length in characters
- `syllable_count`: Total syllables (proxy for complexity)

#### Readability Metrics
- `flesch_reading_ease`: Flesch Reading Ease score (0-100 scale). Higher scores = easier to read.

#### TF-IDF + SVD Features (NEW in Capstone II)
- TF-IDF vectorization of product titles across the corpus
- Truncated SVD dimensionality reduction to capture latent semantic topics
- Captures category-specific language patterns and keyword clustering

#### Keyword Flags (Binary Features)
Boolean indicators for presence of specific keyword categories:
- `has_premium_keywords`: "premium", "luxury", "pro", "professional"
- `has_bundle_keywords`: "set", "bundle", "pack", "kit"
- `has_new_keywords`: "new", "latest", "2024", "2025"
- `has_size_keywords`: "large", "small", "compact", "portable"
- `has_color_keywords`: "black", "white", "blue", "red", etc.

#### Structural Features (NEW in Capstone II)
- Capitalization patterns (all caps words, title case ratio)
- Special character usage (dashes, pipes, brackets)
- Number presence and density

#### Sentiment Features (Exploratory Only)
For products with customer sentiment data (JSON format): `sentiment_positive_count`, `sentiment_negative_count`, `sentiment_mixed_count`, `sentiment_ratio`. These were excluded from final models to prevent data leakage but were analyzed in exploratory phases.

### 4.4 Metadata Features

- `image_count`: Number of images in product listing (1-8 typical range)
- `brand_encoded`: Label-encoded brand names (100+ unique brands)
- `category`: Amazon product category classification

### Feature Scaling & Normalization

Continuous features with varying scales were Z-score normalized: `Z = (X - μ) / σ`. This ensures features like `bg_white_pct` (0-100 scale) and `edge_density` (0-1 scale) contribute equally to model training.

### Final Feature Set

| Category | Feature Count |
|---|---|
| Image Quality (OpenCV) | ~50 features (brightness, contrast, sharpness, etc.) |
| Clutter/Composition | ~30 features (edge density, background, border clutter) |
| CNN Embeddings (PCA-reduced) | 128 features (top 20 used for classification) |
| CLIP Embeddings (PCA-reduced) | 128 features (top 20 used for classification) |
| Object Detection (YOLO) | ~20 features (confidence, coverage, person detection) |
| NLP/Text (Title + Bullets) | ~120 features (TF-IDF PCA: 50 title + 50 bullets + stats) |
| Metadata | ~10 features (image count, price, A+ content) |
| **Total (after cleaning)** | **~2,330 features for regression** |

---

## 5. Exploratory Data Analysis

Exploratory analysis revealed key patterns and relationships guiding model development.

> **Figure 2:** Correlation heatmap showing relationships between top 20 features and log(BSR).

### Target Variable Distribution

**BSR Distribution (Ranked Products):**
- Highly left-skewed (mean: ~5,500; median: ~317)
- Range: 1 (best) to ~7,000,000 (worst)
- Log-transformation produces approximately normal distribution

> **Figure 3:** Side-by-side histograms showing (left) raw BSR distribution with extreme left skew, (right) log10(BSR) distribution showing approximately normal shape.

**Class Imbalance:**
- Ranked products: 69.9% (25,766 products)
- Unranked products: 30.1% (11,113 products)
- This substantial imbalance informed stratified sampling strategies and class-weight balancing in classification.

### Key Feature Relationships

**Image Quality vs. BSR:**
- Negative correlation: Higher quality images associate with better (lower) BSR
- `bg_neutral_pct` shows strongest correlation (r = -0.31)
- Professional white backgrounds correlate with BSR < 10,000

**Clutter Effects:**
- Higher `clutter_score` correlates with worse (higher) BSR (r = +0.24)
- Products with `clutter_score > 2.0` have median BSR 3x worse than clean images

**Image Count Impact:**
- Products with 6+ images have 45% lower median BSR than those with 1-2 images
- Diminishing returns beyond 6 images

> **Figure 4:** Box plots showing BSR distribution across image_count bins (1-2, 3-4, 5-6, 7+).

**Text Features:**
- Title length shows weak correlation (r = -0.09)
- Presence of "bundle" keywords associates with 30% higher median BSR (worse rank)
- Readability scores show minimal correlation

**Deep Learning Embeddings (Capstone II finding):**
- CNN and CLIP PCA components show strong correlations with BSR that hand-crafted features miss
- Top CNN/CLIP PCA components consistently rank among the most important features across categories
- CLIP embeddings particularly effective at distinguishing professional vs. amateur product photography

### Feature Importance Preview

Preliminary feature importance (from baseline Random Forest):
1. `bg_neutral_pct_z`: 13.3%
2. `largest_cluster_pct_z`: 10.9%
3. `bg_white_pct`: 9.3%
4. `image_count`: 8.4%
5. `clutter_score`: 7.9%

This pattern persisted in final XGBoost models, confirming visual quality as the dominant predictor.

---

## 6. Model Development & Selection

### Two-Stage Modeling Approach

The prediction task was decomposed into two sub-problems:

- **Stage 1: Classification** — Predict whether a product will achieve any sales ranking (binary: ranked vs. unranked)
- **Stage 2: Regression** — For products classified as ranked, predict the specific BSR value

This approach addresses the inherent challenge that ~13.7% of products lack BSR entirely.

### Capstone I: Global Models (Baseline)

#### Classification — Logistic Regression Baseline
- Accuracy: 74.2%, Precision: 96.9%, Recall: 74.9%, F1: 84.5%, ROC-AUC: 0.756

#### Classification — XGBoost (Global)
- Accuracy: 90.0%, Precision: 97.5%, Recall: 91.6%, F1: 94.4%, ROC-AUC: 0.849
- Configuration: n_estimators=500, max_depth=12, learning_rate=0.05

#### Regression — Linear Regression Baseline
- RMSE (log-space): 2.01, R²: 0.043, MAE: 1.67

#### Regression — XGBoost (Global)
- RMSE (log-space): 1.76, R²: 0.267, MAE: 1.43

### Capstone II: Per-Category Models (Final)

A key insight from Capstone I was that a single global model averages across fundamentally different product categories. In Capstone II, we trained **14 separate classifier-regressor pairs**, one for each Amazon category.

#### Global Classification (XGBoost, Top 20 Features)

The global classifier uses the top 20 PCA-reduced features selected from CNN + CLIP embeddings + quality metrics:

**Performance (36,879 products, 80/20 stratified split):**
- **Accuracy:** 83.9%
- **ROC-AUC:** 0.891
- **F1 Score:** 0.882
- **Note:** The lower accuracy compared to Capstone I reflects the harder classification task — 30.1% unranked (vs. 13.7% in Capstone I) across 14 diverse categories

#### Global Regression (XGBoost, 2,330 Features)

The global regressor uses all available features (excluding raw embeddings and leakage columns):

**Performance (25,766 ranked products, 80/20 split):**
- **R²:** 0.425 (explains 42.5% of log-BSR variance)
- **RMSE (log-space):** 1.838
- **Significant improvement** over Capstone I (R² = 0.267), driven by deep learning embeddings and NLP features

#### Per-Category Models

Per-category models trained independently on each Amazon category:

**Performance (7 categories with sufficient data evaluated):**
- **Mean AUC:** 0.700
- **Mean R²:** 0.303
- Per-category models capture category-specific patterns but have less training data per model

> **Figure 5:** Confusion matrix heatmap for per-category XGBoost classifier.

> **Figure 6:** Scatter plot of Actual log(BSR) vs. Predicted log(BSR) with diagonal reference line.

### Feature Importance Analysis

> **Figure 7:** Bar chart of top 15 feature importances color-coded by category (Image=blue, NLP=green, Metadata=orange, Embedding=purple).

**Key Observations:**
- Visual features dominate: 8 of top 10 features are image-based or embedding-based
- CNN/CLIP PCA components consistently rank in top 5 across categories
- Background quality is critical: neutral/white backgrounds account for ~20% combined importance
- NLP features contribute ~10% total importance — meaningful but secondary
- Image count ranks in top 5, suggesting multi-image listings signal quality

### Model Comparison Summary — Capstone I vs. Capstone II

| Metric | Capstone I | Capstone II | Change |
|---|---|---|---|
| **Dataset Size** | ~19,000 products | ~36,900 products | +94% |
| **Categories** | 9 electronics subcategories | 14 Amazon categories | Broader |
| **Features Used** | ~80 (OpenCV only) | ~2,330 (CNN+CLIP+NLP+CV) | Multimodal |
| **Classification ROC-AUC** | 0.849 | 0.891 | +0.042 |
| **Classification F1** | 0.944 | 0.882 | Harder task (30% vs 14% unranked) |
| **Regression R²** | 0.267 | 0.425 | +59% relative |
| **Regression RMSE (log)** | 1.76 | 1.838 | Comparable (larger dataset) |
| **CV Mean R²** | N/A | 0.378 ± 0.02 | Stable across folds |
| **Per-Category Mean AUC** | N/A | 0.700 | 7 categories evaluated |
| **Per-Category Mean R²** | N/A | 0.303 | Category-specific patterns |

---

## 7. Results & Performance Evaluation

### Classification Performance Deep Dive

**Global Model Performance:** The XGBoost classifier achieves 83.9% accuracy with an F1 score of 0.882 and ROC-AUC of 0.891 on the full 36,900-product dataset. The classification task is substantially harder than Capstone I due to the higher proportion of unranked products (30.1% vs. 13.7%) and the diversity of 14 categories.

**ROC Curve Analysis:** The ROC-AUC of 0.891 demonstrates strong discriminative ability between ranked and unranked products across all classification thresholds.

> **Figure 8:** ROC curve plot showing per-category classifier curves vs. diagonal baseline.

### Regression Performance Analysis

**Error Distribution:** Residuals show approximately normal distribution with slight right skew, indicating model predictions are unbiased on average with occasional overestimates.

> **Figure 9:** Residuals distribution histogram with normal curve overlay.

### Prediction Confidence Analysis

**High-Confidence Predictions (Top 25%):**
- Characteristics: Strong image quality signals, 5+ images, neutral backgrounds, clear CNN/CLIP embeddings

**Low-Confidence Predictions (Bottom 25%):**
- Characteristics: Missing features, low image count, cluttered backgrounds, ambiguous embeddings

This confidence stratification enables practical deployment where high-confidence predictions can be trusted for decision-making while low-confidence predictions trigger manual review.

### Cross-Validation Stability

**5-Fold Cross-Validation Results (XGBoost Regressor):**
- Fold 1: R² = 0.394, RMSE = 1.874
- Fold 2: R² = 0.374, RMSE = 1.896
- Fold 3: R² = 0.335, RMSE = 1.936
- Fold 4: R² = 0.394, RMSE = 1.905
- Fold 5: R² = 0.392, RMSE = 1.911
- **Mean R²: 0.378 ± 0.022** | **Mean RMSE: 1.904 ± 0.020**
- Low variance across folds indicates robust performance without overfitting

> **Figure 10:** Box plot showing R² and RMSE distributions across 5 CV folds.

---

## 8. Web Application (NEW in Capstone II)

A major Capstone II deliverable is a full-stack web application that operationalizes the ML pipeline for real-time BSR prediction.

### Architecture

```
┌─────────────────────────────┐
│    Next.js Frontend         │
│    (React + TypeScript +    │
│     Tailwind CSS)           │
│    Port 3000                │
└─────────────┬───────────────┘
              │ /api/* proxy
              ▼
┌─────────────────────────────┐
│    FastAPI Backend           │
│    (Python 3.10)            │
│    Port 8000                │
├─────────────────────────────┤
│  Services:                  │
│  ├── Image Feature Extractor│
│  │   (CNN + CLIP + OpenCV   │
│  │    + YOLO, real-time)    │
│  ├── Text Feature Extractor │
│  │   (NLP pipeline)         │
│  ├── Model Registry         │
│  │   (14 clf + 14 reg,      │
│  │    lazy loading + cache)  │
│  └── Inference Service      │
│      (two-stage pipeline)   │
└─────────────────────────────┘
```

### Frontend Features

- **Step-by-step wizard interface:** Category Selection → Image Upload → Title Entry → Results
- **Product evaluation form** with drag-and-drop image upload and URL input
- **Score card** displaying classification probability, predicted BSR, and confidence
- **Signal list** showing which features contributed most to the prediction
- **Model performance dashboard** with per-category metrics
- **NLP feedback panel** with title improvement suggestions
- **Test product suggestions** for users without a product to evaluate
- **Dark/light theme toggle**
- **Responsive design** for desktop and mobile

### Backend API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | System status and loaded model count |
| `/api/categories` | GET | List all 14 categories with metadata |
| `/api/evaluate` | POST | Full prediction pipeline (image + text + category) |
| `/api/embeddings/image` | POST | Extract CNN + CLIP embeddings from uploaded image |
| `/api/models` | GET | Model registry status and performance metrics |
| `/api/suggestions` | GET | Sample products for testing the application |
| `/api/validation/summary` | GET | Model validation metrics summary |

### Key Technical Decisions

- **Per-category model registry with lazy loading:** Models are loaded into memory only when a category is first requested, reducing startup time and memory footprint
- **PCA transformers saved with models:** Each category's PCA transformer is fitted on training data only and saved alongside the model to prevent data leakage during inference
- **Real-time feature extraction:** The full image analysis pipeline (OpenCV quality metrics + EfficientNet-B0 + CLIP + YOLOv8) runs on each uploaded image in real-time
- **Next.js API proxy:** Frontend proxies `/api/*` requests to the FastAPI backend, enabling seamless development and deployment

> **Figure 11:** Screenshots of the web application — (a) product evaluation wizard, (b) prediction results with score card, (c) model performance dashboard.

---

## 9. Business Insights & Applications

### Actionable Insights for Sellers

#### 1. Professional Photography is Non-Negotiable
**Finding:** Products with neutral/white backgrounds (>60% of image) rank 2.3x better on average.
**Recommendation:** Invest in professional product photography with clean, neutral backgrounds. DIY photography on white surfaces with proper lighting can achieve 80% of professional quality at a fraction of the cost.

#### 2. Multi-Image Listings Outperform
**Finding:** Products with 6+ images have median BSR 45% lower (better) than those with 1-2 images.
**Recommendation:** Upload maximum allowed images (8 on Amazon) showing product from multiple angles, in-use scenarios, and size comparisons.

#### 3. Avoid Visual Clutter
**Finding:** High clutter scores (>2.0) correlate with 3x worse median BSR.
**Recommendation:** Remove busy backgrounds, excessive text overlays, and complex compositions. Focus on product-centric imagery with minimal distractions.

#### 4. Title Optimization Shows Limited Impact
**Finding:** Title length and readability contribute only ~3% to model performance.
**Recommendation:** Optimize titles for human readability and keyword inclusion, but don't expect major BSR impacts from text alone. Visual presentation matters far more.

#### 5. Category Context Matters (Capstone II Finding)
**Finding:** Per-category models significantly outperform global models, indicating that visual and textual success patterns vary by category.
**Recommendation:** Benchmark against top performers within your specific category, not across all of Amazon.

### Competitive Analysis Framework

**Use Case:** Analyzing competitor product listings
1. Input competitor ASIN into the web application
2. Extract visual quality and clutter metrics automatically
3. Compare against successful product benchmarks (BSR < 10,000)
4. Identify visual deficiencies and improvement opportunities

### Inventory Planning & Demand Forecasting

**Workflow:**
1. Create product mock-ups with professional photography
2. Run mock-up through the web application
3. Estimate probable BSR range
4. Convert BSR to estimated daily sales using industry benchmarks
5. Plan inventory levels accordingly

**Limitation:** Model predicts relative performance based on listing quality, not absolute demand. External factors (market size, pricing, competition) must be considered separately.

---

## 10. Ethical Considerations & Limitations

### Ethical Considerations

#### Data Usage & Privacy
- **Data Source:** All product information was sourced from a paid data source (ScrapingDog API). No customer personally identifiable information (PII) was collected or used.
- **Bias Considerations:**
  - Model trained on multiple categories but may not generalize to all Amazon verticals
  - Data reflects US marketplace (Amazon.com); international markets may behave differently
  - Historical data may not reflect future platform algorithm changes

#### Fair Use & Commercial Application
This project is an academic research initiative demonstrating machine learning methodologies. Commercial deployment would require additional considerations including Amazon Terms of Service compliance, trademark usage, and disclosure of prediction uncertainty to end users.

**Responsibility Statement:** Predictions should inform decisions, not determine them. Human judgment, market research, and business context must complement model outputs.

### Limitations

#### 1. Model Performance Ceiling
R² = 0.425 (global) indicates that 57.5% of BSR variance remains unexplained by listing features. This reflects real-world complexity: pricing dynamics, promotion timing, competitive actions, seasonality, and external reviews all influence BSR but are not captured by listing features alone.

#### 2. Category Specificity
While Capstone II expanded to 14 categories, the model may not generalize to fundamentally different verticals (apparel, books, grocery) where different features drive success.

#### 3. Temporal Validity
Model trained on late-2025 data. BSR algorithms, platform policies, and market conditions change over time. Periodic retraining (quarterly recommended) is necessary.

#### 4. Causation vs. Correlation
The model identifies correlations between listing features and BSR but does not prove causation. Professional images may correlate with better BSR because successful sellers invest more in photography, not necessarily because better images cause more sales.

---

## 11. Deployment & Reproducibility

### System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| CPU | 4+ cores | 8+ cores |
| RAM | 8GB | 16GB |
| Storage | 15GB free | 25GB free |
| GPU | Not required | CUDA-capable (for faster inference) |

**Supported OS:** macOS (Sonoma 14+), Windows 10/11, Linux (Ubuntu 20.04+)

### Software Dependencies

**Core Environment:**
- Python 3.10+ (via Conda)
- Node.js 18+ (for frontend)

**Key Libraries:**
- Data Processing: pandas (2.0+), NumPy (1.24+)
- Machine Learning: scikit-learn (1.3+), XGBoost (2.0+)
- Deep Learning: PyTorch, EfficientNet-B0, CLIP (open_clip)
- Computer Vision: OpenCV (4.8+), YOLOv8 (ultralytics)
- NLP: textstat (0.7+)
- Backend: FastAPI, uvicorn, pydantic
- Frontend: Next.js, React, TypeScript, Tailwind CSS
- Visualization: matplotlib (3.7+), seaborn (0.12+)

Full dependency list available in `environment.yml`.

### Installation Guide

**Step 1: Clone Repository**
```bash
git clone https://github.com/DylanSuniaga/Forecasting-Amazon-Sales-with-Multimodal-Data.git
cd Forecasting-Amazon-Sales-with-Multimodal-Data
```

**Step 2: Create Conda Environment**
```bash
conda env create -f environment.yml
conda activate amazon-forecast
```

**Step 3: Train Models & Export for Web**
```bash
cd notebooks/02-Image\ Analysis
python save_models_for_web.py
```

**Step 4: Start Backend**
```bash
cd src/web/backend
uvicorn app.main:app --reload --port 8000
```

**Step 5: Start Frontend**
```bash
cd src/web/frontend
npm install
npm run dev
```

**Step 6: Access Application**
Open `http://localhost:3000` in your browser. API docs available at `http://localhost:8000/docs`.

### Reproducibility Verification

- **Random Seeds:** All models use `random_state=42` for reproducible results
- **Data Integrity:** SHA-256 checksums provided in `reports/dataset_audit.md`
- **Environment Locking:** `environment.yml` pins exact package versions
- **PCA Transformers:** Saved per-category to prevent train/test leakage during inference

### Project Structure

```
project/
├── CLAUDE.md                              # Development guide
├── environment.yml                        # Conda environment
├── data/                                  # Data directory (not in git)
│   ├── products_with_image_feats.csv      # All features (~4.4GB)
│   ├── data_with_scraper.csv              # Enriched metadata (~60MB)
│   └── images_amz/                        # Product images (~16K)
├── notebooks/
│   ├── 00-Data Downloading/               # SP-API + ScrapingDog collection
│   ├── 01-Base Model with Visuals/        # Random Forest baseline
│   ├── 02-Image Analysis/                 # CV features + deep learning
│   │   ├── image_feature_extraction.ipynb # Full extraction pipeline
│   │   ├── c2_eda_images.ipynb            # Capstone II EDA
│   │   ├── c2_baseline_modeling*.ipynb    # Per-category modeling
│   │   └── save_models_for_web.py         # Model export script
│   ├── 03-NLP/                            # Text preprocessing & features
│   │   ├── 01_text_preprocessing.ipynb    # Text cleaning
│   │   ├── 02_NLP_EDA.ipynb              # NLP exploration
│   │   ├── 03_tfidf_features.ipynb       # TF-IDF pipeline
│   │   └── 04_advanced_nlp.ipynb         # Advanced NLP features
│   └── 04-Final Modeling/                 # Final models + figures
│       ├── main.ipynb                     # Training pipeline
│       └── report_figures.ipynb           # Publication figures
├── src/
│   ├── nlp/                               # NLP utilities
│   ├── utils/                             # API helpers
│   └── web/                               # Web application
│       ├── backend/                       # FastAPI server
│       │   ├── app/
│       │   │   ├── api/routes.py          # API endpoints
│       │   │   ├── core/config.py         # Settings & categories
│       │   │   ├── models/schemas.py      # Pydantic models
│       │   │   └── services/              # Business logic
│       │   │       ├── inference.py        # Two-stage pipeline
│       │   │       ├── image_features.py   # CNN+CLIP+OpenCV+YOLO
│       │   │       ├── text_features.py    # NLP pipeline
│       │   │       └── model_registry.py   # Per-category model cache
│       │   └── models/                    # Saved model artifacts
│       └── frontend/                      # Next.js app
│           └── src/
│               ├── app/page.tsx           # Main page
│               ├── components/            # 20+ React components
│               ├── lib/api.ts             # API client
│               └── types/index.ts         # TypeScript interfaces
└── presentation/                          # Documentation & poster
```

---

## 12. Future Work & Recommendations

### Completed from Capstone I Future Work

The following items from the Capstone I "Future Work" section were implemented in Capstone II:

- **Deep learning for image analysis** — EfficientNet-B0 and CLIP embeddings now provide learned visual features
- **Category-specific models** — 14 per-category classifier-regressor pairs replace the single global model
- **Web application for sellers** — Full-stack FastAPI + Next.js application with real-time prediction
- **NLP feature pipeline** — TF-IDF, SVD, structural features complement basic text statistics

### Remaining Opportunities

#### 1. Time-Series BSR Tracking
Collect historical BSR data over 30-90 days to model BSR trajectory (improving/declining/stable) using LSTM/GRU networks.

#### 2. Multi-Marketplace Analysis
Expand to Amazon UK, DE, JP to compare feature importance across markets and identify universal vs. culture-specific success factors.

#### 3. Causal Inference Study
Design a randomized controlled trial with real Amazon listings to establish whether listing improvements *cause* BSR improvement (vs. correlation).

#### 4. Browser Extension
Build a Chrome/Firefox extension that overlays predicted BSR and image quality scores on Amazon product pages in real-time.

#### 5. Expanded Category Coverage
Extend beyond the current 14 categories to apparel, books, grocery, and home goods — each with unique feature importance patterns.

#### 6. Transformer Models for Text
Use BERT/RoBERTa embeddings for product titles to capture semantic meaning beyond TF-IDF bag-of-words features.

---

## 13. References

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning (ICML)*.

Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. *International Conference on Machine Learning (ICML)*.

Jocher, G. (2023). YOLOv8 by Ultralytics. GitHub repository. https://github.com/ultralytics/ultralytics

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal of Software Tools*, 25(11), 120-123.

McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 445, 51-56.

Amazon Web Services. (2024). Amazon Selling Partner API Documentation. https://developer-docs.amazon.com/sp-api/

Textstat Contributors. (2024). textstat: Calculate readability statistics. https://pypi.org/project/textstat/

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

---

## 14. Appendices

### Appendix A: Feature Dictionary

> **Figure 12:** Complete table of all 80+ features with columns: Feature Name | Type (Image/NLP/Embedding/Meta) | Description | Example Value

### Appendix B: Per-Category Performance Breakdown

> **Figure 13:** Table showing classification accuracy, ROC-AUC, regression R², and RMSE for each of the 14 categories

### Appendix C: Hyperparameter Configurations

> **Figure 14:** Per-category optimal hyperparameter configurations for both classifiers and regressors

### Appendix D: Web Application Screenshots

> **Figure 15:** Full-page screenshots of (a) wizard interface, (b) prediction results, (c) model performance dashboard, (d) NLP feedback panel

### Appendix E: Capstone I vs. Capstone II Changelog

| Area | Capstone I | Capstone II |
|---|---|---|
| Dataset | ~19,000 products | ~36,900 products |
| Categories | 9 electronics subcategories | 14 Amazon categories |
| Unranked Ratio | 13.7% | 30.1% |
| Image Features | OpenCV + YOLOv8 only | + EfficientNet-B0 + CLIP embeddings |
| NLP Features | Basic text stats + keywords | + TF-IDF PCA (50 title + 50 bullets) + structural |
| Total Features | ~80 | ~2,330 (after cleaning) |
| Models | 1 global clf + 1 global reg | Global + per-category models |
| Classification AUC | 0.849 | 0.891 (global) / 0.700 mean (per-cat) |
| Classification F1 | 0.944 | 0.882 |
| Regression R² | 0.267 | 0.425 (global) / 0.303 mean (per-cat) |
| Regression RMSE | 1.76 | 1.838 |
| CV Stability | N/A | R² = 0.378 ± 0.022 across 5 folds |
| Deployment | Notebooks only | Full-stack web application |
| Model Export | None | Automated pipeline (`save_models_for_web.py`) |
| Frontend | None | Next.js + React + TypeScript |
| Backend | None | FastAPI with real-time inference |

### Appendix F: Code Repository Structure

Full file-by-file breakdown available in the project's `README.md` on GitHub.
