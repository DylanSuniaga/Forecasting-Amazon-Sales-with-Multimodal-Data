# User Manual
## Amazon Best Seller Rank Prediction using Multimodal Analysis

---

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start Guide](#quick-start-guide)
3. [Data Pipeline Overview](#data-pipeline-overview)
4. [Running the Analysis](#running-the-analysis)
5. [Understanding the Output](#understanding-the-output)
6. [Advanced Usage](#advanced-usage)
7. [API Integration](#api-integration)
8. [Common Workflows](#common-workflows)
9. [FAQs](#faqs)

---

## Introduction

This manual provides comprehensive instructions for using the Amazon Best Seller Rank (BSR) prediction system. The system uses multimodal machine learning to predict product sales performance based on images, text, and metadata.

### What This System Does

- **Classifies** products as likely to be ranked or unranked (90% accuracy)
- **Predicts** Best Seller Rank for ranked products (R² = 0.27)
- **Analyzes** image quality and presentation factors
- **Extracts** text features from product titles and descriptions
- **Identifies** key factors that drive sales performance

### Who Should Use This Manual

- Data scientists and ML engineers
- E-commerce analysts
- Product managers
- Researchers studying online marketplaces
- Students learning multimodal machine learning

---

## Quick Start Guide

### 5-Minute Start

1. **Activate environment**:
   ```bash
   conda activate amazon-forecast
   ```

2. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Open final model**:
   - Navigate to `notebooks/04-Final Modeling/`
   - Open `main.ipynb`

4. **Run all cells**:
   - Click: Kernel → Restart & Run All
   - Wait for completion (~5-10 minutes)

5. **View results**:
   - Scroll to bottom for model performance metrics
   - Check classification accuracy and regression R²

### What You'll See

- Classification accuracy: ~90%
- Regression R²: ~0.27
- Feature importance rankings
- Prediction visualizations

---

## Data Pipeline Overview

### System Architecture

```
Input Data
    ├── Product Images (19K images)
    ├── Product Text (titles, descriptions)
    └── Metadata (brand, category, etc.)
           ↓
Feature Extraction
    ├── Image Features (clutter, quality, composition)
    ├── NLP Features (readability, keywords, sentiment)
    └── Numerical Features (image count, ratings)
           ↓
Model Pipeline
    ├── Stage 1: Classification (Ranked vs Unranked)
    └── Stage 2: Regression (Predict BSR value)
           ↓
Predictions & Insights
```

### Data Flow

1. **Raw Data** → `data/` directory
2. **Image Processing** → `notebooks/02-Image Analysis/`
3. **Text Processing** → `notebooks/03-NLP/`
4. **Model Training** → `notebooks/04-Final Modeling/`
5. **Predictions** → Output CSV files

---

## Running the Analysis

### Option 1: Run Complete Pipeline (Start to Finish)

#### Prerequisites
- Amazon SP-API credentials configured in `.env`
- At least 4 hours of processing time
- 16 GB RAM recommended

#### Steps

**Step 1: Data Collection**
```bash
cd notebooks/00-Data\ Downloading/
jupyter notebook data_download.ipynb
```

- Enter search keywords when prompted
- Wait for API data collection (~30-60 minutes)
- Output: CSV files in `data/batches/`

**Step 2: Image Download and Analysis**
```bash
cd ../02-Image\ Analysis/
jupyter notebook main.ipynb
```

- Images will be downloaded automatically
- Computer vision analysis runs (~2-3 hours for 19K images)
- Output: `data/image_analysis_data/yolo_update_data.csv`

**Step 3: NLP Processing**
```bash
cd ../03-NLP/
jupyter notebook 01_text_preprocessing.ipynb
```

- Text features are extracted
- Sentiment analysis runs
- Output: `data/interim/nlp_text_clean.parquet`

**Step 4: Model Training**
```bash
cd ../04-Final\ Modeling/
jupyter notebook main.ipynb
```

- Trains XGBoost classifier and regressor
- Generates performance metrics
- Creates visualizations
- Output: Model predictions and feature importance

### Option 2: Run with Pre-Processed Data (Recommended)

Use existing data files to skip time-consuming steps:

1. **Verify data files exist**:
   ```bash
   ls data/17k_products_amazon_data.csv
   ls data/image_analysis_data/yolo_update_data.csv
   ```

2. **Jump directly to final modeling**:
   ```bash
   cd notebooks/04-Final\ Modeling/
   jupyter notebook main.ipynb
   ```

3. **Run all cells**:
   - Kernel → Restart & Run All
   - Processing time: ~10-15 minutes

### Option 3: Run Individual Components

#### Image Analysis Only

```python
# In notebooks/02-Image Analysis/main.ipynb

import cv2
import pandas as pd
from your_image_functions import tech_quality_for_path

# Load image paths
df = pd.read_csv('../../data/all_keywords_merged.csv')

# Process single image
img_path = df['image_path'].iloc[0]
quality_metrics = tech_quality_for_path(img_path)
print(quality_metrics)
```

#### NLP Analysis Only

```python
# In notebooks/03-NLP/01_text_preprocessing.ipynb

from src.nlp.preprocess import build_preprocessed_frame
import pandas as pd

# Load data
base_df = pd.read_csv('data/all_keywords_merged.csv')
enriched_df = pd.read_csv('data/17k_products_amazon_data.csv')

# Extract features
clean_df = build_preprocessed_frame(base_df, enriched_df)
print(clean_df[['text_clean', 'title_readability']].head())
```

#### Prediction Only (Using Trained Models)

```python
# Load pre-trained model
import pickle
import pandas as pd

# Load model (if saved)
with open('models/xgboost_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load new data
new_data = pd.read_csv('your_new_data.csv')

# Predict
predictions = clf.predict(new_data)
print(f"Predicted ranked: {predictions.sum()}/{len(predictions)}")
```

---

## Understanding the Output

### Classification Model Output

#### Metrics Explained

```python
=== XGBoost Classifier ===
              precision    recall  f1-score   support

           0       0.32      0.63      0.43       234
           1       0.97      0.92      0.94      3634

    accuracy                           0.90      3868
```

**Interpretation**:
- **Accuracy: 0.90** → 90% of predictions are correct
- **Class 0 (Unranked)**: Harder to predict (only 32% precision)
- **Class 1 (Ranked)**: Very accurate (97% precision, 92% recall)
- **F1-Score: 0.94** → Excellent balance for ranked products

#### Confusion Matrix

```
[[ 147   87]     → 147 true unranked, 87 false positives
 [ 307 3327]]    → 307 false negatives, 3327 true ranked
```

**What this means**:
- 3327 products correctly predicted as ranked
- 307 ranked products missed (false negatives)
- 147 unranked products correctly identified
- 87 products wrongly predicted as ranked

### Regression Model Output

#### Metrics Explained

```python
=== XGBoost Regressor ===
RMSE (log-space): 1.76
R²: 0.27
```

**Interpretation**:
- **R² = 0.27** → Model explains 27% of BSR variance
- **RMSE = 1.76** → Average error of ~2,500 BSR points
- Log-space predictions reduce impact of outliers

#### Feature Importance

```
Top Features:
1. bg_neutral_pct_z: 13.3%    → Neutral background is most important
2. largest_cluster_pct_z: 10.9% → Color composition matters
3. bg_white_pct: 9.3%          → Professional white backgrounds
4. image_count: 8.4%           → More images = better ranking
```

**Business Insights**:
- Clean, neutral backgrounds improve sales
- Multiple product images are beneficial
- Professional photography correlates with success
- Visual simplicity outperforms complexity

### Prediction Output Files

#### random_forest_bsr_predictions.csv

```csv
asin,item_name,brand,bsr_best,predicted_bsr,prediction_error,error_percentage
B0CQ2MSP2B,Product Name,Brand,1617,1617,0,0.0
```

**Columns**:
- `asin`: Amazon product identifier
- `item_name`: Product title
- `bsr_best`: True Best Seller Rank
- `predicted_bsr`: Model prediction
- `prediction_error`: Absolute error
- `error_percentage`: Relative error

---

## Advanced Usage

### Custom Model Training

#### Train with Different Parameters

```python
from xgboost import XGBClassifier

# Custom classifier
clf = XGBClassifier(
    n_estimators=500,        # More trees
    max_depth=8,             # Deeper trees
    learning_rate=0.03,      # Slower learning
    subsample=0.7,           # Less data per tree
    colsample_bytree=0.7,    # Less features per tree
    random_state=42
)

clf.fit(X_train, y_train)
```

#### Save Trained Model

```python
import pickle

# Save classifier
with open('models/xgboost_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Save regressor
with open('models/xgboost_regressor.pkl', 'wb') as f:
    pickle.dump(reg, f)
```

#### Load and Use Saved Model

```python
# Load model
with open('models/xgboost_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# Make predictions
predictions = clf.predict(X_new)
probabilities = clf.predict_proba(X_new)
```

### Feature Engineering

#### Add Custom Features

```python
# In notebooks/04-Final Modeling/main.ipynb

# Custom text features
df['title_has_bestseller'] = df['item_name'].str.contains(
    'bestseller|best seller', 
    case=False
).astype(int)

df['title_exclamation_count'] = df['item_name'].str.count('!')

# Custom image features
df['image_aspect_ratio'] = df['image_width'] / df['image_height']
df['is_square_image'] = (df['image_aspect_ratio'] > 0.9) & (df['image_aspect_ratio'] < 1.1)
```

#### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top 50 features
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X_train, y_train)

# Get selected feature names
selected_features = X_train.columns[selector.get_support()].tolist()
print(f"Selected features: {selected_features}")
```

### Batch Predictions

#### Process New Products

```python
def predict_bsr_for_new_products(product_data_file):
    """
    Predict BSR for new products
    
    Args:
        product_data_file: CSV with product data
        
    Returns:
        DataFrame with predictions
    """
    import pandas as pd
    import pickle
    
    # Load models
    with open('models/xgboost_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('models/xgboost_regressor.pkl', 'rb') as f:
        reg = pickle.load(f)
    
    # Load new data
    df = pd.read_csv(product_data_file)
    
    # Extract features (same as training)
    X = extract_features(df)
    
    # Step 1: Predict if product will be ranked
    will_rank = clf.predict(X)
    
    # Step 2: Predict BSR for ranked products
    bsr_predictions = reg.predict(X)
    bsr_predictions = np.expm1(bsr_predictions)  # Convert from log space
    
    # Combine results
    df['predicted_ranked'] = will_rank
    df['predicted_bsr'] = bsr_predictions
    df.loc[will_rank == 0, 'predicted_bsr'] = None  # No BSR for unranked
    
    return df

# Usage
results = predict_bsr_for_new_products('new_products.csv')
results.to_csv('predictions_output.csv', index=False)
```

---

## API Integration

### Amazon SP-API Setup

#### Step 1: Get API Credentials

1. Register as Amazon SP-API developer
2. Create an application
3. Note your credentials:
   - Client ID
   - Client Secret
   - Refresh Token

#### Step 2: Configure Environment

Create `.env` file:
```
CLIENT_ID=amzn1.application-oa2-client.xxxxx
CLIENT_SECRET=xxxxxxxxxxxxx
REFRESH_TOKEN=Atzr|IwEBxxxxxxxxxxxx
```

#### Step 3: Use API in Notebooks

```python
# In notebooks/00-Data Downloading/data_download.ipynb

from dotenv import load_dotenv
import os

# Load credentials
load_dotenv()
CLIENT_ID = os.environ.get('CLIENT_ID')
CLIENT_SECRET = os.environ.get('CLIENT_SECRET')
REFRESH_TOKEN = os.environ.get('REFRESH_TOKEN')

# Get access token
from utils.spapi_helper import get_lwa_access_token
token = get_lwa_access_token(CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN)

# Search for products
from utils.spapi_helper import search_catalog_items
keywords = ["wireless keyboard", "gaming mouse"]
asins = search_catalog_items(keywords, page_size=20, max_pages=5)

print(f"Found {len(asins)} products")
```

### Rate Limiting

The API includes automatic rate limiting:
- Retries on 429 (throttle) errors
- Exponential backoff
- Respects `Retry-After` headers

```python
# Built into sp_get() function
# No additional configuration needed
```

---

## Common Workflows

### Workflow 1: Analyze Competitor Products

```python
# 1. Get competitor ASINs
competitor_asins = ['B08Z6X4NK3', 'B09LK1P1RD', 'B0D14N2QZF']

# 2. Get product data from API
from utils.spapi_helper import hydrate_asins
df_competitors = hydrate_asins(competitor_asins)

# 3. Download images
# (Automated in notebooks/02-Image Analysis/main.ipynb)

# 4. Run prediction
predictions = clf.predict(X_features)

# 5. Compare features
print(df_competitors[['title', 'predicted_bsr', 'image_count']])
```

### Workflow 2: Optimize Product Listing

```python
# 1. Get current product data
current_product = get_product_data('B08HR74WV4')

# 2. Predict current BSR
current_prediction = predict_bsr(current_product)

# 3. Simulate improvements
optimized = current_product.copy()
optimized['image_count'] = 7  # Add more images
optimized['bg_white_pct'] = 0.85  # Improve background
optimized['clutter_score'] = -0.5  # Reduce clutter

# 4. Predict optimized BSR
optimized_prediction = predict_bsr(optimized)

# 5. Compare
print(f"Current BSR: {current_prediction}")
print(f"Optimized BSR: {optimized_prediction}")
print(f"Improvement: {current_prediction - optimized_prediction} positions")
```

### Workflow 3: A/B Test Image Quality

```python
# Compare two image sets
images_a = ['image1_a.jpg', 'image2_a.jpg', 'image3_a.jpg']
images_b = ['image1_b.jpg', 'image2_b.jpg', 'image3_b.jpg']

def analyze_image_set(image_paths):
    metrics = []
    for path in image_paths:
        m = tech_quality_for_path(path)
        metrics.append(m)
    return pd.DataFrame(metrics).mean()

metrics_a = analyze_image_set(images_a)
metrics_b = analyze_image_set(images_b)

print("Set A vs Set B:")
print(metrics_a.compare(metrics_b))
```

---

## FAQs

### General Questions

**Q: Do I need Amazon SP-API credentials to use this system?**  
A: No, the pre-downloaded data in `data/` directory allows you to run all analyses. API credentials are only needed for downloading new product data.

**Q: How long does it take to run the complete pipeline?**  
A: With pre-processed data: 10-15 minutes. From scratch: 4-6 hours (mostly image processing).

**Q: Can I use this for products outside electronics?**  
A: Yes, but you'll need to collect new data for your category. The model architecture works for any category.

**Q: What if I don't have product images?**  
A: You can still use text and metadata features, but accuracy will be lower. Image features are the most important predictors.

### Technical Questions

**Q: Why is classification accuracy (90%) higher than regression R² (0.27)?**  
A: Predicting if a product will rank is easier than predicting exact rank. BSR has high variance and many confounding factors.

**Q: Can I improve model performance?**  
A: Yes, try:
- Adding more features (price, reviews, ratings)
- Using deep learning for images (ResNet, EfficientNet)
- Adding temporal features (seasonality, trends)
- Collecting more training data

**Q: How do I handle missing data?**  
A: The pipeline uses median imputation for missing values. For better results, collect complete data or use advanced imputation methods.

**Q: What's the minimum dataset size?**  
A: For reliable models: 5,000+ products. Current dataset has 19,000 products.

### Data Questions

**Q: Where does the training data come from?**  
A: Amazon SP-API (product listings) + web scraping (images) + manual labeling (quality metrics).

**Q: Is the data biased toward any category?**  
A: Yes, electronics products are over-represented. Results may not generalize to other categories without retraining.

**Q: Can I add my own products to the dataset?**  
A: Yes, follow the data format in `data/17k_products_amazon_data.csv` and add your products.

### Model Questions

**Q: Why XGBoost instead of deep learning?**  
A: XGBoost is faster, more interpretable, and works well with tabular features. Deep learning could improve image analysis.

**Q: Can I deploy this model to production?**  
A: Yes, but consider:
- Model retraining schedule (BSR changes over time)
- API rate limits
- Inference latency requirements
- Model monitoring for drift

**Q: What's the model's business value?**  
A: Predict product success before launch, optimize listings, identify improvement opportunities, benchmark against competitors.

---

## Support and Resources

### Getting Help

**Email**: dsuniaga001@gmail.com

**Documentation**:
- `README.md` - Project overview
- `README.txt` - Directory structure
- `INSTALLATION_GUIDE.md` - Setup instructions
- `reports/dataset_audit.md` - Data validation

### External Resources

- **Amazon SP-API Docs**: https://developer-docs.amazon.com/sp-api/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **OpenCV Tutorials**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- **Pandas Guide**: https://pandas.pydata.org/docs/user_guide/

### Best Practices

1. **Always activate the conda environment** before running code
2. **Save models after training** to avoid retraining
3. **Use Parquet files** for faster loading
4. **Process images in batches** to manage memory
5. **Validate predictions** against known products
6. **Monitor API usage** to avoid rate limits

---

## Appendix

### Glossary

- **ASIN**: Amazon Standard Identification Number
- **BSR**: Best Seller Rank (lower = better sales)
- **SP-API**: Amazon Selling Partner API
- **Clutter Score**: Image complexity metric (higher = more cluttered)
- **Feature Importance**: Measure of feature's contribution to predictions
- **R²**: Coefficient of determination (model fit quality)
- **RMSE**: Root Mean Squared Error (prediction error)
- **F1 Score**: Harmonic mean of precision and recall

### Feature List

**Image Features** (12 features):
- edge_density, bg_white_pct, bg_neutral_pct
- color_entropy, largest_cluster_pct
- clutter_score, image_count
- Plus Z-scored versions of above

**Text Features** (8 features):
- title_char_len, title_word_len, title_avg_word
- title_readability, title_syllables
- title_has_premium, title_has_set, title_has_new

**Metadata Features** (varies):
- Category, brand, release date
- A+ content presence, brand story presence

### Sample Commands Cheatsheet

```bash
# Activate environment
conda activate amazon-forecast

# Start Jupyter
jupyter notebook

# Run specific notebook
jupyter nbconvert --to notebook --execute notebook.ipynb

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Update environment
conda env update -f environment.yml --prune

# Export results
jupyter nbconvert --to html notebook.ipynb --output results.html
```

---

**User Manual Version**: 1.0  
**Last Updated**: December 2025  
**Author**: Dylan Suniaga

