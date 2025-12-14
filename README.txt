================================================================================
PROJECT DIRECTORY STRUCTURE DOCUMENTATION
================================================================================
Project: Amazon Best Seller Rank Prediction using Multimodal Analysis
Author: Dylan Suniaga
Date: December 2025
================================================================================

This document provides a complete overview of the project directory structure,
including all source code, data files, notebooks, and deployment components.

================================================================================
ROOT DIRECTORY STRUCTURE
================================================================================

cis_amazon_forecast_proj/
│
├── README.md                 - Main project documentation and overview
├── README.txt                - This file: Complete directory structure guide
├── INSTALLATION_GUIDE.md     - Step-by-step installation instructions
├── USER_MANUAL.md            - User manual and usage guide
├── LICENSE                   - Non-commercial license terms
├── environment.yml           - Conda environment specification
├── main.ipynb                - Main project notebook (legacy)
│
├── data/                     - All data files and datasets
├── notebooks/                - Jupyter notebooks organized by workflow stage
├── src/                      - Source code modules and utilities
├── models/                   - Saved machine learning models
├── reports/                  - Generated analysis reports
├── docs/                     - Additional documentation
└── tests/                    - Unit tests (placeholder)

================================================================================
1. DATA DIRECTORY (data/)
================================================================================
Contains all raw, intermediate, and processed datasets.

data/
│
├── Raw Data Files:
│   ├── 17k_products_amazon_data.csv         - Main dataset with product info
│   ├── all_keywords_merged.csv              - Merged keyword data (CSV)
│   ├── all_keywords_merged.parquet          - Merged keyword data (Parquet)
│   ├── data_with_scraper.csv                - Data with web scraping results
│   └── data_merger.ipynb                    - Notebook for merging datasets
│
├── batches/                                  - Product data by category
│   ├── audio_headphones_catalog_full_*.csv/parquet
│   ├── computers_catalog_full_*.csv/parquet
│   ├── gaming_catalog_full_*.csv/parquet
│   ├── mobile_phones_catalog_full_*.csv/parquet
│   ├── other_catalog_full_*.csv/parquet
│   ├── photography_catalog_full_*.csv/parquet
│   ├── smart_devices_catalog_full_*.csv/parquet
│   ├── smart_home_catalog_full_*.csv/parquet
│   └── tv_catalog_full_*.csv/parquet
│   Note: Each category has both CSV and Parquet formats
│
├── image_analysis_data/                      - Processed image features
│   ├── df_with_clutter_features.csv         - Clutter analysis results
│   └── yolo_update_data.csv                 - YOLO object detection data
│
└── images_amz/                               - Product images (19K+ images)
    └── images_amz.zip                        - Compressed image archive

DATA FILE DESCRIPTIONS:
- 17k_products_amazon_data.csv: Contains ASIN, title, brand, sentiment, ratings,
  customer reviews, BSR rankings, and Amazon SP-API metadata
- all_keywords_merged: Combined product data across all search keywords
- data_with_scraper.csv: Enhanced data with web-scraped information
- df_with_clutter_features.csv: Image clutter scores and visual quality metrics
- yolo_update_data.csv: Object detection results and image composition data

================================================================================
2. NOTEBOOKS DIRECTORY (notebooks/)
================================================================================
Jupyter notebooks organized by project workflow stages.

notebooks/
│
├── 00-Data Downloading/
│   ├── data_download.ipynb           - Amazon SP-API data collection
│   ├── scrapingdog_api.ipynb         - Web scraping implementation
│   └── columns.csv                   - Column schema documentation
│
├── 01-Base Model with Visuals/
│   ├── bsr_analysis.ipynb            - Initial BSR prediction with Random Forest
│   ├── visual_additions.ipynb        - Visual feature engineering
│   ├── bsr_visual_data.csv           - Visual features dataset
│   └── random_forest_bsr_predictions.csv - Baseline model predictions
│
├── 02-Image Analysis/
│   ├── main.ipynb                    - Image quality analysis pipeline
│   ├── runpod_main.ipynb             - GPU-accelerated processing version
│   ├── yolov8n.pt                    - YOLOv8 model weights
│   └── tech_quality_batches/         - Batch-processed image metrics
│       └── tech_batch_*.parquet      - 36 batches (500 images each)
│
├── 03-NLP/
│   ├── 00_dataset_audit.ipynb        - Data quality checks and validation
│   └── 01_text_preprocessing.ipynb   - Text cleaning and feature extraction
│
└── 04-Final Modeling/
    └── main.ipynb                    - Final XGBoost models (classification + regression)

NOTEBOOK WORKFLOW:
1. Stage 00: Download product data from Amazon SP-API
2. Stage 01: Baseline Random Forest model with visual features
3. Stage 02: Deep image analysis (OpenCV + YOLO)
4. Stage 03: NLP preprocessing and text feature engineering
5. Stage 04: Final multimodal XGBoost models

================================================================================
3. SOURCE CODE DIRECTORY (src/)
================================================================================
Reusable Python modules and utility functions.

src/
│
├── nlp/                              - Natural Language Processing modules
│   ├── __init__.py                   - Package initializer
│   ├── io.py                         - Data loading utilities
│   └── preprocess.py                 - Text preprocessing functions
│
├── utils/                            - General utility functions
│   ├── __init__.py                   - Package initializer
│   ├── spapi_helper.py               - Amazon SP-API wrapper functions
│   └── downloader.ipynb              - Data download utilities
│
├── image_analysis/                   - Image processing modules (placeholder)
│
└── main.ipynb                        - Source code demonstrations

MODULE DESCRIPTIONS:

src/nlp/io.py:
- read_data_pd(): Load CSV/Parquet data into pandas DataFrames
- Data validation and schema checking

src/nlp/preprocess.py:
- build_preprocessed_frame(): Merge and clean text data
- Text tokenization and normalization
- Feature extraction (title length, readability, keywords)
- Sentiment parsing from JSON format

src/utils/spapi_helper.py:
- Amazon SP-API authentication (LWA tokens)
- Catalog item retrieval
- Rate limiting and retry logic
- ASIN search and hydration

================================================================================
4. MODELS DIRECTORY (models/)
================================================================================
Saved machine learning model files.

models/
└── (Empty - models are trained in notebooks and can be saved here)

RECOMMENDED MODEL FILES:
- xgboost_classifier.pkl      - BSR classification model
- xgboost_regressor.pkl        - BSR regression model
- random_forest_baseline.pkl   - Initial baseline model
- feature_scaler.pkl           - Feature preprocessing scaler

================================================================================
5. REPORTS DIRECTORY (reports/)
================================================================================
Generated analysis reports and documentation.

reports/
└── dataset_audit.md           - Data quality audit report

REPORT CONTENTS:
- File checksums (SHA-256) for reproducibility
- Row counts and null value statistics
- Data validation results
- Column schema documentation

================================================================================
6. DOCUMENTATION DIRECTORY (docs/)
================================================================================
Additional project documentation.

docs/
└── (Empty - available for supplementary documentation)

================================================================================
7. TESTS DIRECTORY (tests/)
================================================================================
Unit tests and integration tests.

tests/
└── (Empty - placeholder for test files)

RECOMMENDED TEST STRUCTURE:
- test_nlp.py           - Test NLP preprocessing functions
- test_image.py         - Test image analysis pipeline
- test_models.py        - Test model training and prediction
- test_api.py           - Test SP-API integration

================================================================================
KEY FILES IN ROOT DIRECTORY
================================================================================

README.md:
- Comprehensive project overview
- Key findings and model performance
- Methodology and technical approach
- Setup instructions
- Contact information

environment.yml:
- Conda environment specification
- All Python package dependencies
- Package versions for reproducibility

LICENSE:
- Attribution-NonCommercial license
- Usage rights and restrictions
- Commercial use policy

INSTALLATION_GUIDE.md:
- Step-by-step setup instructions
- Environment configuration
- Troubleshooting tips
- Platform-specific guidance (macOS/Windows)

USER_MANUAL.md:
- How to run the analysis pipeline
- Input data requirements
- Output interpretation
- API configuration

================================================================================
DATA FLOW DIAGRAM
================================================================================

1. DATA COLLECTION:
   Amazon SP-API → notebooks/00-Data Downloading/ → data/batches/

2. DATA MERGING:
   data/batches/*.csv → data_merger.ipynb → data/all_keywords_merged.csv

3. IMAGE ANALYSIS:
   data/images_amz/ → notebooks/02-Image Analysis/ → data/image_analysis_data/

4. NLP PROCESSING:
   data/17k_products_amazon_data.csv → notebooks/03-NLP/ → cleaned text features

5. FEATURE INTEGRATION:
   Image features + Text features + Metadata → notebooks/04-Final Modeling/

6. MODEL TRAINING:
   Combined features → XGBoost Classifier + Regressor → Predictions

7. RESULTS:
   Model predictions → reports/ + notebooks/04-Final Modeling/

================================================================================
FILE FORMATS USED
================================================================================

- .csv         - Comma-separated values (human-readable data)
- .parquet     - Apache Parquet (compressed columnar format)
- .ipynb       - Jupyter Notebook (interactive analysis)
- .py          - Python source code
- .pkl         - Pickle (serialized Python objects)
- .pt          - PyTorch model weights
- .md          - Markdown documentation
- .txt         - Plain text
- .yml/.yaml   - YAML configuration
- .zip         - Compressed archives

================================================================================
IMPORTANT NOTES
================================================================================

1. ENVIRONMENT VARIABLES:
   - SP-API credentials should be stored in .env file (not tracked in git)
   - Required variables: CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN

2. LARGE FILES:
   - images_amz/ directory contains 19,000+ product images (~2GB)
   - Consider using images_amz.zip for distribution
   - Batch files in data/batches/ are provided in both CSV and Parquet

3. REPRODUCIBILITY:
   - Random seeds are set to 42 in all models
   - reports/dataset_audit.md contains file checksums
   - environment.yml specifies exact package versions

4. PERFORMANCE:
   - notebooks/02-Image Analysis/runpod_main.ipynb is optimized for GPU
   - Batch processing saves results incrementally (resumable)
   - Parquet format reduces memory usage vs CSV

5. DATA PRIVACY:
   - No customer PII is included in datasets
   - Only public Amazon product information
   - Sentiment data is aggregated and anonymized

================================================================================
GETTING STARTED
================================================================================

1. Read INSTALLATION_GUIDE.md for environment setup
2. Read USER_MANUAL.md for usage instructions
3. Review notebooks/ in order (00 → 01 → 02 → 03 → 04)
4. Check reports/dataset_audit.md for data validation
5. Contact dsuniaga001@gmail.com for questions

================================================================================
VERSION INFORMATION
================================================================================

Project Version: 1.0
Python Version: 3.10+
Last Updated: December 2025
Documentation Version: 1.0

================================================================================
END OF DIRECTORY STRUCTURE DOCUMENTATION
================================================================================

