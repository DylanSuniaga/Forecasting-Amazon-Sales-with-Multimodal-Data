# Amazon Best Seller Rank Prediction using Multimodal Analysis

This project predicts Amazon product Best Seller Rank (BSR) using a multimodal machine learning approach that combines **product images**, **text/NLP features**, and **metadata**. The goal is to understand which product characteristics drive sales performance on Amazon.

---

## 🎯 Project Overview

This analysis uses **~19,000 Amazon electronics products** to predict sales performance through a two-stage modeling approach:

1. **Classification Model**: Predicts whether a product will be ranked (have BSR) or not
2. **Regression Model**: Predicts the actual BSR value for ranked products

The project demonstrates that multimodal features (visual + text + metadata) can effectively predict product performance, with **image quality** and **product presentation** being key factors.

---

## 📊 Key Findings

### Model Performance

#### Classification (Has BSR Ranking?)
- **XGBoost Classifier**: 90% accuracy, 0.94 F1 score, 0.85 ROC AUC
- Successfully identifies products likely to achieve sales rankings
- Far outperforms Logistic Regression baseline (74% accuracy)

#### Regression (Predict BSR Value)
- **XGBoost Regressor**: R² = 0.27, RMSE = 1.76 (log-space)
- Explains 27% of variance in Best Seller Rank
- Average prediction error: ~2,500 BSR points
- Confidence stratification analysis shows significant performance differences:
  - High-confidence predictions (top 25%): Lower RMSE, higher accuracy
  - Low-confidence predictions (bottom 25%): Higher RMSE, require manual review

### Important Features Discovered

**Top Predictors for BSR:**
1. **Background neutral percentage** (13.3% importance) - Clean, neutral backgrounds correlate with better rankings
2. **Largest cluster percentage** (10.9%) - Image color composition affects sales
3. **Background white percentage** (9.3%) - Professional white backgrounds matter
4. **Image count** (8.4%) - More product images correlate with better performance
5. **Clutter score** (7.9%) - Less cluttered images perform better

**Text & NLP Features:**
- Title length and readability scores
- Keyword presence (premium, bundle, new, size indicators)
- Customer sentiment distribution (positive/negative/mixed)
- Character count and word complexity

**Image Quality Metrics:**
- Sharpness (Laplacian variance)
- Contrast and saturation
- Edge density and color entropy
- Technical quality composite scores

---

## 🔬 Methodology

### 1. Data Collection (`notebooks/00-Data Downloading/`)
- **Initial Collection**: Amazon SP-API for discovering product ASINs and basic metadata
  - Used for keyword-based product discovery across electronics categories
- **Data Enrichment**: ScrapingDog API (paid service) for comprehensive product data scraping
  - Customer sentiments (`sd_customer_sentiments`), prices (`sd_price`, `sd_list_price`)
  - Ratings (`sd_average_rating`, `sd_total_reviews`, `sd_ratings_count`)
  - Sales metrics (`sd_number_bought_past_month`)
  - Additional product details not available via SP-API
- **Keywords**: Electronics categories (keyboards, mice, phones, TVs, gaming, photography, etc.)
- **Final Dataset**: `data_with_scraper.csv` - ~19,000 products with enriched features from both sources

### 2. Image Analysis (`notebooks/02-Image Analysis/`)
- **Computer Vision Pipeline**: OpenCV-based analysis
- **Clutter Detection**: Edge density, color clustering, background analysis
- **Quality Metrics**: Sharpness, exposure, contrast, saturation, noise estimation
- **YOLO Integration**: Object detection for image composition

### 3. NLP Processing (`notebooks/03-NLP/`)
- **Text Preprocessing**: Tokenization and cleaning
- **Feature Extraction**: 
  - Title length, word count, average word length
  - Readability scores (Flesch Reading Ease, syllable count)
  - Keyword flags (premium, bundle, new, size indicators)
- **Sentiment Analysis**: JSON parsing of customer sentiment data from ScrapingDog API
  - Extracts positive, negative, and mixed sentiment counts
  - Sentiment character length as feature

### 4. Base Modeling (`notebooks/01-Base Model with Visuals/`)
- **Random Forest Baseline**: Initial BSR prediction using visual features
- **Results**: R² = 0.255, identified key image quality predictors

### 5. Final Multimodal Model (`notebooks/04-Final Modeling/`)
- **Feature Integration**: Combined 80+ features from images, text, and metadata
- **Data Source**: `data_with_scraper.csv` - enriched dataset with ScrapingDog data
- **Two-Stage Pipeline**:
  1. Classification: Predict ranking eligibility (XGBoost Classifier)
  2. Regression: Predict actual BSR for ranked products (XGBoost Regressor)
- **XGBoost Models**: Optimized with hyperparameter tuning
- **Confidence Stratification**: Analysis of prediction quality by confidence levels

---

## 💡 Business Insights

1. **Image Quality Matters**: Professional product photography with clean backgrounds significantly impacts sales ranking
2. **Visual Simplicity Wins**: Less cluttered images with neutral/white backgrounds perform better
3. **Multiple Images Help**: Products with more images tend to rank better
4. **Title Optimization**: Readable titles with relevant keywords correlate with better performance
5. **Sentiment Signals**: Customer sentiment distribution provides predictive power

---

## 📈 Potential Applications

- **Product Listing Optimization**: Predict BSR before launch to optimize listings
- **Competitive Analysis**: Understand what makes competitor products successful
- **A/B Testing**: Test different image/title combinations
- **Inventory Planning**: Predict demand based on listing characteristics
- **Marketing Strategy**: Focus resources on products with predicted high rankings

---

---

## 🛠️ Technical Stack

- **Python 3.10+**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost
- **Computer Vision**: OpenCV, YOLOv8 (ultralytics)
- **NLP**: textstat, JSON parsing for sentiment
- **Data Collection**: Amazon SP-API, ScrapingDog API (paid)
- **Visualization**: matplotlib, seaborn

---

# 📦 Setup Guide

This document walks you through creating the **Conda environment** from our `.yml` file on **macOS** and **Windows**, optionally setting it up in **VS Code**, and the **Git workflow** for cloning, branching, committing, and pushing to this repo. Please read carefully and follow the steps.

---

## 1) Prerequisites

- **Anaconda** or **Miniconda** installed  
  - Download: https://www.anaconda.com/download or https://docs.conda.io/en/latest/miniconda.html
- **Git** installed  
  - Download: https://git-scm.com/downloads
- **VS Code** (optional, recommended)  
  - Download: https://code.visualstudio.com/

> I'll provide commands for both macOS and Windows. Use the section for your OS.
---

## 2) Create the Conda Environment from `.yml`

**Assumptions:**
- The environment file is named `environment.yml` (if it’s different, replace the filename below).
- The environment name inside the file will be created automatically (you can check the `name:` field in the `.yml`).

### macOS

1. Open **Terminal**.
2. Navigate to the project folder:
   `cd /path/to/your/project`
3. Create the environment from the YAML:
   `conda env create -f environment.yml`
4. Activate it:
   `conda activate <env-name>`
   Replace `<env-name>` with the name specified in `environment.yml` under `name:` (e.g., `amazon-forecast`).
5. (Optional) If we update the YAML later, update your env:
   `conda env update -f environment.yml --prune`

### Windows

> Use **Anaconda Prompt** (recommended) or **PowerShell** with `conda init` configured.

1. Open **Anaconda Prompt**.
2. Navigate to the project folder:
   `cd C:\path\to\your\project`
3. Create the environment from the YAML:
   `conda env create -f environment.yml`
4. Activate it:
   `conda activate <env-name>`
5. (Optional) Update later:
   `conda env update -f environment.yml --prune`

**Troubleshooting tips:**
- If `conda` is not recognized, run `conda init` for your shell and restart it:
  - macOS (zsh/bash): `conda init zsh` or `conda init bash`
  - Windows (PowerShell): `conda init powershell` (then restart PowerShell)
- If package conflicts occur, make sure you are using the latest `conda`:
  conda update -n base -c defaults conda

---

## 3) Optional: Set Up in VS Code

### Install VS Code Extensions
- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **Jupyter** (Microsoft), if you’ll use notebooks

### Select the Conda Interpreter (macOS & Windows)
1. Open the project folder in VS Code.
2. Press **Ctrl/Cmd + Shift + P** → type **“Python: Select Interpreter”**.
3. Pick the interpreter that shows your Conda env name (e.g., `Python 3.x ('amazon-forecast')`).

### VS Code Terminal Uses Conda Env
- Open a new terminal **inside VS Code** (Terminal → New Terminal).
- If it doesn’t auto-activate, run:
  conda activate <env-name>

### Jupyter Notebooks (Optional)
- When opening a `.ipynb`, click the **kernel** (top-right in notebook) and select your Conda env.

---

## 4) Connect to This GitHub Repo (VERY IMPORTANT)

> Choose **HTTPS**.

### One-Time Git Setup (Any OS)
```
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### Clone the Repo

**HTTPS:**
```
git clone https://github.com/DylanSuniaga/Forecasting-Amazon-Sales-with-Multimodal-Data.git

cd Forecasting-Amazon-Sales-with-Multimodal-Data
```

**Keep your local copy updated:**
```
git pull origin main
```

---

## 5) Branching & Commit Workflow (Please follow exactly)

> **I (the maintainer) will handle merges.** You should work on branches and open Pull Requests. Do **not** push directly to `main`.

### When to Create a Branch
- **Always** create a new branch before starting work on a task/feature/bugfix.
- Use clear names:  
  - feature/data-loader  
  - fix/image-preprocessor  
  - docs/readme-setup

### Create & Switch to a New Branch
```
git checkout -b <your-branch-name>
```

### Make Changes, Then Stage & Commit
```
git add .
git commit -m "Short, clear message about what you changed"
```

> Commit often with small, descriptive messages.

### Push Your Branch to GitHub
```
git push origin <your-branch-name>
```

Then go to GitHub and open a **Pull Request** from your branch into `main`.  
**Do not merge** — I will handle the merge. Please send me a text once you do this.

### Pull Latest Changes Before Working
Always do this **before** you start your day’s work to avoid conflicts:
```
git checkout main
git pull origin main
git checkout <your-branch-name>
git merge main
```

Resolve any conflicts locally if they appear, then continue working. If you are unsure, send me a text.

---

## 6) What **NOT** to Commit

Please **do not commit**:
- Large raw datasets (e.g., `data/raw/`), especially anything not meant for version control
- Credentials, API keys, tokens
- `.env` files or secrets
- System files: `.DS_Store`, `Thumbs.db`

---

## 📞 Contact

For questions or collaboration opportunities:

**Dylan Suniaga**  
📧 Email: [dsuniaga001@gmail.com](mailto:dsuniaga001@gmail.com)  
🔗 GitHub: [DylanSuniaga](https://github.com/DylanSuniaga)

---

## 📄 Citation

If you use this work in research or publications, please cite:

```
Dylan Suniaga (2025). Amazon Best Seller Rank Prediction using Multimodal Analysis.
GitHub repository: https://github.com/DylanSuniaga/Forecasting-Amazon-Sales-with-Multimodal-Data
```

---

**© 2025 Dylan Suniaga. See LICENSE file for terms of use.**
