# Forecasting-Amazon-Sales-with-Multimodal-Data
This project focuses on forecasting Amazon product sales by combining reviews, ratings, and product image quality. By analyzing both customer feedback and visual presentation, it highlights how these factors contribute to sales performance.

## Dataset & Preprocessing  
- **Data Sources:** Amazon product listings, reviews, and product images (tentative)
- **Preprocessing:**  
  - Clean and tokenize reviews  
  - Extract features such as ratings and review counts  
  - Assess image quality using computer vision metrics  
  - Merge structured and unstructured data into a single dataset

---

## Methodology  
- **Text Data:** Natural Language Processing (sentiment analysis, embeddings)  
- **Image Data:** Convolutional Neural Networks (CNNs) for quality/feature extraction  
- **Sales Forecasting:** Regression and machine learning models trained on combined features  

---

## Project Completion Guide  

1. **Data Collection** – Gather Amazon product listings, reviews, and images, as well as a desired label (person buys, person leaves a review, etc.)  
2. **Data Cleaning** – Remove duplicates, handle missing values, preprocess text and images  
3. **Feature Engineering** – Create numerical features from ratings, reviews, and image scores  
4. **Exploratory Data Analysis (EDA)** – Identify trends and correlations  
5. **Model Training** – Train models separately on text, images, and combined features  
6. **Model Evaluation** – Compare performance with metrics such as RMSE, R², and accuracy  
7. **Final Integration** – Build a multimodal forecasting model  
8. **Results & Visualization** – Summarize findings with charts and metrics  

---
# Team Setup Guide

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
