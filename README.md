# Forecasting-Amazon-Sales-with-Multimodal-Data
This project focuses on forecasting Amazon product sales by combining reviews, ratings, and product image quality. By analyzing both customer feedback and visual presentation, it highlights how these factors contribute to sales performance.

## 📊 Dataset & Preprocessing  
- **Data Sources:** Amazon product listings, reviews, and product images (tentative)
- **Preprocessing:**  
  - Clean and tokenize reviews  
  - Extract features such as ratings and review counts  
  - Assess image quality using computer vision metrics  
  - Merge structured and unstructured data into a single dataset

---

## 🧠 Methodology  
- **Text Data:** Natural Language Processing (sentiment analysis, embeddings)  
- **Image Data:** Convolutional Neural Networks (CNNs) for quality/feature extraction  
- **Sales Forecasting:** Regression and machine learning models trained on combined features  

---

## ✅ Project Completion Guide  

1. **Data Collection** – Gather Amazon product listings, reviews, and images, as well as a desired label (person buys, person leaves a review, etc.)  
2. **Data Cleaning** – Remove duplicates, handle missing values, preprocess text and images  
3. **Feature Engineering** – Create numerical features from ratings, reviews, and image scores  
4. **Exploratory Data Analysis (EDA)** – Identify trends and correlations  
5. **Model Training** – Train models separately on text, images, and combined features  
6. **Model Evaluation** – Compare performance with metrics such as RMSE, R², and accuracy  
7. **Final Integration** – Build a multimodal forecasting model  
8. **Results & Visualization** – Summarize findings with charts and metrics  

---

## 🚀 Getting Started  

Follow these steps to set up your environment and stay in sync with the team.  

### 1. Clone the Repository  
git clone https://github.com/DylanSuniaga/Forecasting-Amazon-Sales-with-Multimodal-Data.git  
cd repo-name  

### 2. Set Up a Virtual Environment  

#### Option A: pip + venv  
python -m venv venv  
source venv/bin/activate   # On Mac/Linux  
venv\Scripts\activate      # On Windows  

#### Option B: Conda  
conda create -n amazon-forecast python=3.10  
conda activate amazon-forecast  

👉 Once the requirements.txt or environment.yml file is added, you will install dependencies here.  

---

### 3. Git Workflow  

#### Pull the latest changes  
git pull origin main  

#### Create a new branch  
git checkout -b your-branch-name  

#### Stage and commit your changes  
git add .  
git commit -m "Your message about what changed"  

#### Push your changes to GitHub  
git push origin your-branch-name  

Then, open a Pull Request (PR) on GitHub so the team can review and merge.  

---

## ✅ Notes for Teammates  
- Always pull before you start working to get the latest changes.  
- Use branches instead of committing directly to main.  
- Once the project structure and files are created, we will update this README with instructions to run the code.  

---

📜 License: MIT License  
