# POSTER LAYOUT — 48" x 36" Horizontal

> **Print**: 48" wide x 36" tall, landscape. White background.
> **Layout**: Header banner + 3 columns + footer strip.
> **Tool**: PowerPoint, Canva, or Google Slides at 48x36 custom size.
> **Colors**: FIU Blue (#081E3F), FIU Gold (#B6862C), accent green for improvements.

---

## TOP BANNER (full width, ~4")

**Amazon Best Seller Rank Prediction Using Multimodal Machine Learning**

CIS 4911 - Capstone II | Spring 2026 | Florida International University
Dylan Suniaga, Bianca Poggi, Michelle Orozco, Fesal Fayed, Maksim Pikalov | Instructor: Seyedmasoud Sadjadi

> FIU logo left, project icon right. Title 72pt bold. Names 24pt.

**INSERT: `poster_figures/poster_metric_cards.png`** *(report_figures.ipynb — Poster Setup cells)*
> 4 big-number cards: 36.9K Products, 0.891 ROC-AUC, 2,330 Features, R²=0.425

---

## LEFT COLUMN

### The Problem

**INSERT: `poster_figures/poster_problem_flow.png`** *(report_figures.ipynb — Problem Flow cells)*
> 3-box diagram: New Listing → ??? → BSR Rank

- Sellers can't predict if a product will sell **before** investing in inventory
- Amazon BSR only appears **after** sales happen
- **Our goal:** Predict BSR from listing content alone

### Data Overview

**INSERT: `poster_figures/poster_table_data.png`** *(report_figures.ipynb — Poster Table 1)*

**INSERT: `poster_figures/poster_category_breakdown.png`** *(report_figures.ipynb — Data Collection cells)*
> Horizontal bar chart showing product counts per category

### Feature Engineering

**INSERT: `poster_figures/poster_feature_pipeline.png`** *(report_figures.ipynb — Poster Feature Engineering cells)*
> 3-box diagram: Image Features (CNN, CLIP, OpenCV, YOLO) + Text Features (TF-IDF, readability, keywords) + Metadata → 80+ merged features

**INSERT: Clutter Score Comparison** *(report_figures.ipynb, Cell 8)*
> Side-by-side low vs high clutter product images — visually grabs attention

---

## CENTER COLUMN

### Two-Stage Model Pipeline

**INSERT: `poster_figures/poster_pipeline.png`** *(report_figures.ipynb — Poster Pipeline cells)*
> Flowchart: Listing → Feature Extraction → Classifier (95-99%) → YES: Predict BSR / NO: Unlikely to Rank

### Results

**INSERT: `poster_figures/poster_table_results.png`** *(report_figures.ipynb — Poster Table 2)*
> Side-by-side Capstone I vs II with green improvement column

**INSERT: ROC Curve** *(report_figures.ipynb, Cell 22)*
> Shows strong classification AUC = 0.849+ curve vs diagonal

**INSERT: Feature Importance by Category** *(report_figures.ipynb, Cell 19/20)*
> Color-coded bar chart — visual proof that images dominate

### Key Findings

**INSERT: `poster_figures/poster_key_findings.png`** *(report_figures.ipynb — Poster Key Findings cells)*
> 4 callout cards: 2.3x better BSR (backgrounds), 45% lower BSR (6+ images), 3x worse BSR (clutter), 8/10 top features are image-based

---

## RIGHT COLUMN

### Web Application

- Real-time BSR prediction from uploaded images
- Wizard: Category → Image → Title → Results
- NLP feedback with title improvement tips
- Model performance dashboard

**INSERT: `poster_figures/poster_webapp_arch.png`** *(report_figures.ipynb — Web App Architecture cells)*
> Frontend/Backend/ML Registry architecture diagram

**INSERT: Web App Screenshots** *(take from localhost:3000 — product form, score card results, model dashboard)*
> 2-3 screenshots stacked or cascaded

### Tech Stack

**INSERT: `poster_figures/poster_table_techstack.png`** *(report_figures.ipynb — Poster Table 3)*
> Styled table: XGBoost, EfficientNet, CLIP, FastAPI, Next.js, etc.

### Future Work

- Real-time BSR tracking over time
- Multi-marketplace (Amazon UK, DE, JP)
- Causal inference via A/B testing
- Browser extension for live predictions

---

## BOTTOM FOOTER (full width, ~1.5")

GitHub: github.com/DylanSuniaga/Forecasting-Amazon-Sales-with-Multimodal-Data | CIS 4911 Capstone II | Spring 2026 | FIU

> 18pt font. QR code to GitHub repo on the right. FIU logo repeated.

---

# FIGURE PLACEMENT CHEAT SHEET

All poster-specific figures are saved to `notebooks/04-Final Modeling/poster_figures/` after running the notebook. The notebook chart figures need to be right-clicked and saved from the notebook output.

| Where on Poster | Figure File | Notebook Source |
|---|---|---|
| **Banner** — under title | `poster_metric_cards.png` | Poster: Key Metrics cells |
| **Left** — Problem | `poster_problem_flow.png` | Poster: Problem Statement cells |
| **Left** — Data Overview | `poster_table_data.png` | Poster: Table 1 |
| **Left** — Data Overview | `poster_category_breakdown.png` | Poster: Data Collection cells |
| **Left** — Feature Engineering | `poster_feature_pipeline.png` | Poster: Feature Engineering cells |
| **Left** — Feature Engineering | Clutter Comparison | Figure 1 (Cell 6) |
| **Center** — Pipeline | `poster_pipeline.png` | Poster: Two-Stage Pipeline cells |
| **Center** — Solution | `poster_solution_flow.png` | Poster: Solution Flow cells |
| **Center** — Results | `poster_table_results.png` | Poster: Table 2 |
| **Center** — Results | ROC Curve | Figure 8 (Cell 20) |
| **Center** — Results | Feature Importance | Figure 7 (Cell 18) |
| **Center** — Findings | `poster_key_findings.png` | Poster: Key Findings cells |
| **Right** — Web App | `poster_webapp_arch.png` | Poster: Web App Architecture cells |
| **Right** — Web App | Screenshots | localhost:3000 (manual capture) |
| **Right** — Tech Stack | `poster_table_techstack.png` | Poster: Table 3 |

**To generate all poster figures at once:**
1. Open `notebooks/04-Final Modeling/report_figures.ipynb`
2. Run all cells (the poster cells are at the end, after Cell 35)
3. PNGs saved to `notebooks/04-Final Modeling/poster_figures/` at 300 DPI
4. Chart figures (ROC, feature importance, clutter) — right-click output → Save Image As
