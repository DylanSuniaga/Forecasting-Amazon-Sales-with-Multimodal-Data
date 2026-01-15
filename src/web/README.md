# Amazon Product Launch Viability - Web Application

A capstone-grade evaluation console for predicting Amazon product launch viability using multimodal image analysis.

## Overview

This web application provides:
- **Product Evaluation**: Upload product images and metadata to receive launch viability predictions
- **Category Context**: View competitive difficulty and baseline metrics by category
- **Image Evidence**: Detailed image quality analysis and comparison to category medians
- **Model Overview**: Architecture diagram and API integration examples
- **Validation Snapshot**: Model performance metrics and calibration data

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js UI    │────▶│   FastAPI API   │────▶│  ML Models      │
│   (Bento Grid)  │     │   (REST)        │     │  (XGBoost)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         │                      │                       │
         ▼                      ▼                       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    Feature Extraction                            │
  │  - CNN Embeddings (EfficientNet B0)                             │
  │  - CLIP Embeddings (ViT-B-32)                                   │
  │  - Quality Metrics (brightness, contrast, sharpness, etc.)      │
  │  - Composition Metrics (edge density, white bg %, occupancy)    │
  └─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- (Optional) CUDA-capable GPU for faster inference

### 1. Backend Setup

```bash
cd src/web/backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For full functionality with embeddings:
pip install torch torchvision timm open-clip-torch

# Start the API server
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/health`

### 2. Frontend Setup

```bash
cd src/web/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The UI will be available at `http://localhost:3000`

### 3. (Optional) Export Models from Notebooks

For real model predictions instead of placeholders:

```bash
cd notebooks/02-Image\ Analysis

# Option 1: Run the export script
python save_models_for_web.py

# Option 2: Add export cells to notebooks and run them
# See save_models_for_web.py for the cell code to add
```

Models will be saved to `src/web/backend/models/`

## Directory Structure

```
src/web/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py          # API endpoints
│   │   ├── core/
│   │   │   └── config.py          # Configuration
│   │   ├── models/
│   │   │   └── schemas.py         # Pydantic models
│   │   ├── services/
│   │   │   ├── image_features.py  # Feature extraction
│   │   │   ├── model_registry.py  # Model loading
│   │   │   └── inference.py       # Prediction logic
│   │   └── main.py                # FastAPI app
│   ├── models/                    # Trained models (.pkl)
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx           # Main page
│   │   │   ├── layout.tsx         # Root layout
│   │   │   └── globals.css        # Global styles
│   │   ├── components/            # React components
│   │   ├── lib/
│   │   │   ├── api.ts             # API client
│   │   │   └── utils.ts           # Utility functions
│   │   └── types/
│   │       └── index.ts           # TypeScript types
│   ├── package.json
│   └── tailwind.config.ts
│
└── README.md                      # This file
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/categories` | GET | List all categories |
| `/api/subcategories?category=...` | GET | Get subcategories |
| `/api/evaluate` | POST | Evaluate a product |
| `/api/embeddings/image` | POST | Extract image embeddings |
| `/api/models` | GET | List loaded models |
| `/api/validation/summary` | GET | Get validation metrics |

### Example: Evaluate Product

```bash
curl -X POST "http://localhost:8000/api/evaluate" \
  -F "title=Premium Water Bottle 32oz" \
  -F "description=Insulated stainless steel" \
  -F "category=Home & Kitchen" \
  -F "images=@product_image.jpg"
```

Response:
```json
{
  "launch_viability_score": 72.5,
  "bsr_entry_probability": 0.85,
  "expected_rank_band": "Top 25%",
  "competitive_intensity": 68.0,
  "confidence": 0.82,
  "strengths": [...],
  "risks": [...],
  "image_quality": {...},
  "product_hash": "abc123def456"
}
```

## Configuration

### Backend Environment Variables

Create `.env` in `src/web/backend/`:

```env
# Device for ML inference (cpu or cuda)
DEVICE=cpu

# Debug mode
DEBUG=true
```

### Frontend Environment Variables

Create `.env.local` in `src/web/frontend/`:

```env
# API base URL (default uses proxy in next.config.js)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Placeholder Mode

When models are not exported from notebooks, the system runs in placeholder mode:

- **Feature Extraction**: Works fully (quality metrics, composition metrics)
- **Embeddings**: Returns random vectors (still demonstrates the pipeline)
- **Predictions**: Returns baseline-informed mock predictions

To enable real predictions:
1. Run classification notebooks to train models
2. Export models using `save_models_for_web.py`
3. Restart the backend server

## Design System

### Colors (Tailwind Custom Palette)

| Name | Hex | Usage |
|------|-----|-------|
| Background | `#0B0F14` | Page background |
| Surface | `#111827` | Card background |
| Surface-2 | `#0F172A` | Secondary surface |
| Primary | `#D4AF37` | Gold accent |
| Primary-2 | `#C8A951` | Gold hover |
| Text | `#E5E7EB` | Main text |
| Muted | `#9CA3AF` | Secondary text |
| Border | `#1F2937` | Borders |
| Success | `#22C55E` | Positive signals |
| Warning | `#F59E0B` | Warnings |
| Danger | `#EF4444` | Errors/risks |

### Components

- **Bento Cards**: Primary container with hover effects
- **Score Cards**: Large numbers with explanatory text
- **Signal Lists**: Strengths/risks with impact scores
- **Tabs**: Navigate between screens

## Development

### Adding New Features

1. **Backend**: Add endpoint in `routes.py`, service logic in `services/`
2. **Frontend**: Add component in `components/`, wire up in `page.tsx`
3. **Types**: Update `types/index.ts` and `models/schemas.py`

### Running Tests

```bash
# Backend
cd src/web/backend
pytest

# Frontend
cd src/web/frontend
npm test
```

## Notebooks Reference

The web application is grounded in these notebooks:

| Notebook | Purpose |
|----------|---------|
| `image_feature_extraction.ipynb` | Feature extraction pipeline |
| `c2_baseline_modeling_clf.ipynb` | BSR classification models |
| `c2_baseline_modeling_keywords_clf.ipynb` | Per-category models |
| `c2_eda_images_clf.ipynb` | Feature ranking and EDA |

## Known Limitations

1. **NLP Features**: Not implemented (notebooks focus on image analysis)
2. **Model Loading**: Requires manual export from notebooks
3. **Embeddings**: GPU recommended for real-time CNN/CLIP extraction
4. **Categories**: Limited to 14 main BSR groups with 300+ products

## License

Part of the Amazon Product Launch Viability Capstone Project.
