"""
Amazon Product Launch Viability API

FastAPI application for evaluating prospective Amazon products.
Provides multimodal analysis using image embeddings and quality metrics.

Run with: uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import settings
from .api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description="""
# Amazon Product Launch Viability API

Evaluate prospective Amazon products for launch viability using multimodal analysis.

## Features

- **Image Analysis**: CNN and CLIP embeddings, quality metrics, composition analysis
- **Category Modeling**: Per-category classification models for BSR entry prediction
- **Signal Decomposition**: Human-readable strengths and risks
- **Integration Ready**: Clean API for integration with other systems

## Key Endpoints

- `POST /api/evaluate` - Evaluate a product
- `GET /api/categories` - List available categories
- `GET /api/models` - View model registry
- `GET /api/validation/summary` - Model performance metrics

## Notes

This API is designed for internal evaluation and industry demonstration.
Some features use placeholder logic pending full model deployment.
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # CORS middleware for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # Next.js dev server
            "http://127.0.0.1:3000",
            "http://localhost:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router, prefix=settings.API_PREFIX)
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.API_TITLE,
            "version": settings.API_VERSION,
            "docs": "/docs",
            "health": "/api/health"
        }
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        print(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
        print(f"Models directory: {settings.MODELS_DIR}")
        print(f"Uploads directory: {settings.UPLOADS_DIR}")
        
        # Pre-load feature extractor
        from .services.image_features import get_feature_extractor
        extractor = get_feature_extractor(device=settings.DEVICE)
        print(f"Feature extractor initialized (device: {settings.DEVICE})")
        
        # Initialize model registry
        from .services.model_registry import get_model_registry
        registry = get_model_registry()
        models = registry.list_models()
        loaded = sum(1 for m in models if m.get('clf_loaded', False))
        print(f"Model registry initialized: {loaded}/{len(models)} category models loaded")

        # Initialize text feature extractor
        from .services.text_features import get_text_feature_extractor
        text_extractor = get_text_feature_extractor()
        print(f"Text feature extractor initialized (loaded: {text_extractor.loaded})")
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
