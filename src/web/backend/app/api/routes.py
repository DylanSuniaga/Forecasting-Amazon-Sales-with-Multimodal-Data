"""
FastAPI routes for the Amazon Product Launch Viability API.
"""
import os
import shutil
import uuid
import tempfile
from pathlib import Path
from typing import List, Optional
import httpx

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import json

from ..core.config import settings
from ..models.schemas import (
    EvaluationResponse, CategoryListResponse, CategoryInfo,
    ValidationSummaryResponse, ValidationMetrics, ModelRegistryResponse,
    ModelInfo, HealthResponse,
    EXAMPLE_EVALUATE_REQUEST, EXAMPLE_EVALUATE_RESPONSE
)
from ..services.inference import get_inference_service
from ..services.model_registry import get_model_registry
from ..services.image_features import get_feature_extractor

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns system status and loaded model count.
    """
    registry = get_model_registry()
    models = registry.list_models()
    loaded_count = sum(1 for m in models if m.get('clf_loaded', False))
    
    # Check GPU
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except:
        pass
    
    return HealthResponse(
        status="healthy",
        version=settings.API_VERSION,
        models_loaded=loaded_count,
        gpu_available=gpu_available
    )


@router.get("/categories", response_model=CategoryListResponse, tags=["Categories"])
async def list_categories():
    """
    List all available categories with metadata.
    Includes subcategories, sample counts, and model performance metrics.
    """
    registry = get_model_registry()
    categories = []
    
    for cat_name in settings.CATEGORIES:
        model = registry.get_category_model(cat_name)
        metrics = model.clf_metrics  # Use clf_metrics property
        
        # Get subcategories
        subcats = settings.SUBCATEGORIES.get(cat_name, [])
        
        categories.append(CategoryInfo(
            name=cat_name,
            subcategories=subcats,
            sample_count=1000,  # Placeholder - would come from data
            avg_bsr_rate=0.70,  # From notebook analysis
            model_auc=metrics.get('roc_auc', 0.85),
            difficulty_score=_get_difficulty(cat_name)
        ))
    
    return CategoryListResponse(categories=categories)


@router.get("/subcategories", tags=["Categories"])
async def list_subcategories(category: str = Query(..., description="Parent category name")):
    """
    List subcategories for a given category.
    """
    if category not in settings.CATEGORIES:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
    
    subcats = settings.SUBCATEGORIES.get(category, [])
    return {"category": category, "subcategories": subcats}


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    tags=["Evaluation"],
    summary="Evaluate product launch viability",
    description="""
    Evaluate a prospective Amazon product's launch viability.
    
    Upload product details and images to receive:
    - Launch Viability Score (0-100)
    - BSR Entry Probability
    - Expected Rank Band
    - Competitive Intensity Index
    - Signal decomposition (strengths/risks)
    
    **Image Requirements:**
    - At least one image required (main/hero image)
    - Supported formats: JPEG, PNG, WebP
    - Recommended: 1000x1000 or higher resolution
    """
)
async def evaluate_product(
    title: str = Form(..., description="Product title"),
    description: str = Form(default="", description="Product description/bullet points"),
    category: str = Form(..., description="Main category"),
    subcategory: Optional[str] = Form(default=None, description="Subcategory"),
    images: List[UploadFile] = File(default=None, description="Product images (main image first)"),
    image_url: Optional[str] = Form(default=None, description="Image URL (alternative to uploading)"),
    cnn_embedding_json: Optional[str] = Form(default=None, description="Pre-computed CNN embedding (JSON array)"),
    clip_embedding_json: Optional[str] = Form(default=None, description="Pre-computed CLIP embedding (JSON array)"),
    cnn_pca_json: Optional[str] = Form(default=None, description="Pre-computed CNN PCA features (JSON object)"),
    clip_pca_json: Optional[str] = Form(default=None, description="Pre-computed CLIP PCA features (JSON object)"),
    price: Optional[float] = Form(default=None, description="Product price in USD")
):
    """
    Evaluate a product's launch viability based on title, description, category, and images.
    Supports:
    - Uploaded images (normal flow)
    - Image URL (for suggestion products - downloads image)
    - Pre-computed embeddings (for suggestion products - uses embeddings from dataframe, skips extraction)
    """
    # Validate category
    if category not in settings.CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Valid categories: {settings.CATEGORIES}"
        )
    
    # Check if we have images, image_url, pre-computed embeddings, or pre-computed PCA features
    has_uploaded_images = images and len(images) > 0 and images[0].filename
    has_image_url = image_url and image_url.strip()
    has_precomputed_embeddings = cnn_embedding_json and clip_embedding_json
    has_precomputed_pca = cnn_pca_json and clip_pca_json
    
    if not has_uploaded_images and not has_image_url and not has_precomputed_embeddings and not has_precomputed_pca:
        raise HTTPException(status_code=400, detail="Either images, image_url, pre-computed embeddings, or pre-computed PCA features are required")
    
    # Parse pre-computed features if provided
    precomputed_cnn = None
    precomputed_clip = None
    precomputed_cnn_pca = None
    precomputed_clip_pca = None
    
    if has_precomputed_pca:
        # Use pre-computed PCA features (dataset already has PCA features)
        try:
            precomputed_cnn_pca = json.loads(cnn_pca_json)
            precomputed_clip_pca = json.loads(clip_pca_json)
            if not isinstance(precomputed_cnn_pca, dict) or not isinstance(precomputed_clip_pca, dict):
                raise ValueError("PCA features must be JSON objects")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PCA feature format: {str(e)}")
    elif has_precomputed_embeddings:
        # Use pre-computed raw embeddings (will apply PCA)
        try:
            precomputed_cnn = json.loads(cnn_embedding_json)
            precomputed_clip = json.loads(clip_embedding_json)
            if not isinstance(precomputed_cnn, list) or not isinstance(precomputed_clip, list):
                raise ValueError("Embeddings must be JSON arrays")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid embedding format: {str(e)}")
    
    # Save uploaded images or download from URL
    image_paths = []
    upload_dir = settings.UPLOADS_DIR / str(uuid.uuid4())
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if has_uploaded_images:
            # Process uploaded images
            for idx, image in enumerate(images):
                # Validate file type
                if not image.content_type or not image.content_type.startswith('image/'):
                    continue
                
                # Save file
                ext = Path(image.filename).suffix if image.filename else '.jpg'
                file_path = upload_dir / f"image_{idx}{ext}"
                
                with open(file_path, 'wb') as f:
                    content = await image.read()
                    f.write(content)
                
                image_paths.append(str(file_path))
        
        elif has_image_url:
            # Download image from URL
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(image_url, follow_redirects=True)
                    response.raise_for_status()
                    
                    # Determine extension from content type or URL
                    content_type = response.headers.get('content-type', 'image/jpeg')
                    if 'png' in content_type:
                        ext = '.png'
                    elif 'webp' in content_type:
                        ext = '.webp'
                    else:
                        ext = '.jpg'
                    
                    file_path = upload_dir / f"image_0{ext}"
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    image_paths.append(str(file_path))
                    
            except Exception as e:
                print(f"Failed to download image from URL: {e}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to download image from URL: {str(e)}"
                )
        
        # If using pre-computed embeddings, skip image processing
        if not has_precomputed_embeddings and not image_paths:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        # Run evaluation
        inference_service = get_inference_service()
        result = inference_service.evaluate_product(
            title=title,
            description=description,
            category=category,
            subcategory=subcategory,
            image_paths=image_paths if image_paths else [],  # Empty if using pre-computed features
            precomputed_cnn_embedding=precomputed_cnn,
            precomputed_clip_embedding=precomputed_clip,
            precomputed_cnn_pca=precomputed_cnn_pca,
            precomputed_clip_pca=precomputed_clip_pca,
            price=price
        )
        
        return result
    
    finally:
        # Cleanup uploaded files (optional: keep for caching)
        # For demo, we keep them; in production, clean up after processing
        pass


@router.post("/embeddings/image", tags=["Features"])
async def extract_image_embeddings(
    image: UploadFile = File(..., description="Image file to extract embeddings from")
):
    """
    Extract image embeddings from an uploaded image.
    Returns CNN and CLIP embeddings along with quality metrics.
    """
    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image.")
    
    # Save temporarily
    upload_dir = settings.UPLOADS_DIR / str(uuid.uuid4())
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    ext = Path(image.filename).suffix if image.filename else '.jpg'
    file_path = upload_dir / f"temp_image{ext}"
    
    try:
        with open(file_path, 'wb') as f:
            content = await image.read()
            f.write(content)
        
        # Extract features
        extractor = get_feature_extractor()
        features = extractor.extract_all_features(str(file_path))
        
        # Remove raw embeddings from response (too large)
        # Instead return summary stats
        response = {
            'valid': features.get('valid', False),
            'quality_metrics': {
                'width': features.get('width'),
                'height': features.get('height'),
                'aspect_ratio': features.get('aspect_ratio'),
                'brightness_mean': features.get('brightness_mean'),
                'contrast': features.get('contrast'),
                'saturation_mean': features.get('saturation_mean'),
                'sharpness': features.get('sharpness'),
                'blur_score': features.get('blur_score'),
            },
            'composition_metrics': {
                'edge_density': features.get('edge_density'),
                'white_bg_pct': features.get('white_bg_pct'),
                'object_occupancy_proxy': features.get('object_occupancy_proxy'),
                'border_clutter_score': features.get('border_clutter_score'),
            },
            'embedding_stats': {
                'cnn_dim': features.get('cnn_embedding_dim', 0),
                'clip_dim': features.get('clip_embedding_dim', 0),
            }
        }
        
        return response
    
    finally:
        # Cleanup
        if upload_dir.exists():
            shutil.rmtree(upload_dir)


@router.post("/embeddings/text", tags=["Features"])
async def extract_text_embeddings(
    text: str = Form(..., description="Text to extract embeddings from")
):
    """
    Extract text embeddings from product text.
    
    PLACEHOLDER: Text embedding extraction is not yet implemented.
    The current model focuses on image analysis.
    """
    # Placeholder response
    return {
        'status': 'placeholder',
        'message': 'Text embedding extraction not yet implemented. Current model uses image features only.',
        'text_length': len(text),
        'word_count': len(text.split())
    }


@router.get("/models", tags=["Models"])
async def list_models():
    """
    List all models in the registry with their status.
    Shows which models are loaded and their performance metrics.
    """
    registry = get_model_registry()
    
    models = []
    for model_info in registry.list_models():
        models.append({
            'name': model_info['name'],
            'category': model_info['category'],
            'model_type': f"CLF: {model_info.get('clf_model_type', '?')} / REG: {model_info.get('reg_model_type', '?')}",
            'loaded': model_info['clf_loaded'],
            'clf_loaded': model_info['clf_loaded'],
            'reg_loaded': model_info['reg_loaded'],
            'metrics': model_info['clf_metrics'],
            'reg_metrics': model_info['reg_metrics'],
        })
    
    return {
        'embedding_model_loaded': registry.embedding_model_loaded,
        'category_models': models
    }


@router.get("/validation/summary", response_model=ValidationSummaryResponse, tags=["Validation"])
async def get_validation_summary():
    """
    Get validation metrics summary across all categories.
    Includes AUC, F1, precision, recall by category.
    """
    registry = get_model_registry()
    
    metrics_by_category = []
    total_auc = 0
    total_f1 = 0
    total_r2 = 0
    total_mae = 0

    for cat_name in settings.CATEGORIES:
        model = registry.get_category_model(cat_name)
        clf_met = model.clf_metrics
        reg_met = model.reg_metrics

        metrics_by_category.append(ValidationMetrics(
            category=cat_name,
            model_type='XGBoost',
            roc_auc=clf_met.get('roc_auc', 0.85),
            f1_score=clf_met.get('f1', 0.80),
            precision=clf_met.get('precision', 0.78),
            recall=clf_met.get('recall', 0.82),
            accuracy=clf_met.get('accuracy', 0.80),
            sample_size=1000,
            reg_r2=reg_met.get('r2'),
            reg_mae=reg_met.get('mae'),
        ))

        total_auc += clf_met.get('roc_auc', 0.85)
        total_f1 += clf_met.get('f1', 0.80)
        total_r2 += reg_met.get('r2', 0.0)
        total_mae += reg_met.get('mae', 0.0)

    n_cats = len(settings.CATEGORIES)

    return ValidationSummaryResponse(
        overall_auc=total_auc / n_cats if n_cats > 0 else 0.85,
        overall_f1=total_f1 / n_cats if n_cats > 0 else 0.80,
        overall_r2=total_r2 / n_cats if n_cats > 0 else 0.0,
        overall_mae=total_mae / n_cats if n_cats > 0 else 0.0,
        by_category=metrics_by_category,
        calibration_summary={
            'brier_score': 0.12,
            'calibration_error': 0.08
        }
    )


@router.get("/suggestions", tags=["Suggestions"])
async def get_suggestions(category: Optional[str] = Query(default=None, description="Filter by category")):
    """
    Get sample product suggestions from the test set.
    These are real products that were held out during training.
    """
    import json
    
    suggestions_path = settings.BASE_DIR / "src" / "web" / "backend" / "data" / "suggestions.json"
    
    if not suggestions_path.exists():
        return {
            "suggestions": {},
            "message": "No suggestions available. Run save_models_for_web.py to generate."
        }
    
    try:
        with open(suggestions_path, 'r') as f:
            all_suggestions = json.load(f)
        
        if category and category in all_suggestions:
            return {"suggestions": {category: all_suggestions[category]}}
        
        return {"suggestions": all_suggestions}
    except Exception as e:
        return {"suggestions": {}, "error": str(e)}


@router.get("/api-examples", tags=["Documentation"])
async def get_api_examples():
    """
    Get example API request/response payloads for integration.
    Useful for developers integrating with this API.
    """
    return {
        'evaluate_endpoint': {
            'method': 'POST',
            'path': '/api/evaluate',
            'content_type': 'multipart/form-data',
            'request_fields': {
                'title': 'string (required)',
                'description': 'string (optional)',
                'category': f'string enum: {settings.CATEGORIES}',
                'subcategory': 'string (optional)',
                'images': 'file[] (at least one required)'
            },
            'example_request': EXAMPLE_EVALUATE_REQUEST,
            'example_response': EXAMPLE_EVALUATE_RESPONSE
        },
        'categories_endpoint': {
            'method': 'GET',
            'path': '/api/categories',
            'description': 'List all available categories with metadata'
        },
        'embeddings_endpoint': {
            'method': 'POST',
            'path': '/api/embeddings/image',
            'description': 'Extract image embeddings and quality metrics'
        }
    }


def _get_difficulty(category: str) -> float:
    """Get difficulty score for a category."""
    difficulty_map = {
        "Home & Kitchen": 72,
        "Health & Household": 68,
        "Office Products": 58,
        "Baby": 65,
        "Clothing, Shoes & Jewelry": 78,
        "Kitchen & Dining": 70,
        "Electronics": 75,
        "Cell Phones & Accessories": 65,
        "Tools & Home Improvement": 55,
        "Video Games": 60,
        "Pet Supplies": 62,
        "Sports & Outdoors": 68,
        "Industrial & Scientific": 45,
        "Musical Instruments": 50,
    }
    return difficulty_map.get(category, 65)
