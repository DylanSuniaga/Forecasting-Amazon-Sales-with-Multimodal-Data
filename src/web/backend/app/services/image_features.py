"""
Image feature extraction service.
Adapted from notebooks/02-Image Analysis/image_feature_extraction.ipynb

This module extracts:
- Quality metrics (brightness, contrast, saturation, sharpness, etc.)
- Composition metrics (edge density, background uniformity, white bg %, etc.)
- CNN embeddings (EfficientNet) [PLACEHOLDER - requires model download]
- CLIP embeddings [PLACEHOLDER - requires model download]
"""
import os
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

warnings.filterwarnings('ignore')

# Core image processing (always available)
import cv2
from PIL import Image

# Optional deep learning imports
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. Embeddings will use placeholders.")

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: timm not available. CNN embeddings will use placeholders.")

try:
    import open_clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("Warning: open_clip not available. CLIP embeddings will use placeholders.")

# Optional for advanced features
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class CNNEmbedder:
    """
    CNN embedding extractor using EfficientNet.
    Adapted from notebook: image_feature_extraction.ipynb
    """
    
    def __init__(self, backbone: str = "efficientnet_b0", device: str = "cpu"):
        self.device = device
        self.backbone = backbone
        self.model = None
        self.transform = None
        self.embed_dim = 1280  # EfficientNet-B0 output dimension
        
        if HAS_TORCH and HAS_TIMM:
            self._load_model()
        else:
            print(f"CNNEmbedder: Running in placeholder mode (torch={HAS_TORCH}, timm={HAS_TIMM})")
    
    def _load_model(self):
        """Load the EfficientNet model."""
        try:
            device = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
            self.device = str(device)
            
            self.model = timm.create_model(
                self.backbone,
                pretrained=True,
                num_classes=0,  # Remove classification head
                global_pool="avg"
            )
            self.model.eval()
            self.model.to(device)
            
            # Get actual embedding dimension
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224).to(device)
                self.embed_dim = self.model(dummy).shape[1]
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print(f"CNNEmbedder loaded: {self.backbone}, dim={self.embed_dim}, device={self.device}")
        except Exception as e:
            print(f"Failed to load CNN model: {e}")
            self.model = None
    
    def extract(self, img_path: str) -> Optional[np.ndarray]:
        """Extract CNN embedding from image."""
        if self.model is None:
            # Return placeholder embedding
            return np.random.randn(self.embed_dim).astype(np.float32) * 0.1
        
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0)
            device = torch.device(self.device)
            tensor = tensor.to(device)
            
            with torch.no_grad():
                embedding = self.model(tensor).cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            print(f"CNN embedding extraction failed: {e}")
            return None


class CLIPEmbedder:
    """
    CLIP embedding extractor.
    Adapted from notebook: image_feature_extraction.ipynb
    """
    
    def __init__(self, model_name: str = "ViT-B-32", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.embed_dim = 512  # CLIP ViT-B-32 output dimension
        
        if HAS_TORCH and HAS_CLIP:
            self._load_model()
        else:
            print(f"CLIPEmbedder: Running in placeholder mode (torch={HAS_TORCH}, clip={HAS_CLIP})")
    
    def _load_model(self):
        """Load the CLIP model."""
        try:
            device = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
            self.device = str(device)
            
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained="openai"
            )
            self.model.eval()
            self.model.to(device)
            
            # Get actual embedding dimension
            with torch.no_grad():
                dummy = self.preprocess(Image.new('RGB', (224, 224))).unsqueeze(0).to(device)
                self.embed_dim = self.model.encode_image(dummy).shape[1]
            
            print(f"CLIPEmbedder loaded: {self.model_name}, dim={self.embed_dim}, device={self.device}")
        except Exception as e:
            print(f"Failed to load CLIP model: {e}")
            self.model = None
    
    def extract(self, img_path: str) -> Optional[np.ndarray]:
        """Extract CLIP embedding from image."""
        if self.model is None:
            # Return placeholder embedding
            return np.random.randn(self.embed_dim).astype(np.float32) * 0.1
        
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = self.preprocess(img).unsqueeze(0)
            device = torch.device(self.device)
            tensor = tensor.to(device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(tensor).cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            print(f"CLIP embedding extraction failed: {e}")
            return None


class ImageFeatureExtractor:
    """
    Complete image feature extraction pipeline.
    Extracts quality, composition, and embedding features from product images.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.cnn_embedder = CNNEmbedder(device=device)
        self.clip_embedder = CLIPEmbedder(device=device)
    
    def compute_quality_features(self, img_path: str) -> Optional[Dict[str, float]]:
        """
        Compute technical quality features for an image.
        Adapted from notebook: image_feature_extraction.ipynb
        """
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            # File size
            filesize_kb = os.path.getsize(img_path) / 1024.0
            
            h, w = img.shape[:2]
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Convert to different color spaces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Brightness
            brightness_mean = float(gray.mean())
            brightness_std = float(gray.std())
            
            # Contrast (same as brightness_std)
            contrast = brightness_std
            
            # Saturation
            saturation_mean = float(hsv[:, :, 1].mean())
            saturation_std = float(hsv[:, :, 1].std())
            
            # Colorfulness (Hasler-Süsstrunk metric)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rg = rgb[:, :, 0].astype(np.float32) - rgb[:, :, 1].astype(np.float32)
            yb = (rgb[:, :, 0].astype(np.float32) + rgb[:, :, 1].astype(np.float32)) / 2 - rgb[:, :, 2].astype(np.float32)
            colorfulness = float(np.sqrt(np.mean(rg**2) + np.mean(yb**2)))
            
            # Sharpness (variance of Laplacian)
            sharpness_laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # Blur score (inverse of sharpness)
            blur_score = 1.0 / (1.0 + sharpness_laplacian_var)
            
            # Entropy
            hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
            hist = hist[hist > 0]
            hist = hist / hist.sum()
            entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
            
            # Exposure clipping
            exposure_clipped_high_pct = float(np.sum(gray > 250) / gray.size * 100)
            exposure_clipped_low_pct = float(np.sum(gray < 5) / gray.size * 100)
            
            return {
                'width': w,
                'height': h,
                'aspect_ratio': aspect_ratio,
                'filesize_kb': filesize_kb,
                'brightness_mean': brightness_mean,
                'brightness_std': brightness_std,
                'contrast': contrast,
                'saturation_mean': saturation_mean,
                'saturation_std': saturation_std,
                'colorfulness': colorfulness,
                'sharpness_laplacian_var': sharpness_laplacian_var,
                'sharpness': sharpness_laplacian_var,  # Alias
                'blur_score': blur_score,
                'entropy': entropy,
                'exposure_clipped_high_pct': exposure_clipped_high_pct,
                'exposure_clipped_low_pct': exposure_clipped_low_pct
            }
        except Exception as e:
            print(f"Quality feature extraction failed: {e}")
            return None
    
    def compute_composition_features(self, img_path: str) -> Optional[Dict[str, float]]:
        """
        Compute composition and background features.
        Adapted from notebook: image_feature_extraction.ipynb
        """
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Edge density (Canny edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.sum(edges > 0) / (w * h))
            
            # Background uniformity (std of border pixels)
            border_width = min(20, w // 10, h // 10)
            if border_width > 0:
                top_border = gray[:border_width, :].flatten()
                bottom_border = gray[-border_width:, :].flatten()
                left_border = gray[:, :border_width].flatten()
                right_border = gray[:, -border_width:].flatten()
                all_borders = np.concatenate([top_border, bottom_border, left_border, right_border])
                bg_uniformity = float(all_borders.std())
            else:
                bg_uniformity = 0.0
            
            # White background percentage
            white_threshold = 240
            white_pixels = np.sum(gray > white_threshold)
            white_bg_pct = float(white_pixels / (w * h) * 100)
            
            # Dominant color count (simplified)
            dominant_color_count = 3  # Default
            if HAS_SKLEARN:
                try:
                    sample_size = min(10000, w * h)
                    indices = np.random.choice(w * h, sample_size, replace=False)
                    pixels = img.reshape(-1, 3)[indices]
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    kmeans.fit(pixels)
                    dominant_color_count = len(np.unique(kmeans.labels_))
                except:
                    pass
            
            # Saliency (simplified: use variance of gradients)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            saliency_map = np.sqrt(grad_x**2 + grad_y**2)
            
            saliency_mean = float(saliency_map.mean())
            saliency_max = float(saliency_map.max())
            saliency_peak_ratio = saliency_max / (saliency_mean + 1e-10)
            
            # Saliency center distance
            saliency_norm = saliency_map / (saliency_map.sum() + 1e-10)
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            center_y = float(np.sum(y_coords * saliency_norm))
            center_x = float(np.sum(x_coords * saliency_norm))
            img_center_y, img_center_x = h / 2, w / 2
            saliency_center_dist = float(np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2))
            
            # Object occupancy proxy
            salient_threshold = saliency_mean * 2
            salient_area = np.sum(saliency_map > salient_threshold)
            object_occupancy_proxy = float(salient_area / (w * h))
            
            # Border clutter score
            border_width_clutter = min(30, w // 10, h // 10)
            if border_width_clutter > 0:
                border_mask = np.zeros((h, w), dtype=bool)
                border_mask[:border_width_clutter, :] = True
                border_mask[-border_width_clutter:, :] = True
                border_mask[:, :border_width_clutter] = True
                border_mask[:, -border_width_clutter:] = True
                border_edges = np.sum(edges[border_mask] > 0)
                border_clutter_score = float(border_edges / (border_mask.sum() + 1e-10))
            else:
                border_clutter_score = 0.0
            
            return {
                'edge_density': edge_density,
                'bg_uniformity': bg_uniformity,
                'white_bg_pct': white_bg_pct,
                'dominant_color_count': dominant_color_count,
                'saliency_peak_ratio': saliency_peak_ratio,
                'saliency_center_dist': saliency_center_dist,
                'object_occupancy_proxy': object_occupancy_proxy,
                'border_clutter_score': border_clutter_score
            }
        except Exception as e:
            print(f"Composition feature extraction failed: {e}")
            return None
    
    def extract_all_features(self, img_path: str) -> Dict[str, Any]:
        """
        Extract all features from an image.
        Returns a dictionary with quality, composition, and embedding features.
        """
        features = {
            'image_path': str(img_path),
            'valid': True,
            'error': None
        }
        
        # Quality features
        quality = self.compute_quality_features(img_path)
        if quality:
            features.update(quality)
        else:
            features['valid'] = False
            features['error'] = 'Quality feature extraction failed'
            return features
        
        # Composition features
        composition = self.compute_composition_features(img_path)
        if composition:
            features.update(composition)
        
        # CNN embeddings
        cnn_embedding = self.cnn_embedder.extract(img_path)
        if cnn_embedding is not None:
            features['cnn_embedding'] = cnn_embedding.tolist()
            features['cnn_embedding_dim'] = len(cnn_embedding)
        
        # CLIP embeddings
        clip_embedding = self.clip_embedder.extract(img_path)
        if clip_embedding is not None:
            features['clip_embedding'] = clip_embedding.tolist()
            features['clip_embedding_dim'] = len(clip_embedding)
        
        return features
    
    def compute_image_hash(self, img_path: str) -> str:
        """Compute a hash of the image for caching."""
        try:
            with open(img_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return hashlib.md5(str(img_path).encode()).hexdigest()


# Singleton instance for use across the application
_extractor_instance = None

def get_feature_extractor(device: str = "cpu") -> ImageFeatureExtractor:
    """Get or create the singleton feature extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ImageFeatureExtractor(device=device)
    return _extractor_instance
