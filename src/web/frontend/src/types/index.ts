// NLP Feedback types
export interface KeywordSuggestion {
  keyword: string;
  prevalence: number;
  suggestion: string;
  source?: string;
}

export interface NLPFeedback {
  title_score: number;
  bullets_score: number;
  missing_keywords: KeywordSuggestion[];
  weak_keywords: KeywordSuggestion[];
  title_stats: Record<string, any>;
  bullets_stats: Record<string, any>;
}

// Wizard types
export type WizardStep = 'category' | 'images' | 'text' | 'results';
export type UIMode = 'wizard' | 'quick' | 'models';

// API Types matching backend schemas

export type RankBand = 
  | "Top 1%"
  | "Top 5%"
  | "Top 10%"
  | "Top 25%"
  | "Top 50%"
  | "Bottom 50%";

export type SignalType = "strength" | "risk";

export interface Signal {
  type: SignalType;
  label: string;
  description: string;
  impact: number;
}

export interface ImageQualityMetrics {
  width: number;
  height: number;
  aspect_ratio: number;
  brightness_mean: number;
  brightness_std: number;
  contrast: number;
  saturation_mean: number;
  colorfulness: number;
  sharpness: number;
  blur_score: number;
  white_bg_pct: number;
  edge_density: number;
  vs_category_median?: Record<string, number>;
}

export interface FeatureImportance {
  feature: string;
  display_name: string;
  importance: number;
  type: 'image' | 'text' | 'price';
}

export interface EvaluationResponse {
  launch_viability_score: number;
  bsr_entry_probability: number;
  expected_rank_band: RankBand;
  competitive_intensity: number;
  confidence: number;
  strengths: Signal[];
  risks: Signal[];
  category_notes: string;
  category_baseline_viability: number;
  percentile_in_category: number;
  image_quality: ImageQualityMetrics;
  top_feature_importances?: FeatureImportance[];
  feature_summary: {
    top_features_used: number;
    embedding_dimension: number;
    images_processed: number;
    model_loaded: boolean;
    is_placeholder: boolean;
    estimated_rank?: number;
  };
  product_hash: string;
  model_version: string;
  nlp_feedback?: NLPFeedback;
}

export interface CategoryInfo {
  name: string;
  subcategories: string[];
  sample_count: number;
  avg_bsr_rate: number;
  model_auc: number;
  difficulty_score: number;
}

export interface ValidationMetrics {
  category: string;
  model_type: string;
  roc_auc: number;
  f1_score: number;
  precision: number;
  recall: number;
  accuracy: number;
  sample_size: number;
  reg_r2?: number;
  reg_mae?: number;
}

export interface ModelInfo {
  name: string;
  category: string;
  model_type: string;
  loaded: boolean;
  metrics: {
    roc_auc: number;
    f1: number;
    precision: number;
    recall: number;
  };
  reg_metrics?: {
    r2: number;
    mae: number;
  };
}

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: number;
  gpu_available: boolean;
}

// Form state
export interface ProductFormData {
  title: string;
  description: string;
  category: string;
  subcategory: string;
  images: File[];
  price?: number; // Product price in USD
  imageUrl?: string; // For suggestion products with external image URLs
  cnnEmbedding?: number[]; // Pre-computed CNN embedding (for suggestions)
  clipEmbedding?: number[]; // Pre-computed CLIP embedding (for suggestions)
  cnnPcaFeatures?: Record<string, number>; // Pre-computed CNN PCA features (for suggestions with existing dataset)
  clipPcaFeatures?: Record<string, number>; // Pre-computed CLIP PCA features (for suggestions with existing dataset)
}

// UI State
export type TabId = 'evaluate' | 'category' | 'image' | 'model' | 'validation';

export interface TabConfig {
  id: TabId;
  label: string;
  description: string;
}

export const TABS: TabConfig[] = [
  { id: 'evaluate', label: 'Evaluate', description: 'Product evaluation' },
  { id: 'category', label: 'Category Context', description: 'Baseline & difficulty' },
  { id: 'image', label: 'Image Evidence', description: 'Visual signal analysis' },
  { id: 'model', label: 'Model & API', description: 'Architecture & integration' },
  { id: 'validation', label: 'Validation', description: 'Performance metrics' },
];
