'use client';

import { useState, useEffect, useCallback } from 'react';
import { Activity, Sparkles, ChevronDown, ChevronUp, ExternalLink, Loader2, BarChart3 } from 'lucide-react';
import { WizardShell, QuickMode, ModelPerformanceTab, ThemeToggle } from '@/components';
import {
  checkHealth,
  getCategories,
  evaluateProduct,
  getModels,
  getSuggestions,
} from '@/lib/api';
import { cn } from '@/lib/utils';
import type {
  EvaluationResponse,
  CategoryInfo,
  ProductFormData,
  ModelInfo,
  UIMode,
} from '@/types';

interface Suggestion {
  title: string;
  description: string;
  category: string;
  subcategory?: string;
  image_url: string;
  asin: string;
  actual_bsr: number | null;
  has_bsr: boolean;
  price?: number | null;
  cnn_embedding?: number[];
  clip_embedding?: number[];
  cnn_pca_features?: Record<string, number>;
  clip_pca_features?: Record<string, number>;
}

interface AllSuggestions {
  [category: string]: Suggestion[];
}

export default function Home() {
  // UI mode
  const [mode, setMode] = useState<UIMode>('wizard');

  // Data state
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [categories, setCategories] = useState<CategoryInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);

  // Evaluation state
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);

  // Suggestions state
  const [allSuggestions, setAllSuggestions] = useState<AllSuggestions>({});
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null);

  // Selected suggestion for pre-fill
  const [selectedSuggestion, setSelectedSuggestion] = useState<Suggestion | null>(null);
  const [initialFormData, setInitialFormData] = useState<Partial<ProductFormData> | undefined>(undefined);

  // Load initial data
  useEffect(() => {
    async function loadInitialData() {
      try {
        const health = await checkHealth();
        setIsHealthy(health.status === 'healthy');

        try {
          const catData = await getCategories();
          if (catData.categories && catData.categories.length > 0) {
            setCategories(catData.categories);
          }
        } catch (e) {
          console.warn('Failed to load categories from API');
        }

        try {
          const modelData = await getModels();
          setModels(modelData.category_models || []);
        } catch (e) {
          console.warn('Failed to load models from API');
        }
      } catch (err) {
        console.error('Failed to load initial data:', err);
        setIsHealthy(false);
      }
    }

    loadInitialData();
  }, []);

  // Load suggestions
  useEffect(() => {
    async function loadSuggestions() {
      setLoadingSuggestions(true);
      try {
        const data = await getSuggestions();
        setAllSuggestions(data.suggestions || {});
      } catch (err) {
        console.warn('Failed to load suggestions:', err);
        setAllSuggestions({});
      } finally {
        setLoadingSuggestions(false);
      }
    }
    loadSuggestions();
  }, []);

  // Handle evaluation
  const handleEvaluate = useCallback(async (data: ProductFormData) => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await evaluateProduct(
        data.title,
        data.description,
        data.category,
        data.subcategory || null,
        data.images,
        data.imageUrl,
        data.cnnEmbedding,
        data.clipEmbedding,
        data.cnnPcaFeatures,
        data.clipPcaFeatures,
        data.price
      );
      setEvaluation(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Evaluation failed');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Apply suggestion
  const applySuggestion = useCallback((suggestion: Suggestion) => {
    setSelectedSuggestion(suggestion);
    setInitialFormData({
      title: suggestion.title,
      description: suggestion.description || '',
      category: suggestion.category,
      subcategory: suggestion.subcategory || '',
      images: [],
      imageUrl: suggestion.image_url,
      price: suggestion.price ?? undefined,
      cnnEmbedding: suggestion.cnn_embedding,
      clipEmbedding: suggestion.clip_embedding,
      cnnPcaFeatures: suggestion.cnn_pca_features,
      clipPcaFeatures: suggestion.clip_pca_features,
    });
    setShowSuggestions(false);
    setEvaluation(null);
    setError(null);
  }, []);

  // Clear suggestion
  const clearSuggestion = useCallback(() => {
    setSelectedSuggestion(null);
    setInitialFormData({
      title: '',
      description: '',
      category: '',
      subcategory: '',
      images: [],
    });
    setEvaluation(null);
    setError(null);
  }, []);

  const categoriesWithSuggestions = Object.keys(allSuggestions).filter(
    cat => allSuggestions[cat] && allSuggestions[cat].length > 0
  );

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-surface/80 backdrop-blur-lg sticky top-0 z-50" style={{ boxShadow: 'var(--shadow-bento)' }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Left: Title */}
            <div className="flex items-center gap-3">
              <Activity className="w-6 h-6 text-primary" />
              <div>
                <h1 className="text-lg font-semibold text-text">
                  Amazon Launch Viability
                </h1>
                <p className="text-xs text-muted hidden sm:block">
                  Product Evaluation Console
                </p>
              </div>
            </div>

            {/* Center: Mode Toggle */}
            <div className="flex items-center gap-1 p-1 bg-surface-2 rounded-lg">
              <button
                onClick={() => setMode('wizard')}
                className={cn(
                  'px-4 py-1.5 text-sm font-medium rounded-md transition-all',
                  mode === 'wizard'
                    ? 'bg-primary text-white'
                    : 'text-muted hover:text-text'
                )}
              >
                Wizard
              </button>
              <button
                onClick={() => setMode('quick')}
                className={cn(
                  'px-4 py-1.5 text-sm font-medium rounded-md transition-all',
                  mode === 'quick'
                    ? 'bg-primary text-white'
                    : 'text-muted hover:text-text'
                )}
              >
                Quick
              </button>
              <button
                onClick={() => setMode('models')}
                className={cn(
                  'px-4 py-1.5 text-sm font-medium rounded-md transition-all flex items-center gap-1.5',
                  mode === 'models'
                    ? 'bg-primary text-white'
                    : 'text-muted hover:text-text'
                )}
              >
                <BarChart3 className="w-3.5 h-3.5" />
                Models
              </button>
            </div>

            {/* Right: Status + Theme */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className={cn(
                  'w-2 h-2 rounded-full',
                  isHealthy === null ? 'bg-muted' :
                  isHealthy ? 'bg-success' : 'bg-danger'
                )} />
                <span className="text-xs text-muted hidden sm:block">
                  {isHealthy === null ? 'Connecting...' :
                   isHealthy ? 'API Connected' : 'API Offline'}
                </span>
              </div>
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Suggestions Panel (hidden in models tab) */}
        {mode !== 'models' && categoriesWithSuggestions.length > 0 && (
          <div className={cn(
            'mb-8 rounded-xl border overflow-hidden transition-all',
            !evaluation
              ? 'bg-gradient-to-br from-primary/10 to-surface-2 border-primary/30'
              : 'bg-surface-2 border-border'
          )}>
            <button
              type="button"
              onClick={() => setShowSuggestions(!showSuggestions)}
              className="w-full p-4 flex items-center justify-between hover:bg-surface/50 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className={cn(
                  'p-2 rounded-lg',
                  !evaluation ? 'bg-primary/20' : 'bg-surface'
                )}>
                  <Sparkles className={cn(
                    'w-5 h-5',
                    !evaluation ? 'text-primary' : 'text-muted'
                  )} />
                </div>
                <div className="text-left">
                  <p className={cn(
                    'font-medium',
                    !evaluation ? 'text-primary' : 'text-text'
                  )}>
                    {!evaluation ? "Don't have a product? Try these examples" : 'Example Products'}
                  </p>
                  <p className="text-xs text-muted">
                    {Object.values(allSuggestions).flat().length} real products from our test set
                  </p>
                </div>
              </div>
              {selectedSuggestion && (
                <span className="text-xs text-success bg-success/10 px-2 py-1 rounded mr-3 hidden sm:block">
                  Using: {selectedSuggestion.title.slice(0, 40)}...
                </span>
              )}
              {showSuggestions ? (
                <ChevronUp className="w-5 h-5 text-muted flex-shrink-0" />
              ) : (
                <ChevronDown className="w-5 h-5 text-muted flex-shrink-0" />
              )}
            </button>

            {showSuggestions && (
              <div className="border-t border-border p-4 space-y-3 max-h-[400px] overflow-y-auto">
                {loadingSuggestions ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="w-6 h-6 animate-spin text-muted" />
                    <span className="ml-2 text-muted">Loading suggestions...</span>
                  </div>
                ) : (
                  categoriesWithSuggestions.map((category) => (
                    <div key={category} className="rounded-lg border border-border overflow-hidden">
                      <button
                        type="button"
                        onClick={() => setExpandedCategory(expandedCategory === category ? null : category)}
                        className="w-full p-3 flex items-center justify-between bg-surface hover:bg-surface/80 transition-colors"
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-text">{category}</span>
                          <span className="text-xs text-muted bg-surface-2 px-2 py-0.5 rounded">
                            {allSuggestions[category].length} examples
                          </span>
                        </div>
                        {expandedCategory === category ? (
                          <ChevronUp className="w-4 h-4 text-muted" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-muted" />
                        )}
                      </button>

                      {expandedCategory === category && (
                        <div className="p-2 space-y-2 bg-background">
                          {allSuggestions[category].map((suggestion, idx) => (
                            <button
                              key={idx}
                              type="button"
                              onClick={() => applySuggestion(suggestion)}
                              className={cn(
                                'w-full text-left p-3 rounded-lg border transition-all group',
                                selectedSuggestion?.asin === suggestion.asin
                                  ? 'bg-primary/10 border-primary/50'
                                  : 'bg-surface border-border hover:border-primary/50 hover:bg-surface/80'
                              )}
                            >
                              <div className="flex items-start justify-between gap-2">
                                <p className="text-sm text-text line-clamp-2 flex-1 group-hover:text-primary transition-colors">
                                  {suggestion.title}
                                </p>
                                {suggestion.image_url && (
                                  <a
                                    href={suggestion.image_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    onClick={(e) => e.stopPropagation()}
                                    className="text-muted hover:text-primary"
                                    title="View original image"
                                  >
                                    <ExternalLink className="w-3 h-3" />
                                  </a>
                                )}
                              </div>
                              <div className="flex items-center gap-3 mt-2">
                                {suggestion.price != null && (
                                  <span className="text-xs text-text bg-surface-2 px-2 py-0.5 rounded">
                                    ${suggestion.price.toFixed(2)}
                                  </span>
                                )}
                                {suggestion.has_bsr && suggestion.actual_bsr && (
                                  <span className="text-xs text-success bg-success/10 px-2 py-0.5 rounded">
                                    BSR: #{suggestion.actual_bsr.toLocaleString()}
                                  </span>
                                )}
                                {!suggestion.has_bsr && (
                                  <span className="text-xs text-warning bg-warning/10 px-2 py-0.5 rounded">
                                    No BSR
                                  </span>
                                )}
                                {selectedSuggestion?.asin === suggestion.asin && (
                                  <span className="text-xs text-primary font-medium">Selected</span>
                                )}
                              </div>
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  ))
                )}
                <p className="text-xs text-muted text-center pt-2">
                  Click a product to auto-fill the evaluation form.
                </p>
              </div>
            )}
          </div>
        )}

        {/* Mode Content */}
        {mode === 'wizard' ? (
          <WizardShell
            categories={categories}
            onEvaluate={handleEvaluate}
            isLoading={isLoading}
            evaluation={evaluation}
            error={error}
            initialFormData={initialFormData}
            suggestionImageUrl={selectedSuggestion?.image_url}
            onClearSuggestion={clearSuggestion}
          />
        ) : mode === 'quick' ? (
          <QuickMode
            categories={categories}
            onEvaluate={handleEvaluate}
            isLoading={isLoading}
            evaluation={evaluation}
            error={error}
            initialFormData={initialFormData}
            suggestionImageUrl={selectedSuggestion?.image_url}
            onClearSuggestion={clearSuggestion}
          />
        ) : (
          <ModelPerformanceTab onBack={() => setMode('wizard')} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-surface mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
            <p className="text-sm text-muted">
              Amazon Product Launch Viability Capstone Project
            </p>
            <div className="flex items-center gap-4 text-xs text-muted">
              <span>v1.0.0</span>
              <span>|</span>
              <a href="/docs" className="hover:text-primary transition-colors">API Docs</a>
              <span>|</span>
              <span>{models.length} category models</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
