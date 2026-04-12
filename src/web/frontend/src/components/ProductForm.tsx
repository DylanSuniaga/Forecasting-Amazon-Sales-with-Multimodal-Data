'use client';

import { useState, useEffect } from 'react';
import { Send, Loader2, Sparkles, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import { ImageUpload } from './ImageUpload';
import { getSuggestions } from '@/lib/api';
import type { CategoryInfo, ProductFormData } from '@/types';
import { cn } from '@/lib/utils';

interface Suggestion {
  title: string;
  description: string;
  category: string;
  subcategory?: string; // Subcategory from dataset
  image_url: string;
  asin: string;
  actual_bsr: number | null;
  has_bsr: boolean;
  cnn_embedding?: number[]; // Pre-computed embeddings from dataframe (fallback)
  clip_embedding?: number[]; // Pre-computed embeddings from dataframe (fallback)
  cnn_pca_features?: Record<string, number>; // Pre-computed PCA features (preferred - dataset already has PCA)
  clip_pca_features?: Record<string, number>; // Pre-computed PCA features (preferred - dataset already has PCA)
}

interface AllSuggestions {
  [category: string]: Suggestion[];
}

// Default categories (from notebook analysis - 14 main BSR groups)
const DEFAULT_CATEGORIES = [
  "Home & Kitchen",
  "Health & Household",
  "Office Products",
  "Baby",
  "Clothing, Shoes & Jewelry",
  "Kitchen & Dining",
  "Electronics",
  "Cell Phones & Accessories",
  "Tools & Home Improvement",
  "Video Games",
  "Pet Supplies",
  "Sports & Outdoors",
  "Industrial & Scientific",
  "Musical Instruments",
];

// Default subcategories
const DEFAULT_SUBCATEGORIES: Record<string, string[]> = {
  "Home & Kitchen": ["Kitchen & Dining", "Bedding", "Bath", "Furniture", "Home Decor", "Storage & Organization"],
  "Health & Household": ["Vitamins & Supplements", "Personal Care", "Household Supplies", "Medical Supplies", "Health Care"],
  "Office Products": ["Office Electronics", "Office Furniture", "Writing Supplies", "Filing Products", "Desk Accessories"],
  "Baby": ["Feeding", "Diapering", "Nursery", "Baby Care", "Strollers & Accessories", "Baby Toys"],
  "Clothing, Shoes & Jewelry": ["Women's Clothing", "Men's Clothing", "Kids' Clothing", "Shoes", "Jewelry", "Watches"],
  "Kitchen & Dining": ["Cookware", "Dinnerware", "Kitchen Utensils", "Storage & Organization", "Small Appliances"],
  "Electronics": ["Computers & Accessories", "TV & Video", "Audio & Home Theater", "Cameras", "Wearables"],
  "Cell Phones & Accessories": ["Cell Phones", "Cases & Covers", "Chargers", "Screen Protectors", "Mounts"],
  "Tools & Home Improvement": ["Power Tools", "Hand Tools", "Hardware", "Electrical", "Plumbing"],
  "Video Games": ["PlayStation", "Xbox", "Nintendo", "PC Gaming", "Gaming Accessories"],
  "Pet Supplies": ["Dogs", "Cats", "Fish & Aquatic", "Birds", "Small Animals"],
  "Sports & Outdoors": ["Exercise & Fitness", "Outdoor Recreation", "Team Sports", "Camping & Hiking"],
  "Industrial & Scientific": ["Lab Equipment", "Safety Equipment", "Industrial Hardware", "Measuring Tools"],
  "Musical Instruments": ["Guitars & Basses", "Keyboards & Pianos", "Drums & Percussion", "Recording Equipment"],
};

interface ProductFormProps {
  onSubmit: (data: ProductFormData) => void;
  isLoading?: boolean;
  categories?: CategoryInfo[];
}

export function ProductForm({ 
  onSubmit, 
  isLoading = false,
  categories 
}: ProductFormProps) {
  const [formData, setFormData] = useState<ProductFormData>({
    title: '',
    description: '',
    category: '',
    subcategory: '',
    images: []
  });

  const [subcategories, setSubcategories] = useState<string[]>([]);
  const [allSuggestions, setAllSuggestions] = useState<AllSuggestions>({});
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null);
  
  // Track if using a suggestion (with external image URL)
  const [selectedSuggestion, setSelectedSuggestion] = useState<Suggestion | null>(null);

  // Load all suggestions on mount
  useEffect(() => {
    loadAllSuggestions();
  }, []);

  // Update subcategories when category changes
  useEffect(() => {
    if (formData.category) {
      const cat = categories?.find(c => c.name === formData.category);
      if (cat?.subcategories && cat.subcategories.length > 0) {
        setSubcategories(cat.subcategories);
      } else {
        setSubcategories(DEFAULT_SUBCATEGORIES[formData.category] || []);
      }
      setFormData(prev => ({ ...prev, subcategory: '' }));
      // Auto-expand the selected category in suggestions
      setExpandedCategory(formData.category);
    } else {
      setSubcategories([]);
    }
  }, [formData.category, categories]);

  // Load all suggestions from API
  const loadAllSuggestions = async () => {
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
  };

  // Apply a suggestion to the form
  const applySuggestion = (suggestion: Suggestion) => {
    setFormData(prev => ({
      ...prev,
      title: suggestion.title,
      description: suggestion.description || '',
      category: suggestion.category,
      subcategory: suggestion.subcategory || '', // Auto-fill subcategory from dataset
      images: [], // Clear any uploaded images
      imageUrl: suggestion.image_url, // Store image URL
      cnnEmbedding: suggestion.cnn_embedding, // Pre-computed embeddings (fallback)
      clipEmbedding: suggestion.clip_embedding, // Pre-computed embeddings (fallback)
      cnnPcaFeatures: suggestion.cnn_pca_features, // Pre-computed PCA features (preferred)
      clipPcaFeatures: suggestion.clip_pca_features // Pre-computed PCA features (preferred)
    }));
    setSelectedSuggestion(suggestion);
    // Collapse suggestions after selection
    setShowSuggestions(false);
  };

  // Clear the selected suggestion (go back to manual mode)
  const clearSuggestion = () => {
    setSelectedSuggestion(null);
    setFormData(prev => ({
      ...prev,
      title: '',
      description: '',
      category: '',
      subcategory: '',
      images: []
    }));
    setShowSuggestions(true);
  };

  // Check if form is empty (to show suggestions prominently)
  const isFormEmpty = !formData.title && !formData.description && formData.images.length === 0 && !selectedSuggestion;
  
  // Categories that have suggestions
  const categoriesWithSuggestions = Object.keys(allSuggestions).filter(
    cat => allSuggestions[cat] && allSuggestions[cat].length > 0
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Valid if: has title + category + (has uploaded image OR has suggestion with image URL)
    const hasImage = formData.images.length > 0 || (selectedSuggestion && selectedSuggestion.image_url);
    if (formData.title && formData.category && hasImage) {
      // If using suggestion, pass the image URL in a special way
      const submitData = {
        ...formData,
        // Add the image URL for backend to fetch if no uploaded image
        imageUrl: selectedSuggestion?.image_url || undefined
      };
      onSubmit(submitData as ProductFormData);
    }
  };

  // Use categories from API if available, otherwise use defaults
  const categoryOptions = (categories && categories.length > 0) 
    ? categories.map(c => c.name) 
    : DEFAULT_CATEGORIES;

  // Valid if: has title + category + (has uploaded image OR using suggestion with image)
  const hasImage = formData.images.length > 0 || (selectedSuggestion && selectedSuggestion.image_url);
  const isValid = formData.title.trim() && formData.category && hasImage;

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Suggestions Panel - Prominent when form is empty */}
      {categoriesWithSuggestions.length > 0 && (
        <div className={cn(
          "rounded-lg border overflow-hidden transition-all",
          isFormEmpty 
            ? "bg-gradient-to-br from-primary/10 to-surface-2 border-primary/30" 
            : "bg-surface-2 border-border"
        )}>
          <button
            type="button"
            onClick={() => setShowSuggestions(!showSuggestions)}
            className="w-full p-4 flex items-center justify-between hover:bg-surface/50 transition-colors"
          >
            <div className="flex items-center gap-3">
              <div className={cn(
                "p-2 rounded-lg",
                isFormEmpty ? "bg-primary/20" : "bg-surface"
              )}>
                <Sparkles className={cn(
                  "w-5 h-5",
                  isFormEmpty ? "text-primary" : "text-muted"
                )} />
              </div>
              <div className="text-left">
                <p className={cn(
                  "font-medium",
                  isFormEmpty ? "text-primary" : "text-text"
                )}>
                  {isFormEmpty ? "Don't have a product? Try these examples" : "Example Products"}
                </p>
                <p className="text-xs text-muted">
                  {Object.values(allSuggestions).flat().length} real products from our test set
                </p>
              </div>
            </div>
            {showSuggestions ? (
              <ChevronUp className="w-5 h-5 text-muted" />
            ) : (
              <ChevronDown className="w-5 h-5 text-muted" />
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
                            className="w-full text-left p-3 bg-surface rounded-lg border border-border hover:border-primary/50 hover:bg-surface/80 transition-all group"
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
                              <span className="text-xs text-muted">
                                Click to use this product
                              </span>
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
              
              <p className="text-xs text-muted text-center pt-2">
                These are real products from our held-out test set. Click to auto-fill the form.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Title */}
      <div>
        <label htmlFor="title" className="form-label">
          Product Title <span className="text-danger">*</span>
        </label>
        <input
          id="title"
          type="text"
          value={formData.title}
          onChange={(e) => setFormData(prev => ({ ...prev, title: e.target.value }))}
          placeholder="Premium Stainless Steel Water Bottle 32oz..."
          className="form-input"
          maxLength={500}
        />
        <p className="text-xs text-muted mt-1">
          {formData.title.length}/500 characters
        </p>
      </div>

      {/* Description */}
      <div>
        <label htmlFor="description" className="form-label">
          Description / Bullet Points
        </label>
        <textarea
          id="description"
          value={formData.description}
          onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
          placeholder="Key features, benefits, and specifications..."
          className="form-input min-h-[120px] resize-y"
          maxLength={5000}
        />
        <p className="text-xs text-muted mt-1">
          {formData.description.length}/5000 characters
        </p>
      </div>

      {/* Category & Subcategory */}
      <div className="grid sm:grid-cols-2 gap-4">
        <div>
          <label htmlFor="category" className="form-label">
            Category <span className="text-danger">*</span>
          </label>
          <select
            id="category"
            value={formData.category}
            onChange={(e) => setFormData(prev => ({ ...prev, category: e.target.value }))}
            className="form-select"
          >
            <option value="">Select category...</option>
            {categoryOptions.map((cat) => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="subcategory" className="form-label">
            Subcategory
          </label>
          <select
            id="subcategory"
            value={formData.subcategory}
            onChange={(e) => setFormData(prev => ({ ...prev, subcategory: e.target.value }))}
            className="form-select"
            disabled={!formData.category || subcategories.length === 0}
          >
            <option value="">Select subcategory...</option>
            {subcategories.map((subcat) => (
              <option key={subcat} value={subcat}>{subcat}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Image Upload / Suggestion Image */}
      <div>
        <label className="form-label">
          Product Image <span className="text-danger">*</span>
        </label>
        
        {selectedSuggestion && selectedSuggestion.image_url ? (
          // Show suggestion image (read-only)
          <div className="space-y-3">
            <div className="p-4 bg-surface-2 rounded-lg border border-primary/30">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium text-primary">Using example product image</span>
              </div>
              <div className="relative aspect-square w-32 rounded-lg overflow-hidden border border-border bg-surface">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={selectedSuggestion.image_url}
                  alt="Product"
                  className="w-full h-full object-contain"
                  onError={(e) => {
                    // Hide broken images
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              </div>
              <div className="flex items-center gap-2 mt-3">
                <a
                  href={selectedSuggestion.image_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-primary hover:underline flex items-center gap-1"
                >
                  <ExternalLink className="w-3 h-3" />
                  View full image
                </a>
                <span className="text-xs text-muted">•</span>
                <button
                  type="button"
                  onClick={clearSuggestion}
                  className="text-xs text-muted hover:text-danger transition-colors"
                >
                  Clear and upload your own
                </button>
              </div>
            </div>
            <p className="text-xs text-muted">
              Image from Amazon product listing. Click &quot;Clear&quot; to upload your own image instead.
            </p>
          </div>
        ) : (
          // Normal image upload
          <>
            <p className="text-xs text-muted mb-3">
              Upload main product image (multiple images to be added later)
            </p>
            <ImageUpload
              images={formData.images}
              onChange={(images) => setFormData(prev => ({ ...prev, images }))}
              maxImages={1}
            />
          </>
        )}
      </div>

      {/* Selected Suggestion Info Banner */}
      {selectedSuggestion && (
        <div className="p-3 bg-success/10 border border-success/30 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-success" />
              <span className="text-sm text-success font-medium">Using example product</span>
              {selectedSuggestion.has_bsr && selectedSuggestion.actual_bsr && (
                <span className="text-xs bg-success/20 text-success px-2 py-0.5 rounded">
                  Actual BSR: #{selectedSuggestion.actual_bsr.toLocaleString()}
                </span>
              )}
            </div>
            <button
              type="button"
              onClick={clearSuggestion}
              className="text-xs text-muted hover:text-danger transition-colors"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {/* Submit Button */}
      <button
        type="submit"
        disabled={!isValid || isLoading}
        className={cn(
          'btn-primary w-full flex items-center justify-center gap-2',
          (!isValid || isLoading) && 'opacity-50 cursor-not-allowed'
        )}
      >
        {isLoading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Analyzing...
          </>
        ) : (
          <>
            <Send className="w-5 h-5" />
            Evaluate Product
          </>
        )}
      </button>

      {/* Validation hints */}
      {!isValid && (
        <p className="text-xs text-muted text-center">
          {!formData.title && 'Title required. '}
          {!formData.category && 'Category required. '}
          {!hasImage && 'Image required (upload or use example). '}
        </p>
      )}
    </form>
  );
}
