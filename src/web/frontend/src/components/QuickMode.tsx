'use client';

import { useState, useEffect } from 'react';
import { Send, Loader2, AlertCircle, Zap } from 'lucide-react';
import { ImageUpload } from './ImageUpload';
import { StepResults } from './StepResults';
import { getSubcategories } from '@/lib/api';
import type { CategoryInfo, ProductFormData, EvaluationResponse } from '@/types';
import { cn } from '@/lib/utils';

// Default categories
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

interface QuickModeProps {
  categories: CategoryInfo[];
  onEvaluate: (data: ProductFormData) => Promise<void>;
  isLoading: boolean;
  evaluation: EvaluationResponse | null;
  error: string | null;
  initialFormData?: Partial<ProductFormData>;
  suggestionImageUrl?: string;
  onClearSuggestion?: () => void;
}

export function QuickMode({
  categories,
  onEvaluate,
  isLoading,
  evaluation,
  error,
  initialFormData,
  suggestionImageUrl,
  onClearSuggestion,
}: QuickModeProps) {
  const [formData, setFormData] = useState<ProductFormData>({
    title: initialFormData?.title || '',
    description: initialFormData?.description || '',
    category: initialFormData?.category || '',
    subcategory: initialFormData?.subcategory || '',
    images: initialFormData?.images || [],
    imageUrl: initialFormData?.imageUrl,
    price: initialFormData?.price,
    cnnEmbedding: initialFormData?.cnnEmbedding,
    clipEmbedding: initialFormData?.clipEmbedding,
    cnnPcaFeatures: initialFormData?.cnnPcaFeatures,
    clipPcaFeatures: initialFormData?.clipPcaFeatures,
  });
  const [subcategories, setSubcategories] = useState<string[]>([]);

  // Sync when initialFormData changes
  const [lastInitial, setLastInitial] = useState(initialFormData);
  if (initialFormData !== lastInitial) {
    setLastInitial(initialFormData);
    if (initialFormData) {
      setFormData({
        title: initialFormData.title || '',
        description: initialFormData.description || '',
        category: initialFormData.category || '',
        subcategory: initialFormData.subcategory || '',
        images: initialFormData.images || [],
        imageUrl: initialFormData.imageUrl,
        price: initialFormData.price,
        cnnEmbedding: initialFormData.cnnEmbedding,
        clipEmbedding: initialFormData.clipEmbedding,
        cnnPcaFeatures: initialFormData.cnnPcaFeatures,
        clipPcaFeatures: initialFormData.clipPcaFeatures,
      });
    }
  }

  const categoryOptions = (categories && categories.length > 0)
    ? categories.map(c => c.name)
    : DEFAULT_CATEGORIES;

  // Update subcategories when category changes
  useEffect(() => {
    if (formData.category) {
      const cat = categories.find(c => c.name === formData.category);
      if (cat?.subcategories && cat.subcategories.length > 0) {
        setSubcategories(cat.subcategories);
      } else {
        getSubcategories(formData.category)
          .then(data => {
            if (data.subcategories && data.subcategories.length > 0) {
              setSubcategories(data.subcategories);
            } else {
              setSubcategories(DEFAULT_SUBCATEGORIES[formData.category] || []);
            }
          })
          .catch(() => {
            setSubcategories(DEFAULT_SUBCATEGORIES[formData.category] || []);
          });
      }
      setFormData(prev => ({ ...prev, subcategory: '' }));
    } else {
      setSubcategories([]);
    }
  }, [formData.category, categories]);

  const hasImage = formData.images.length > 0 || !!suggestionImageUrl || !!formData.imageUrl;
  const isValid = !!formData.title.trim() && !!formData.category && hasImage;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isValid || isLoading) return;
    await onEvaluate(formData);
  };

  return (
    <div className="space-y-8">
      {/* Quick mode header */}
      <div className="text-center max-w-2xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-primary/15 border border-primary/30 rounded-full text-primary text-sm font-medium mb-4">
          <Zap className="w-4 h-4" />
          Quick Mode
        </div>
        <h2 className="text-2xl font-bold text-text mb-2">Fast Product Evaluation</h2>
        <p className="text-muted">All inputs in one view for rapid testing</p>
      </div>

      <div className="max-w-3xl mx-auto">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Category & Subcategory */}
          <div className="grid sm:grid-cols-2 gap-4">
            <div>
              <label htmlFor="quick-category" className="form-label">
                Category <span className="text-danger">*</span>
              </label>
              <select
                id="quick-category"
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
              <label htmlFor="quick-subcategory" className="form-label">
                Subcategory
              </label>
              <select
                id="quick-subcategory"
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

          {/* Title */}
          <div>
            <label htmlFor="quick-title" className="form-label">
              Product Title <span className="text-danger">*</span>
            </label>
            <input
              id="quick-title"
              type="text"
              value={formData.title}
              onChange={(e) => setFormData(prev => ({ ...prev, title: e.target.value }))}
              placeholder="Premium Stainless Steel Water Bottle 32oz..."
              className="form-input"
              maxLength={500}
            />
            <p className="text-xs text-muted mt-1">{formData.title.length}/500</p>
          </div>

          {/* Description */}
          <div>
            <label htmlFor="quick-description" className="form-label">
              Description / Bullet Points
            </label>
            <textarea
              id="quick-description"
              value={formData.description}
              onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
              placeholder="Key features, benefits, and specifications..."
              className="form-input min-h-[120px] resize-y"
              maxLength={5000}
            />
            <p className="text-xs text-muted mt-1">{formData.description.length}/5000</p>
          </div>

          {/* Price */}
          <div>
            <label htmlFor="quick-price" className="form-label">
              Price (USD) <span className="text-xs text-muted font-normal ml-1">Optional</span>
            </label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted">$</span>
              <input
                id="quick-price"
                type="number"
                min="0"
                step="0.01"
                value={formData.price ?? ''}
                onChange={(e) => {
                  const val = e.target.value;
                  setFormData(prev => ({ ...prev, price: val === '' ? undefined : parseFloat(val) }));
                }}
                placeholder="29.99"
                className="form-input pl-7 max-w-[200px]"
              />
            </div>
          </div>

          {/* Image */}
          <div>
            <label className="form-label">
              Product Image <span className="text-danger">*</span>
            </label>
            {(suggestionImageUrl || formData.imageUrl) && !formData.images.length ? (
              <div className="p-4 bg-surface-2 rounded-lg border border-primary/30">
                <div className="flex items-center gap-3">
                  <div className="w-16 h-16 rounded-lg overflow-hidden bg-surface border border-border flex-shrink-0">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={suggestionImageUrl || formData.imageUrl}
                      alt="Product"
                      className="w-full h-full object-contain"
                      onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                    />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-primary font-medium">Example product image</p>
                    {onClearSuggestion && (
                      <button
                        type="button"
                        onClick={onClearSuggestion}
                        className="text-xs text-muted hover:text-danger mt-1"
                      >
                        Clear and upload your own
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <ImageUpload
                images={formData.images}
                onChange={(images) => setFormData(prev => ({ ...prev, images }))}
                maxImages={1}
              />
            )}
          </div>

          {/* Submit */}
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

          {!isValid && (
            <p className="text-xs text-muted text-center">
              {!formData.title.trim() && 'Title required. '}
              {!formData.category && 'Category required. '}
              {!hasImage && 'Image required. '}
            </p>
          )}
        </form>

        {/* Error */}
        {error && (
          <div className="mt-6 bento-card border-danger/50 bg-danger/10">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-danger" />
              <p className="text-danger text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Loading */}
        {isLoading && !evaluation && (
          <div className="mt-8 text-center py-12">
            <Loader2 className="w-8 h-8 text-primary animate-spin mx-auto mb-4" />
            <p className="text-muted">Analyzing product...</p>
          </div>
        )}

        {/* Results */}
        {evaluation && !isLoading && (
          <div className="mt-8">
            <StepResults evaluation={evaluation} category={formData.category} />
          </div>
        )}
      </div>
    </div>
  );
}
