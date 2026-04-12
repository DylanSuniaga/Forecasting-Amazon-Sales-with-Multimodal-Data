'use client';

import { useState, useEffect } from 'react';
import { Target, BarChart3, TrendingUp } from 'lucide-react';
import { getSubcategories } from '@/lib/api';
import type { CategoryInfo } from '@/types';
import { cn } from '@/lib/utils';

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

interface StepCategoryProps {
  category: string;
  subcategory: string;
  onCategoryChange: (category: string) => void;
  onSubcategoryChange: (subcategory: string) => void;
  categories: CategoryInfo[];
}

export function StepCategory({
  category,
  subcategory,
  onCategoryChange,
  onSubcategoryChange,
  categories,
}: StepCategoryProps) {
  const [subcategories, setSubcategories] = useState<string[]>([]);

  const categoryOptions = (categories && categories.length > 0)
    ? categories.map(c => c.name)
    : DEFAULT_CATEGORIES;

  const selectedCategoryInfo = categories.find(c => c.name === category);

  // Update subcategories when category changes
  useEffect(() => {
    if (category) {
      const cat = categories.find(c => c.name === category);
      if (cat?.subcategories && cat.subcategories.length > 0) {
        setSubcategories(cat.subcategories);
      } else {
        // Try API, fall back to defaults
        getSubcategories(category)
          .then(data => {
            if (data.subcategories && data.subcategories.length > 0) {
              setSubcategories(data.subcategories);
            } else {
              setSubcategories(DEFAULT_SUBCATEGORIES[category] || []);
            }
          })
          .catch(() => {
            setSubcategories(DEFAULT_SUBCATEGORIES[category] || []);
          });
      }
      onSubcategoryChange('');
    } else {
      setSubcategories([]);
    }
  }, [category, categories]); // eslint-disable-line react-hooks/exhaustive-deps

  // Category difficulty
  const difficultyTier = selectedCategoryInfo
    ? selectedCategoryInfo.difficulty_score >= 70 ? 'High'
      : selectedCategoryInfo.difficulty_score >= 50 ? 'Medium'
      : 'Low'
    : null;

  const difficultyColor = selectedCategoryInfo
    ? selectedCategoryInfo.difficulty_score >= 70 ? 'text-danger'
      : selectedCategoryInfo.difficulty_score >= 50 ? 'text-warning'
      : 'text-success'
    : '';

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Heading */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-text mb-2">Select Your Category</h2>
        <p className="text-muted">
          Choose the Amazon category and subcategory for your product
        </p>
      </div>

      {/* Category dropdown */}
      <div>
        <label htmlFor="wizard-category" className="form-label">
          Category <span className="text-danger">*</span>
        </label>
        <select
          id="wizard-category"
          value={category}
          onChange={(e) => onCategoryChange(e.target.value)}
          className="form-select text-lg"
        >
          <option value="">Select a category...</option>
          {categoryOptions.map((cat) => (
            <option key={cat} value={cat}>{cat}</option>
          ))}
        </select>
      </div>

      {/* Subcategory dropdown */}
      <div>
        <label htmlFor="wizard-subcategory" className="form-label">
          Subcategory <span className="text-xs text-muted font-normal">(optional)</span>
        </label>
        <select
          id="wizard-subcategory"
          value={subcategory}
          onChange={(e) => onSubcategoryChange(e.target.value)}
          className="form-select"
          disabled={!category || subcategories.length === 0}
        >
          <option value="">Select subcategory...</option>
          {subcategories.map((subcat) => (
            <option key={subcat} value={subcat}>{subcat}</option>
          ))}
        </select>
      </div>

      {/* Category difficulty preview */}
      {selectedCategoryInfo && (
        <div className="rounded-xl border border-border bg-surface p-6 space-y-4 animate-in fade-in duration-300">
          <h3 className="text-sm font-medium text-muted uppercase tracking-wider">
            Category Overview
          </h3>
          <div className="grid sm:grid-cols-3 gap-4">
            <div className="p-4 bg-surface-2 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-4 h-4 text-primary" />
                <span className="text-xs text-muted uppercase">Model AUC</span>
              </div>
              <p className="text-2xl font-bold text-text">
                {(selectedCategoryInfo.model_auc * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-muted mt-1">Classification accuracy</p>
            </div>

            <div className="p-4 bg-surface-2 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className="w-4 h-4 text-primary" />
                <span className="text-xs text-muted uppercase">BSR Rate</span>
              </div>
              <p className="text-2xl font-bold text-text">
                {(selectedCategoryInfo.avg_bsr_rate * 100).toFixed(0)}%
              </p>
              <p className="text-xs text-muted mt-1">Products with BSR rank</p>
            </div>

            <div className="p-4 bg-surface-2 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-primary" />
                <span className="text-xs text-muted uppercase">Difficulty</span>
              </div>
              <p className={cn('text-2xl font-bold', difficultyColor)}>
                {difficultyTier}
              </p>
              <p className="text-xs text-muted mt-1">
                {selectedCategoryInfo.difficulty_score}/100 score
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
