'use client';

import { BarChart3, Target, TrendingUp } from 'lucide-react';
import type { CategoryInfo, EvaluationResponse } from '@/types';
import { cn, formatPercent, formatNumber } from '@/lib/utils';

interface CategoryContextProps {
  category: string | null;
  categories: CategoryInfo[];
  evaluation: EvaluationResponse | null;
}

export function CategoryContext({ category, categories, evaluation }: CategoryContextProps) {
  const selectedCategory = categories.find(c => c.name === category);

  if (!category || !selectedCategory) {
    return (
      <div className="bento-card text-center py-12">
        <BarChart3 className="w-12 h-12 text-muted mx-auto mb-4" />
        <p className="text-muted">
          Select a category to view baseline difficulty and distribution
        </p>
      </div>
    );
  }

  // Category difficulty tier
  const difficultyTier = selectedCategory.difficulty_score >= 70 ? 'High' :
                         selectedCategory.difficulty_score >= 50 ? 'Medium' : 'Low';
  
  const difficultyColor = selectedCategory.difficulty_score >= 70 ? 'text-danger' :
                          selectedCategory.difficulty_score >= 50 ? 'text-warning' : 'text-success';

  return (
    <div className="space-y-6">
      {/* Category Overview */}
      <div className="bento-card">
        <h3 className="text-lg font-semibold text-text mb-4">
          {selectedCategory.name}
        </h3>
        
        <div className="grid sm:grid-cols-3 gap-4">
          <div className="p-4 bg-surface-2 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-primary" />
              <span className="text-xs text-muted uppercase">Model AUC</span>
            </div>
            <p className="text-2xl font-bold text-text">
              {(selectedCategory.model_auc * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-muted mt-1">Classification accuracy</p>
          </div>

          <div className="p-4 bg-surface-2 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-primary" />
              <span className="text-xs text-muted uppercase">BSR Rate</span>
            </div>
            <p className="text-2xl font-bold text-text">
              {formatPercent(selectedCategory.avg_bsr_rate)}
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
              {selectedCategory.difficulty_score}/100 competitive score
            </p>
          </div>
        </div>
      </div>

      {/* Position Indicator */}
      {evaluation && (
        <div className="bento-card">
          <h3 className="text-lg font-semibold text-text mb-4">Your Position</h3>
          
          <div className="relative">
            {/* Distribution visualization */}
            <div className="h-16 bg-surface-2 rounded-lg overflow-hidden relative">
              {/* Gradient showing distribution */}
              <div 
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-danger via-warning to-success opacity-30"
                style={{ width: '100%' }}
              />
              
              {/* Percentile marker */}
              <div 
                className="absolute top-0 bottom-0 w-1 bg-primary"
                style={{ left: `${evaluation.percentile_in_category}%` }}
              >
                <div className="absolute -top-6 left-1/2 -translate-x-1/2 whitespace-nowrap">
                  <span className="text-xs font-medium text-primary">
                    {evaluation.percentile_in_category.toFixed(0)}th percentile
                  </span>
                </div>
              </div>
              
              {/* Labels */}
              <div className="absolute bottom-1 left-2 text-xs text-muted">0%</div>
              <div className="absolute bottom-1 right-2 text-xs text-muted">100%</div>
            </div>
            
            <p className="text-sm text-muted mt-4">
              Your product scores in the <strong className="text-text">{evaluation.expected_rank_band}</strong> of {selectedCategory.name} based on image analysis.
            </p>
          </div>
        </div>
      )}

      {/* Category Notes */}
      <div className="bento-card">
        <h3 className="text-lg font-semibold text-text mb-4">Category Insights</h3>
        
        <div className="space-y-3">
          <CategoryInsight 
            label="Image Sensitivity"
            value={selectedCategory.difficulty_score > 60 ? 'High' : 'Moderate'}
            description="How much image quality affects BSR entry in this category"
          />
          <CategoryInsight 
            label="Competition Level"
            value={difficultyTier}
            description="Based on number of products and BSR distribution"
          />
          <CategoryInsight 
            label="Subcategories"
            value={selectedCategory.subcategories.length.toString()}
            description={selectedCategory.subcategories.slice(0, 3).join(', ')}
          />
        </div>
      </div>

      {/* All Categories Comparison */}
      <details className="bento-card">
        <summary className="collapsible-header cursor-pointer list-none">
          <span>Compare All Categories</span>
        </summary>
        <div className="mt-4 space-y-2">
          {categories
            .sort((a, b) => b.model_auc - a.model_auc)
            .map((cat) => (
              <div 
                key={cat.name}
                className={cn(
                  'flex items-center justify-between p-2 rounded',
                  cat.name === category && 'bg-surface-2'
                )}
              >
                <span className={cn(
                  'text-sm',
                  cat.name === category ? 'text-primary font-medium' : 'text-muted'
                )}>
                  {cat.name}
                </span>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-muted">
                    AUC: {(cat.model_auc * 100).toFixed(0)}%
                  </span>
                  <span className={cn(
                    'w-16 text-right',
                    cat.difficulty_score >= 70 ? 'text-danger' :
                    cat.difficulty_score >= 50 ? 'text-warning' : 'text-success'
                  )}>
                    {cat.difficulty_score >= 70 ? 'Hard' :
                     cat.difficulty_score >= 50 ? 'Medium' : 'Easy'}
                  </span>
                </div>
              </div>
            ))}
        </div>
      </details>
    </div>
  );
}

interface CategoryInsightProps {
  label: string;
  value: string;
  description: string;
}

function CategoryInsight({ label, value, description }: CategoryInsightProps) {
  return (
    <div className="flex items-start justify-between">
      <div>
        <p className="text-sm text-muted">{label}</p>
        <p className="text-xs text-muted mt-0.5">{description}</p>
      </div>
      <span className="text-sm font-medium text-text">{value}</span>
    </div>
  );
}
