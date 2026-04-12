'use client';

import { useState } from 'react';
import { Info, ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '@/lib/utils';

export function HowItWorksPanel({ className }: { className?: string }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={cn('rounded-xl border border-border bg-surface overflow-hidden', className)}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-5 hover:bg-surface-2/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Info className="w-5 h-5 text-primary" />
          <span className="text-lg font-semibold text-text">How It Works</span>
        </div>
        {expanded ? (
          <ChevronUp className="w-5 h-5 text-muted" />
        ) : (
          <ChevronDown className="w-5 h-5 text-muted" />
        )}
      </button>

      {expanded && (
        <div className="px-5 pb-5 border-t border-border pt-5 space-y-5">
          {/* Pipeline steps */}
          <div className="space-y-4">
            <PipelineStep
              number={1}
              title="Image Feature Extraction"
              description="Your product image is processed through EfficientNet-B0 (CNN) and CLIP (ViT-B-32) to extract visual embeddings, plus OpenCV for quality metrics (sharpness, contrast, background analysis)."
              detail="1,792 raw dimensions reduced to 256 via PCA"
            />
            <PipelineStep
              number={2}
              title="Text Analysis (TF-IDF + NLP)"
              description="Title and bullet points are vectorized with TF-IDF (5,000 terms, bigrams), reduced to 100 features via SVD. Text statistics like readability, keyword density, and structure are also extracted."
              detail="100 TF-IDF + 15 text statistics"
            />
            <PipelineStep
              number={3}
              title="Feature Engineering"
              description="Metadata features are computed: log price, price-vs-category, text-image interactions, product completeness signals. All features are pre-listing (causal, no look-ahead)."
              detail="~10 engineered features"
            />
            <PipelineStep
              number={4}
              title="Two-Stage XGBoost Prediction"
              description="Stage 1: Per-category XGBoost classifier predicts if the product will achieve a BSR ranking. Stage 2: If likely (>50%), a regression model estimates the actual BSR rank."
              detail="14 category-specific model pairs"
            />
            <PipelineStep
              number={5}
              title="Keyword Feedback"
              description="Your text is compared against TF-IDF profiles of top-performing products (BSR top 25%) in the same category. Missing high-prevalence keywords and weak terms are identified."
              detail="Cosine similarity scoring"
            />
          </div>

          {/* Data note */}
          <div className="p-3 bg-surface-2 rounded-lg">
            <p className="text-xs text-muted">
              <span className="text-text font-medium">Data:</span> Trained on ~16,000 Amazon products across 14 categories, collected via Amazon SP-API and enriched with ScrapingDog. All features are causal (available before listing). Models are retrained with train-only fitting to prevent data leakage.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function PipelineStep({
  number,
  title,
  description,
  detail,
}: {
  number: number;
  title: string;
  description: string;
  detail: string;
}) {
  return (
    <div className="flex gap-4">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center text-sm font-bold">
        {number}
      </div>
      <div className="flex-1 min-w-0">
        <h4 className="text-sm font-medium text-text">{title}</h4>
        <p className="text-xs text-muted mt-1 leading-relaxed">{description}</p>
        <span className="inline-block text-[10px] text-primary/80 bg-primary/10 px-2 py-0.5 rounded mt-1.5">
          {detail}
        </span>
      </div>
    </div>
  );
}
