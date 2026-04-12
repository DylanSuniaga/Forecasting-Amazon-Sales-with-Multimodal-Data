'use client';

import { useMemo } from 'react';
import { Type, AlignLeft, DollarSign } from 'lucide-react';
import { cn } from '@/lib/utils';

interface StepTextProps {
  title: string;
  description: string;
  price?: number;
  onTitleChange: (title: string) => void;
  onDescriptionChange: (description: string) => void;
  onPriceChange?: (price: number | undefined) => void;
}

// Detect common product spec patterns
function detectSpecs(title: string, description: string): { label: string; color: string }[] {
  const combined = `${title} ${description}`.toLowerCase();
  const specs: { label: string; color: string }[] = [];

  // Size patterns
  if (/\d+\s*(oz|ml|l|gal|qt|fl\s*oz|liter|litre|inch|in|ft|cm|mm|"\s)/i.test(combined)) {
    specs.push({ label: 'Size', color: 'bg-blue-500/20 text-blue-400 border-blue-500/30' });
  }

  // Color patterns
  const colorWords = ['black', 'white', 'red', 'blue', 'green', 'silver', 'gold', 'pink', 'purple', 'gray', 'grey', 'brown', 'navy', 'beige'];
  if (colorWords.some(c => combined.includes(c))) {
    specs.push({ label: 'Color', color: 'bg-purple-500/20 text-purple-400 border-purple-500/30' });
  }

  // Brand detection (capitalized words at start, or "by X")
  if (/^[A-Z][a-zA-Z]+\s/.test(title) || /\bby\s+[A-Z]/i.test(combined)) {
    specs.push({ label: 'Brand', color: 'bg-primary/20 text-primary border-primary/30' });
  }

  // Material patterns
  const materials = ['stainless steel', 'silicone', 'bamboo', 'wood', 'plastic', 'glass', 'ceramic', 'leather', 'cotton', 'polyester', 'nylon', 'aluminum', 'titanium', 'rubber'];
  if (materials.some(m => combined.includes(m))) {
    specs.push({ label: 'Material', color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' });
  }

  // Quantity/pack patterns
  if (/\d+\s*(-|\s)?(pack|piece|set|count|ct|pk|pcs)/i.test(combined)) {
    specs.push({ label: 'Quantity', color: 'bg-orange-500/20 text-orange-400 border-orange-500/30' });
  }

  // Weight patterns
  if (/\d+(\.\d+)?\s*(lb|lbs|kg|g|oz|pound|ounce)/i.test(combined)) {
    specs.push({ label: 'Weight', color: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30' });
  }

  return specs;
}

export function StepText({
  title,
  description,
  price,
  onTitleChange,
  onDescriptionChange,
  onPriceChange,
}: StepTextProps) {
  const bulletCount = useMemo(() => {
    if (!description.trim()) return 0;
    return description.split('\n').filter(line => line.trim().length > 0).length;
  }, [description]);

  const detectedSpecs = useMemo(() => {
    return detectSpecs(title, description);
  }, [title, description]);

  const wordCount = useMemo(() => {
    const combined = `${title} ${description}`.trim();
    if (!combined) return 0;
    return combined.split(/\s+/).length;
  }, [title, description]);

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Heading */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-text mb-2">Product Text</h2>
        <p className="text-muted">
          Enter your product title and bullet points for NLP analysis
        </p>
      </div>

      {/* Title input */}
      <div>
        <label htmlFor="wizard-title" className="form-label flex items-center gap-2">
          <Type className="w-4 h-4 text-primary" />
          Product Title <span className="text-danger">*</span>
        </label>
        <input
          id="wizard-title"
          type="text"
          value={title}
          onChange={(e) => onTitleChange(e.target.value)}
          placeholder="Premium Stainless Steel Water Bottle 32oz..."
          className="form-input text-lg"
          maxLength={500}
        />
        <div className="flex items-center justify-between mt-2">
          <p className={cn(
            'text-xs',
            title.length > 200 ? 'text-warning' : 'text-muted'
          )}>
            {title.length}/500 characters
            {title.length > 0 && title.length < 50 && (
              <span className="text-warning ml-2">Short titles may rank lower</span>
            )}
          </p>
          {title.length >= 80 && title.length <= 200 && (
            <span className="text-xs text-success">Optimal length</span>
          )}
        </div>
      </div>

      {/* Description / Bullet Points */}
      <div>
        <label htmlFor="wizard-description" className="form-label flex items-center gap-2">
          <AlignLeft className="w-4 h-4 text-primary" />
          Bullet Points / Description
        </label>
        <textarea
          id="wizard-description"
          value={description}
          onChange={(e) => onDescriptionChange(e.target.value)}
          placeholder={"Key features and benefits, one per line:\n- BPA-free Tritan plastic construction\n- Keeps drinks cold 24hrs or hot 12hrs\n- Leak-proof double-wall vacuum insulation\n- Fits standard cup holders\n- Lifetime warranty included"}
          className="form-input min-h-[200px] resize-y font-mono text-sm leading-relaxed"
          maxLength={5000}
        />
        <div className="flex items-center justify-between mt-2">
          <p className="text-xs text-muted">
            {description.length}/5000 characters
          </p>
          <p className="text-xs text-muted">
            {bulletCount} {bulletCount === 1 ? 'line' : 'lines'}
          </p>
        </div>
      </div>

      {/* Price (optional) */}
      <div>
        <label htmlFor="wizard-price" className="form-label flex items-center gap-2">
          <DollarSign className="w-4 h-4 text-primary" />
          Price (USD) <span className="text-xs text-muted font-normal ml-1">Optional</span>
        </label>
        <div className="relative">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted">$</span>
          <input
            id="wizard-price"
            type="number"
            min="0"
            step="0.01"
            value={price ?? ''}
            onChange={(e) => {
              const val = e.target.value;
              onPriceChange?.(val === '' ? undefined : parseFloat(val));
            }}
            placeholder="29.99"
            className="form-input pl-7 max-w-[200px]"
          />
        </div>
        <p className="text-xs text-muted mt-1">
          Helps the model compare your pricing against category benchmarks
        </p>
      </div>

      {/* Live analysis preview */}
      {(title.trim() || description.trim()) && (
        <div className="rounded-xl border border-border bg-surface p-5 space-y-4">
          <h3 className="text-sm font-medium text-muted uppercase tracking-wider">
            Live Analysis Preview
          </h3>

          {/* Detected specs */}
          {detectedSpecs.length > 0 && (
            <div>
              <p className="text-xs text-muted mb-2">Detected Attributes</p>
              <div className="flex flex-wrap gap-2">
                {detectedSpecs.map((spec) => (
                  <span
                    key={spec.label}
                    className={cn(
                      'inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium border',
                      spec.color
                    )}
                  >
                    {spec.label}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Quick stats */}
          <div className="grid grid-cols-3 gap-3">
            <div className="p-3 bg-surface-2 rounded-lg text-center">
              <p className="text-lg font-bold text-text">{wordCount}</p>
              <p className="text-xs text-muted">Words</p>
            </div>
            <div className="p-3 bg-surface-2 rounded-lg text-center">
              <p className="text-lg font-bold text-text">{title.length}</p>
              <p className="text-xs text-muted">Title Chars</p>
            </div>
            <div className="p-3 bg-surface-2 rounded-lg text-center">
              <p className="text-lg font-bold text-text">{bulletCount}</p>
              <p className="text-xs text-muted">Bullet Lines</p>
            </div>
          </div>

          {detectedSpecs.length === 0 && (title.trim() || description.trim()) && (
            <p className="text-xs text-warning">
              No product attributes detected yet. Including size, material, color, and quantity helps improve ranking.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
