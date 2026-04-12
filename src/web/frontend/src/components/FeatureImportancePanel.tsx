'use client';

import { BarChart3 } from 'lucide-react';
import type { FeatureImportance } from '@/types';
import { cn } from '@/lib/utils';

interface FeatureImportancePanelProps {
  features: FeatureImportance[] | null | undefined;
  className?: string;
}

const TYPE_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  image: { bg: 'bg-blue-500/15', text: 'text-blue-400', label: 'Image' },
  text: { bg: 'bg-emerald-500/15', text: 'text-emerald-400', label: 'Text' },
  price: { bg: 'bg-amber-500/15', text: 'text-amber-400', label: 'Price' },
};

export function FeatureImportancePanel({ features, className }: FeatureImportancePanelProps) {
  if (!features || features.length === 0) return null;

  const maxImportance = Math.max(...features.map(f => f.importance));

  return (
    <div className={cn('bento-card', className)}>
      <div className="flex items-center gap-3 mb-5">
        <BarChart3 className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-semibold text-text">What Drove This Prediction</h3>
      </div>

      <div className="space-y-3">
        {features.map((feat, idx) => {
          const pct = maxImportance > 0 ? (feat.importance / maxImportance) * 100 : 0;
          const typeInfo = TYPE_COLORS[feat.type] || TYPE_COLORS.image;

          return (
            <div key={feat.feature} className="group">
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-sm text-text truncate">{feat.display_name}</span>
                  <span className={cn(
                    'text-[10px] px-1.5 py-0.5 rounded-full font-medium flex-shrink-0',
                    typeInfo.bg, typeInfo.text
                  )}>
                    {typeInfo.label}
                  </span>
                </div>
                <span className="text-xs text-muted ml-2 flex-shrink-0 tabular-nums">
                  {(feat.importance * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 bg-surface-2 rounded-full overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded-full transition-all duration-700 ease-out',
                    idx === 0 ? 'bg-primary' : idx < 3 ? 'bg-primary/70' : 'bg-primary/40'
                  )}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-xs text-muted mt-4">
        Feature importance from XGBoost classification model (gain-based)
      </p>
    </div>
  );
}
