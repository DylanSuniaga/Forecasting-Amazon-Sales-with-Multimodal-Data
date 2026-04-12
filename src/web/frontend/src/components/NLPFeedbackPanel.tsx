'use client';

import { useState } from 'react';
import { FileText, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';
import type { NLPFeedback } from '@/types';
import { cn } from '@/lib/utils';

interface NLPFeedbackPanelProps {
  feedback: NLPFeedback | null | undefined;
  className?: string;
}

// Circular gauge component for scores
function CircularGauge({ value, label, size = 96 }: { value: number; label: string; size?: number }) {
  const radius = (size - 12) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = (value / 100) * circumference;
  const remaining = circumference - progress;

  const getColor = (v: number) => {
    if (v >= 75) return { stroke: 'var(--color-success)', text: 'text-success' };
    if (v >= 50) return { stroke: 'var(--color-primary)', text: 'text-primary' };
    if (v >= 25) return { stroke: 'var(--color-warning)', text: 'text-warning' };
    return { stroke: 'var(--color-danger)', text: 'text-danger' };
  };

  const color = getColor(value);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          {/* Background circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="var(--gauge-track)"
            strokeWidth="6"
          />
          {/* Progress circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color.stroke}
            strokeWidth="6"
            strokeDasharray={`${progress} ${remaining}`}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={cn('text-xl font-bold', color.text)}>
            {Math.round(value)}
          </span>
        </div>
      </div>
      <span className="text-xs text-muted font-medium">{label}</span>
    </div>
  );
}

export function NLPFeedbackPanel({ feedback, className }: NLPFeedbackPanelProps) {
  const [showStats, setShowStats] = useState(false);

  if (!feedback) {
    return (
      <div className={cn('bento-card', className)}>
        <div className="flex items-center gap-3 mb-4">
          <FileText className="w-5 h-5 text-muted" />
          <h3 className="text-lg font-semibold text-text">Text Analysis</h3>
        </div>
        <div className="flex items-center gap-3 py-6 justify-center">
          <AlertCircle className="w-5 h-5 text-muted" />
          <p className="text-sm text-muted">Text analysis not available for this evaluation</p>
        </div>
      </div>
    );
  }

  const titleStats = feedback.title_stats || {};
  const bulletsStats = feedback.bullets_stats || {};

  return (
    <div className={cn('bento-card', className)}>
      <div className="flex items-center gap-3 mb-6">
        <FileText className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-semibold text-text">Text Analysis</h3>
      </div>

      {/* Score Gauges */}
      <div className="flex items-center justify-center gap-10 mb-8">
        <CircularGauge value={feedback.title_score} label="Title Score" />
        <CircularGauge value={feedback.bullets_score} label="Bullets Score" />
      </div>

      {/* Keywords to Add */}
      {feedback.missing_keywords && feedback.missing_keywords.length > 0 && (
        <div className="mb-5">
          <h4 className="text-sm font-medium text-text mb-3">Keywords to Add</h4>
          <div className="flex flex-wrap gap-2">
            {feedback.missing_keywords.map((kw, idx) => (
              <div
                key={idx}
                className="group relative inline-flex items-center gap-1.5 px-3 py-1.5 bg-success/15 border border-success/30 text-success rounded-full text-sm cursor-default"
              >
                <span className="font-medium">{kw.keyword}</span>
                <span className="text-xs opacity-75">({Math.round(kw.prevalence)}%)</span>
                {/* Tooltip */}
                {kw.suggestion && (
                  <div className="tooltip -top-10 left-1/2 -translate-x-1/2 whitespace-nowrap">
                    {kw.suggestion}
                  </div>
                )}
              </div>
            ))}
          </div>
          <p className="text-xs text-muted mt-2">
            High-value keywords found in successful products within this category
          </p>
        </div>
      )}

      {/* Keywords to Reconsider */}
      {feedback.weak_keywords && feedback.weak_keywords.length > 0 && (
        <div className="mb-5">
          <h4 className="text-sm font-medium text-text mb-3">Keywords to Reconsider</h4>
          <div className="flex flex-wrap gap-2">
            {feedback.weak_keywords.map((kw, idx) => (
              <div
                key={idx}
                className="group relative inline-flex items-center gap-1.5 px-3 py-1.5 bg-warning/15 border border-warning/30 text-warning rounded-full text-sm cursor-default"
              >
                <span className="font-medium">{kw.keyword}</span>
                <span className="text-xs opacity-75">({Math.round(kw.prevalence)}%)</span>
                {kw.suggestion && (
                  <div className="tooltip -top-10 left-1/2 -translate-x-1/2 whitespace-nowrap">
                    {kw.suggestion}
                  </div>
                )}
              </div>
            ))}
          </div>
          <p className="text-xs text-muted mt-2">
            These keywords may be hurting your listing or are associated with lower-performing products
          </p>
        </div>
      )}

      {/* No keywords message */}
      {(!feedback.missing_keywords || feedback.missing_keywords.length === 0) &&
       (!feedback.weak_keywords || feedback.weak_keywords.length === 0) && (
        <p className="text-sm text-muted mb-5 text-center py-3">
          No specific keyword recommendations available
        </p>
      )}

      {/* Text Statistics (expandable) */}
      <div className="border-t border-border pt-4">
        <button
          onClick={() => setShowStats(!showStats)}
          className="w-full flex items-center justify-between text-sm text-muted hover:text-text transition-colors"
        >
          <span className="font-medium">Text Statistics</span>
          {showStats ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>

        {showStats && (
          <div className="mt-4 space-y-4">
            {/* Title Stats */}
            <div>
              <p className="text-xs text-muted uppercase tracking-wider mb-2">Title</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {Object.entries(titleStats).map(([key, value]) => (
                  <div key={key} className="p-2 bg-surface-2 rounded-lg">
                    <p className="text-xs text-muted capitalize">{key.replace(/_/g, ' ')}</p>
                    <p className="text-sm font-medium text-text">
                      {typeof value === 'number' ? (Number.isInteger(value) ? value : (value as number).toFixed(2)) : String(value)}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Bullets Stats */}
            {Object.keys(bulletsStats).length > 0 && (
              <div>
                <p className="text-xs text-muted uppercase tracking-wider mb-2">Bullet Points</p>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {Object.entries(bulletsStats).map(([key, value]) => (
                    <div key={key} className="p-2 bg-surface-2 rounded-lg">
                      <p className="text-xs text-muted capitalize">{key.replace(/_/g, ' ')}</p>
                      <p className="text-sm font-medium text-text">
                        {typeof value === 'number' ? (Number.isInteger(value) ? value : (value as number).toFixed(2)) : String(value)}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
