'use client';

import { Image as ImageIcon, Check, X, Minus } from 'lucide-react';
import type { ImageQualityMetrics } from '@/types';
import { cn, formatNumber } from '@/lib/utils';

interface ImageEvidenceProps {
  metrics: ImageQualityMetrics | null;
  category?: string;
}

export function ImageEvidence({ metrics, category }: ImageEvidenceProps) {
  if (!metrics) {
    return (
      <div className="bento-card text-center py-12">
        <ImageIcon className="w-12 h-12 text-muted mx-auto mb-4" />
        <p className="text-muted">
          Upload and evaluate a product to see image analysis
        </p>
      </div>
    );
  }

  // Category medians from API response (vs_category_median contains % differences)
  // Compute actual medians from current values and % differences
  const categoryMedians = metrics.vs_category_median ? (() => {
    const medians: Record<string, number> = {};
    // Reverse calculate: if vs_median = (current - median) / median * 100
    // Then: median = current / (1 + vs_median / 100)
    if (metrics.vs_category_median.brightness_mean !== undefined) {
      medians.brightness_mean = metrics.brightness_mean / (1 + metrics.vs_category_median.brightness_mean / 100);
    }
    if (metrics.vs_category_median.contrast !== undefined) {
      medians.contrast = metrics.contrast / (1 + metrics.vs_category_median.contrast / 100);
    }
    if (metrics.vs_category_median.sharpness !== undefined) {
      medians.sharpness = metrics.sharpness / (1 + metrics.vs_category_median.sharpness / 100);
    }
    if (metrics.vs_category_median.white_bg_pct !== undefined) {
      medians.white_bg_pct = metrics.white_bg_pct / (1 + metrics.vs_category_median.white_bg_pct / 100);
    }
    if (metrics.vs_category_median.edge_density !== undefined) {
      medians.edge_density = metrics.edge_density / (1 + metrics.vs_category_median.edge_density / 100);
    }
    // Defaults if not available
    return {
      brightness_mean: medians.brightness_mean || 180,
      contrast: medians.contrast || 50,
      saturation_mean: 100,
      sharpness: medians.sharpness || 400,
      white_bg_pct: medians.white_bg_pct || 45,
      edge_density: medians.edge_density || 0.12,
    };
  })() : {
    // Fallback if no medians available
    brightness_mean: 180,
    contrast: 50,
    saturation_mean: 100,
    sharpness: 400,
    white_bg_pct: 45,
    edge_density: 0.12,
  };

  const indicators = [
    {
      label: 'Resolution',
      value: `${metrics.width} × ${metrics.height}`,
      status: metrics.width >= 1000 && metrics.height >= 1000 ? 'good' : 
              metrics.width >= 500 && metrics.height >= 500 ? 'ok' : 'bad',
      description: metrics.width >= 1000 ? 'Meets Amazon requirements' : 'Below recommended minimum'
    },
    {
      label: 'Aspect Ratio',
      value: metrics.aspect_ratio.toFixed(2),
      status: Math.abs(metrics.aspect_ratio - 1) < 0.1 ? 'good' : 
              Math.abs(metrics.aspect_ratio - 1) < 0.3 ? 'ok' : 'bad',
      description: Math.abs(metrics.aspect_ratio - 1) < 0.1 ? 'Square (ideal)' : 'Non-square ratio'
    },
    {
      label: 'Brightness',
      value: formatNumber(metrics.brightness_mean),
      status: metrics.brightness_mean > 150 && metrics.brightness_mean < 220 ? 'good' : 
              metrics.brightness_mean > 100 && metrics.brightness_mean < 240 ? 'ok' : 'bad',
      description: `Category median: ${categoryMedians.brightness_mean}`,
      vsMedian: ((metrics.brightness_mean - categoryMedians.brightness_mean) / categoryMedians.brightness_mean * 100).toFixed(0)
    },
    {
      label: 'Contrast',
      value: formatNumber(metrics.contrast),
      status: metrics.contrast > 40 && metrics.contrast < 80 ? 'good' : 
              metrics.contrast > 20 && metrics.contrast < 100 ? 'ok' : 'bad',
      description: `Category median: ${categoryMedians.contrast}`,
      vsMedian: ((metrics.contrast - categoryMedians.contrast) / categoryMedians.contrast * 100).toFixed(0)
    },
    {
      label: 'Sharpness',
      value: formatNumber(metrics.sharpness, 0),
      status: metrics.sharpness > 500 ? 'good' : 
              metrics.sharpness > 200 ? 'ok' : 'bad',
      description: metrics.sharpness > 500 ? 'Sharp and clear' : 'May appear soft',
      vsMedian: ((metrics.sharpness - categoryMedians.sharpness) / categoryMedians.sharpness * 100).toFixed(0)
    },
    {
      label: 'White Background %',
      value: `${formatNumber(metrics.white_bg_pct, 1)}%`,
      status: metrics.white_bg_pct > 50 ? 'good' : 
              metrics.white_bg_pct > 30 ? 'ok' : 'bad',
      description: metrics.white_bg_pct > 50 ? 'Clean background' : 'Consider white background',
      vsMedian: ((metrics.white_bg_pct - categoryMedians.white_bg_pct) / categoryMedians.white_bg_pct * 100).toFixed(0)
    },
  ];

  return (
    <div className="space-y-6">
      {/* Main metrics grid */}
      <div className="bento-card">
        <h3 className="text-lg font-semibold text-text mb-4">Image Quality Indicators</h3>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {indicators.map((indicator) => (
            <QualityIndicator key={indicator.label} {...indicator} />
          ))}
        </div>
      </div>

      {/* Category comparison */}
      <div className="bento-card">
        <h3 className="text-lg font-semibold text-text mb-4">
          vs Category Median {category && <span className="text-muted font-normal">({category})</span>}
        </h3>
        <div className="space-y-3">
          <ComparisonBar 
            label="Sharpness" 
            value={metrics.sharpness} 
            median={categoryMedians.sharpness}
            max={1000}
          />
          <ComparisonBar 
            label="White Background" 
            value={metrics.white_bg_pct} 
            median={categoryMedians.white_bg_pct}
            max={100}
            unit="%"
          />
          <ComparisonBar 
            label="Edge Density" 
            value={metrics.edge_density} 
            median={categoryMedians.edge_density}
            max={0.3}
          />
        </div>
        <p className="text-xs text-muted mt-4">
          Note: Category medians are illustrative. Real values computed from training data.
        </p>
      </div>

      {/* Raw metrics (collapsible) */}
      <details className="bento-card">
        <summary className="collapsible-header cursor-pointer list-none">
          <span>Raw Metrics</span>
        </summary>
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
          <div><span className="text-muted">Width:</span> {metrics.width}px</div>
          <div><span className="text-muted">Height:</span> {metrics.height}px</div>
          <div><span className="text-muted">Aspect:</span> {metrics.aspect_ratio.toFixed(3)}</div>
          <div><span className="text-muted">Brightness:</span> {metrics.brightness_mean.toFixed(1)}</div>
          <div><span className="text-muted">Contrast:</span> {metrics.contrast.toFixed(1)}</div>
          <div><span className="text-muted">Saturation:</span> {metrics.saturation_mean.toFixed(1)}</div>
          <div><span className="text-muted">Colorfulness:</span> {metrics.colorfulness.toFixed(1)}</div>
          <div><span className="text-muted">Sharpness:</span> {metrics.sharpness.toFixed(1)}</div>
          <div><span className="text-muted">Blur Score:</span> {metrics.blur_score.toFixed(4)}</div>
          <div><span className="text-muted">White BG:</span> {metrics.white_bg_pct.toFixed(1)}%</div>
          <div><span className="text-muted">Edge Density:</span> {metrics.edge_density.toFixed(4)}</div>
        </div>
      </details>
    </div>
  );
}

interface QualityIndicatorProps {
  label: string;
  value: string;
  status: 'good' | 'ok' | 'bad';
  description: string;
  vsMedian?: string;
}

function QualityIndicator({ label, value, status, description, vsMedian }: QualityIndicatorProps) {
  const statusConfig = {
    good: { icon: Check, color: 'text-success', bg: 'bg-success/10' },
    ok: { icon: Minus, color: 'text-warning', bg: 'bg-warning/10' },
    bad: { icon: X, color: 'text-danger', bg: 'bg-danger/10' },
  };

  const { icon: Icon, color, bg } = statusConfig[status];

  return (
    <div className={cn('p-3 rounded-lg', bg)}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-muted uppercase tracking-wider">{label}</span>
        <Icon className={cn('w-4 h-4', color)} />
      </div>
      <p className="text-lg font-semibold text-text">{value}</p>
      <p className="text-xs text-muted mt-1">{description}</p>
      {vsMedian && (
        <p className={cn('text-xs mt-1', Number(vsMedian) >= 0 ? 'text-success' : 'text-warning')}>
          {Number(vsMedian) >= 0 ? '+' : ''}{vsMedian}% vs median
        </p>
      )}
    </div>
  );
}

interface ComparisonBarProps {
  label: string;
  value: number;
  median: number;
  max: number;
  unit?: string;
}

function ComparisonBar({ label, value, median, max, unit = '' }: ComparisonBarProps) {
  const valuePercent = Math.min((value / max) * 100, 100);
  const medianPercent = (median / max) * 100;

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-muted">{label}</span>
        <span className="text-text">{formatNumber(value, 2)}{unit}</span>
      </div>
      <div className="relative h-2 bg-surface-2 rounded-full">
        {/* Value bar */}
        <div 
          className={cn(
            'absolute h-full rounded-full',
            value >= median ? 'bg-success' : 'bg-warning'
          )}
          style={{ width: `${valuePercent}%` }}
        />
        {/* Median marker */}
        <div 
          className="absolute w-0.5 h-4 -top-1 bg-primary"
          style={{ left: `${medianPercent}%` }}
        />
      </div>
    </div>
  );
}
