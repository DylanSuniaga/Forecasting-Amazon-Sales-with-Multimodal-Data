'use client';

import { TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';
import type { EvaluationResponse } from '@/types';
import { cn } from '@/lib/utils';

interface ImprovementBarProps {
  evaluation: EvaluationResponse;
}

interface ImprovementItem {
  label: string;
  current: string;
  recommendation: string;
  priority: 'high' | 'medium' | 'low';
}

export function ImprovementBar({ evaluation }: ImprovementBarProps) {
  const items: ImprovementItem[] = [];

  // ── NLP-based improvements ──────────────────────────────────────────
  const nlp = evaluation.nlp_feedback;
  if (nlp) {
    // Title keyword match
    if (nlp.title_score < 40) {
      const topMissing = (nlp.missing_keywords ?? [])
        .filter(k => k.source === 'title')
        .slice(0, 3)
        .map(k => `"${k.keyword}"`)
        .join(', ');
      items.push({
        label: 'Title Keywords',
        current: `${Math.round(nlp.title_score)}% match`,
        recommendation: topMissing
          ? `Add these high-value keywords to your title: ${topMissing}. Top sellers in this category use them frequently.`
          : 'Rewrite your title to include category-specific keywords that top sellers use.',
        priority: 'high',
      });
    } else if (nlp.title_score < 65) {
      const topMissing = (nlp.missing_keywords ?? [])
        .filter(k => k.source === 'title')
        .slice(0, 2)
        .map(k => `"${k.keyword}"`)
        .join(', ');
      items.push({
        label: 'Title Keywords',
        current: `${Math.round(nlp.title_score)}% match`,
        recommendation: topMissing
          ? `Consider adding ${topMissing} to your title to better match top performers.`
          : 'Review the missing keywords above and incorporate the most relevant ones.',
        priority: 'medium',
      });
    }

    // Title structure
    const stats = nlp.title_stats;
    if (stats) {
      if ((stats.word_count ?? 0) < 8) {
        items.push({
          label: 'Title Too Short',
          current: `${stats.word_count} words`,
          recommendation: 'Expand your title to 15-25 words. Include brand, key feature, size/quantity, and material.',
          priority: 'high',
        });
      } else if ((stats.word_count ?? 0) > 30) {
        items.push({
          label: 'Title Too Long',
          current: `${stats.word_count} words`,
          recommendation: 'Trim your title to under 25 words. Focus on the most important features and remove filler.',
          priority: 'medium',
        });
      }
      if (stats.has_size_spec === false) {
        items.push({
          label: 'Add Size/Quantity',
          current: 'Not found',
          recommendation: 'Include a size or quantity spec (e.g., "32oz", "12-Pack", "6ft") — top sellers nearly always do.',
          priority: 'low',
        });
      }
    }

    // Bullets
    if (nlp.bullets_score < 30 && (nlp.bullets_stats?.total_word_count ?? 0) < 10) {
      items.push({
        label: 'Bullet Points Missing',
        current: 'None provided',
        recommendation: 'Add 5 bullet points (150+ chars each) covering: key benefit, materials, dimensions, use case, and warranty/guarantee.',
        priority: 'high',
      });
    } else if (nlp.bullets_score < 50) {
      const bulletMissing = (nlp.missing_keywords ?? [])
        .filter(k => k.source === 'bullets')
        .slice(0, 3)
        .map(k => `"${k.keyword}"`)
        .join(', ');
      items.push({
        label: 'Bullet Points',
        current: `${Math.round(nlp.bullets_score)}% match`,
        recommendation: bulletMissing
          ? `Strengthen bullets by mentioning: ${bulletMissing}. Focus on concrete benefits, not just features.`
          : 'Expand bullets with more specific details — dimensions, materials, warranty, and use cases.',
        priority: 'medium',
      });
    }
  }

  // ── Image-based improvements ────────────────────────────────────────
  const img = evaluation.image_quality;
  if (img) {
    if (img.width < 1000 || img.height < 1000) {
      items.push({
        label: 'Image Resolution',
        current: `${img.width}×${img.height}px`,
        recommendation: `Reshoot or upscale to at least 1500×1500px. Amazon requires 1000px minimum, but 2000px+ enables zoom.`,
        priority: 'high',
      });
    }

    if (img.white_bg_pct < 30) {
      items.push({
        label: 'White Background',
        current: `${img.white_bg_pct.toFixed(0)}% white`,
        recommendation: 'Amazon requires a pure white background (RGB 255,255,255) for main images. Use a photo editing tool or reshoot on white.',
        priority: 'high',
      });
    } else if (img.white_bg_pct < 60) {
      items.push({
        label: 'Background Cleanliness',
        current: `${img.white_bg_pct.toFixed(0)}% white`,
        recommendation: 'Your background isn\'t fully clean. Remove shadows, props, or color casts to get closer to pure white.',
        priority: 'medium',
      });
    }

    if (img.sharpness < 200) {
      items.push({
        label: 'Image Sharpness',
        current: 'Below average',
        recommendation: 'Your image looks soft or blurry. Use a tripod, better lighting, or apply sharpening in post-processing.',
        priority: 'medium',
      });
    }

    if (img.brightness_mean < 120) {
      items.push({
        label: 'Image Too Dark',
        current: `Brightness: ${img.brightness_mean.toFixed(0)}`,
        recommendation: 'Increase lighting or adjust exposure. Product images should be bright and well-lit (aim for 170+ brightness).',
        priority: 'medium',
      });
    } else if (img.brightness_mean > 240) {
      items.push({
        label: 'Image Overexposed',
        current: `Brightness: ${img.brightness_mean.toFixed(0)}`,
        recommendation: 'Image appears washed out. Reduce exposure to preserve product detail and color accuracy.',
        priority: 'medium',
      });
    }

    if (img.contrast < 20) {
      items.push({
        label: 'Low Contrast',
        current: `Contrast: ${img.contrast.toFixed(0)}`,
        recommendation: 'Increase contrast so the product pops against the background. Adjust levels or curves in your editor.',
        priority: 'low',
      });
    }

    if (img.edge_density > 0.15) {
      items.push({
        label: 'Image Clutter',
        current: 'High edge density',
        recommendation: 'Too many visual elements in the frame. Simplify: one product, clean background, minimal props.',
        priority: 'low',
      });
    }

    // Category median comparisons
    const vs = img.vs_category_median;
    if (vs) {
      if (vs.sharpness && vs.sharpness < -0.3) {
        items.push({
          label: 'Sharpness vs Category',
          current: `${((1 + vs.sharpness) * 100).toFixed(0)}% of avg`,
          recommendation: 'Your image is significantly less sharp than competitors. Invest in better product photography.',
          priority: 'medium',
        });
      }
      if (vs.colorfulness && vs.colorfulness < -0.3) {
        items.push({
          label: 'Color Vibrancy',
          current: `${((1 + vs.colorfulness) * 100).toFixed(0)}% of avg`,
          recommendation: 'Your image has less color pop than top sellers. Adjust saturation or use more vibrant product styling.',
          priority: 'low',
        });
      }
    }
  }

  // ── Price-based improvements ────────────────────────────────────────
  // Check if price was provided in the feature summary
  const feats = evaluation.feature_summary;
  const risks = evaluation.risks ?? [];
  const priceRisk = risks.find(r =>
    r.label.toLowerCase().includes('price') ||
    r.description.toLowerCase().includes('price')
  );
  if (priceRisk) {
    items.push({
      label: 'Pricing',
      current: priceRisk.description.includes('high') ? 'Above average' : 'Review pricing',
      recommendation: priceRisk.description.includes('high') || priceRisk.description.includes('above')
        ? 'Your price is above category average. Consider reducing by 10-15% or bundling extras to justify premium pricing.'
        : priceRisk.description.includes('low') || priceRisk.description.includes('below')
          ? 'Your price is well below average, which can signal low quality. Consider raising it slightly or adding a "was $X" reference.'
          : 'Review your pricing relative to competitors — it may be impacting your predicted rank.',
      priority: priceRisk.impact > 0.5 ? 'high' : 'medium',
    });
  }

  // ── Score-based fallback ────────────────────────────────────────────
  if (evaluation.bsr_entry_probability < 0.4 && items.length < 2) {
    items.push({
      label: 'Low BSR Probability',
      current: `${(evaluation.bsr_entry_probability * 100).toFixed(0)}%`,
      recommendation: 'This product has a low chance of ranking. Focus on the highest-priority items above, or consider pivoting to a less competitive subcategory.',
      priority: 'high',
    });
  }

  if (items.length === 0) {
    return (
      <div className="bento-card bg-success/5 border-success/20">
        <div className="flex items-center gap-3">
          <CheckCircle className="w-5 h-5 text-success" />
          <div>
            <p className="text-sm font-medium text-success">Looking Good</p>
            <p className="text-xs text-muted mt-0.5">No major improvements identified. Your listing appears well-optimized.</p>
          </div>
        </div>
      </div>
    );
  }

  // Sort by priority
  const priorityOrder = { high: 0, medium: 1, low: 2 };
  items.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

  // Limit to top 6 to avoid overwhelming
  const shown = items.slice(0, 6);

  return (
    <div className="bento-card">
      <div className="flex items-center gap-3 mb-4">
        <TrendingUp className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-semibold text-text">How to Improve</h3>
        <span className="text-xs bg-primary/15 text-primary px-2 py-0.5 rounded-full">
          {items.length} {items.length === 1 ? 'suggestion' : 'suggestions'}
        </span>
      </div>

      <div className="space-y-3">
        {shown.map((item, idx) => (
          <div
            key={idx}
            className={cn(
              'flex items-start gap-3 p-3 rounded-lg border',
              item.priority === 'high'
                ? 'bg-danger/5 border-danger/20'
                : item.priority === 'medium'
                  ? 'bg-warning/5 border-warning/20'
                  : 'bg-surface-2 border-border'
            )}
          >
            <AlertTriangle className={cn(
              'w-4 h-4 mt-0.5 flex-shrink-0',
              item.priority === 'high' ? 'text-danger' :
              item.priority === 'medium' ? 'text-warning' : 'text-muted'
            )} />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-0.5">
                <span className="text-sm font-medium text-text">{item.label}</span>
                <span className={cn(
                  'text-[10px] px-1.5 py-0.5 rounded-full font-medium uppercase',
                  item.priority === 'high'
                    ? 'bg-danger/15 text-danger'
                    : item.priority === 'medium'
                      ? 'bg-warning/15 text-warning'
                      : 'bg-muted/15 text-muted'
                )}>
                  {item.priority}
                </span>
                <span className="text-xs text-muted ml-auto">{item.current}</span>
              </div>
              <p className="text-xs text-muted leading-relaxed">{item.recommendation}</p>
            </div>
          </div>
        ))}
      </div>
      {items.length > shown.length && (
        <p className="text-xs text-muted mt-3 text-center">
          + {items.length - shown.length} more suggestions
        </p>
      )}
    </div>
  );
}
