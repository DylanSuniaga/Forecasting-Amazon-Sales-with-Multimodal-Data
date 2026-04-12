'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp, Image as ImageIcon } from 'lucide-react';
import { ScoreCard, RankBandCard } from './ScoreCard';
import { SignalPanel } from './SignalList';
import { NLPFeedbackPanel } from './NLPFeedbackPanel';
import { ImageEvidence } from './ImageEvidence';
import { FeatureImportancePanel } from './FeatureImportancePanel';
import { HowItWorksPanel } from './HowItWorksPanel';
import { ImprovementBar } from './ImprovementBar';
import { getRankBandPercent } from '@/lib/utils';
import type { EvaluationResponse } from '@/types';

interface StepResultsProps {
  evaluation: EvaluationResponse;
  category?: string;
}

export function StepResults({ evaluation, category }: StepResultsProps) {
  const [showImageQuality, setShowImageQuality] = useState(false);

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Top: Score Cards Row */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <ScoreCard
          title="Viability Score"
          value={evaluation.launch_viability_score}
          subtitle="Overall (0-100)"
          type="score"
        />
        <ScoreCard
          title="BSR Probability"
          value={evaluation.bsr_entry_probability}
          subtitle="Likelihood of ranking"
          type="probability"
        />
        <RankBandCard
          title="Rank Band"
          band={evaluation.expected_rank_band}
          percentile={getRankBandPercent(evaluation.expected_rank_band)}
          estimatedRank={evaluation.feature_summary?.estimated_rank}
        />
        <ScoreCard
          title="Competition"
          value={evaluation.competitive_intensity}
          subtitle="Category intensity"
          type="percent"
        />
      </div>

      {/* Estimated BSR Rank (if probability > 50%) */}
      {evaluation.bsr_entry_probability > 0.5 && evaluation.feature_summary?.estimated_rank && (
        <div className="bento-card bg-primary/10 border border-primary/30">
          <p className="text-sm text-muted mb-2">Estimated BSR Rank</p>
          <p className="text-3xl font-bold text-primary">
            #{Math.round(evaluation.feature_summary.estimated_rank).toLocaleString()}
          </p>
          <p className="text-xs text-muted mt-2">
            Based on regression model (BSR probability: {(evaluation.bsr_entry_probability * 100).toFixed(1)}%)
          </p>
        </div>
      )}

      {/* Improvement Bar - actionable feedback */}
      <ImprovementBar evaluation={evaluation} />

      {/* Middle: Three-column layout */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Left: NLP Feedback */}
        <NLPFeedbackPanel feedback={evaluation.nlp_feedback} />

        {/* Right: Signal Decomposition */}
        <SignalPanel
          strengths={evaluation.strengths}
          risks={evaluation.risks}
          categoryNotes={evaluation.category_notes}
          defaultCollapsed={false}
        />
      </div>

      {/* Feature Importance */}
      <FeatureImportancePanel features={evaluation.top_feature_importances} />

      {/* Placeholder warning */}
      {evaluation.feature_summary?.is_placeholder && (
        <div className="text-xs text-warning bg-warning/10 px-4 py-2 rounded-lg">
          Note: Using placeholder predictions. Save models from notebooks for real inference.
        </div>
      )}

      {/* Bottom: Image Quality Summary (collapsible) */}
      <div className="rounded-xl border border-border bg-surface overflow-hidden">
        <button
          onClick={() => setShowImageQuality(!showImageQuality)}
          className="w-full flex items-center justify-between p-6 hover:bg-surface-2/30 transition-colors"
        >
          <div className="flex items-center gap-3">
            <ImageIcon className="w-5 h-5 text-primary" />
            <span className="text-lg font-semibold text-text">Image Quality Analysis</span>
          </div>
          {showImageQuality ? (
            <ChevronUp className="w-5 h-5 text-muted" />
          ) : (
            <ChevronDown className="w-5 h-5 text-muted" />
          )}
        </button>

        {showImageQuality && (
          <div className="px-6 pb-6 border-t border-border pt-6">
            <ImageEvidence
              metrics={evaluation.image_quality || null}
              category={category}
            />
          </div>
        )}
      </div>

      {/* How It Works */}
      <HowItWorksPanel />
    </div>
  );
}
