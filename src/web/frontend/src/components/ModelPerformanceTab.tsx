'use client';

import { useState, useEffect } from 'react';
import { Activity, Loader2, ArrowLeft } from 'lucide-react';
import { getValidationSummary, getModels } from '@/lib/api';
import type { ValidationMetrics, ModelInfo } from '@/types';
import { cn } from '@/lib/utils';

interface ModelPerformanceTabProps {
  onBack: () => void;
}

export function ModelPerformanceTab({ onBack }: ModelPerformanceTabProps) {
  const [metrics, setMetrics] = useState<ValidationMetrics[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [overallAuc, setOverallAuc] = useState(0);
  const [overallF1, setOverallF1] = useState(0);
  const [overallR2, setOverallR2] = useState(0);
  const [overallMae, setOverallMae] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const [valData, modelData] = await Promise.all([
          getValidationSummary().catch(() => null),
          getModels().catch(() => null),
        ]);
        if (valData) {
          setMetrics(valData.by_category || []);
          setOverallAuc(valData.overall_auc);
          setOverallF1(valData.overall_f1);
          setOverallR2(valData.overall_r2 ?? 0);
          setOverallMae(valData.overall_mae ?? 0);
        }
        if (modelData) {
          setModels(modelData.category_models || []);
        }
      } catch {
        // silently handle
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={onBack} className="btn-secondary flex items-center gap-2">
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <div>
          <h2 className="text-2xl font-bold text-text">Model Performance</h2>
          <p className="text-sm text-muted">Classification and regression metrics across 14 categories</p>
        </div>
      </div>

      {/* Overall summary cards */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bento-card text-center">
          <p className="text-xs text-muted uppercase tracking-wider mb-1">CLF ROC-AUC</p>
          <p className={cn(
            'text-3xl font-bold',
            overallAuc >= 0.9 ? 'text-success' : overallAuc >= 0.8 ? 'text-primary' : 'text-warning'
          )}>
            {overallAuc.toFixed(3)}
          </p>
        </div>
        <div className="bento-card text-center">
          <p className="text-xs text-muted uppercase tracking-wider mb-1">CLF F1 Score</p>
          <p className={cn(
            'text-3xl font-bold',
            overallF1 >= 0.9 ? 'text-success' : overallF1 >= 0.8 ? 'text-primary' : 'text-warning'
          )}>
            {overallF1.toFixed(3)}
          </p>
        </div>
        <div className="bento-card text-center">
          <p className="text-xs text-muted uppercase tracking-wider mb-1">REG R² Score</p>
          <p className={cn(
            'text-3xl font-bold',
            overallR2 >= 0.3 ? 'text-success' : overallR2 >= 0.15 ? 'text-primary' : 'text-warning'
          )}>
            {overallR2.toFixed(3)}
          </p>
        </div>
        <div className="bento-card text-center">
          <p className="text-xs text-muted uppercase tracking-wider mb-1">REG MAE (log)</p>
          <p className={cn(
            'text-3xl font-bold',
            overallMae <= 1.0 ? 'text-success' : overallMae <= 1.5 ? 'text-primary' : 'text-warning'
          )}>
            {overallMae.toFixed(3)}
          </p>
        </div>
        <div className="bento-card text-center">
          <p className="text-xs text-muted uppercase tracking-wider mb-1">Models Loaded</p>
          <p className="text-3xl font-bold text-text">
            {models.filter(m => m.loaded).length}/{models.length}
          </p>
        </div>
      </div>

      {/* Per-category classification table */}
      <div className="bento-card overflow-hidden">
        <h3 className="text-lg font-semibold text-text mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-primary" />
          Classification — Has BSR? (per category)
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-3 text-muted font-medium">Category</th>
                <th className="text-center py-3 px-3 text-muted font-medium">Best Model</th>
                <th className="text-right py-3 px-3 text-muted font-medium">ROC-AUC</th>
                <th className="text-right py-3 px-3 text-muted font-medium">F1</th>
                <th className="text-right py-3 px-3 text-muted font-medium">Precision</th>
                <th className="text-right py-3 px-3 text-muted font-medium">Recall</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m) => {
                const model = models.find(mod => mod.category === m.category);
                const modelType = model?.model_type?.split('/')[0]?.replace('CLF:', '').trim() || '—';
                return (
                  <tr key={m.category} className="border-b border-border/50 hover:bg-surface-2/30 transition-colors">
                    <td className="py-3 px-3 text-text font-medium">{m.category}</td>
                    <td className="py-3 px-3 text-center">
                      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-primary/10 text-primary">
                        {modelType}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right">
                      <MetricBadge value={m.roc_auc} />
                    </td>
                    <td className="py-3 px-3 text-right">
                      <MetricBadge value={m.f1_score} />
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums text-muted">
                      {m.precision.toFixed(3)}
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums text-muted">
                      {m.recall.toFixed(3)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-category regression table */}
      <div className="bento-card overflow-hidden">
        <h3 className="text-lg font-semibold text-text mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-primary" />
          Regression — BSR Rank Prediction (per category)
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-3 text-muted font-medium">Category</th>
                <th className="text-center py-3 px-3 text-muted font-medium">Best Model</th>
                <th className="text-right py-3 px-3 text-muted font-medium">R² Score</th>
                <th className="text-right py-3 px-3 text-muted font-medium">MAE (log)</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m) => {
                const model = models.find(mod => mod.category === m.category);
                const r2 = m.reg_r2 ?? model?.reg_metrics?.r2 ?? 0;
                const mae = m.reg_mae ?? model?.reg_metrics?.mae ?? 0;
                const regType = model?.model_type?.split('/')[1]?.replace('REG:', '').trim() || '—';
                return (
                  <tr key={m.category} className="border-b border-border/50 hover:bg-surface-2/30 transition-colors">
                    <td className="py-3 px-3 text-text font-medium">{m.category}</td>
                    <td className="py-3 px-3 text-center">
                      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-primary/10 text-primary">
                        {regType}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right">
                      <span className={cn(
                        'inline-block tabular-nums font-medium',
                        r2 >= 0.3 ? 'text-success' :
                        r2 >= 0.15 ? 'text-primary' :
                        r2 >= 0.05 ? 'text-warning' : 'text-danger'
                      )}>
                        {r2.toFixed(3)}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right">
                      <span className={cn(
                        'inline-block tabular-nums font-medium',
                        mae <= 1.0 ? 'text-success' :
                        mae <= 1.3 ? 'text-primary' :
                        mae <= 1.6 ? 'text-warning' : 'text-danger'
                      )}>
                        {mae.toFixed(3)}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Architecture note */}
      <div className="bento-card">
        <h3 className="text-sm font-medium text-text mb-3">Architecture</h3>
        <div className="grid sm:grid-cols-2 gap-4 text-xs text-muted leading-relaxed">
          <div>
            <p className="text-text font-medium mb-1">Model Selection</p>
            <p>For each category, 6 candidates are trained (XGBoost small/med/large, LightGBM, RandomForest, ExtraTrees) and the best is kept by ROC-AUC (clf) or R² (reg). Zero-importance features are pruned before final training.</p>
          </div>
          <div>
            <p className="text-text font-medium mb-1">Two-Stage Pipeline</p>
            <p>Stage 1: Classification predicts BSR entry probability. Stage 2: If probability &gt; 50%, regression predicts log-transformed BSR rank. Features: PCA embeddings + image quality + TF-IDF text + price. 80/20 train/test split.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricBadge({ value }: { value: number }) {
  return (
    <span className={cn(
      'inline-block tabular-nums font-medium',
      value >= 0.95 ? 'text-success' :
      value >= 0.9 ? 'text-emerald-400' :
      value >= 0.85 ? 'text-primary' :
      value >= 0.8 ? 'text-warning' : 'text-danger'
    )}>
      {value.toFixed(3)}
    </span>
  );
}
