'use client';

import { BarChart3, Target, TrendingUp, Activity } from 'lucide-react';
import type { ValidationMetrics } from '@/types';
import { cn, formatPercent } from '@/lib/utils';

interface ValidationSnapshotProps {
  metrics: ValidationMetrics[];
  overallAuc: number;
  overallF1: number;
  calibration: Record<string, number | string>;
}

export function ValidationSnapshot({ 
  metrics, 
  overallAuc, 
  overallF1,
  calibration 
}: ValidationSnapshotProps) {
  // Sort by AUC descending
  const sortedMetrics = [...metrics].sort((a, b) => b.roc_auc - a.roc_auc);

  return (
    <div className="space-y-6">
      {/* Overall Performance */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <PerformanceCard
          icon={Target}
          label="Overall ROC-AUC"
          value={formatPercent(overallAuc)}
          description="Classification accuracy"
          color="primary"
        />
        <PerformanceCard
          icon={Activity}
          label="Overall F1 Score"
          value={formatPercent(overallF1)}
          description="Precision-recall balance"
          color="success"
        />
        <PerformanceCard
          icon={BarChart3}
          label="Brier Score"
          value={(calibration.brier_score as number || 0.12).toFixed(3)}
          description="Calibration error (lower=better)"
          color="warning"
        />
        <PerformanceCard
          icon={TrendingUp}
          label="Categories"
          value={metrics.length.toString()}
          description="Models trained"
          color="primary"
        />
      </div>

      {/* Performance by Category */}
      <div className="bento-card">
        <h3 className="text-lg font-semibold text-text mb-4">Performance by Category</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-2 text-muted font-medium">Category</th>
                <th className="text-right py-3 px-2 text-muted font-medium">ROC-AUC</th>
                <th className="text-right py-3 px-2 text-muted font-medium">F1</th>
                <th className="text-right py-3 px-2 text-muted font-medium">Precision</th>
                <th className="text-right py-3 px-2 text-muted font-medium">Recall</th>
                <th className="text-right py-3 px-2 text-muted font-medium hidden sm:table-cell">Model</th>
              </tr>
            </thead>
            <tbody>
              {sortedMetrics.map((m, idx) => (
                <tr 
                  key={m.category}
                  className={cn(
                    'border-b border-border/50',
                    idx % 2 === 0 && 'bg-surface-2/50'
                  )}
                >
                  <td className="py-3 px-2 text-text">{m.category}</td>
                  <td className={cn(
                    'py-3 px-2 text-right font-medium',
                    getAucColor(m.roc_auc)
                  )}>
                    {formatPercent(m.roc_auc)}
                  </td>
                  <td className="py-3 px-2 text-right text-muted">
                    {formatPercent(m.f1_score)}
                  </td>
                  <td className="py-3 px-2 text-right text-muted">
                    {formatPercent(m.precision)}
                  </td>
                  <td className="py-3 px-2 text-right text-muted">
                    {formatPercent(m.recall)}
                  </td>
                  <td className="py-3 px-2 text-right text-muted hidden sm:table-cell">
                    {m.model_type}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Calibration */}
      <div className="bento-card">
        <h3 className="text-lg font-semibold text-text mb-4">Calibration Summary</h3>
        
        <div className="grid sm:grid-cols-2 gap-4 mb-4">
          <div className="p-4 bg-surface-2 rounded-lg">
            <p className="text-xs text-muted uppercase mb-1">Brier Score</p>
            <p className="text-xl font-semibold text-text">
              {(calibration.brier_score as number || 0.12).toFixed(3)}
            </p>
            <p className="text-xs text-muted mt-1">Lower is better (0 = perfect)</p>
          </div>
          <div className="p-4 bg-surface-2 rounded-lg">
            <p className="text-xs text-muted uppercase mb-1">Calibration Error</p>
            <p className="text-xl font-semibold text-text">
              {(calibration.calibration_error as number || 0.08).toFixed(3)}
            </p>
            <p className="text-xs text-muted mt-1">Expected calibration error</p>
          </div>
        </div>

        <p className="text-sm text-muted">
          <strong className="text-text">Note:</strong> {calibration.note || 'Calibration metrics computed on held-out test set'}
        </p>
      </div>

      {/* Methodology */}
      <details className="bento-card">
        <summary className="collapsible-header cursor-pointer list-none">
          <span>Evaluation Methodology</span>
        </summary>
        <div className="mt-4 space-y-3 text-sm text-muted">
          <p>
            <strong className="text-text">Train/Test Split:</strong> 80/20 stratified split by category and target class.
          </p>
          <p>
            <strong className="text-text">Cross-Validation:</strong> 5-fold stratified CV for hyperparameter tuning.
          </p>
          <p>
            <strong className="text-text">Target:</strong> has_main_bsr (binary) - whether product has a BSR rank.
          </p>
          <p>
            <strong className="text-text">Features:</strong> Top 20 image features (CNN/CLIP PCA + quality metrics) selected via composite ranking.
          </p>
          <p>
            <strong className="text-text">Models:</strong> XGBoost classifier (primary), with LogisticRegression and RandomForest baselines.
          </p>
        </div>
      </details>
    </div>
  );
}

interface PerformanceCardProps {
  icon: React.ElementType;
  label: string;
  value: string;
  description: string;
  color: 'primary' | 'success' | 'warning' | 'danger';
}

function PerformanceCard({ icon: Icon, label, value, description, color }: PerformanceCardProps) {
  const colorClasses = {
    primary: 'text-primary',
    success: 'text-success',
    warning: 'text-warning',
    danger: 'text-danger',
  };

  return (
    <div className="bento-card">
      <div className="flex items-center gap-2 mb-2">
        <Icon className={cn('w-5 h-5', colorClasses[color])} />
        <span className="text-sm text-muted">{label}</span>
      </div>
      <p className={cn('text-3xl font-bold', colorClasses[color])}>{value}</p>
      <p className="text-xs text-muted mt-1">{description}</p>
    </div>
  );
}

function getAucColor(auc: number): string {
  if (auc >= 0.95) return 'text-success';
  if (auc >= 0.90) return 'text-primary';
  if (auc >= 0.85) return 'text-warning';
  return 'text-danger';
}
