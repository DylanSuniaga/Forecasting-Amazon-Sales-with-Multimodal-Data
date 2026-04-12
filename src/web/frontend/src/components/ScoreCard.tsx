'use client';

import { cn, getScoreColor, formatScore, formatPercent } from '@/lib/utils';

interface ScoreCardProps {
  title: string;
  value: number;
  subtitle?: string;
  type?: 'score' | 'probability' | 'percent';
  className?: string;
}

export function ScoreCard({ 
  title, 
  value, 
  subtitle,
  type = 'score',
  className 
}: ScoreCardProps) {
  const displayValue = type === 'probability' 
    ? formatPercent(value)
    : type === 'percent'
    ? `${formatScore(value)}%`
    : formatScore(value);
  
  const colorClass = type === 'probability'
    ? getScoreColor(value * 100)
    : getScoreColor(value);

  return (
    <div className={cn('bento-card', className)}>
      <p className="text-sm text-muted mb-2">{title}</p>
      <p className={cn('score-large', colorClass)}>
        {displayValue}
      </p>
      {subtitle && (
        <p className="text-sm text-muted mt-2">{subtitle}</p>
      )}
    </div>
  );
}

interface RankBandCardProps {
  title: string;
  band: string;
  percentile: number;
  estimatedRank?: number | null;
  className?: string;
}

export function RankBandCard({ 
  title, 
  band, 
  percentile,
  estimatedRank,
  className 
}: RankBandCardProps) {
  return (
    <div className={cn('bento-card', className)}>
      <p className="text-sm text-muted mb-2">{title}</p>
      <p className="text-2xl font-bold text-text mb-3">{band}</p>
      {estimatedRank && (
        <p className="text-lg font-semibold text-primary mb-2">
          Estimated Rank: #{Math.round(estimatedRank).toLocaleString()}
        </p>
      )}
      <div className="rank-band">
        <div 
          className="rank-band-fill bg-primary"
          style={{ width: `${percentile}%` }}
        />
      </div>
      <p className="text-xs text-muted mt-2">
        {percentile.toFixed(0)}th percentile in category
      </p>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export function MetricCard({
  label,
  value,
  subtext,
  trend,
  className
}: MetricCardProps) {
  const trendColors = {
    up: 'text-success',
    down: 'text-danger',
    neutral: 'text-muted'
  };

  return (
    <div className={cn('p-4 bg-surface-2 rounded-lg', className)}>
      <p className="text-xs text-muted uppercase tracking-wider mb-1">{label}</p>
      <p className="text-xl font-semibold text-text">{value}</p>
      {subtext && (
        <p className={cn('text-xs mt-1', trend ? trendColors[trend] : 'text-muted')}>
          {subtext}
        </p>
      )}
    </div>
  );
}
