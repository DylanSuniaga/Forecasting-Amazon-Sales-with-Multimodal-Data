'use client';

import { CheckCircle, AlertTriangle } from 'lucide-react';
import type { Signal } from '@/types';
import { cn } from '@/lib/utils';

interface SignalListProps {
  signals: Signal[];
  type: 'strength' | 'risk';
  className?: string;
}

export function SignalList({ signals, type, className }: SignalListProps) {
  const Icon = type === 'strength' ? CheckCircle : AlertTriangle;
  const iconColor = type === 'strength' ? 'text-success' : 'text-warning';
  const title = type === 'strength' ? 'Strengths' : 'Risks';

  if (signals.length === 0) {
    return null;
  }

  return (
    <div className={cn('space-y-3', className)}>
      <h4 className="text-sm font-medium text-muted uppercase tracking-wider">
        {title}
      </h4>
      <ul className="space-y-2">
        {signals.map((signal, idx) => (
          <li key={idx} className="signal-item">
            <Icon className={cn('signal-icon', iconColor)} />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-text">{signal.label}</p>
              <p className="text-xs text-muted">{signal.description}</p>
            </div>
            <div className="flex-shrink-0">
              <span 
                className={cn(
                  'text-xs font-medium px-2 py-0.5 rounded',
                  type === 'strength' ? 'bg-success/10 text-success' : 'bg-warning/10 text-warning'
                )}
              >
                {(signal.impact * 100).toFixed(0)}%
              </span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

interface SignalPanelProps {
  strengths: Signal[];
  risks: Signal[];
  categoryNotes?: string;
  defaultCollapsed?: boolean;
}

export function SignalPanel({ 
  strengths, 
  risks, 
  categoryNotes,
  defaultCollapsed = true 
}: SignalPanelProps) {
  return (
    <div className="bento-card">
      <details open={!defaultCollapsed}>
        <summary className="collapsible-header cursor-pointer list-none">
          <span className="text-lg font-semibold">Signal Decomposition</span>
          <span className="text-xs text-muted ml-2">
            ({strengths.length} strengths, {risks.length} risks)
          </span>
        </summary>
        
        <div className="mt-4 space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <SignalList signals={strengths} type="strength" />
            <SignalList signals={risks} type="risk" />
          </div>
          
          {categoryNotes && (
            <div className="pt-4 border-t border-border">
              <p className="text-sm text-muted">
                <span className="font-medium text-text">Category Note:</span>{' '}
                {categoryNotes}
              </p>
            </div>
          )}
        </div>
      </details>
    </div>
  );
}
