'use client';

import { useState, useEffect } from 'react';
import { Box, Code, Server, Check, X, Copy, CheckCheck } from 'lucide-react';
import type { ModelInfo } from '@/types';
import { cn } from '@/lib/utils';

interface ModelOverviewProps {
  models: ModelInfo[];
  embeddingLoaded: boolean;
}

export function ModelOverview({ models, embeddingLoaded }: ModelOverviewProps) {
  const [copiedRequest, setCopiedRequest] = useState(false);
  const [copiedResponse, setCopiedResponse] = useState(false);

  const copyToClipboard = async (text: string, type: 'request' | 'response') => {
    await navigator.clipboard.writeText(text);
    if (type === 'request') {
      setCopiedRequest(true);
      setTimeout(() => setCopiedRequest(false), 2000);
    } else {
      setCopiedResponse(true);
      setTimeout(() => setCopiedResponse(false), 2000);
    }
  };

  const requestExample = `POST /api/evaluate
Content-Type: multipart/form-data

{
  "title": "Premium Stainless Steel Water Bottle 32oz",
  "description": "Keep drinks cold 24hrs or hot 12hrs...",
  "category": "Home & Kitchen",
  "subcategory": "Kitchen & Dining",
  "images": [<file1>, <file2>]
}`;

  const responseExample = `{
  "launch_viability_score": 72.5,
  "bsr_entry_probability": 0.85,
  "expected_rank_band": "Top 25%",
  "competitive_intensity": 68.0,
  "confidence": 0.82,
  "strengths": [...],
  "risks": [...],
  "category_notes": "...",
  "image_quality": {...},
  "product_hash": "abc123def456",
  "model_version": "1.0.0"
}`;

  return (
    <div className="space-y-6">
      {/* Architecture Diagram */}
      <div className="bento-card">
        <h3 className="text-lg font-semibold text-text mb-4">System Architecture</h3>
        
        <div className="bg-surface-2 rounded-lg p-6">
          <div className="flex flex-col lg:flex-row items-center gap-4 lg:gap-8">
            {/* Input */}
            <div className="flex-shrink-0 text-center">
              <div className="w-24 h-24 bg-surface rounded-lg border border-border flex items-center justify-center mb-2">
                <Box className="w-8 h-8 text-primary" />
              </div>
              <p className="text-sm text-muted">Input</p>
              <p className="text-xs text-muted">Title, Images, Category</p>
            </div>
            
            {/* Arrow */}
            <div className="text-muted">→</div>
            
            {/* Feature Extraction */}
            <div className="flex-1 text-center">
              <div className="bg-surface rounded-lg border border-border p-4 mb-2">
                <p className="text-sm font-medium text-text mb-2">Feature Extraction</p>
                <div className="flex flex-wrap justify-center gap-2 text-xs">
                  <span className="px-2 py-1 bg-surface-2 rounded">CNN (EfficientNet)</span>
                  <span className="px-2 py-1 bg-surface-2 rounded">CLIP</span>
                  <span className="px-2 py-1 bg-surface-2 rounded">Quality Metrics</span>
                  <span className="px-2 py-1 bg-surface-2 rounded">Composition</span>
                </div>
              </div>
              <p className="text-xs text-muted">PCA → 128 dim</p>
            </div>
            
            {/* Arrow */}
            <div className="text-muted">→</div>
            
            {/* Model */}
            <div className="flex-shrink-0 text-center">
              <div className="w-24 h-24 bg-surface rounded-lg border border-border flex items-center justify-center mb-2">
                <Server className="w-8 h-8 text-primary" />
              </div>
              <p className="text-sm text-muted">Category Model</p>
              <p className="text-xs text-muted">XGBoost Classifier</p>
            </div>
            
            {/* Arrow */}
            <div className="text-muted">→</div>
            
            {/* Output */}
            <div className="flex-shrink-0 text-center">
              <div className="w-24 h-24 bg-surface rounded-lg border border-primary flex items-center justify-center mb-2">
                <Code className="w-8 h-8 text-primary" />
              </div>
              <p className="text-sm text-muted">Output</p>
              <p className="text-xs text-muted">Scores + Signals</p>
            </div>
          </div>
        </div>
        
        <p className="text-xs text-muted mt-4">
          Note: NLP features are placeholders. Current model focuses on image analysis.
        </p>
      </div>

      {/* API Examples */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Request */}
        <div className="bento-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text">Request</h3>
            <button
              onClick={() => copyToClipboard(requestExample, 'request')}
              className="btn-secondary text-xs flex items-center gap-1"
            >
              {copiedRequest ? <CheckCheck className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
              {copiedRequest ? 'Copied' : 'Copy'}
            </button>
          </div>
          <pre className="code-block text-xs overflow-x-auto">
            {requestExample}
          </pre>
        </div>

        {/* Response */}
        <div className="bento-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text">Response</h3>
            <button
              onClick={() => copyToClipboard(responseExample, 'response')}
              className="btn-secondary text-xs flex items-center gap-1"
            >
              {copiedResponse ? <CheckCheck className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
              {copiedResponse ? 'Copied' : 'Copy'}
            </button>
          </div>
          <pre className="code-block text-xs overflow-x-auto">
            {responseExample}
          </pre>
        </div>
      </div>

      {/* Model Registry */}
      <div className="bento-card">
        <h3 className="text-lg font-semibold text-text mb-4">Model Registry</h3>
        
        {/* Embedding status */}
        <div className="flex items-center gap-3 p-3 bg-surface-2 rounded-lg mb-4">
          {embeddingLoaded ? (
            <Check className="w-5 h-5 text-success" />
          ) : (
            <X className="w-5 h-5 text-warning" />
          )}
          <div>
            <p className="text-sm font-medium text-text">Embedding Models</p>
            <p className="text-xs text-muted">
              {embeddingLoaded ? 'CNN + CLIP loaded' : 'Running in placeholder mode'}
            </p>
          </div>
        </div>

        {/* Category models */}
        <div className="space-y-2">
          <p className="text-sm text-muted mb-2">Category Classification Models</p>
          <div className="grid sm:grid-cols-2 gap-2">
            {models.map((model) => (
              <div 
                key={model.category}
                className="flex items-center justify-between p-2 bg-surface-2 rounded"
              >
                <div className="flex items-center gap-2">
                  {model.loaded ? (
                    <Check className="w-4 h-4 text-success" />
                  ) : (
                    <X className="w-4 h-4 text-muted" />
                  )}
                  <span className="text-sm text-text truncate">{model.category}</span>
                </div>
                <span className="text-xs text-muted">
                  {(model.metrics.roc_auc * 100).toFixed(0)}% AUC
                </span>
              </div>
            ))}
          </div>
        </div>

        <p className="text-xs text-muted mt-4">
          Models not loaded will use placeholder predictions based on category baseline metrics.
        </p>
      </div>
    </div>
  );
}
