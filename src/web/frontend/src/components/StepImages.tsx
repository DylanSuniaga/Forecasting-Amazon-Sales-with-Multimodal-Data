'use client';

import { ExternalLink, Sparkles, Image as ImageIcon } from 'lucide-react';
import { ImageUpload } from './ImageUpload';
import { cn } from '@/lib/utils';

interface StepImagesProps {
  images: File[];
  onImagesChange: (images: File[]) => void;
  suggestionImageUrl?: string;
  onClearSuggestion?: () => void;
}

export function StepImages({
  images,
  onImagesChange,
  suggestionImageUrl,
  onClearSuggestion,
}: StepImagesProps) {
  const hasSuggestionImage = !!suggestionImageUrl;

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Heading */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-text mb-2">Product Images</h2>
        <p className="text-muted">
          Upload your main product image for visual analysis
        </p>
      </div>

      {/* Suggestion image display */}
      {hasSuggestionImage ? (
        <div className="space-y-4">
          <div className="p-6 bg-surface border border-primary/30 rounded-xl">
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium text-primary">Using example product image</span>
            </div>
            <div className="flex justify-center">
              <div className="relative aspect-square w-48 rounded-lg overflow-hidden border border-border bg-surface-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={suggestionImageUrl}
                  alt="Product"
                  className="w-full h-full object-contain"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              </div>
            </div>
            <div className="flex items-center justify-center gap-4 mt-4">
              <a
                href={suggestionImageUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-primary hover:underline flex items-center gap-1"
              >
                <ExternalLink className="w-3 h-3" />
                View full image
              </a>
              {onClearSuggestion && (
                <>
                  <span className="text-xs text-muted">|</span>
                  <button
                    type="button"
                    onClick={onClearSuggestion}
                    className="text-xs text-muted hover:text-danger transition-colors"
                  >
                    Clear and upload your own
                  </button>
                </>
              )}
            </div>
          </div>
          <p className="text-xs text-muted text-center">
            Image from Amazon product listing. The model will analyze this image for quality and composition signals.
          </p>
        </div>
      ) : (
        /* Normal image upload */
        <div className="space-y-4">
          <p className="text-sm text-muted text-center">
            Upload your main product image (JPEG, PNG, or WebP, max 10MB)
          </p>
          <ImageUpload
            images={images}
            onChange={onImagesChange}
            maxImages={1}
          />
        </div>
      )}

      {/* Image tips */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <h3 className="text-sm font-medium text-text mb-4 flex items-center gap-2">
          <ImageIcon className="w-4 h-4 text-primary" />
          Image Tips for Better Results
        </h3>
        <div className="grid sm:grid-cols-2 gap-3">
          {[
            { label: 'White background', desc: 'Amazon requires pure white (RGB 255,255,255) backgrounds' },
            { label: 'High resolution', desc: 'At least 1000x1000 pixels for zoom functionality' },
            { label: 'Product fills frame', desc: 'Product should occupy 85%+ of the image area' },
            { label: 'Sharp and well-lit', desc: 'Clear focus, even lighting, no shadows' },
          ].map((tip) => (
            <div key={tip.label} className="flex items-start gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
              <div>
                <p className="text-sm text-text">{tip.label}</p>
                <p className="text-xs text-muted">{tip.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
