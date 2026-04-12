'use client';

import { useState, useCallback } from 'react';
import { ArrowLeft, ArrowRight, Loader2, Send, RotateCcw, Check } from 'lucide-react';
import { StepCategory } from './StepCategory';
import { StepImages } from './StepImages';
import { StepText } from './StepText';
import { StepResults } from './StepResults';
import type { WizardStep, CategoryInfo, ProductFormData, EvaluationResponse } from '@/types';
import { cn } from '@/lib/utils';

const STEPS: { id: WizardStep; label: string; number: number }[] = [
  { id: 'category', label: 'Category', number: 1 },
  { id: 'images', label: 'Images', number: 2 },
  { id: 'text', label: 'Text', number: 3 },
  { id: 'results', label: 'Results', number: 4 },
];

interface WizardShellProps {
  categories: CategoryInfo[];
  onEvaluate: (data: ProductFormData) => Promise<void>;
  isLoading: boolean;
  evaluation: EvaluationResponse | null;
  error: string | null;
  // Suggestion pre-fill support
  initialFormData?: Partial<ProductFormData>;
  suggestionImageUrl?: string;
  onClearSuggestion?: () => void;
}

export function WizardShell({
  categories,
  onEvaluate,
  isLoading,
  evaluation,
  error,
  initialFormData,
  suggestionImageUrl,
  onClearSuggestion,
}: WizardShellProps) {
  const [currentStep, setCurrentStep] = useState<WizardStep>('category');
  const [formData, setFormData] = useState<ProductFormData>({
    title: initialFormData?.title || '',
    description: initialFormData?.description || '',
    category: initialFormData?.category || '',
    subcategory: initialFormData?.subcategory || '',
    images: initialFormData?.images || [],
    imageUrl: initialFormData?.imageUrl,
    price: initialFormData?.price,
    cnnEmbedding: initialFormData?.cnnEmbedding,
    clipEmbedding: initialFormData?.clipEmbedding,
    cnnPcaFeatures: initialFormData?.cnnPcaFeatures,
    clipPcaFeatures: initialFormData?.clipPcaFeatures,
  });

  // Sync when initialFormData changes (from suggestion selection)
  const [lastInitial, setLastInitial] = useState(initialFormData);
  if (initialFormData !== lastInitial) {
    setLastInitial(initialFormData);
    if (initialFormData) {
      setFormData({
        title: initialFormData.title || '',
        description: initialFormData.description || '',
        category: initialFormData.category || '',
        subcategory: initialFormData.subcategory || '',
        images: initialFormData.images || [],
        imageUrl: initialFormData.imageUrl,
        price: initialFormData.price,
        cnnEmbedding: initialFormData.cnnEmbedding,
        clipEmbedding: initialFormData.clipEmbedding,
        cnnPcaFeatures: initialFormData.cnnPcaFeatures,
        clipPcaFeatures: initialFormData.clipPcaFeatures,
      });
    }
  }

  const currentStepIndex = STEPS.findIndex(s => s.id === currentStep);

  // Validation per step
  const isStepValid = useCallback((step: WizardStep): boolean => {
    switch (step) {
      case 'category':
        return !!formData.category.trim();
      case 'images':
        return formData.images.length > 0 || !!suggestionImageUrl || !!formData.imageUrl;
      case 'text':
        return !!formData.title.trim();
      case 'results':
        return true;
      default:
        return false;
    }
  }, [formData, suggestionImageUrl]);

  const canGoNext = isStepValid(currentStep);

  const handleNext = async () => {
    if (currentStep === 'text') {
      // Submit the evaluation
      await onEvaluate(formData);
      setCurrentStep('results');
    } else {
      const nextIdx = currentStepIndex + 1;
      if (nextIdx < STEPS.length) {
        setCurrentStep(STEPS[nextIdx].id);
      }
    }
  };

  const handleBack = () => {
    const prevIdx = currentStepIndex - 1;
    if (prevIdx >= 0) {
      setCurrentStep(STEPS[prevIdx].id);
    }
  };

  const handleNewEvaluation = () => {
    setFormData({
      title: '',
      description: '',
      category: '',
      subcategory: '',
      images: [],
    });
    setCurrentStep('category');
    if (onClearSuggestion) onClearSuggestion();
  };

  const handleStepClick = (step: WizardStep) => {
    const targetIdx = STEPS.findIndex(s => s.id === step);
    // Can only go to completed steps or current step
    // For results, only if we have an evaluation
    if (step === 'results' && !evaluation) return;
    if (targetIdx <= currentStepIndex) {
      setCurrentStep(step);
    }
    // Allow jumping forward only if all previous steps are valid
    if (targetIdx > currentStepIndex) {
      let allValid = true;
      for (let i = 0; i < targetIdx; i++) {
        if (!isStepValid(STEPS[i].id)) {
          allValid = false;
          break;
        }
      }
      if (allValid) {
        setCurrentStep(step);
      }
    }
  };

  return (
    <div className="space-y-8">
      {/* Progress Bar */}
      <div className="max-w-3xl mx-auto">
        <div className="flex items-center justify-between relative">
          {/* Connecting line */}
          <div className="absolute top-5 left-0 right-0 h-0.5 bg-border" />
          <div
            className="absolute top-5 left-0 h-0.5 bg-primary transition-all duration-500"
            style={{
              width: currentStep === 'results'
                ? '100%'
                : `${(currentStepIndex / (STEPS.length - 1)) * 100}%`
            }}
          />

          {STEPS.map((step, idx) => {
            const isActive = step.id === currentStep;
            const isCompleted = idx < currentStepIndex || (currentStep === 'results' && idx < STEPS.length);
            const isClickable = idx <= currentStepIndex || (step.id === 'results' && !!evaluation);

            return (
              <button
                key={step.id}
                onClick={() => handleStepClick(step.id)}
                disabled={!isClickable}
                className={cn(
                  'relative z-10 flex flex-col items-center gap-2 group',
                  isClickable ? 'cursor-pointer' : 'cursor-default'
                )}
              >
                <div
                  className={cn(
                    'w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-300 border-2',
                    isActive
                      ? 'bg-primary text-white border-primary scale-110 shadow-lg shadow-primary/25'
                      : isCompleted
                        ? 'bg-primary/20 text-primary border-primary'
                        : 'bg-surface-2 text-muted border-border'
                  )}
                >
                  {isCompleted && !isActive ? (
                    <Check className="w-4 h-4" />
                  ) : (
                    step.number
                  )}
                </div>
                <span
                  className={cn(
                    'text-xs font-medium transition-colors',
                    isActive ? 'text-primary' : isCompleted ? 'text-text' : 'text-muted'
                  )}
                >
                  {step.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Step Content */}
      <div className="min-h-[400px]">
        {/* Error display */}
        {error && (
          <div className="max-w-2xl mx-auto mb-6 bento-card border-danger/50 bg-danger/10">
            <p className="text-danger text-sm">{error}</p>
          </div>
        )}

        {/* Loading state */}
        {isLoading && currentStep === 'results' && (
          <div className="max-w-2xl mx-auto text-center py-16">
            <Loader2 className="w-10 h-10 text-primary animate-spin mx-auto mb-4" />
            <p className="text-text font-medium">Analyzing your product...</p>
            <p className="text-sm text-muted mt-2">
              Extracting image features, running model inference, and generating signals
            </p>
          </div>
        )}

        {/* Step: Category */}
        {currentStep === 'category' && (
          <StepCategory
            category={formData.category}
            subcategory={formData.subcategory}
            onCategoryChange={(cat) => setFormData(prev => ({ ...prev, category: cat }))}
            onSubcategoryChange={(sub) => setFormData(prev => ({ ...prev, subcategory: sub }))}
            categories={categories}
          />
        )}

        {/* Step: Images */}
        {currentStep === 'images' && (
          <StepImages
            images={formData.images}
            onImagesChange={(images) => setFormData(prev => ({ ...prev, images }))}
            suggestionImageUrl={suggestionImageUrl || formData.imageUrl}
            onClearSuggestion={onClearSuggestion}
          />
        )}

        {/* Step: Text */}
        {currentStep === 'text' && (
          <StepText
            title={formData.title}
            description={formData.description}
            price={formData.price}
            onTitleChange={(title) => setFormData(prev => ({ ...prev, title }))}
            onDescriptionChange={(description) => setFormData(prev => ({ ...prev, description }))}
            onPriceChange={(price) => setFormData(prev => ({ ...prev, price }))}
          />
        )}

        {/* Step: Results */}
        {currentStep === 'results' && !isLoading && evaluation && (
          <StepResults
            evaluation={evaluation}
            category={formData.category}
          />
        )}

        {/* Empty results state (shouldn't normally happen) */}
        {currentStep === 'results' && !isLoading && !evaluation && !error && (
          <div className="max-w-2xl mx-auto text-center py-16">
            <p className="text-muted">No results yet. Please complete the previous steps.</p>
          </div>
        )}
      </div>

      {/* Bottom Navigation */}
      <div className="flex items-center justify-between max-w-3xl mx-auto pt-4 border-t border-border">
        {/* Back Button */}
        <div>
          {currentStepIndex > 0 && currentStep !== 'results' && (
            <button
              onClick={handleBack}
              className="btn-secondary flex items-center gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </button>
          )}
          {currentStep === 'results' && (
            <button
              onClick={handleBack}
              className="btn-secondary flex items-center gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Edit Inputs
            </button>
          )}
        </div>

        {/* Next / Submit / New Evaluation Button */}
        <div>
          {currentStep === 'results' ? (
            <button
              onClick={handleNewEvaluation}
              className="btn-primary flex items-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              New Evaluation
            </button>
          ) : currentStep === 'text' ? (
            <button
              onClick={handleNext}
              disabled={!canGoNext || isLoading}
              className={cn(
                'btn-primary flex items-center gap-2',
                (!canGoNext || isLoading) && 'opacity-50 cursor-not-allowed'
              )}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  Evaluate Product
                </>
              )}
            </button>
          ) : (
            <button
              onClick={handleNext}
              disabled={!canGoNext}
              className={cn(
                'btn-primary flex items-center gap-2',
                !canGoNext && 'opacity-50 cursor-not-allowed'
              )}
            >
              Next
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
