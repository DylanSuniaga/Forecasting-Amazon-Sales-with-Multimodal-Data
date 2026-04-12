'use client';

import { useState, useCallback, useRef } from 'react';
import { Upload, X, Image as ImageIcon } from 'lucide-react';
import { cn, formatFileSize, validateImageFile } from '@/lib/utils';

interface ImageUploadProps {
  images: File[];
  onChange: (images: File[]) => void;
  maxImages?: number;
  className?: string;
}

export function ImageUpload({ 
  images, 
  onChange, 
  maxImages = 5,
  className 
}: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback((files: FileList | null) => {
    if (!files) return;
    
    setError(null);
    const newFiles: File[] = [];
    
    for (let i = 0; i < files.length && images.length + newFiles.length < maxImages; i++) {
      const file = files[i];
      const validation = validateImageFile(file);
      
      if (validation.valid) {
        newFiles.push(file);
      } else {
        setError(validation.error || 'Invalid file');
      }
    }
    
    if (newFiles.length > 0) {
      onChange([...images, ...newFiles]);
    }
  }, [images, maxImages, onChange]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
  }, []);

  const removeImage = useCallback((index: number) => {
    onChange(images.filter((_, i) => i !== index));
  }, [images, onChange]);

  return (
    <div className={cn('space-y-4', className)}>
      {/* Upload Zone */}
      <div
        className={cn(
          'upload-zone',
          dragActive && 'upload-zone-active'
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        
        <Upload className="w-8 h-8 text-muted mx-auto mb-3" />
        <p className="text-text font-medium">
          Drop image here or click to upload
        </p>
        <p className="text-sm text-muted mt-1">
          JPEG, PNG, or WebP • Max 10MB
        </p>
        {images.length > 0 && (
          <p className="text-xs text-primary mt-2">
            Image uploaded
          </p>
        )}
      </div>

      {/* Error message */}
      {error && (
        <p className="text-sm text-danger">{error}</p>
      )}

      {/* Image Previews */}
      {images.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
          {images.map((file, index) => (
            <ImagePreview
              key={`${file.name}-${index}`}
              file={file}
              isHero={index === 0}
              onRemove={() => removeImage(index)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface ImagePreviewProps {
  file: File;
  isHero?: boolean;
  onRemove: () => void;
}

function ImagePreview({ file, isHero, onRemove }: ImagePreviewProps) {
  const [preview, setPreview] = useState<string | null>(null);
  
  // Generate preview URL
  useState(() => {
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  });

  return (
    <div className="relative group">
      <div className={cn(
        'aspect-square rounded-lg overflow-hidden bg-surface-2 border',
        isHero ? 'border-primary' : 'border-border'
      )}>
        {preview ? (
          <img
            src={preview}
            alt={file.name}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <ImageIcon className="w-8 h-8 text-muted" />
          </div>
        )}
      </div>
      
      {/* Remove button */}
      <button
        onClick={onRemove}
        className="absolute -top-2 -right-2 w-6 h-6 bg-danger rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
      >
        <X className="w-4 h-4 text-white" />
      </button>
      
      {/* Hero badge */}
      {isHero && (
        <span className="absolute bottom-1 left-1 text-[10px] font-medium px-1.5 py-0.5 bg-primary text-white rounded">
          MAIN
        </span>
      )}
      
      {/* File info */}
      <p className="text-[10px] text-muted mt-1 truncate">
        {formatFileSize(file.size)}
      </p>
    </div>
  );
}
