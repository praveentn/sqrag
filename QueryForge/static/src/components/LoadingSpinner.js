// static/src/components/LoadingSpinner.js
import React from 'react';
import { Loader2 } from 'lucide-react';

const LoadingSpinner = ({ 
  size = 'md', 
  message = 'Loading...', 
  className = '',
  showMessage = true,
  color = 'blue'
}) => {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6', 
    lg: 'h-8 w-8',
    xl: 'h-12 w-12'
  };

  const colorClasses = {
    blue: 'text-blue-500',
    gray: 'text-gray-500',
    green: 'text-green-500',
    red: 'text-red-500',
    white: 'text-white'
  };

  const messageSize = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
    xl: 'text-lg'
  };

  return (
    <div className={`flex flex-col items-center justify-center space-y-2 ${className}`}>
      <Loader2 
        className={`animate-spin ${sizeClasses[size]} ${colorClasses[color]}`}
      />
      {showMessage && message && (
        <p className={`${messageSize[size]} text-gray-600 font-medium`}>
          {message}
        </p>
      )}
    </div>
  );
};

// Inline spinner for buttons and small spaces
export const InlineSpinner = ({ 
  size = 'sm', 
  className = '',
  color = 'white'
}) => {
  const sizeClasses = {
    xs: 'h-3 w-3',
    sm: 'h-4 w-4',
    md: 'h-5 w-5'
  };

  const colorClasses = {
    blue: 'text-blue-500',
    gray: 'text-gray-500',
    white: 'text-white',
    current: 'text-current'
  };

  return (
    <Loader2 
      className={`animate-spin ${sizeClasses[size]} ${colorClasses[color]} ${className}`}
    />
  );
};

// Full page loading overlay
export const LoadingOverlay = ({ 
  message = 'Loading...', 
  isVisible = true,
  backdrop = true 
}) => {
  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {backdrop && (
        <div className="absolute inset-0 bg-black bg-opacity-25 backdrop-blur-sm" />
      )}
      <div className="relative bg-white rounded-lg shadow-lg p-8 max-w-sm mx-4">
        <LoadingSpinner 
          size="lg" 
          message={message}
          className="py-4"
        />
      </div>
    </div>
  );
};

// Skeleton loader for content placeholders
export const SkeletonLoader = ({ 
  lines = 3, 
  className = '',
  avatar = false 
}) => {
  return (
    <div className={`animate-pulse ${className}`}>
      {avatar && (
        <div className="flex items-center space-x-4 mb-4">
          <div className="rounded-full bg-gray-300 h-10 w-10"></div>
          <div className="space-y-2 flex-1">
            <div className="h-4 bg-gray-300 rounded w-3/4"></div>
            <div className="h-3 bg-gray-300 rounded w-1/2"></div>
          </div>
        </div>
      )}
      <div className="space-y-3">
        {Array.from({ length: lines }).map((_, index) => (
          <div key={index} className="space-y-2">
            <div className="h-4 bg-gray-300 rounded"></div>
            {index === lines - 1 && (
              <div className="h-4 bg-gray-300 rounded w-2/3"></div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

// Button with loading state
export const LoadingButton = ({ 
  loading = false, 
  children, 
  className = '',
  disabled = false,
  loadingText = 'Loading...',
  ...props 
}) => {
  return (
    <button
      className={`btn ${className} ${loading ? 'cursor-not-allowed' : ''}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <div className="flex items-center space-x-2">
          <InlineSpinner size="sm" color="current" />
          <span>{loadingText}</span>
        </div>
      ) : (
        children
      )}
    </button>
  );
};

// New: a simple progress notification bar
export const ProgressNotification = ({
  progress = 0,            // number from 0 to 100
  message = 'Processing…', // optional label
  className = ''           // extra tailwind classes
}) => (
  <div className={`flex flex-col space-y-2 ${className}`}>
    <p className="text-sm text-gray-700 font-medium">{message}</p>
    <div className="w-full bg-gray-200 rounded-full h-2">
      <div
        className="bg-blue-500 h-2 rounded-full"
        style={{ width: `${progress}%` }}
      />
    </div>
  </div>
);


export default LoadingSpinner;