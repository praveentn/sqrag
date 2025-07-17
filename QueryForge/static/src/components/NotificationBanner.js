// static/src/components/NotificationBanner.js
import React, { useEffect, useState } from 'react';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react';

const NotificationBanner = ({ 
  message, 
  type = 'info', 
  onClose, 
  autoClose = true,
  duration = 5000,
  position = 'top-right',
  className = ''
}) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    if (autoClose && duration > 0) {
      const timer = setTimeout(() => {
        handleClose();
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [autoClose, duration]);

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => {
      if (onClose) onClose();
    }, 300); // Wait for animation to complete
  };

  const getTypeConfig = () => {
    switch (type) {
      case 'success':
        return {
          bgColor: 'bg-green-50',
          borderColor: 'border-green-200',
          textColor: 'text-green-800',
          iconColor: 'text-green-400',
          Icon: CheckCircle
        };
      case 'error':
        return {
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200',
          textColor: 'text-red-800',
          iconColor: 'text-red-400',
          Icon: AlertCircle
        };
      case 'warning':
        return {
          bgColor: 'bg-yellow-50',
          borderColor: 'border-yellow-200',
          textColor: 'text-yellow-800',
          iconColor: 'text-yellow-400',
          Icon: AlertTriangle
        };
      default: // info
        return {
          bgColor: 'bg-blue-50',
          borderColor: 'border-blue-200',
          textColor: 'text-blue-800',
          iconColor: 'text-blue-400',
          Icon: Info
        };
    }
  };

  const getPositionClasses = () => {
    const baseClasses = 'fixed z-50';
    
    switch (position) {
      case 'top-left':
        return `${baseClasses} top-4 left-4`;
      case 'top-center':
        return `${baseClasses} top-4 left-1/2 transform -translate-x-1/2`;
      case 'top-right':
        return `${baseClasses} top-4 right-4`;
      case 'bottom-left':
        return `${baseClasses} bottom-4 left-4`;
      case 'bottom-center':
        return `${baseClasses} bottom-4 left-1/2 transform -translate-x-1/2`;
      case 'bottom-right':
        return `${baseClasses} bottom-4 right-4`;
      default:
        return `${baseClasses} top-4 right-4`;
    }
  };

  const config = getTypeConfig();
  const { bgColor, borderColor, textColor, iconColor, Icon } = config;

  if (!isVisible) return null;

  return (
    <div 
      className={`
        ${getPositionClasses()} 
        max-w-sm w-full
        transition-all duration-300 ease-in-out
        ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-2'}
        ${className}
      `}
    >
      <div className={`
        ${bgColor} ${borderColor} ${textColor}
        border rounded-lg shadow-lg p-4
      `}>
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <Icon className={`h-5 w-5 ${iconColor}`} />
          </div>
          <div className="ml-3 flex-1">
            <p className={`text-sm font-medium ${textColor}`}>
              {message}
            </p>
          </div>
          <div className="ml-4 flex-shrink-0 flex">
            <button
              className={`
                ${bgColor} rounded-md inline-flex ${textColor} 
                hover:${textColor} focus:outline-none focus:ring-2 
                focus:ring-offset-2 focus:ring-offset-${bgColor.split('-')[1]}-50 
                focus:ring-${iconColor.split('-')[1]}-600
              `}
              onClick={handleClose}
            >
              <span className="sr-only">Close</span>
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Toast container for multiple notifications
export const ToastContainer = ({ notifications = [], onRemove }) => {
  return (
    <div className="fixed top-4 right-4 z-50 space-y-2">
      {notifications.map((notification) => (
        <NotificationBanner
          key={notification.id}
          message={notification.message}
          type={notification.type}
          onClose={() => onRemove(notification.id)}
          autoClose={notification.autoClose !== false}
          duration={notification.duration || 5000}
          className="relative"
        />
      ))}
    </div>
  );
};

// Hook for managing multiple notifications
export const useNotifications = () => {
  const [notifications, setNotifications] = useState([]);

  const addNotification = (message, type = 'info', options = {}) => {
    const id = Date.now() + Math.random();
    const notification = {
      id,
      message,
      type,
      ...options
    };

    setNotifications(prev => [...prev, notification]);

    // Auto-remove if specified
    if (options.autoClose !== false) {
      setTimeout(() => {
        removeNotification(id);
      }, options.duration || 5000);
    }

    return id;
  };

  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const clearAll = () => {
    setNotifications([]);
  };

  return {
    notifications,
    addNotification,
    removeNotification,
    clearAll,
    // Convenience methods
    success: (message, options) => addNotification(message, 'success', options),
    error: (message, options) => addNotification(message, 'error', options),
    warning: (message, options) => addNotification(message, 'warning', options),
    info: (message, options) => addNotification(message, 'info', options)
  };
};

// Progress notification for long-running operations
export const ProgressNotification = ({ 
  message, 
  progress = 0, 
  onClose,
  type = 'info',
  showPercentage = true 
}) => {
  const config = {
    info: { 
      bgColor: 'bg-blue-50', 
      progressColor: 'bg-blue-500',
      textColor: 'text-blue-800'
    },
    success: { 
      bgColor: 'bg-green-50', 
      progressColor: 'bg-green-500',
      textColor: 'text-green-800'
    },
    warning: { 
      bgColor: 'bg-yellow-50', 
      progressColor: 'bg-yellow-500',
      textColor: 'text-yellow-800'
    }
  };

  const { bgColor, progressColor, textColor } = config[type] || config.info;

  return (
    <div className={`${bgColor} border border-gray-200 rounded-lg shadow-lg p-4 max-w-sm`}>
      <div className="flex items-center justify-between mb-2">
        <p className={`text-sm font-medium ${textColor}`}>
          {message}
        </p>
        {onClose && (
          <button
            onClick={onClose}
            className={`${textColor} hover:opacity-75`}
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div 
          className={`${progressColor} h-2 rounded-full transition-all duration-300`}
          style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
        />
      </div>
      {showPercentage && (
        <p className={`text-xs ${textColor} mt-1 text-right`}>
          {Math.round(progress)}%
        </p>
      )}
    </div>
  );
};

export default NotificationBanner;