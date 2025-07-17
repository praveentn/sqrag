// static/src/index.js
import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './App.css';

// Error boundary component for better error handling
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error,
      errorInfo
    });
    
    // Log error to console
    console.error('React Error Boundary caught an error:', error, errorInfo);
    
    // You could send this to an error tracking service
    // Example: Sentry, LogRocket, etc.
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
          <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6 text-center">
            <div className="mb-4">
              <div className="mx-auto w-16 h-16 bg-red-100 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              </div>
            </div>
            
            <h1 className="text-xl font-bold text-gray-900 mb-2">
              Something went wrong
            </h1>
            
            <p className="text-gray-600 mb-6">
              We're sorry, but something unexpected happened. Please refresh the page to try again.
            </p>
            
            <div className="space-y-3">
              <button
                onClick={() => window.location.reload()}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors"
              >
                Refresh Page
              </button>
              
              <details className="text-left">
                <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
                  Show error details
                </summary>
                <div className="mt-2 p-3 bg-gray-50 rounded text-xs font-mono text-gray-700 overflow-auto max-h-32">
                  {this.state.error && this.state.error.toString()}
                  <br />
                  <pre>{this.state.errorInfo?.componentStack || "no stack trace available"}</pre>
                </div>
              </details>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Get the root element
const container = document.getElementById('root');

if (!container) {
  throw new Error('Root element not found! Make sure you have a div with id="root" in your HTML.');
}

// Create React root
const root = createRoot(container);

// Performance monitoring
const startTime = performance.now();

// Render the app
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);

// Log render time
if (process.env.NODE_ENV === 'development') {
  const endTime = performance.now();
  console.log(`React app rendered in ${(endTime - startTime).toFixed(2)}ms`);
}

// Hot module replacement for development
if (process.env.NODE_ENV === 'development' && module.hot) {
  module.hot.accept('./App', () => {
    const NextApp = require('./App').default;
    root.render(
      <React.StrictMode>
        <ErrorBoundary>
          <NextApp />
        </ErrorBoundary>
      </React.StrictMode>
    );
  });
}

// Export for testing
export default App;