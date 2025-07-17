// static/src/components/SuggestionsModal.js
import React, { useState } from 'react';
import { 
  X, 
  CheckCircle, 
  AlertTriangle, 
  Sparkles, 
  BookOpen,
  Tag,
  Filter,
  Search
} from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';

const SuggestionsModal = ({ suggestions = [], onClose, onCreateEntries }) => {
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [creating, setCreating] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterCategory, setFilterCategory] = useState('all');
  const [sortBy, setSortBy] = useState('confidence');

  // Ensure suggestions is always an array
  const suggestionsList = Array.isArray(suggestions) ? suggestions : [];

  const toggleSelect = (index) => {
    const updated = new Set(selectedIds);
    if (updated.has(index)) {
      updated.delete(index);
    } else {
      updated.add(index);
    }
    setSelectedIds(updated);
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === filteredSuggestions.length) {
      setSelectedIds(new Set());
    } else {
      const allIds = new Set(filteredSuggestions.map((_, index) => suggestionsList.indexOf(_)));
      setSelectedIds(allIds);
    }
  };

  const handleCreate = async () => {
    if (selectedIds.size === 0) return;
    
    setCreating(true);
    try {
      const selectedSuggestions = Array.from(selectedIds).map(id => suggestionsList[id]);
      await onCreateEntries(selectedSuggestions);
    } finally {
      setCreating(false);
    }
  };

  // Filter and sort suggestions
  const filteredSuggestions = suggestionsList.filter(suggestion => {
    const matchesSearch = suggestion.term?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         suggestion.auto_definition?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         suggestion.enhanced_definition?.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesCategory = filterCategory === 'all' || suggestion.category === filterCategory;
    
    return matchesSearch && matchesCategory;
  });

  const sortedSuggestions = [...filteredSuggestions].sort((a, b) => {
    switch (sortBy) {
      case 'confidence':
        return (b.confidence || 0) - (a.confidence || 0);
      case 'term':
        return (a.term || '').localeCompare(b.term || '');
      case 'category':
        return (a.category || '').localeCompare(b.category || '');
      default:
        return 0;
    }
  });

  const categories = [...new Set(suggestionsList.map(s => s.category))].filter(Boolean);

  const getCategoryColor = (category) => {
    switch (category) {
      case 'business_term':
        return 'bg-blue-100 text-blue-800';
      case 'technical_term':
        return 'bg-purple-100 text-purple-800';
      case 'abbreviation':
        return 'bg-orange-100 text-orange-800';
      case 'metric':
        return 'bg-green-100 text-green-800';
      case 'dimension':
        return 'bg-indigo-100 text-indigo-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" onClick={onClose}></div>

        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-4xl sm:w-full">
          {/* Header */}
          <div className="bg-white px-6 py-4 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Sparkles className="h-6 w-6 text-purple-500 mr-3" />
                <div>
                  <h3 className="text-lg font-medium text-gray-900">
                    Dictionary Suggestions
                  </h3>
                  <p className="text-sm text-gray-500">
                    Review and select terms to add to your data dictionary
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="h-6 w-6" />
              </button>
            </div>
          </div>

          {/* Filters and Search */}
          <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-3 sm:space-y-0 sm:space-x-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search suggestions..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="form-input pl-10 w-full"
                />
              </div>
              
              <div className="flex items-center space-x-3">
                <select
                  value={filterCategory}
                  onChange={(e) => setFilterCategory(e.target.value)}
                  className="form-select"
                >
                  <option value="all">All Categories</option>
                  {categories.map(category => (
                    <option key={category} value={category}>
                      {category.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </option>
                  ))}
                </select>
                
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className="form-select"
                >
                  <option value="confidence">Sort by Confidence</option>
                  <option value="term">Sort by Term</option>
                  <option value="category">Sort by Category</option>
                </select>
              </div>
            </div>

            {/* Selection Controls */}
            <div className="flex items-center justify-between mt-4">
              <div className="flex items-center space-x-4">
                <button
                  onClick={toggleSelectAll}
                  className="text-sm text-blue-600 hover:text-blue-800 font-medium"
                >
                  {selectedIds.size === filteredSuggestions.length ? 'Deselect All' : 'Select All'}
                </button>
                <span className="text-sm text-gray-500">
                  {selectedIds.size} of {sortedSuggestions.length} selected
                </span>
              </div>
              
              <div className="text-sm text-gray-500">
                {suggestionsList.length} total suggestions
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="max-h-96 overflow-y-auto bg-white">
            {sortedSuggestions.length === 0 ? (
              <div className="text-center py-12">
                <BookOpen className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900">No suggestions found</h3>
                <p className="mt-1 text-sm text-gray-500">
                  {searchTerm || filterCategory !== 'all' 
                    ? 'Try adjusting your search or filter criteria.' 
                    : 'No suggestions were generated.'}
                </p>
              </div>
            ) : (
              <div className="divide-y divide-gray-200">
                {sortedSuggestions.map((suggestion, index) => {
                  const originalIndex = suggestionsList.indexOf(suggestion);
                  const isSelected = selectedIds.has(originalIndex);
                  
                  return (
                    <div
                      key={originalIndex}
                      className={`p-4 hover:bg-gray-50 transition-colors ${
                        isSelected ? 'bg-blue-50 border-l-4 border-blue-500' : ''
                      }`}
                    >
                      <div className="flex items-start space-x-3">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleSelect(originalIndex)}
                          className="mt-1 h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                        />
                        
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                              <h4 className="text-sm font-medium text-gray-900">
                                {suggestion.term}
                              </h4>
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getCategoryColor(suggestion.category)}`}>
                                {suggestion.category?.replace('_', ' ')}
                              </span>
                              {suggestion.confidence && (
                                <span className={`text-xs font-medium ${getConfidenceColor(suggestion.confidence)}`}>
                                  {Math.round(suggestion.confidence * 100)}%
                                </span>
                              )}
                            </div>
                          </div>
                          
                          <p className="mt-1 text-sm text-gray-600">
                            {suggestion.enhanced_definition || suggestion.auto_definition || suggestion.definition}
                          </p>
                          
                          {suggestion.suggested_synonyms && suggestion.suggested_synonyms.length > 0 && (
                            <div className="mt-2">
                              <span className="text-xs font-medium text-gray-500">Synonyms: </span>
                              <span className="text-xs text-gray-600">
                                {suggestion.suggested_synonyms.join(', ')}
                              </span>
                            </div>
                          )}
                          
                          {suggestion.suggested_domain && (
                            <div className="mt-1">
                              <span className="text-xs font-medium text-gray-500">Domain: </span>
                              <span className="text-xs text-gray-600">{suggestion.suggested_domain}</span>
                            </div>
                          )}
                          
                          {suggestion.context && (
                            <div className="mt-2 p-2 bg-gray-100 rounded text-xs text-gray-600">
                              <span className="font-medium">Context: </span>
                              {suggestion.context}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="bg-gray-50 px-6 py-4 flex items-center justify-between">
            <div className="text-sm text-gray-500">
              {selectedIds.size > 0 && (
                <span>
                  {selectedIds.size} term{selectedIds.size === 1 ? '' : 's'} selected
                </span>
              )}
            </div>
            
            <div className="flex items-center space-x-3">
              <button
                onClick={onClose}
                className="btn btn-secondary"
                disabled={creating}
              >
                Cancel
              </button>
              <button
                onClick={handleCreate}
                disabled={creating || selectedIds.size === 0}
                className="btn btn-primary flex items-center"
              >
                {creating ? (
                  <>
                    <LoadingSpinner size="sm" color="white" className="mr-2" />
                    Creating...
                  </>
                ) : (
                  <>
                    <CheckCircle className="h-4 w-4 mr-2" />
                    Create {selectedIds.size} Term{selectedIds.size === 1 ? '' : 's'}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SuggestionsModal;