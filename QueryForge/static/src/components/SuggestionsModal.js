import React, { useState } from 'react';
import { X, CheckCircle, AlertTriangle } from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';

/**
 * Props:
 * - suggestions: Array of suggestion objects { term, auto_definition, category, confidence, context }
 * - onClose: () => void
 * - onCreateEntries: (selectedSuggestions) => void
 */
export default function SuggestionsModal({ suggestions = [], onClose, onCreateEntries }) {
  const list = Array.isArray(suggestions) ? suggestions : [];
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [creating, setCreating] = useState(false);

  const toggleSelect = (idx) => {
    const updated = new Set(selectedIds);
    if (updated.has(idx)) updated.delete(idx);
    else updated.add(idx);
    setSelectedIds(updated);
  };

  const handleCreate = async () => {
    if (selectedIds.size === 0) return;
    const toCreate = Array.from(selectedIds).map(i => suggestions[i]);
    setCreating(true);
    await onCreateEntries(toCreate);
    setCreating(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
      <div className="fixed inset-0 bg-black bg-opacity-50 transition-opacity" onClick={onClose} />
      <div className="bg-white rounded-lg shadow-lg overflow-hidden max-w-2xl w-full mx-auto relative z-10">
        <div className="flex justify-between items-center px-6 py-4 border-b">
          <h3 className="text-lg font-medium text-gray-900">Suggestion Review</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="p-4 max-h-96 overflow-y-auto">
          {suggestions.length === 0 ? (
            <div className="text-center text-gray-500 py-12">
              No suggestions available.
            </div>
          ) : (
            <ul className="space-y-4">
              {suggestions.map((sugg, idx) => (
                <li key={idx} className="flex items-start space-x-3 p-3 border rounded hover:bg-gray-50">
                  <input
                    type="checkbox"
                    className="mt-1 h-4 w-4 text-blue-600 border-gray-300 rounded"
                    checked={selectedIds.has(idx)}
                    onChange={() => toggleSelect(idx)}
                  />
                  <div className="flex-1">
                    <div className="flex justify-between items-center">
                      <h4 className="font-medium text-gray-900">{sugg.term}</h4>
                      <span className="text-xs text-gray-500">{(sugg.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{sugg.auto_definition}</p>
                    <div className="mt-1 flex space-x-2 text-xs text-gray-500">
                      <span className="capitalize">{sugg.category.replace('_', ' ')}</span>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="px-6 py-4 bg-gray-50 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="btn btn-secondary"
            disabled={creating}
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            className="btn btn-primary flex items-center"
            disabled={creating || selectedIds.size === 0}
          >
            {creating ? <LoadingSpinner size="sm" color="white" /> : <CheckCircle className="h-4 w-4 mr-2" />}
            Create {selectedIds.size} Term{selectedIds.size === 1 ? '' : 's'}
          </button>
        </div>
      </div>
    </div>
  );
}
