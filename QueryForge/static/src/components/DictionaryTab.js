// static/src/components/DictionaryTab.js
import React, { useState, useEffect } from 'react';
import { 
  BookOpen, 
  Plus, 
  Search, 
  Filter, 
  Edit3, 
  Trash2, 
  Sparkles,
  Download,
  Upload,
  Tag,
  Users,
  CheckCircle,
  Clock,
  AlertTriangle, 
  X,
  Check,
  ThumbsUp,
  ThumbsDown
} from 'lucide-react';
import LoadingSpinner, { LoadingButton, SkeletonLoader } from './LoadingSpinner';
import SuggestionsModal from './SuggestionsModal';

const DictionaryTab = ({ projectId, apiUrl, onNotification }) => {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterCategory, setFilterCategory] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [selectedEntry, setSelectedEntry] = useState(null);
  const [showSuggestionsModal, setShowSuggestionsModal] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);

  useEffect(() => {
    if (projectId) {
      fetchDictionaryEntries();
    }
  }, [projectId]);

  const fetchDictionaryEntries = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiUrl}/projects/${projectId}/dictionary`);
      if (!response.ok) throw new Error('Failed to fetch dictionary entries');
      
      const data = await response.json();
      setEntries(data.entries || []);
    } catch (error) {
      onNotification('Error fetching dictionary: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const generateSuggestions = async () => {
    try {
      setLoadingSuggestions(true);
      const response = await fetch(`${apiUrl}/projects/${projectId}/dictionary/suggest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) throw new Error('Failed to generate suggestions');
      
      const data = await response.json();
      
      // Process suggestions into a flat array
      const allSuggestions = [];
      
      // Add business terms
      if (data.suggestions.business_terms) {
        allSuggestions.push(...data.suggestions.business_terms);
      }
      
      // Add technical terms
      if (data.suggestions.technical_terms) {
        allSuggestions.push(...data.suggestions.technical_terms);
      }
      
      // Add abbreviations
      if (data.suggestions.abbreviations) {
        allSuggestions.push(...data.suggestions.abbreviations);
      }
      
      // Add domain terms
      if (data.suggestions.domain_terms) {
        Object.values(data.suggestions.domain_terms).forEach(domainTerms => {
          allSuggestions.push(...domainTerms);
        });
      }
      
      setSuggestions(allSuggestions);
      setShowSuggestionsModal(true);
      onNotification(`Generated ${allSuggestions.length} suggestions`, 'success');
    } catch (error) {
      onNotification('Error generating suggestions: ' + error.message, 'error');
    } finally {
      setLoadingSuggestions(false);
    }
  };

  const createEntry = async (entryData) => {
    try {
      const response = await fetch(`${apiUrl}/projects/${projectId}/dictionary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entryData)
      });

      if (!response.ok) throw new Error('Failed to create entry');
      
      const data = await response.json();
      onNotification(`Dictionary entry "${data.entry.term}" created successfully`, 'success');
      setShowCreateModal(false);
      fetchDictionaryEntries();
    } catch (error) {
      onNotification('Error creating entry: ' + error.message, 'error');
    }
  };

  const updateEntry = async (entryId, entryData) => {
    try {
      const response = await fetch(`${apiUrl}/dictionary/${entryId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entryData)
      });

      if (!response.ok) throw new Error('Failed to update entry');
      
      const data = await response.json();
      onNotification(`Dictionary entry "${data.entry.term}" updated successfully`, 'success');
      setShowEditModal(false);
      setSelectedEntry(null);
      fetchDictionaryEntries();
    } catch (error) {
      onNotification('Error updating entry: ' + error.message, 'error');
    }
  };

  const deleteEntry = async (entryId, term) => {
    if (!window.confirm(`Are you sure you want to delete the term "${term}"?`)) {
      return;
    }

    try {
      const response = await fetch(`${apiUrl}/dictionary/${entryId}`, {
        method: 'DELETE'
      });

      if (!response.ok) throw new Error('Failed to delete entry');
      
      onNotification(`Dictionary entry "${term}" deleted successfully`, 'success');
      fetchDictionaryEntries();
    } catch (error) {
      onNotification('Error deleting entry: ' + error.message, 'error');
    }
  };

  const approveEntry = async (entryId) => {
    try {
      const response = await fetch(`${apiUrl}/dictionary/${entryId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'approved' })
      });

      if (!response.ok) throw new Error('Failed to approve entry');
      
      onNotification('Dictionary entry approved', 'success');
      fetchDictionaryEntries();
    } catch (error) {
      onNotification('Error approving entry: ' + error.message, 'error');
    }
  };

  const rejectEntry = async (entryId) => {
    try {
      const response = await fetch(`${apiUrl}/dictionary/${entryId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'rejected' })
      });

      if (!response.ok) throw new Error('Failed to reject entry');
      
      onNotification('Dictionary entry rejected', 'success');
      fetchDictionaryEntries();
    } catch (error) {
      onNotification('Error rejecting entry: ' + error.message, 'error');
    }
  };

  const filteredEntries = entries.filter(entry => {
    const matchesSearch = entry.term.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         entry.definition.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         (entry.synonyms && entry.synonyms.some(syn => syn.toLowerCase().includes(searchTerm.toLowerCase())));
    
    const matchesCategory = filterCategory === 'all' || entry.category === filterCategory;
    const matchesStatus = filterStatus === 'all' || entry.status === filterStatus;
    
    return matchesSearch && matchesCategory && matchesStatus;
  });

  const categories = [...new Set(entries.map(entry => entry.category))].filter(Boolean);
  const statuses = [...new Set(entries.map(entry => entry.status))].filter(Boolean);

  if (!projectId) {
    return (
      <div className="p-6 text-center">
        <BookOpen className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No Project Selected</h3>
        <p className="text-gray-500">Please select a project to manage the data dictionary.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="p-6">
        <div className="flex justify-between items-center mb-6">
          <SkeletonLoader lines={1} className="w-64" />
          <SkeletonLoader lines={1} className="w-48" />
        </div>
        <SkeletonLoader lines={10} />
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
        <div>
          <h3 className="text-lg font-medium text-gray-900">Data Dictionary</h3>
          <p className="mt-1 text-sm text-gray-500">
            Manage business terms, definitions, and metadata
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex space-x-3">
          <LoadingButton
            onClick={generateSuggestions}
            loading={loadingSuggestions}
            className="btn btn-secondary"
            loadingText="Generating..."
          >
            <Sparkles className="h-4 w-4 mr-2" />
            Auto-Generate
          </LoadingButton>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn btn-primary"
          >
            <Plus className="h-4 w-4 mr-2" />
            New Term
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row sm:items-center space-y-3 sm:space-y-0 sm:space-x-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search terms, definitions, or synonyms..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="form-input pl-10"
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
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="form-select"
          >
            <option value="all">All Status</option>
            {statuses.map(status => (
              <option key={status} value={status}>
                {status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center">
            <BookOpen className="h-8 w-8 text-blue-500" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-500">Total Terms</p>
              <p className="text-2xl font-semibold text-gray-900">{entries.length}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center">
            <CheckCircle className="h-8 w-8 text-green-500" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-500">Approved</p>
              <p className="text-2xl font-semibold text-gray-900">
                {entries.filter(e => e.status === 'approved').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center">
            <Clock className="h-8 w-8 text-yellow-500" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-500">Draft</p>
              <p className="text-2xl font-semibold text-gray-900">
                {entries.filter(e => e.status === 'draft').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center">
            <Sparkles className="h-8 w-8 text-purple-500" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-500">Auto-Generated</p>
              <p className="text-2xl font-semibold text-gray-900">
                {entries.filter(e => e.is_auto_generated).length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Dictionary Entries */}
      {filteredEntries.length === 0 ? (
        <div className="text-center py-12">
          <BookOpen className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No dictionary entries</h3>
          <p className="mt-1 text-sm text-gray-500">
            {searchTerm || filterCategory !== 'all' || filterStatus !== 'all' 
              ? 'No entries match your search criteria.' 
              : 'Get started by creating terms or auto-generating suggestions.'}
          </p>
          {(!searchTerm && filterCategory === 'all' && filterStatus === 'all') && (
            <div className="mt-6 flex justify-center space-x-3">
              <LoadingButton
                onClick={generateSuggestions}
                loading={loadingSuggestions}
                className="btn btn-secondary"
                loadingText="Generating..."
              >
                <Sparkles className="h-4 w-4 mr-2" />
                Auto-Generate
              </LoadingButton>
              <button
                onClick={() => setShowCreateModal(true)}
                className="btn btn-primary"
              >
                <Plus className="h-4 w-4 mr-2" />
                New Term
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {filteredEntries.map((entry) => (
            <DictionaryEntryCard
              key={entry.id}
              entry={entry}
              onEdit={() => {
                setSelectedEntry(entry);
                setShowEditModal(true);
              }}
              onDelete={() => deleteEntry(entry.id, entry.term)}
              onApprove={() => approveEntry(entry.id)}
              onReject={() => rejectEntry(entry.id)}
            />
          ))}
        </div>
      )}

      {/* Create Entry Modal */}
      {showCreateModal && (
        <DictionaryEntryModal
          title="Create New Term"
          onSubmit={createEntry}
          onCancel={() => setShowCreateModal(false)}
        />
      )}

      {/* Edit Entry Modal */}
      {showEditModal && selectedEntry && (
        <DictionaryEntryModal
          title="Edit Term"
          entry={selectedEntry}
          onSubmit={(data) => updateEntry(selectedEntry.id, data)}
          onCancel={() => {
            setShowEditModal(false);
            setSelectedEntry(null);
          }}
        />
      )}

      {/* Suggestions Modal */}
      {showSuggestionsModal && suggestions && (
        <SuggestionsModal
          suggestions={suggestions}
          onClose={() => {
            setShowSuggestionsModal(false);
            setSuggestions([]);
          }}
          onCreateEntries={async (selectedSuggestions) => {
            // Create entries in bulk
            for (const suggestion of selectedSuggestions) {
              await createEntry({
                term: suggestion.term,
                definition: suggestion.enhanced_definition || suggestion.auto_definition,
                category: suggestion.category || 'business_term',
                domain: suggestion.suggested_domain,
                synonyms: suggestion.suggested_synonyms || [],
                status: 'draft'
              });
            }
            setShowSuggestionsModal(false);
            setSuggestions([]);
          }}
        />
      )}
    </div>
  );
};

// Dictionary Entry Card Component
const DictionaryEntryCard = ({ entry, onEdit, onDelete, onApprove, onReject }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'approved':
        return 'bg-green-100 text-green-800';
      case 'draft':
        return 'bg-yellow-100 text-yellow-800';
      case 'rejected':
        return 'bg-red-100 text-red-800';
      case 'archived':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-blue-100 text-blue-800';
    }
  };

  const getCategoryColor = (category) => {
    switch (category) {
      case 'business_term':
        return 'bg-blue-100 text-blue-800';
      case 'technical_term':
        return 'bg-purple-100 text-purple-800';
      case 'abbreviation':
        return 'bg-orange-100 text-orange-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const showApprovalButtons = entry.status === 'draft' || entry.status === 'rejected';

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-3 mb-2">
            <h3 className="text-lg font-semibold text-gray-900">{entry.term}</h3>
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(entry.status)}`}>
              {entry.status}
            </span>
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getCategoryColor(entry.category)}`}>
              {entry.category?.replace('_', ' ')}
            </span>
            {entry.is_auto_generated && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                <Sparkles className="h-3 w-3 mr-1" />
                Auto
              </span>
            )}
          </div>
          
          <p className="text-gray-700 mb-3">{entry.definition}</p>
          
          {entry.synonyms && entry.synonyms.length > 0 && (
            <div className="mb-2">
              <span className="text-sm font-medium text-gray-500">Synonyms: </span>
              <span className="text-sm text-gray-600">{entry.synonyms.join(', ')}</span>
            </div>
          )}
          
          {entry.domain && (
            <div className="mb-2">
              <span className="text-sm font-medium text-gray-500">Domain: </span>
              <span className="text-sm text-gray-600">{entry.domain}</span>
            </div>
          )}
          
          <div className="flex items-center text-xs text-gray-500 space-x-4">
            <span>Created: {new Date(entry.created_at).toLocaleDateString()}</span>
            {entry.created_by && <span>By: {entry.created_by}</span>}
            {entry.confidence_score && (
              <span>Confidence: {Math.round(entry.confidence_score * 100)}%</span>
            )}
          </div>
        </div>
        
        <div className="flex items-center space-x-2 ml-4">
          {showApprovalButtons && (
            <>
              <button
                onClick={onApprove}
                className="p-2 text-green-600 hover:text-green-800 rounded-full hover:bg-green-50"
                title="Approve"
              >
                <ThumbsUp className="h-4 w-4" />
              </button>
              <button
                onClick={onReject}
                className="p-2 text-red-600 hover:text-red-800 rounded-full hover:bg-red-50"
                title="Reject"
              >
                <ThumbsDown className="h-4 w-4" />
              </button>
            </>
          )}
          <button
            onClick={onEdit}
            className="p-2 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100"
            title="Edit"
          >
            <Edit3 className="h-4 w-4" />
          </button>
          <button
            onClick={onDelete}
            className="p-2 text-gray-400 hover:text-red-600 rounded-full hover:bg-red-50"
            title="Delete"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

// Dictionary Entry Modal Component
const DictionaryEntryModal = ({ title, entry, onSubmit, onCancel }) => {
  const [formData, setFormData] = useState({
    term: entry?.term || '',
    definition: entry?.definition || '',
    category: entry?.category || 'business_term',
    domain: entry?.domain || '',
    synonyms: entry?.synonyms?.join(', ') || '',
    abbreviations: entry?.abbreviations?.join(', ') || '',
    status: entry?.status || 'draft'
  });
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validation
    const newErrors = {};
    if (!formData.term.trim()) {
      newErrors.term = 'Term is required';
    }
    if (!formData.definition.trim()) {
      newErrors.definition = 'Definition is required';
    }
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setLoading(true);
    try {
      const submitData = {
        ...formData,
        synonyms: formData.synonyms.split(',').map(s => s.trim()).filter(Boolean),
        abbreviations: formData.abbreviations.split(',').map(s => s.trim()).filter(Boolean)
      };
      await onSubmit(submitData);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"></div>

        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
          <form onSubmit={handleSubmit}>
            <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
                {title}
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="form-label">Term *</label>
                  <input
                    type="text"
                    value={formData.term}
                    onChange={(e) => setFormData({ ...formData, term: e.target.value })}
                    className={`form-input ${errors.term ? 'border-red-300' : ''}`}
                    placeholder="Enter term"
                  />
                  {errors.term && <p className="form-error">{errors.term}</p>}
                </div>

                <div>
                  <label className="form-label">Definition *</label>
                  <textarea
                    value={formData.definition}
                    onChange={(e) => setFormData({ ...formData, definition: e.target.value })}
                    className={`form-textarea ${errors.definition ? 'border-red-300' : ''}`}
                    rows="3"
                    placeholder="Enter definition"
                  />
                  {errors.definition && <p className="form-error">{errors.definition}</p>}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="form-label">Category</label>
                    <select
                      value={formData.category}
                      onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                      className="form-select"
                    >
                      <option value="business_term">Business Term</option>
                      <option value="technical_term">Technical Term</option>
                      <option value="abbreviation">Abbreviation</option>
                      <option value="metric">Metric</option>
                      <option value="dimension">Dimension</option>
                    </select>
                  </div>

                  <div>
                    <label className="form-label">Status</label>
                    <select
                      value={formData.status}
                      onChange={(e) => setFormData({ ...formData, status: e.target.value })}
                      className="form-select"
                    >
                      <option value="draft">Draft</option>
                      <option value="approved">Approved</option>
                      <option value="rejected">Rejected</option>
                      <option value="archived">Archived</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="form-label">Domain</label>
                  <input
                    type="text"
                    value={formData.domain}
                    onChange={(e) => setFormData({ ...formData, domain: e.target.value })}
                    className="form-input"
                    placeholder="e.g., finance, hr"
                  />
                </div>

                <div>
                  <label className="form-label">Synonyms</label>
                  <input
                    type="text"
                    value={formData.synonyms}
                    onChange={(e) => setFormData({ ...formData, synonyms: e.target.value })}
                    className="form-input"
                    placeholder="Comma-separated synonyms"
                  />
                </div>

                <div>
                  <label className="form-label">Abbreviations</label>
                  <input
                    type="text"
                    value={formData.abbreviations}
                    onChange={(e) => setFormData({ ...formData, abbreviations: e.target.value })}
                    className="form-input"
                    placeholder="Comma-separated abbreviations"
                  />
                </div>
              </div>
            </div>

            <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
              <LoadingButton
                type="submit"
                loading={loading}
                className="btn-primary w-full sm:ml-3 sm:w-auto"
                loadingText="Saving..."
              >
                {entry ? 'Update' : 'Create'} Term
              </LoadingButton>
              <button
                type="button"
                onClick={onCancel}
                className="btn btn-secondary mt-3 w-full sm:mt-0 sm:w-auto"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default DictionaryTab;