// components/EmbeddingsTab.js
import React, { useState, useEffect, useRef } from 'react';
import { 
  Brain, 
  Zap, 
  Database, 
  Play, 
  Pause, 
  RefreshCw, 
  CheckCircle, 
  XCircle,
  Clock,
  BarChart3,
  Settings,
  Download
} from 'lucide-react';

const EmbeddingsTab = ({ projectId, apiUrl, onNotification }) => {
  const [embeddings, setEmbeddings] = useState([]);
  const [indexes, setIndexes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [jobStatuses, setJobStatuses] = useState({});
  const [availableModels, setAvailableModels] = useState({});
  const [showCreateEmbeddingModal, setShowCreateEmbeddingModal] = useState(false);
  const [showCreateIndexModal, setShowCreateIndexModal] = useState(false);
  const [embeddingStats, setEmbeddingStats] = useState(null);
  const [activePollingJobs, setActivePollingJobs] = useState(new Set());

  // Refs for polling intervals
  const pollingIntervals = useRef(new Map());

  useEffect(() => {
    fetchData();
    fetchAvailableModels();
    
    // Cleanup intervals on unmount
    return () => {
      pollingIntervals.current.forEach(interval => clearInterval(interval));
    };
  }, [projectId]);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      const [embeddingsResponse, indexesResponse] = await Promise.all([
        fetch(`${apiUrl}/projects/${projectId}/embeddings`),
        fetch(`${apiUrl}/projects/${projectId}/indexes`)
      ]);

      if (embeddingsResponse.ok) {
        const embeddingsData = await embeddingsResponse.json();
        if (embeddingsData.success) {
          setEmbeddingStats(embeddingsData);
          setEmbeddings(embeddingsData.embeddings || []);
        }
      }

      if (indexesResponse.ok) {
        const indexesData = await indexesResponse.json();
        if (indexesData.success) {
          setIndexes(indexesData.indexes || []);
        }
      }
    } catch (error) {
      onNotification('Error fetching data: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch(`${apiUrl}/embeddings/models`);
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setAvailableModels(data.models);
        }
      }
    } catch (error) {
      console.error('Error fetching available models:', error);
      // Set default models if API fails
      setAvailableModels({
        sentence_transformers: [
          'sentence-transformers/all-MiniLM-L6-v2',
          'sentence-transformers/all-mpnet-base-v2',
          'sentence-transformers/distilbert-base-nli-mean-tokens'
        ],
        openai: [
          'openai/text-embedding-ada-002',
          'openai/text-embedding-3-small'
        ]
      });
    }
  };

  const createEmbeddings = async (embeddingConfig) => {
    try {
      const response = await fetch(`${apiUrl}/projects/${projectId}/embeddings/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(embeddingConfig)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to create embeddings');
      }
      
      const data = await response.json();
      const jobId = data.job_id;
      
      setJobStatuses(prev => ({
        ...prev,
        [jobId]: { 
          status: 'running', 
          progress: 0, 
          type: 'embedding',
          message: 'Starting embedding creation...',
          created_embeddings: 0,
          total_objects: 0
        }
      }));

      onNotification('Embedding creation started', 'success');
      setShowCreateEmbeddingModal(false);
      
      // Start polling for status
      startJobPolling(jobId);
      
    } catch (error) {
      onNotification('Error creating embeddings: ' + error.message, 'error');
    }
  };

  const startJobPolling = (jobId) => {
    // Prevent duplicate polling
    if (activePollingJobs.current.has(jobId)) {
      return;
    }

    activePollingJobs.current.add(jobId);
    
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${apiUrl}/embeddings/job/${jobId}/status`);
        if (response.ok) {
          const data = await response.json();
          if (data.success) {
            const status = data.status;
            
            setJobStatuses(prev => ({
              ...prev,
              [jobId]: { 
                ...status, 
                type: 'embedding' 
              }
            }));

            // Stop polling if job is complete or failed
            if (status.status === 'completed' || status.status === 'failed') {
              clearInterval(pollInterval);
              pollingIntervals.current.delete(jobId);
              activePollingJobs.current.delete(jobId);
              
              if (status.status === 'completed') {
                onNotification(`Embedding creation completed! Created ${status.created_embeddings} embeddings.`, 'success');
                // Refresh data to show new embeddings
                fetchData();
              } else {
                onNotification(`Embedding creation failed: ${status.error || status.message}`, 'error');
              }
            }
          }
        }
      } catch (error) {
        console.error('Error polling job status:', error);
      }
    }, 2000); // Poll every 2 seconds

    pollingIntervals.current.set(jobId, pollInterval);
  };

  const createIndex = async (indexConfig) => {
    try {
      const response = await fetch(`${apiUrl}/projects/${projectId}/indexes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(indexConfig)
      });

      if (!response.ok) throw new Error('Failed to create index');
      
      const data = await response.json();
      onNotification(`Index "${data.index.name}" created successfully`, 'success');
      setShowCreateIndexModal(false);
      fetchData();
    } catch (error) {
      onNotification('Error creating index: ' + error.message, 'error');
    }
  };

  const deleteEmbeddings = async (objectType = null) => {
    if (!confirm(`Are you sure you want to delete ${objectType ? objectType : 'all'} embeddings?`)) {
      return;
    }

    try {
      const url = objectType 
        ? `${apiUrl}/projects/${projectId}/embeddings?object_type=${objectType}`
        : `${apiUrl}/projects/${projectId}/embeddings`;
        
      const response = await fetch(url, { method: 'DELETE' });
      
      if (response.ok) {
        onNotification('Embeddings deleted successfully', 'success');
        fetchData();
      } else {
        throw new Error('Failed to delete embeddings');
      }
    } catch (error) {
      onNotification('Error deleting embeddings: ' + error.message, 'error');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-medium text-gray-900">Embeddings & Search</h3>
          <p className="mt-1 text-sm text-gray-500">
            Create vector embeddings and search indexes for semantic search
          </p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={() => setShowCreateEmbeddingModal(true)}
            className="btn btn-primary"
          >
            <Brain className="h-4 w-4 mr-2" />
            Create Embeddings
          </button>
          <button
            onClick={() => setShowCreateIndexModal(true)}
            className="btn btn-secondary"
            disabled={!embeddingStats || embeddingStats.total_embeddings === 0}
          >
            <Database className="h-4 w-4 mr-2" />
            Create Index
          </button>
        </div>
      </div>

      {/* Statistics Cards */}
      {embeddingStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-blue-500" />
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Total Embeddings</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {embeddingStats.total_embeddings || 0}
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center">
              <Settings className="h-8 w-8 text-green-500" />
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Models Used</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {embeddingStats.models_used || 0}
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center">
              <BarChart3 className="h-8 w-8 text-purple-500" />
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Object Types</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {embeddingStats.object_types || 0}
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center">
              <Database className="h-8 w-8 text-orange-500" />
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Search Indexes</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {indexes.length || 0}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Active Jobs */}
      {Object.keys(jobStatuses).length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h4 className="text-lg font-medium text-gray-900">Active Jobs</h4>
          </div>
          <div className="p-6 space-y-4">
            {Object.entries(jobStatuses).map(([jobId, status]) => (
              <JobStatusCard 
                key={jobId} 
                jobId={jobId} 
                status={status} 
              />
            ))}
          </div>
        </div>
      )}

      {/* Embeddings Overview */}
      {embeddingStats && embeddingStats.total_embeddings > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
            <h4 className="text-lg font-medium text-gray-900">Current Embeddings</h4>
            <button
              onClick={() => deleteEmbeddings()}
              className="btn btn-sm btn-danger"
            >
              Delete All
            </button>
          </div>
          <div className="p-6">
            {/* Model Breakdown */}
            {embeddingStats.model_breakdown && (
              <div className="mb-6">
                <h5 className="font-medium text-gray-900 mb-3">By Model</h5>
                <div className="space-y-2">
                  {Object.entries(embeddingStats.model_breakdown).map(([model, count]) => (
                    <div key={model} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                      <span className="font-mono text-sm">{model}</span>
                      <div className="flex items-center space-x-3">
                        <span className="text-gray-600">{count} embeddings</span>
                        <button
                          onClick={() => deleteEmbeddings(model)}
                          className="text-red-600 hover:text-red-800 text-sm"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Object Type Breakdown */}
            {embeddingStats.object_type_breakdown && (
              <div>
                <h5 className="font-medium text-gray-900 mb-3">By Object Type</h5>
                <div className="space-y-2">
                  {Object.entries(embeddingStats.object_type_breakdown).map(([type, count]) => (
                    <div key={type} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                      <span className="capitalize">{type.replace('_', ' ')}</span>
                      <div className="flex items-center space-x-3">
                        <span className="text-gray-600">{count} embeddings</span>
                        <button
                          onClick={() => deleteEmbeddings(type)}
                          className="text-red-600 hover:text-red-800 text-sm"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Search Indexes */}
      <div className="bg-white border border-gray-200 rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h4 className="text-lg font-medium text-gray-900">Search Indexes</h4>
        </div>
        <div className="p-6">
          {indexes.length === 0 ? (
            <div className="text-center py-8">
              <Database className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500 mb-4">No search indexes created yet</p>
              {embeddingStats && embeddingStats.total_embeddings > 0 ? (
                <button
                  onClick={() => setShowCreateIndexModal(true)}
                  className="btn btn-primary"
                >
                  Create Your First Index
                </button>
              ) : (
                <p className="text-sm text-gray-400">Create embeddings first to build search indexes</p>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              {indexes.map((index) => (
                <IndexCard key={index.id} index={index} onRefresh={fetchData} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Modals */}
      {showCreateEmbeddingModal && (
        <CreateEmbeddingModal
          availableModels={availableModels}
          onClose={() => setShowCreateEmbeddingModal(false)}
          onSubmit={createEmbeddings}
        />
      )}

      {showCreateIndexModal && (
        <CreateIndexModal
          onClose={() => setShowCreateIndexModal(false)}
          onSubmit={createIndex}
          availableModels={embeddingStats?.model_breakdown || {}}
        />
      )}
    </div>
  );
};

// Job Status Card Component
const JobStatusCard = ({ jobId, status }) => {
  const getStatusIcon = () => {
    switch (status.status) {
      case 'running':
        return <RefreshCw className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Clock className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'running':
        return 'bg-blue-50 border-blue-200';
      case 'completed':
        return 'bg-green-50 border-green-200';
      case 'failed':
        return 'bg-red-50 border-red-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className={`p-4 rounded-lg border ${getStatusColor()}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {getStatusIcon()}
          <div>
            <p className="font-medium text-gray-900">
              {status.type === 'embedding' ? 'Embedding Creation' : 'Index Building'}
            </p>
            <p className="text-sm text-gray-600">{status.message}</p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-sm font-medium text-gray-900">
            {Math.round((status.progress || 0) * 100)}%
          </p>
          {status.total_objects > 0 && (
            <p className="text-xs text-gray-500">
              {status.created_embeddings || 0} / {status.total_objects}
            </p>
          )}
        </div>
      </div>
      
      {status.status === 'running' && (
        <div className="mt-3">
          <div className="bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.round((status.progress || 0) * 100)}%` }}
            />
          </div>
        </div>
      )}

      {status.status === 'failed' && status.error && (
        <div className="mt-2 p-2 bg-red-100 rounded text-sm text-red-700">
          Error: {status.error}
        </div>
      )}
    </div>
  );
};

// Index Card Component  
const IndexCard = ({ index, onRefresh }) => {
  const getStatusColor = () => {
    switch (index.status) {
      case 'ready':
        return 'text-green-600 bg-green-100';
      case 'building':
        return 'text-blue-600 bg-blue-100';
      case 'error':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="p-4 border border-gray-200 rounded-lg">
      <div className="flex items-center justify-between">
        <div>
          <h6 className="font-medium text-gray-900">{index.name}</h6>
          <div className="flex items-center space-x-4 mt-1">
            <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${getStatusColor()}`}>
              {index.status}
            </span>
            <span className="text-sm text-gray-500">
              Type: {index.index_type}
            </span>
            <span className="text-sm text-gray-500">
              Vectors: {index.total_vectors || 0}
            </span>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {index.status === 'building' && (
            <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />
          )}
          <button
            onClick={onRefresh}
            className="text-gray-400 hover:text-gray-600"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

// Create Embedding Modal Component
const CreateEmbeddingModal = ({ availableModels, onClose, onSubmit }) => {
  const [selectedModel, setSelectedModel] = useState('sentence-transformers/all-MiniLM-L6-v2');
  const [selectedTypes, setSelectedTypes] = useState(['tables', 'columns', 'dictionary']);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      model_name: selectedModel,
      object_types: selectedTypes
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-md">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Create Embeddings</h3>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="form-label">Embedding Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="form-select"
            >
              <optgroup label="Sentence Transformers">
                {(availableModels.sentence_transformers || []).map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </optgroup>
              {availableModels.openai && availableModels.openai.length > 0 && (
                <optgroup label="OpenAI">
                  {availableModels.openai.map(model => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </optgroup>
              )}
            </select>
          </div>

          <div>
            <label className="form-label">Object Types</label>
            <div className="space-y-2">
              {[
                { value: 'tables', label: 'Tables' },
                { value: 'columns', label: 'Columns' }, 
                { value: 'dictionary', label: 'Dictionary Entries' }
              ].map(type => (
                <label key={type.value} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={selectedTypes.includes(type.value)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedTypes([...selectedTypes, type.value]);
                      } else {
                        setSelectedTypes(selectedTypes.filter(t => t !== type.value));
                      }
                    }}
                    className="form-checkbox"
                  />
                  <span className="ml-2">{type.label}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="flex justify-end space-x-3 pt-4">
            <button type="button" onClick={onClose} className="btn btn-secondary">
              Cancel
            </button>
            <button 
              type="submit" 
              className="btn btn-primary"
              disabled={selectedTypes.length === 0}
            >
              Create Embeddings
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// Create Index Modal Component
const CreateIndexModal = ({ onClose, onSubmit, availableModels }) => {
  const [indexName, setIndexName] = useState('');
  const [indexType, setIndexType] = useState('faiss');
  const [embeddingModel, setEmbeddingModel] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      name: indexName,
      index_type: indexType,
      embedding_model: embeddingModel || undefined
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-md">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Create Search Index</h3>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="form-label">Index Name</label>
            <input
              type="text"
              value={indexName}
              onChange={(e) => setIndexName(e.target.value)}
              className="form-input"
              placeholder="e.g., semantic-search-index"
              required
            />
          </div>

          <div>
            <label className="form-label">Index Type</label>
            <select
              value={indexType}
              onChange={(e) => setIndexType(e.target.value)}
              className="form-select"
            >
              <option value="faiss">FAISS (Vector Similarity)</option>
              <option value="tfidf">TF-IDF (Keyword)</option>
              <option value="bm25">BM25 (Ranking)</option>
            </select>
          </div>

          <div>
            <label className="form-label">Embedding Model (Optional)</label>
            <select
              value={embeddingModel}
              onChange={(e) => setEmbeddingModel(e.target.value)}
              className="form-select"
            >
              <option value="">All Models</option>
              {Object.keys(availableModels).map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>

          <div className="flex justify-end space-x-3 pt-4">
            <button type="button" onClick={onClose} className="btn btn-secondary">
              Cancel
            </button>
            <button 
              type="submit" 
              className="btn btn-primary"
              disabled={!indexName.trim()}
            >
              Create Index
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default EmbeddingsTab;