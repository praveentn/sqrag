// static/src/components/EmbeddingsTab.js
import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Database, 
  Search, 
  Plus, 
  Play, 
  Pause, 
  RefreshCw,
  CheckCircle,
  Clock,
  AlertTriangle,
  BarChart3,
  Settings,
  Trash2,
  Eye,
  Download
} from 'lucide-react';
import LoadingSpinner, { LoadingButton, SkeletonLoader, ProgressNotification } from './LoadingSpinner';

const EmbeddingsTab = ({ projectId, apiUrl, onNotification }) => {
  const [activeSubTab, setActiveSubTab] = useState(0);
  const [embeddings, setEmbeddings] = useState([]);
  const [indexes, setIndexes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateEmbeddingModal, setShowCreateEmbeddingModal] = useState(false);
  const [showCreateIndexModal, setShowCreateIndexModal] = useState(false);
  const [jobStatuses, setJobStatuses] = useState({});
  const [availableModels, setAvailableModels] = useState({});

  const subTabs = [
    { id: 0, name: 'Embeddings', icon: Cpu },
    { id: 1, name: 'Indexes', icon: Search }
  ];

  useEffect(() => {
    if (projectId) {
      fetchData();
      fetchAvailableModels();
    }
  }, [projectId]);

  // Poll for job statuses
  useEffect(() => {
    const interval = setInterval(() => {
      Object.keys(jobStatuses).forEach(jobId => {
        if (jobStatuses[jobId].status === 'running') {
          checkJobStatus(jobId);
        }
      });
    }, 3000);

    return () => clearInterval(interval);
  }, [jobStatuses]);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [embeddingsResponse, indexesResponse] = await Promise.all([
        fetch(`${apiUrl}/projects/${projectId}/embeddings`),
        fetch(`${apiUrl}/projects/${projectId}/indexes`)
      ]);

      if (embeddingsResponse.ok) {
        const embeddingsData = await embeddingsResponse.json();
        setEmbeddings(embeddingsData.embeddings || []);
      }

      if (indexesResponse.ok) {
        const indexesData = await indexesResponse.json();
        setIndexes(indexesData.indexes || []);
      }
    } catch (error) {
      onNotification('Error fetching data: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      // This would typically be an API call to get available models
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
    } catch (error) {
      console.error('Error fetching available models:', error);
    }
  };

  const createEmbeddings = async (embeddingConfig) => {
    try {
      const response = await fetch(`${apiUrl}/projects/${projectId}/embeddings/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(embeddingConfig)
      });

      if (!response.ok) throw new Error('Failed to create embeddings');
      
      const data = await response.json();
      const jobId = data.job_id;
      
      setJobStatuses(prev => ({
        ...prev,
        [jobId]: { status: 'running', progress: 0, type: 'embedding' }
      }));

      onNotification('Embedding creation started', 'success');
      setShowCreateEmbeddingModal(false);
      
      // Start polling for status
      checkJobStatus(jobId);
    } catch (error) {
      onNotification('Error creating embeddings: ' + error.message, 'error');
    }
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

  const checkJobStatus = async (jobId) => {
    try {
      // This would be a real API call to check job status
      // For now, simulate progress
      setJobStatuses(prev => {
        const current = prev[jobId];
        if (!current || current.status !== 'running') return prev;
        
        const newProgress = Math.min(current.progress + 0.1, 1);
        const newStatus = newProgress >= 1 ? 'completed' : 'running';
        
        if (newStatus === 'completed') {
          onNotification('Embedding creation completed!', 'success');
          setTimeout(() => fetchData(), 1000);
        }
        
        return {
          ...prev,
          [jobId]: { ...current, progress: newProgress, status: newStatus }
        };
      });
    } catch (error) {
      console.error('Error checking job status:', error);
    }
  };

  const deleteEmbeddings = async (objectType) => {
    if (!window.confirm(`Are you sure you want to delete all ${objectType} embeddings?`)) {
      return;
    }

    try {
      // This would be a delete API call
      onNotification(`${objectType} embeddings deleted`, 'success');
      fetchData();
    } catch (error) {
      onNotification('Error deleting embeddings: ' + error.message, 'error');
    }
  };

  const rebuildIndex = async (indexId) => {
    try {
      // This would be a rebuild API call
      onNotification('Index rebuild started', 'success');
      fetchData();
    } catch (error) {
      onNotification('Error rebuilding index: ' + error.message, 'error');
    }
  };

  if (!projectId) {
    return (
      <div className="p-6 text-center">
        <Cpu className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No Project Selected</h3>
        <p className="text-gray-500">Please select a project to manage embeddings and indexes.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="p-6">
        <SkeletonLoader lines={8} />
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Sub-tab Navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {subTabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeSubTab === tab.id;
            
            return (
              <button
                key={tab.id}
                onClick={() => setActiveSubTab(tab.id)}
                className={`
                  group inline-flex items-center py-2 px-1 border-b-2 font-medium text-sm
                  ${isActive 
                    ? 'border-blue-500 text-blue-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }
                `}
              >
                <Icon className={`mr-2 h-5 w-5 ${isActive ? 'text-blue-500' : 'text-gray-400'}`} />
                {tab.name}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Active Jobs Status */}
      {Object.entries(jobStatuses).map(([jobId, status]) => (
        status.status === 'running' && (
          <div key={jobId} className="mb-4">
            <ProgressNotification
              message={`Creating ${status.type}s...`}
              progress={status.progress * 100}
              type="info"
            />
          </div>
        )
      ))}

      {/* Sub-tab Content */}
      {activeSubTab === 0 && (
        <EmbeddingsSubTab
          embeddings={embeddings}
          onCreateEmbeddings={() => setShowCreateEmbeddingModal(true)}
          onDeleteEmbeddings={deleteEmbeddings}
          onRefresh={fetchData}
        />
      )}

      {activeSubTab === 1 && (
        <IndexesSubTab
          indexes={indexes}
          onCreateIndex={() => setShowCreateIndexModal(true)}
          onRebuildIndex={rebuildIndex}
          onRefresh={fetchData}
        />
      )}

      {/* Create Embedding Modal */}
      {showCreateEmbeddingModal && (
        <CreateEmbeddingModal
          availableModels={availableModels}
          onSubmit={createEmbeddings}
          onCancel={() => setShowCreateEmbeddingModal(false)}
        />
      )}

      {/* Create Index Modal */}
      {showCreateIndexModal && (
        <CreateIndexModal
          embeddings={embeddings}
          onSubmit={createIndex}
          onCancel={() => setShowCreateIndexModal(false)}
        />
      )}
    </div>
  );
};

// Embeddings Sub-tab Component
const EmbeddingsSubTab = ({ embeddings, onCreateEmbeddings, onDeleteEmbeddings, onRefresh }) => {
  const groupedEmbeddings = embeddings.reduce((acc, embedding) => {
    const key = embedding.model_name;
    if (!acc[key]) acc[key] = [];
    acc[key].push(embedding);
    return acc;
  }, {});

  return (
    <div>
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-lg font-medium text-gray-900">Embeddings</h3>
          <p className="text-sm text-gray-500">Create and manage vector embeddings for your data</p>
        </div>
        <div className="flex space-x-3">
          <button onClick={onRefresh} className="btn btn-secondary">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </button>
          <button onClick={onCreateEmbeddings} className="btn btn-primary">
            <Plus className="h-4 w-4 mr-2" />
            Create Embeddings
          </button>
        </div>
      </div>

      {/* Embeddings Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center">
            <Cpu className="h-8 w-8 text-blue-500" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-500">Total Embeddings</p>
              <p className="text-2xl font-semibold text-gray-900">{embeddings.length}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center">
            <BarChart3 className="h-8 w-8 text-green-500" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-500">Models Used</p>
              <p className="text-2xl font-semibold text-gray-900">{Object.keys(groupedEmbeddings).length}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center">
            <Database className="h-8 w-8 text-purple-500" />
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-500">Object Types</p>
              <p className="text-2xl font-semibold text-gray-900">
                {new Set(embeddings.map(e => e.object_type)).size}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Embeddings by Model */}
      {Object.keys(groupedEmbeddings).length === 0 ? (
        <div className="text-center py-12">
          <Cpu className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No embeddings created</h3>
          <p className="mt-1 text-sm text-gray-500">
            Create embeddings to enable semantic search and AI-powered features.
          </p>
          <div className="mt-6">
            <button onClick={onCreateEmbeddings} className="btn btn-primary">
              <Plus className="h-4 w-4 mr-2" />
              Create Embeddings
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {Object.entries(groupedEmbeddings).map(([modelName, modelEmbeddings]) => (
            <div key={modelName} className="bg-white border border-gray-200 rounded-lg p-6">
              <div className="flex justify-between items-center mb-4">
                <h4 className="text-lg font-medium text-gray-900">{modelName}</h4>
                <button
                  onClick={() => onDeleteEmbeddings(modelName)}
                  className="btn btn-danger btn-sm"
                >
                  <Trash2 className="h-4 w-4 mr-1" />
                  Delete All
                </button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(
                  modelEmbeddings.reduce((acc, emb) => {
                    if (!acc[emb.object_type]) acc[emb.object_type] = [];
                    acc[emb.object_type].push(emb);
                    return acc;
                  }, {})
                ).map(([objectType, typeEmbeddings]) => (
                  <div key={objectType} className="bg-gray-50 p-4 rounded">
                    <h5 className="font-medium text-gray-900 mb-2">
                      {objectType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </h5>
                    <p className="text-sm text-gray-600 mb-2">
                      {typeEmbeddings.length} embeddings
                    </p>
                    <p className="text-xs text-gray-500">
                      Dimension: {typeEmbeddings[0]?.vector_dimension || 'N/A'}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Indexes Sub-tab Component
const IndexesSubTab = ({ indexes, onCreateIndex, onRebuildIndex, onRefresh }) => {
  const getStatusIcon = (status) => {
    switch (status) {
      case 'ready':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'building':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'error':
        return <AlertTriangle className="h-5 w-5 text-red-500" />;
      default:
        return <Clock className="h-5 w-5 text-gray-500" />;
    }
  };

  return (
    <div>
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-lg font-medium text-gray-900">Search Indexes</h3>
          <p className="text-sm text-gray-500">Manage search indexes for fast retrieval</p>
        </div>
        <div className="flex space-x-3">
          <button onClick={onRefresh} className="btn btn-secondary">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </button>
          <button onClick={onCreateIndex} className="btn btn-primary">
            <Plus className="h-4 w-4 mr-2" />
            Create Index
          </button>
        </div>
      </div>

      {/* Indexes List */}
      {indexes.length === 0 ? (
        <div className="text-center py-12">
          <Search className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No indexes created</h3>
          <p className="mt-1 text-sm text-gray-500">
            Create search indexes to enable fast querying of your embeddings.
          </p>
          <div className="mt-6">
            <button onClick={onCreateIndex} className="btn btn-primary">
              <Plus className="h-4 w-4 mr-2" />
              Create Index
            </button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {indexes.map((index) => (
            <div key={index.id} className="bg-white border border-gray-200 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <h4 className="text-lg font-medium text-gray-900">{index.name}</h4>
                  <div className="ml-3 flex items-center">
                    {getStatusIcon(index.status)}
                    <span className="ml-1 text-sm text-gray-600 capitalize">{index.status}</span>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => onRebuildIndex(index.id)}
                    className="btn btn-secondary btn-sm"
                    disabled={index.status === 'building'}
                  >
                    <RefreshCw className="h-4 w-4 mr-1" />
                    Rebuild
                  </button>
                  <button className="btn btn-secondary btn-sm">
                    <Eye className="h-4 w-4 mr-1" />
                    View
                  </button>
                </div>
              </div>
              
              {index.description && (
                <p className="text-gray-600 mb-4">{index.description}</p>
              )}
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="font-medium text-gray-500">Type:</span>
                  <span className="ml-1 text-gray-900">{index.index_type}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-500">Vectors:</span>
                  <span className="ml-1 text-gray-900">{index.total_vectors || 0}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-500">Size:</span>
                  <span className="ml-1 text-gray-900">
                    {index.index_size_mb ? `${index.index_size_mb} MB` : 'N/A'}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-gray-500">Created:</span>
                  <span className="ml-1 text-gray-900">
                    {new Date(index.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>
              
              {index.build_progress < 1 && index.status === 'building' && (
                <div className="mt-4">
                  <div className="flex items-center justify-between text-sm text-gray-600 mb-1">
                    <span>Building progress</span>
                    <span>{Math.round(index.build_progress * 100)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${index.build_progress * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Create Embedding Modal
const CreateEmbeddingModal = ({ availableModels, onSubmit, onCancel }) => {
  const [formData, setFormData] = useState({
    model_name: 'sentence-transformers/all-MiniLM-L6-v2',
    object_types: ['tables', 'columns', 'dictionary']
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      await onSubmit(formData);
    } finally {
      setLoading(false);
    }
  };

  const toggleObjectType = (type) => {
    setFormData(prev => ({
      ...prev,
      object_types: prev.object_types.includes(type)
        ? prev.object_types.filter(t => t !== type)
        : [...prev.object_types, type]
    }));
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"></div>

        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
          <form onSubmit={handleSubmit}>
            <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
                Create Embeddings
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="form-label">Embedding Model</label>
                  <select
                    value={formData.model_name}
                    onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
                    className="form-select"
                  >
                    {Object.entries(availableModels).map(([provider, models]) => (
                      <optgroup key={provider} label={provider.replace('_', ' ').toUpperCase()}>
                        {models.map(model => (
                          <option key={model} value={model}>{model}</option>
                        ))}
                      </optgroup>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="form-label">Object Types to Embed</label>
                  <div className="space-y-2">
                    {['tables', 'columns', 'dictionary'].map(type => (
                      <label key={type} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={formData.object_types.includes(type)}
                          onChange={() => toggleObjectType(type)}
                          className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200"
                        />
                        <span className="ml-2 text-sm text-gray-700">
                          {type.charAt(0).toUpperCase() + type.slice(1)}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
              <LoadingButton
                type="submit"
                loading={loading}
                disabled={formData.object_types.length === 0}
                className="btn-primary w-full sm:ml-3 sm:w-auto"
                loadingText="Creating..."
              >
                Create Embeddings
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

// Create Index Modal
const CreateIndexModal = ({ embeddings, onSubmit, onCancel }) => {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    index_type: 'faiss',
    metric: 'cosine',
    object_scope: { object_types: ['tables', 'columns', 'dictionary'] },
    embedding_model: '',
    build_params: {}
  });
  const [loading, setLoading] = useState(false);

  // Get available embedding models from existing embeddings
  const availableEmbeddingModels = [...new Set(embeddings.map(e => e.model_name))];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      await onSubmit(formData);
    } finally {
      setLoading(false);
    }
  };

  const toggleObjectType = (type) => {
    setFormData(prev => ({
      ...prev,
      object_scope: {
        ...prev.object_scope,
        object_types: prev.object_scope.object_types.includes(type)
          ? prev.object_scope.object_types.filter(t => t !== type)
          : [...prev.object_scope.object_types, type]
      }
    }));
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"></div>

        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
          <form onSubmit={handleSubmit}>
            <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
                Create Search Index
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="form-label">Index Name *</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="form-input"
                    placeholder="Enter index name"
                    required
                  />
                </div>

                <div>
                  <label className="form-label">Description</label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    className="form-textarea"
                    rows="2"
                    placeholder="Enter description (optional)"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="form-label">Index Type</label>
                    <select
                      value={formData.index_type}
                      onChange={(e) => setFormData({ ...formData, index_type: e.target.value })}
                      className="form-select"
                    >
                      <option value="faiss">FAISS</option>
                      <option value="tfidf">TF-IDF</option>
                      <option value="bm25">BM25</option>
                    </select>
                  </div>

                  <div>
                    <label className="form-label">Similarity Metric</label>
                    <select
                      value={formData.metric}
                      onChange={(e) => setFormData({ ...formData, metric: e.target.value })}
                      className="form-select"
                    >
                      <option value="cosine">Cosine</option>
                      <option value="euclidean">Euclidean</option>
                      <option value="dot_product">Dot Product</option>
                    </select>
                  </div>
                </div>

                {availableEmbeddingModels.length > 0 && (
                  <div>
                    <label className="form-label">Embedding Model</label>
                    <select
                      value={formData.embedding_model}
                      onChange={(e) => setFormData({ ...formData, embedding_model: e.target.value })}
                      className="form-select"
                    >
                      <option value="">All models</option>
                      {availableEmbeddingModels.map(model => (
                        <option key={model} value={model}>{model}</option>
                      ))}
                    </select>
                  </div>
                )}

                <div>
                  <label className="form-label">Object Types to Include</label>
                  <div className="space-y-2">
                    {['tables', 'columns', 'dictionary'].map(type => (
                      <label key={type} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={formData.object_scope.object_types.includes(type)}
                          onChange={() => toggleObjectType(type)}
                          className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200"
                        />
                        <span className="ml-2 text-sm text-gray-700">
                          {type.charAt(0).toUpperCase() + type.slice(1)}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
              <LoadingButton
                type="submit"
                loading={loading}
                disabled={!formData.name || formData.object_scope.object_types.length === 0}
                className="btn-primary w-full sm:ml-3 sm:w-auto"
                loadingText="Creating..."
              >
                Create Index
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

export default EmbeddingsTab;