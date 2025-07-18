// static/src/components/DataSourcesTab.js
import React, { useState, useEffect } from 'react';
import { 
  Upload, 
  Database, 
  FileText, 
  Table, 
  Eye, 
  Edit3, 
  Trash2,
  Plus,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  Download,
  Server,
  File
} from 'lucide-react';
import LoadingSpinner, { LoadingButton, SkeletonLoader } from './LoadingSpinner';

const DataSourcesTab = ({ projectId, apiUrl, onNotification }) => {
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showDatabaseModal, setShowDatabaseModal] = useState(false);
  const [selectedSource, setSelectedSource] = useState(null);
  const [showTableModal, setShowTableModal] = useState(false);

  useEffect(() => {
    if (projectId) {
      fetchDataSources();
    }
  }, [projectId]);

  const fetchDataSources = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiUrl}/projects/${projectId}/sources`);
      if (!response.ok) throw new Error('Failed to fetch data sources');
      
      const data = await response.json();
      setSources(data.sources || []);
    } catch (error) {
      onNotification('Error fetching data sources: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (files) => {
    const formData = new FormData();
    Array.from(files).forEach(file => {
      formData.append('file', file);
    });

    try {
      const response = await fetch(`${apiUrl}/projects/${projectId}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      
      const data = await response.json();
      onNotification(`File uploaded successfully. Created ${data.tables_created} tables.`, 'success');
      setShowUploadModal(false);
      fetchDataSources();
    } catch (error) {
      onNotification('Upload failed: ' + error.message, 'error');
    }
  };

  const handleDatabaseAdd = async (dbConfig) => {
    try {
      const response = await fetch(`${apiUrl}/projects/${projectId}/sources/database`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(dbConfig),
      });

      if (!response.ok) throw new Error('Failed to add database');
      
      const data = await response.json();
      onNotification(`Database added successfully. Imported ${data.tables_imported} tables.`, 'success');
      setShowDatabaseModal(false);
      fetchDataSources();
    } catch (error) {
      onNotification('Database connection failed: ' + error.message, 'error');
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'processing':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <AlertTriangle className="h-5 w-5 text-gray-500" />;
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return 'N/A';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
  };

  if (!projectId) {
    return (
      <div className="p-6 text-center">
        <Database className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No Project Selected</h3>
        <p className="text-gray-500">Please select a project to manage data sources.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="p-6">
        <div className="flex justify-between items-center mb-6">
          <SkeletonLoader lines={1} className="w-64" />
          <SkeletonLoader lines={1} className="w-32" />
        </div>
        <SkeletonLoader lines={8} />
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
        <div>
          <h3 className="text-lg font-medium text-gray-900">Data Sources</h3>
          <p className="mt-1 text-sm text-gray-500">
            Upload files and connect databases to your project
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex space-x-3">
          <button
            onClick={() => setShowUploadModal(true)}
            className="btn btn-secondary"
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload Files
          </button>
          <button
            onClick={() => setShowDatabaseModal(true)}
            className="btn btn-primary"
          >
            <Database className="h-4 w-4 mr-2" />
            Add Database
          </button>
        </div>
      </div>

      {/* Sources List */}
      {sources.length === 0 ? (
        <div className="text-center py-12">
          <Database className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No data sources</h3>
          <p className="mt-1 text-sm text-gray-500">
            Get started by uploading files or connecting a database.
          </p>
          <div className="mt-6 flex justify-center space-x-3">
            <button
              onClick={() => setShowUploadModal(true)}
              className="btn btn-secondary"
            >
              <Upload className="h-4 w-4 mr-2" />
              Upload Files
            </button>
            <button
              onClick={() => setShowDatabaseModal(true)}
              className="btn btn-primary"
            >
              <Database className="h-4 w-4 mr-2" />
              Add Database
            </button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {sources.map((source) => (
            <DataSourceCard
              key={source.id}
              source={source}
              onViewTables={() => {
                setSelectedSource(source);
                setShowTableModal(true);
              }}
              getStatusIcon={getStatusIcon}
              formatFileSize={formatFileSize}
            />
          ))}
        </div>
      )}

      {/* Upload Modal */}
      {showUploadModal && (
        <FileUploadModal
          onUpload={handleFileUpload}
          onCancel={() => setShowUploadModal(false)}
        />
      )}

      {/* Database Modal */}
      {showDatabaseModal && (
        <DatabaseConnectionModal
          onSubmit={handleDatabaseAdd}
          onCancel={() => setShowDatabaseModal(false)}
        />
      )}

      {/* Table View Modal */}
      {showTableModal && selectedSource && (
        <TableViewModal
          source={selectedSource}
          apiUrl={apiUrl}
          onClose={() => {
            setShowTableModal(false);
            setSelectedSource(null);
          }}
          onNotification={onNotification}
        />
      )}
    </div>
  );
};

// Data Source Card Component
const DataSourceCard = ({ source, onViewTables, getStatusIcon, formatFileSize }) => {
  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="p-6">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            {source.type === 'file' ? (
              <File className="h-8 w-8 text-blue-500" />
            ) : (
              <Server className="h-8 w-8 text-green-500" />
            )}
          </div>
          <div className="ml-4 flex-1">
            <h3 className="text-lg font-medium text-gray-900">{source.name}</h3>
            <div className="flex items-center mt-1">
              <span className="text-sm text-gray-500">
                {source.type} / {source.subtype}
              </span>
              <div className="ml-2 flex items-center">
                {getStatusIcon(source.ingest_status)}
                <span className="ml-1 text-sm text-gray-500 capitalize">
                  {source.ingest_status}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <span>Progress: {Math.round((source.ingest_progress || 0) * 100)}%</span>
            <span>Tables: {source.tables_count || 0}</span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(source.ingest_progress || 0) * 100}%` }}
            />
          </div>
        </div>

        {source.file_size && (
          <div className="mt-2 text-sm text-gray-500">
            Size: {formatFileSize(source.file_size)}
          </div>
        )}

        {source.error_message && (
          <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded">
            <p className="text-sm text-red-600">{source.error_message}</p>
          </div>
        )}

        <div className="mt-4 flex justify-between">
          <button
            onClick={onViewTables}
            className="btn btn-secondary btn-sm"
            disabled={source.ingest_status !== 'completed'}
          >
            <Table className="h-4 w-4 mr-1" />
            View Tables
          </button>
          
          <div className="text-xs text-gray-500">
            Created: {new Date(source.created_at).toLocaleDateString()}
          </div>
        </div>
      </div>
    </div>
  );
};

// File Upload Modal
const FileUploadModal = ({ onUpload, onCancel }) => {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    setSelectedFiles(files);
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    setSelectedFiles(files);
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;
    
    setUploading(true);
    try {
      await onUpload(selectedFiles);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"></div>

        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
          <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
              Upload Files
            </h3>

            <div
              className={`upload-area ${dragOver ? 'dragover' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="text-center">
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <div className="mt-4">
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <span className="mt-2 block text-sm font-medium text-gray-900">
                      Drop files here or click to browse
                    </span>
                    <input
                      id="file-upload"
                      type="file"
                      className="sr-only"
                      multiple
                      accept=".csv,.xlsx,.xls,.json"
                      onChange={handleFileSelect}
                    />
                  </label>
                  <p className="mt-1 text-xs text-gray-500">
                    Supported: CSV, Excel (.xlsx, .xls), JSON
                  </p>
                </div>
              </div>
            </div>

            {selectedFiles.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium text-gray-900 mb-2">Selected Files:</h4>
                <ul className="space-y-1">
                  {selectedFiles.map((file, index) => (
                    <li key={index} className="text-sm text-gray-600 flex items-center">
                      <FileText className="h-4 w-4 mr-2" />
                      {file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
            <LoadingButton
              onClick={handleUpload}
              loading={uploading}
              disabled={selectedFiles.length === 0}
              className="btn-primary w-full sm:ml-3 sm:w-auto"
              loadingText="Uploading..."
            >
              Upload Files
            </LoadingButton>
            <button
              onClick={onCancel}
              className="btn btn-secondary mt-3 w-full sm:mt-0 sm:w-auto"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Database Connection Modal
const DatabaseConnectionModal = ({ onSubmit, onCancel }) => {
  const [formData, setFormData] = useState({
    name: '',
    type: 'postgresql',
    host: '',
    port: 5432,
    database: '',
    username: '',
    password: ''
  });
  const [loading, setLoading] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionValid, setConnectionValid] = useState(null);

  const dbTypes = [
    { value: 'postgresql', label: 'PostgreSQL', defaultPort: 5432 },
    { value: 'mysql', label: 'MySQL', defaultPort: 3306 },
    { value: 'sqlite', label: 'SQLite', defaultPort: null },
    { value: 'mssql', label: 'SQL Server', defaultPort: 1433 }
  ];

  const handleTypeChange = (type) => {
    const dbType = dbTypes.find(db => db.value === type);
    setFormData({
      ...formData,
      type,
      port: dbType.defaultPort || formData.port
    });
  };

  const testConnection = async () => {
    setTestingConnection(true);
    try {
      // This would typically call a test endpoint
      // For now, we'll simulate the test
      await new Promise(resolve => setTimeout(resolve, 2000));
      setConnectionValid(true);
    } catch (error) {
      setConnectionValid(false);
    } finally {
      setTestingConnection(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const connection_config = {
        type: formData.type,
        host: formData.host,
        port: formData.port,
        database: formData.database,
        username: formData.username,
        password: formData.password
      };

      await onSubmit({
        name: formData.name,
        db_type: formData.type,
        connection_config
      });
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
                Add Database Connection
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="form-label">Connection Name</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="form-input"
                    required
                  />
                </div>

                <div>
                  <label className="form-label">Database Type</label>
                  <select
                    value={formData.type}
                    onChange={(e) => handleTypeChange(e.target.value)}
                    className="form-select"
                  >
                    {dbTypes.map(type => (
                      <option key={type.value} value={type.value}>
                        {type.label}
                      </option>
                    ))}
                  </select>
                </div>

                {formData.type !== 'sqlite' && (
                  <>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="form-label">Host</label>
                        <input
                          type="text"
                          value={formData.host}
                          onChange={(e) => setFormData({ ...formData, host: e.target.value })}
                          className="form-input"
                          required
                        />
                      </div>
                      <div>
                        <label className="form-label">Port</label>
                        <input
                          type="number"
                          value={formData.port}
                          onChange={(e) => setFormData({ ...formData, port: parseInt(e.target.value) })}
                          className="form-input"
                          required
                        />
                      </div>
                    </div>

                    <div>
                      <label className="form-label">Database Name</label>
                      <input
                        type="text"
                        value={formData.database}
                        onChange={(e) => setFormData({ ...formData, database: e.target.value })}
                        className="form-input"
                        required
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="form-label">Username</label>
                        <input
                          type="text"
                          value={formData.username}
                          onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                          className="form-input"
                          required
                        />
                      </div>
                      <div>
                        <label className="form-label">Password</label>
                        <input
                          type="password"
                          value={formData.password}
                          onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                          className="form-input"
                          required
                        />
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      <LoadingButton
                        type="button"
                        onClick={testConnection}
                        loading={testingConnection}
                        className="btn-secondary btn-sm"
                        loadingText="Testing..."
                      >
                        Test Connection
                      </LoadingButton>
                      
                      {connectionValid === true && (
                        <div className="flex items-center text-green-600">
                          <CheckCircle className="h-4 w-4 mr-1" />
                          <span className="text-sm">Connection successful</span>
                        </div>
                      )}
                      
                      {connectionValid === false && (
                        <div className="flex items-center text-red-600">
                          <XCircle className="h-4 w-4 mr-1" />
                          <span className="text-sm">Connection failed</span>
                        </div>
                      )}
                    </div>
                  </>
                )}

                {formData.type === 'sqlite' && (
                  <div>
                    <label className="form-label">Database File Path</label>
                    <input
                      type="text"
                      value={formData.database}
                      onChange={(e) => setFormData({ ...formData, database: e.target.value })}
                      className="form-input"
                      placeholder="/path/to/database.db"
                      required
                    />
                  </div>
                )}
              </div>
            </div>

            <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
              <LoadingButton
                type="submit"
                loading={loading}
                className="btn-primary w-full sm:ml-3 sm:w-auto"
                loadingText="Connecting..."
              >
                Add Database
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

// Table View Modal
const TableViewModal = ({ source, apiUrl, onClose, onNotification }) => {
  const [tables, setTables] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTable, setSelectedTable] = useState(null);
  const [columns, setColumns] = useState([]);

  useEffect(() => {
    fetchTables();
  }, []);

  const fetchTables = async () => {
    try {
      const response = await fetch(`${apiUrl}/sources/${source.id}/tables`);
      if (!response.ok) throw new Error('Failed to fetch tables');
      
      const data = await response.json();
      setTables(data.tables || []);
    } catch (error) {
      onNotification('Error fetching tables: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchColumns = async (tableId) => {
    try {
      const response = await fetch(`${apiUrl}/tables/${tableId}/columns`);
      if (!response.ok) throw new Error('Failed to fetch columns');
      
      const data = await response.json();
      setColumns(data.columns || []);
    } catch (error) {
      onNotification('Error fetching columns: ' + error.message, 'error');
    }
  };

  const handleTableSelect = (table) => {
    setSelectedTable(table);
    fetchColumns(table.id);
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"></div>

        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-4xl sm:w-full">
          <div className="bg-white px-4 pt-5 pb-4 sm:p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Tables in {source.name}
              </h3>
              <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
                <XCircle className="h-6 w-6" />
              </button>
            </div>

            {loading ? (
              <LoadingSpinner message="Loading tables..." />
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Tables List */}
                <div>
                  <h4 className="text-md font-medium text-gray-900 mb-3">Tables</h4>
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {tables.map((table) => (
                      <div
                        key={table.id}
                        onClick={() => handleTableSelect(table)}
                        className={`p-3 border rounded cursor-pointer hover:bg-gray-50 ${
                          selectedTable?.id === table.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                        }`}
                      >
                        <h5 className="font-medium text-gray-900">{table.name}</h5>
                        <p className="text-sm text-gray-500">
                          {table.row_count} rows, {table.column_count} columns
                        </p>
                        {table.description && (
                          <p className="text-xs text-gray-400 mt-1">{table.description}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Columns List */}
                <div>
                  <h4 className="text-md font-medium text-gray-900 mb-3">
                    {selectedTable ? `Columns in ${selectedTable.name}` : 'Select a table'}
                  </h4>
                  {selectedTable && (
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {columns.map((column) => (
                        <div key={column.id} className="p-3 border border-gray-200 rounded">
                          <div className="flex justify-between items-start">
                            <h6 className="font-medium text-gray-900">{column.name}</h6>
                            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                              {column.data_type}
                            </span>
                          </div>
                          {column.description && (
                            <p className="text-sm text-gray-600 mt-1">{column.description}</p>
                          )}
                          <div className="flex items-center mt-2 space-x-4 text-xs text-gray-500">
                            {column.is_primary_key && (
                              <span className="bg-blue-100 text-blue-600 px-2 py-1 rounded">PK</span>
                            )}
                            {column.is_foreign_key && (
                              <span className="bg-green-100 text-green-600 px-2 py-1 rounded">FK</span>
                            )}
                            {column.pii_flag && (
                              <span className="bg-red-100 text-red-600 px-2 py-1 rounded">PII</span>
                            )}
                            {column.business_category && (
                              <span className="bg-gray-100 text-gray-600 px-2 py-1 rounded">
                                {column.business_category}
                              </span>
                            )}
                          </div>
                          {column.sample_values && column.sample_values.length > 0 && (
                            <div className="mt-2">
                              <p className="text-xs text-gray-500 mb-1">Sample values:</p>
                              <p className="text-xs text-gray-600">
                                {column.sample_values.slice(0, 3).join(', ')}
                                {column.sample_values.length > 3 && '...'}
                              </p>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataSourcesTab;