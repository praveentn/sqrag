// static/src/components/AdminTab.js
import React, { useState, useEffect } from 'react';
import { 
  Settings, 
  Database, 
  Monitor, 
  Users, 
  Shield, 
  Activity,
  Code,
  Download,
  Upload,
  RefreshCw,
  Trash2,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  BarChart3,
  Cpu,
  HardDrive,
  MemoryStick, // Replaced Memory with MemoryStick
  Network,
  Server,
  Eye, X
} from 'lucide-react';
import LoadingSpinner, { LoadingButton, SkeletonLoader } from './LoadingSpinner';

const AdminTab = ({ apiUrl, onNotification }) => {
  const [activeSection, setActiveSection] = useState('overview');
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  const sections = [
    { id: 'overview', name: 'System Overview', icon: Monitor },
    { id: 'database', name: 'Database Browser', icon: Database },
    { id: 'queries', name: 'SQL Executor', icon: Code },
    { id: 'monitoring', name: 'System Monitoring', icon: Activity },
    { id: 'users', name: 'User Management', icon: Users },
    { id: 'maintenance', name: 'Maintenance', icon: Settings }
  ];

  useEffect(() => {
    fetchSystemHealth();
  }, []);

  const fetchSystemHealth = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiUrl}/admin/health`);
      if (response.ok) {
        const data = await response.json();
        setSystemHealth(data.health);
      }
    } catch (error) {
      onNotification('Error fetching system health: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const renderSection = () => {
    switch (activeSection) {
      case 'overview':
        return <SystemOverview systemHealth={systemHealth} onRefresh={fetchSystemHealth} />;
      case 'database':
        return <DatabaseBrowser apiUrl={apiUrl} onNotification={onNotification} />;
      case 'queries':
        return <SQLExecutor apiUrl={apiUrl} onNotification={onNotification} />;
      case 'monitoring':
        return <SystemMonitoring systemHealth={systemHealth} onRefresh={fetchSystemHealth} />;
      case 'users':
        return <UserManagement apiUrl={apiUrl} onNotification={onNotification} />;
      case 'maintenance':
        return <MaintenancePanel apiUrl={apiUrl} onNotification={onNotification} />;
      default:
        return <SystemOverview systemHealth={systemHealth} onRefresh={fetchSystemHealth} />;
    }
  };

  if (loading) {
    return (
      <div className="p-6">
        <SkeletonLoader lines={8} />
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-lg font-medium text-gray-900">Admin Control Panel</h3>
          <p className="text-sm text-gray-500">System administration and monitoring</p>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
            systemHealth?.status === 'healthy' 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            {systemHealth?.status === 'healthy' ? (
              <CheckCircle className="h-3 w-3 mr-1" />
            ) : (
              <XCircle className="h-3 w-3 mr-1" />
            )}
            {systemHealth?.status || 'Unknown'}
          </div>
          <LoadingButton
            onClick={fetchSystemHealth}
            loading={loading}
            className="btn btn-secondary btn-sm"
            loadingText=""
          >
            <RefreshCw className="h-4 w-4" />
          </LoadingButton>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Sidebar Navigation */}
        <div className="lg:col-span-1">
          <nav className="space-y-1">
            {sections.map((section) => {
              const Icon = section.icon;
              const isActive = activeSection === section.id;
              
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`
                    w-full group flex items-center px-3 py-2 text-sm font-medium rounded-md
                    ${isActive 
                      ? 'bg-blue-50 text-blue-700 border-r-2 border-blue-700' 
                      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                    }
                  `}
                >
                  <Icon className={`mr-3 h-5 w-5 ${isActive ? 'text-blue-500' : 'text-gray-400'}`} />
                  {section.name}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-4">
          {renderSection()}
        </div>
      </div>
    </div>
  );
};

// System Overview Component
const SystemOverview = ({ systemHealth, onRefresh }) => {
  const getServiceStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      case 'error':
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Clock className="h-5 w-5 text-gray-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* System Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center">
            <Cpu className="h-8 w-8 text-blue-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">CPU Usage</p>
              <p className="text-2xl font-semibold text-gray-900">
                {systemHealth?.system?.cpu?.usage_percent?.toFixed(1) || 'N/A'}%
              </p>
              <p className="text-sm text-gray-500">
                {systemHealth?.system?.cpu?.count || 0} cores
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center">
            <MemoryStick className="h-8 w-8 text-green-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Memory Usage</p>
              <p className="text-2xl font-semibold text-gray-900">
                {systemHealth?.system?.memory?.usage_percent?.toFixed(1) || 'N/A'}%
              </p>
              <p className="text-sm text-gray-500">
                {systemHealth?.system?.memory?.available_gb?.toFixed(1) || 0} GB available
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center">
            <HardDrive className="h-8 w-8 text-purple-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Disk Usage</p>
              <p className="text-2xl font-semibold text-gray-900">
                {systemHealth?.system?.disk?.usage_percent?.toFixed(1) || 'N/A'}%
              </p>
              <p className="text-sm text-gray-500">
                {systemHealth?.system?.disk?.free_gb?.toFixed(1) || 0} GB free
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Services Status */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h4 className="text-lg font-medium text-gray-900">Service Status</h4>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {systemHealth?.services && Object.entries(systemHealth.services).map(([service, status]) => (
              <div key={service} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center">
                  {getServiceStatusIcon(status.status)}
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-900 capitalize">
                      {service.replace('_', ' ')}
                    </p>
                    <p className="text-sm text-gray-500">{status.message}</p>
                  </div>
                </div>
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  status.status === 'healthy' 
                    ? 'bg-green-100 text-green-800' 
                    : status.status === 'warning'
                    ? 'bg-yellow-100 text-yellow-800'
                    : 'bg-red-100 text-red-800'
                }`}>
                  {status.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Database Status */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h4 className="text-lg font-medium text-gray-900">Database Status</h4>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {systemHealth?.database?.records && Object.entries(systemHealth.database.records).map(([table, count]) => (
              <div key={table} className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-semibold text-gray-900">{count.toLocaleString()}</p>
                <p className="text-sm text-gray-500 capitalize">{table}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Application Metrics */}
      {systemHealth?.application && (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200">
            <h4 className="text-lg font-medium text-gray-900">Application Metrics</h4>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <p className="text-2xl font-semibold text-blue-900">
                  {systemHealth.application.embeddings_count || 0}
                </p>
                <p className="text-sm text-blue-600">Embeddings</p>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <p className="text-2xl font-semibold text-green-900">
                  {systemHealth.application.indexes_count || 0}
                </p>
                <p className="text-sm text-green-600">Indexes</p>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <p className="text-2xl font-semibold text-purple-900">
                  {systemHealth.application.activity_24h?.searches || 0}
                </p>
                <p className="text-sm text-purple-600">Searches (24h)</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Database Browser Component
const DatabaseBrowser = ({ apiUrl, onNotification }) => {
  const [tables, setTables] = useState([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [pagination, setPagination] = useState({});
  const [selectedTable, setSelectedTable] = useState(null);

  useEffect(() => {
    fetchTables();
  }, [currentPage]);

  const fetchTables = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiUrl}/admin/tables?page=${currentPage}&per_page=10`);
      if (response.ok) {
        const data = await response.json();
        setTables(data.tables || []);
        setPagination(data.pagination || {});
      }
    } catch (error) {
      onNotification('Error fetching tables: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <LoadingSpinner size="lg" message="Loading database tables..." />;
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h4 className="text-lg font-medium text-gray-900">Database Tables</h4>
        <button onClick={fetchTables} className="btn btn-secondary btn-sm">
          <RefreshCw className="h-4 w-4 mr-1" />
          Refresh
        </button>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Table Name
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Project
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Rows
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Columns
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Source
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {tables.map((table) => (
              <tr key={table.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  {table.name}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {table.project_name}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {table.row_count?.toLocaleString() || 0}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {table.column_count || 0}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {table.source_name} ({table.source_type})
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <button
                    onClick={() => setSelectedTable(table)}
                    className="text-blue-600 hover:text-blue-900"
                  >
                    <Eye className="h-4 w-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {pagination.pages > 1 && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-700">
            Showing page {pagination.page} of {pagination.pages} ({pagination.total} total tables)
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
              disabled={!pagination.has_prev}
              className="btn btn-secondary btn-sm"
            >
              Previous
            </button>
            <button
              onClick={() => setCurrentPage(Math.min(pagination.pages, currentPage + 1))}
              disabled={!pagination.has_next}
              className="btn btn-secondary btn-sm"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// SQL Executor Component
const SQLExecutor = ({ apiUrl, onNotification }) => {
  const [sql, setSql] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const executeSQL = async () => {
    if (!sql.trim()) {
      onNotification('Please enter a SQL query', 'warning');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${apiUrl}/admin/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sql: sql.trim() })
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data.results);
        
        if (data.results.success) {
          onNotification(`Query executed successfully in ${data.results.execution_time_seconds}s`, 'success');
          
          // Add to history
          setHistory(prev => [{
            sql: sql.trim(),
            timestamp: new Date().toISOString(),
            success: true,
            rowCount: data.results.row_count
          }, ...prev.slice(0, 9)]);
        } else {
          onNotification('Query failed: ' + data.results.error, 'error');
        }
      }
    } catch (error) {
      onNotification('Error executing query: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h4 className="text-lg font-medium text-gray-900 mb-4">SQL Query Executor</h4>
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 mb-4">
          <div className="flex">
            <AlertTriangle className="h-5 w-5 text-yellow-400" />
            <div className="ml-3">
              <p className="text-sm text-yellow-800">
                <strong>Warning:</strong> Only SELECT queries are allowed. 
                Destructive operations are blocked for safety.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* SQL Input */}
        <div className="lg:col-span-2 space-y-4">
          <div>
            <label className="form-label">SQL Query</label>
            <textarea
              value={sql}
              onChange={(e) => setSql(e.target.value)}
              className="form-textarea font-mono"
              rows="10"
              placeholder="SELECT * FROM tables LIMIT 10;"
            />
          </div>
          
          <div className="flex space-x-2">
            <LoadingButton
              onClick={executeSQL}
              loading={loading}
              disabled={!sql.trim()}
              className="btn-primary"
              loadingText="Executing..."
            >
              <Code className="h-4 w-4 mr-2" />
              Execute Query
            </LoadingButton>
            <button
              onClick={() => setSql('')}
              className="btn btn-secondary"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Query History */}
        <div>
          <h5 className="font-medium text-gray-900 mb-3">Query History</h5>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {history.map((item, index) => (
              <button
                key={index}
                onClick={() => setSql(item.sql)}
                className="w-full text-left p-3 bg-gray-50 hover:bg-gray-100 rounded text-sm transition-colors"
              >
                <div className="font-mono text-gray-900 truncate">{item.sql}</div>
                <div className="text-xs text-gray-500 mt-1">
                  {item.success ? (
                    <span className="text-green-600">{item.rowCount} rows</span>
                  ) : (
                    <span className="text-red-600">Failed</span>
                  )}
                  {' • '}
                  {new Date(item.timestamp).toLocaleTimeString()}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Results */}
      {results && (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200">
            <h5 className="font-medium text-gray-900">Query Results</h5>
          </div>
          <div className="p-6">
            {results.success ? (
              results.data && results.data.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="bg-gray-50">
                        {results.columns.map((col, index) => (
                          <th key={index} className="px-3 py-2 text-left font-medium text-gray-900">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.data.map((row, index) => (
                        <tr key={index} className="border-t border-gray-200">
                          {results.columns.map((col, colIndex) => (
                            <td key={colIndex} className="px-3 py-2 text-gray-900">
                              {row[col] !== null ? String(row[col]) : 'NULL'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-gray-500">No data returned</p>
              )
            ) : (
              <div className="text-red-600">
                <p className="font-medium">Error:</p>
                <p className="text-sm">{results.error}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// System Monitoring Component  
const SystemMonitoring = ({ systemHealth, onRefresh }) => {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h4 className="text-lg font-medium text-gray-900">System Monitoring</h4>
        <button onClick={onRefresh} className="btn btn-secondary btn-sm">
          <RefreshCw className="h-4 w-4 mr-1" />
          Refresh
        </button>
      </div>

      {/* Real-time Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">CPU Usage</p>
              <p className="text-2xl font-semibold text-gray-900">
                {systemHealth?.system?.cpu?.usage_percent?.toFixed(1) || 0}%
              </p>
            </div>
            <Cpu className="h-8 w-8 text-blue-500" />
          </div>
          <div className="mt-4 w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${systemHealth?.system?.cpu?.usage_percent || 0}%` }}
            />
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Memory Usage</p>
              <p className="text-2xl font-semibold text-gray-900">
                {systemHealth?.system?.memory?.usage_percent?.toFixed(1) || 0}%
              </p>
            </div>
            <MemoryStick className="h-8 w-8 text-green-500" />
          </div>
          <div className="mt-4 w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-green-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${systemHealth?.system?.memory?.usage_percent || 0}%` }}
            />
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Disk Usage</p>
              <p className="text-2xl font-semibold text-gray-900">
                {systemHealth?.system?.disk?.usage_percent?.toFixed(1) || 0}%
              </p>
            </div>
            <HardDrive className="h-8 w-8 text-purple-500" />
          </div>
          <div className="mt-4 w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-purple-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${systemHealth?.system?.disk?.usage_percent || 0}%` }}
            />
          </div>
        </div>
      </div>

      {/* Placeholder for charts */}
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h5 className="font-medium text-gray-900 mb-4">Performance Charts</h5>
        <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-center text-gray-500">
            <BarChart3 className="h-12 w-12 mx-auto mb-2" />
            <p>Performance charts would be displayed here</p>
            <p className="text-sm">Integration with monitoring tools like Prometheus/Grafana</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// User Management Component
const UserManagement = ({ apiUrl, onNotification }) => {
  return (
    <div className="space-y-6">
      <h4 className="text-lg font-medium text-gray-900">User Management</h4>
      
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <div className="text-center text-gray-500">
          <Users className="h-12 w-12 mx-auto mb-4" />
          <h5 className="text-lg font-medium text-gray-900 mb-2">User Management</h5>
          <p className="text-gray-600">
            User management features would be implemented here including:
          </p>
          <ul className="text-left mt-4 space-y-2 max-w-md mx-auto">
            <li>• User authentication and authorization</li>
            <li>• Role-based access control (RBAC)</li>
            <li>• User registration and profile management</li>
            <li>• Session management</li>
            <li>• Audit logs</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

// Maintenance Panel Component
const MaintenancePanel = ({ apiUrl, onNotification }) => {
  return (
    <div className="space-y-6">
      <h4 className="text-lg font-medium text-gray-900">System Maintenance</h4>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h5 className="font-medium text-gray-900 mb-4">Database Maintenance</h5>
          <div className="space-y-3">
            <button className="btn btn-secondary w-full">
              <Database className="h-4 w-4 mr-2" />
              Optimize Database
            </button>
            <button className="btn btn-secondary w-full">
              <Download className="h-4 w-4 mr-2" />
              Backup Database
            </button>
            <button className="btn btn-secondary w-full">
              <Upload className="h-4 w-4 mr-2" />
              Restore Database
            </button>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h5 className="font-medium text-gray-900 mb-4">Index Maintenance</h5>
          <div className="space-y-3">
            <button className="btn btn-secondary w-full">
              <RefreshCw className="h-4 w-4 mr-2" />
              Rebuild All Indexes
            </button>
            <button className="btn btn-secondary w-full">
              <Trash2 className="h-4 w-4 mr-2" />
              Clean Unused Indexes
            </button>
            <button className="btn btn-secondary w-full">
              <Activity className="h-4 w-4 mr-2" />
              Index Performance Report
            </button>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h5 className="font-medium text-gray-900 mb-4">Data Cleanup</h5>
          <div className="space-y-3">
            <button className="btn btn-secondary w-full">
              <Trash2 className="h-4 w-4 mr-2" />
              Clean Old Logs
            </button>
            <button className="btn btn-secondary w-full">
              <Trash2 className="h-4 w-4 mr-2" />
              Remove Orphaned Data
            </button>
            <button className="btn btn-secondary w-full">
              <Download className="h-4 w-4 mr-2" />
              Export Data
            </button>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h5 className="font-medium text-gray-900 mb-4">System Actions</h5>
          <div className="space-y-3">
            <button className="btn btn-secondary w-full">
              <RefreshCw className="h-4 w-4 mr-2" />
              Restart Services
            </button>
            <button className="btn btn-secondary w-full">
              <Settings className="h-4 w-4 mr-2" />
              Update Configuration
            </button>
            <button className="btn btn-danger w-full">
              <AlertTriangle className="h-4 w-4 mr-2" />
              Emergency Shutdown
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminTab;