// static/src/components/SearchTab.js
import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Filter, 
  BarChart3, 
  Clock, 
  Target,
  Download,
  RefreshCw,
  Settings,
  Eye,
  Zap,
  Database,
  FileText,
  BookOpen,
  Star,
  TrendingUp
} from 'lucide-react';
import LoadingSpinner, { LoadingButton } from './LoadingSpinner';

const SearchTab = ({ projectId, apiUrl, onNotification }) => {
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState('hybrid');
  const [selectedIndex, setSelectedIndex] = useState('');
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState([]);
  const [searchTime, setSearchTime] = useState(0);
  const [loading, setLoading] = useState(false);
  const [indexes, setIndexes] = useState([]);
  const [searchHistory, setSearchHistory] = useState([]);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    if (projectId) {
      fetchIndexes();
      loadSearchHistory();
    }
  }, [projectId]);

  const fetchIndexes = async () => {
    try {
      const response = await fetch(`${apiUrl}/projects/${projectId}/indexes`);
      if (response.ok) {
        const data = await response.json();
        setIndexes(data.indexes?.filter(idx => idx.status === 'ready') || []);
      }
    } catch (error) {
      console.error('Error fetching indexes:', error);
    }
  };

  const loadSearchHistory = () => {
    const history = localStorage.getItem(`search_history_${projectId}`);
    if (history) {
      setSearchHistory(JSON.parse(history));
    }
  };

  const saveSearchToHistory = (searchData) => {
    const newHistory = [searchData, ...searchHistory.slice(0, 9)]; // Keep last 10
    setSearchHistory(newHistory);
    localStorage.setItem(`search_history_${projectId}`, JSON.stringify(newHistory));
  };

  const performSearch = async () => {
    if (!query.trim()) {
      onNotification('Please enter a search query', 'warning');
      return;
    }

    setLoading(true);
    try {
      const searchParams = {
        query: query.trim(),
        search_type: searchType,
        top_k: topK
      };

      if (selectedIndex) {
        searchParams.index_id = selectedIndex;
      } else {
        searchParams.project_id = projectId;
      }

      const response = await fetch(`${apiUrl}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(searchParams)
      });

      if (!response.ok) throw new Error('Search failed');

      const data = await response.json();
      setResults(data.results || []);
      setSearchTime(data.search_time_ms || 0);

      // Save to history
      saveSearchToHistory({
        query: query.trim(),
        searchType,
        selectedIndex,
        timestamp: new Date().toISOString(),
        resultCount: data.total_results,
        searchTime: data.search_time_ms
      });

      onNotification(`Found ${data.total_results} results in ${data.search_time_ms}ms`, 'success');
    } catch (error) {
      onNotification('Search failed: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };

  const handleHistoryClick = (historyItem) => {
    setQuery(historyItem.query);
    setSearchType(historyItem.searchType);
    setSelectedIndex(historyItem.selectedIndex);
  };

  const clearResults = () => {
    setResults([]);
    setQuery('');
    setSearchTime(0);
  };

  const exportResults = () => {
    if (results.length === 0) {
      onNotification('No results to export', 'warning');
      return;
    }

    const exportData = {
      query,
      searchType,
      searchTime,
      timestamp: new Date().toISOString(),
      results: results.map(result => ({
        rank: result.rank,
        score: result.score,
        object_type: result.object_type,
        name: result.table_name || result.column_name || result.term || 'Unknown',
        text: result.object_text,
        table: result.table_name,
        search_method: result.search_method
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `search_results_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    onNotification('Results exported successfully', 'success');
  };

  if (!projectId) {
    return (
      <div className="p-6 text-center">
        <Search className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No Project Selected</h3>
        <p className="text-gray-500">Please select a project to start searching your data.</p>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-lg font-medium text-gray-900">Search Playground</h3>
          <p className="text-sm text-gray-500">Test different search methods across your data</p>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="btn btn-secondary"
          >
            <Settings className="h-4 w-4 mr-2" />
            {showAdvanced ? 'Hide' : 'Show'} Advanced
          </button>
          <button onClick={clearResults} className="btn btn-secondary">
            <RefreshCw className="h-4 w-4 mr-2" />
            Clear
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Search Controls */}
        <div className="lg:col-span-1 space-y-6">
          {/* Search Input */}
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <h4 className="font-medium text-gray-900 mb-3">Search Query</h4>
            <div className="space-y-3">
              <textarea
                value={query}
                onChange={handleQueryChange}
                placeholder="Enter your search query..."
                className="form-textarea"
                rows="3"
              />
              <LoadingButton
                onClick={performSearch}
                loading={loading}
                disabled={!query.trim()}
                className="btn-primary w-full"
                loadingText="Searching..."
              >
                <Search className="h-4 w-4 mr-2" />
                Search
              </LoadingButton>
            </div>
          </div>

          {/* Search Configuration */}
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <h4 className="font-medium text-gray-900 mb-3">Configuration</h4>
            <div className="space-y-4">
              <div>
                <label className="form-label">Search Type</label>
                <select
                  value={searchType}
                  onChange={(e) => setSearchType(e.target.value)}
                  className="form-select"
                >
                  <option value="hybrid">Hybrid (Semantic + Lexical)</option>
                  <option value="semantic">Semantic (Vector)</option>
                  <option value="keyword">Keyword (TF-IDF)</option>
                  <option value="fuzzy">Fuzzy Matching</option>
                </select>
              </div>

              <div>
                <label className="form-label">Target Index</label>
                <select
                  value={selectedIndex}
                  onChange={(e) => setSelectedIndex(e.target.value)}
                  className="form-select"
                >
                  <option value="">All Indexes</option>
                  {indexes.map(index => (
                    <option key={index.id} value={index.id}>
                      {index.name} ({index.index_type})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="form-label">Max Results</label>
                <select
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="form-select"
                >
                  <option value={5}>5 results</option>
                  <option value={10}>10 results</option>
                  <option value={20}>20 results</option>
                  <option value={50}>50 results</option>
                </select>
              </div>
            </div>
          </div>

          {/* Advanced Options */}
          {showAdvanced && (
            <div className="bg-white p-4 rounded-lg border border-gray-200">
              <h4 className="font-medium text-gray-900 mb-3">Advanced Options</h4>
              <div className="space-y-4">
                <div>
                  <label className="form-label">Score Threshold</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    defaultValue="0.0"
                    className="form-input"
                    placeholder="0.0 - 1.0"
                  />
                </div>
                
                <div>
                  <label className="form-label">Object Type Filter</label>
                  <select className="form-select">
                    <option value="">All Types</option>
                    <option value="tables">Tables Only</option>
                    <option value="columns">Columns Only</option>
                    <option value="dictionary">Dictionary Only</option>
                  </select>
                </div>

                <div>
                  <label className="flex items-center">
                    <input type="checkbox" className="rounded border-gray-300" />
                    <span className="ml-2 text-sm text-gray-700">Include metadata</span>
                  </label>
                </div>

                <div>
                  <label className="flex items-center">
                    <input type="checkbox" className="rounded border-gray-300" />
                    <span className="ml-2 text-sm text-gray-700">Boost exact matches</span>
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Search History */}
          {searchHistory.length > 0 && (
            <div className="bg-white p-4 rounded-lg border border-gray-200">
              <h4 className="font-medium text-gray-900 mb-3">Recent Searches</h4>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {searchHistory.map((item, index) => (
                  <button
                    key={index}
                    onClick={() => handleHistoryClick(item)}
                    className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded transition-colors"
                  >
                    <div className="font-medium text-gray-900 truncate">{item.query}</div>
                    <div className="text-xs text-gray-500">
                      {item.resultCount} results • {item.searchTime}ms
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Results Area */}
        <div className="lg:col-span-3">
          {/* Results Header */}
          {results.length > 0 && (
            <div className="bg-white p-4 rounded-lg border border-gray-200 mb-4">
              <div className="flex justify-between items-center">
                <div>
                  <h4 className="font-medium text-gray-900">Search Results</h4>
                  <p className="text-sm text-gray-500">
                    {results.length} results for "{query}" in {searchTime}ms
                  </p>
                </div>
                <div className="flex space-x-2">
                  <button onClick={exportResults} className="btn btn-secondary btn-sm">
                    <Download className="h-4 w-4 mr-1" />
                    Export
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Results List */}
          {loading ? (
            <div className="bg-white p-8 rounded-lg border border-gray-200">
              <LoadingSpinner size="lg" message="Searching your data..." />
            </div>
          ) : results.length > 0 ? (
            <div className="space-y-4">
              {results.map((result, index) => (
                <SearchResultCard key={index} result={result} query={query} />
              ))}
            </div>
          ) : query ? (
            <div className="bg-white p-8 rounded-lg border border-gray-200 text-center">
              <Search className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Results Found</h3>
              <p className="text-gray-500 mb-4">
                No matches found for "{query}". Try different keywords or search methods.
              </p>
              <div className="space-y-2 text-sm text-gray-600">
                <p>💡 Try these search tips:</p>
                <ul className="text-left max-w-md mx-auto space-y-1">
                  <li>• Use different keywords or synonyms</li>
                  <li>• Try a broader search term</li>
                  <li>• Switch to fuzzy matching for typos</li>
                  <li>• Use semantic search for conceptual matches</li>
                </ul>
              </div>
            </div>
          ) : (
            <div className="bg-white p-8 rounded-lg border border-gray-200 text-center">
              <Target className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Ready to Search</h3>
              <p className="text-gray-500 mb-6">
                Enter a query to test different search methods across your data.
              </p>
              
              {/* Example Queries */}
              <div className="max-w-2xl mx-auto">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Try these example queries:</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {[
                    "customer information",
                    "sales revenue data", 
                    "user demographics",
                    "product inventory",
                    "financial metrics",
                    "order details"
                  ].map((example) => (
                    <button
                      key={example}
                      onClick={() => setQuery(example)}
                      className="p-3 text-sm bg-gray-50 hover:bg-gray-100 rounded border text-gray-700 transition-colors"
                    >
                      "{example}"
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Search Result Card Component
const SearchResultCard = ({ result, query }) => {
  const getObjectIcon = (type) => {
    switch (type) {
      case 'table':
        return <Database className="h-5 w-5 text-blue-500" />;
      case 'column':
        return <BarChart3 className="h-5 w-5 text-green-500" />;
      case 'dictionary_entry':
        return <BookOpen className="h-5 w-5 text-purple-500" />;
      default:
        return <FileText className="h-5 w-5 text-gray-500" />;
    }
  };

  const getSearchMethodBadge = (method) => {
    const badges = {
      semantic: { color: 'bg-blue-100 text-blue-800', icon: <Zap className="h-3 w-3" /> },
      lexical: { color: 'bg-green-100 text-green-800', icon: <Search className="h-3 w-3" /> },
      fuzzy: { color: 'bg-yellow-100 text-yellow-800', icon: <Target className="h-3 w-3" /> },
      hybrid: { color: 'bg-purple-100 text-purple-800', icon: <TrendingUp className="h-3 w-3" /> }
    };

    const badge = badges[method] || badges.hybrid;
    
    return (
      <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${badge.color}`}>
        {badge.icon}
        <span className="ml-1">{method}</span>
      </span>
    );
  };

  const highlightText = (text, query) => {
    if (!query || !text) return text;
    
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = text.split(regex);
    
    return parts.map((part, index) => 
      regex.test(part) ? (
        <mark key={index} className="bg-yellow-200 px-1 rounded">{part}</mark>
      ) : (
        part
      )
    );
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            {getObjectIcon(result.object_type)}
            <span className="font-medium text-gray-900">
              #{result.rank}
            </span>
          </div>
          
          <div className="flex items-center space-x-2">
            <Star className="h-4 w-4 text-yellow-500" />
            <span className="text-sm font-medium text-gray-700">
              {(result.score * 100).toFixed(1)}%
            </span>
          </div>
          
          {getSearchMethodBadge(result.search_method)}
        </div>
        
        <button className="text-gray-400 hover:text-gray-600">
          <Eye className="h-5 w-5" />
        </button>
      </div>

      <div className="space-y-2">
        {/* Title/Name */}
        <h4 className="text-lg font-medium text-gray-900">
          {result.table_name || result.column_name || result.term || 'Unknown'}
        </h4>

        {/* Context Information */}
        <div className="flex items-center space-x-4 text-sm text-gray-500">
          <span className="capitalize">{result.object_type.replace('_', ' ')}</span>
          {result.table_name && result.object_type === 'column' && (
            <>
              <span>•</span>
              <span>Table: {result.table_name}</span>
            </>
          )}
          {result.data_type && (
            <>
              <span>•</span>
              <span>Type: {result.data_type}</span>
            </>
          )}
          {result.business_category && (
            <>
              <span>•</span>
              <span>Category: {result.business_category}</span>
            </>
          )}
        </div>

        {/* Object Text */}
        {result.object_text && (
          <div className="text-gray-700">
            <p className="line-clamp-3">
              {highlightText(result.object_text, query)}
            </p>
          </div>
        )}

        {/* Additional Context */}
        {(result.column_description || result.table_description || result.definition) && (
          <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded">
            <p className="line-clamp-2">
              {result.column_description || result.table_description || result.definition}
            </p>
          </div>
        )}

        {/* Metadata */}
        <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-100">
          <div className="flex items-center space-x-4">
            {result.row_count && (
              <span>{result.row_count.toLocaleString()} rows</span>
            )}
            {result.sample_values && result.sample_values.length > 0 && (
              <span>Examples: {result.sample_values.slice(0, 3).join(', ')}</span>
            )}
            {result.synonyms && result.synonyms.length > 0 && (
              <span>Synonyms: {result.synonyms.slice(0, 2).join(', ')}</span>
            )}
          </div>
          
          {result.index_name && (
            <span className="text-gray-400">
              Index: {result.index_name}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default SearchTab;