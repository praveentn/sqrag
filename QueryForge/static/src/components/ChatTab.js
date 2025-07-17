// static/src/components/ChatTab.js
import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, 
  MessageSquare, 
  Database, 
  Code, 
  Play, 
  CheckCircle, 
  XCircle, 
  ThumbsUp, 
  ThumbsDown,
  Copy,
  Download,
  RefreshCw,
  Zap,
  Brain,
  Link,
  Eye,
  EyeOff
} from 'lucide-react';
import LoadingSpinner, { LoadingButton, InlineSpinner } from './LoadingSpinner';

const ChatTab = ({ projectId, apiUrl, onNotification }) => {
  const [messages, setMessages] = useState([]);
  const [currentQuery, setCurrentQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(null);
  const [sessionData, setSessionData] = useState({});
  const messagesEndRef = useRef(null);

  // Pipeline state
  const [entities, setEntities] = useState([]);
  const [mappings, setMappings] = useState([]);
  const [generatedSQL, setGeneratedSQL] = useState('');
  const [sqlResults, setSqlResults] = useState(null);
  const [showSQL, setShowSQL] = useState(true);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (projectId) {
      // Reset session when project changes
      resetSession();
    }
  }, [projectId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const resetSession = () => {
    setMessages([]);
    setEntities([]);
    setMappings([]);
    setGeneratedSQL('');
    setSqlResults(null);
    setSessionData({});
    setCurrentStep(null);
  };

  const addMessage = (type, content, metadata = {}) => {
    const message = {
      id: Date.now() + Math.random(),
      type,
      content,
      metadata,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, message]);
    return message;
  };

  const handleSubmitQuery = async () => {
    if (!currentQuery.trim() || loading) return;
    
    const query = currentQuery.trim();
    setCurrentQuery('');
    setLoading(true);
    
    // Add user message
    addMessage('user', query);
    
    try {
      await processNLQuery(query);
    } catch (error) {
      addMessage('error', `Error processing query: ${error.message}`);
      onNotification('Query processing failed: ' + error.message, 'error');
    } finally {
      setLoading(false);
      setCurrentStep(null);
    }
  };

  const processNLQuery = async (query) => {
    // Step 1: Extract entities
    setCurrentStep('entities');
    addMessage('system', 'Extracting entities from your query...', { step: 'entities' });

    const entityResponse = await fetch(`${apiUrl}/chat/entities`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, project_id: projectId })
    });

    if (!entityResponse.ok) throw new Error('Entity extraction failed');
    
    const entityData = await entityResponse.json();
    const extractedEntities = entityData.entities || [];
    setEntities(extractedEntities);

    addMessage('entities', 'Entities extracted successfully', {
      entities: extractedEntities,
      extractionTime: entityData.extraction_time_ms
    });

    if (extractedEntities.length === 0) {
      addMessage('warning', 'No entities found in your query. Please try rephrasing.');
      return;
    }

    // Step 2: Map entities to schema
    setCurrentStep('mapping');
    addMessage('system', 'Mapping entities to your data schema...', { step: 'mapping' });

    const mappingResponse = await fetch(`${apiUrl}/chat/mapping`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ entities: extractedEntities, project_id: projectId })
    });

    if (!mappingResponse.ok) throw new Error('Entity mapping failed');
    
    const mappingData = await mappingResponse.json();
    const entityMappings = mappingData.mappings || [];
    setMappings(entityMappings);

    addMessage('mappings', 'Entity mapping completed', {
      mappings: entityMappings,
      mappingTime: mappingData.mapping_time_ms
    });

    if (entityMappings.length === 0) {
      addMessage('warning', 'No matching tables or columns found for your entities.');
      return;
    }

    // Step 3: Generate SQL
    setCurrentStep('sql');
    addMessage('system', 'Generating SQL query...', { step: 'sql' });

    const sqlResponse = await fetch(`${apiUrl}/chat/sql`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        query, 
        entities: extractedEntities, 
        mappings: entityMappings, 
        project_id: projectId 
      })
    });

    if (!sqlResponse.ok) throw new Error('SQL generation failed');
    
    const sqlData = await sqlResponse.json();
    const sql = sqlData.sql_result?.sql || '';
    setGeneratedSQL(sql);

    if (!sql) {
      addMessage('error', 'Failed to generate SQL query. Please try rephrasing your question.');
      return;
    }

    addMessage('sql', 'SQL query generated successfully', {
      sql: sql,
      rationale: sqlData.sql_result.rationale,
      confidence: sqlData.sql_result.confidence,
      tablesUsed: sqlData.sql_result.tables_used,
      generationTime: sqlData.sql_result.generation_time_ms
    });

    // Show confirmation step
    addMessage('confirmation', 'Review the generated SQL query above. Would you like to execute it?', {
      sql: sql
    });
  };

  const executeSQL = async () => {
    if (!generatedSQL) return;

    setLoading(true);
    addMessage('system', 'Executing SQL query...', { step: 'execution' });

    try {
      const response = await fetch(`${apiUrl}/chat/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          sql_query: generatedSQL, 
          project_id: projectId 
        })
      });

      if (!response.ok) throw new Error('SQL execution failed');
      
      const data = await response.json();
      setSqlResults(data.results);

      if (data.results?.success) {
        addMessage('results', 'Query executed successfully', {
          results: data.results,
          sql: generatedSQL
        });
      } else {
        addMessage('error', `SQL execution failed: ${data.results?.error || 'Unknown error'}`);
      }
    } catch (error) {
      addMessage('error', `Error executing SQL: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async (rating, comment = '', feedbackType = 'overall') => {
    try {
      await fetch(`${apiUrl}/chat/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_id: projectId,
          nlq_text: messages.find(m => m.type === 'user')?.content || '',
          extracted_entities: entities,
          mapped_tables: mappings,
          generated_sql: generatedSQL,
          sql_results: sqlResults,
          rating,
          feedback_type: feedbackType,
          comment
        })
      });
      
      onNotification('Thank you for your feedback!', 'success');
    } catch (error) {
      onNotification('Failed to submit feedback', 'error');
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    onNotification('Copied to clipboard', 'success');
  };

  if (!projectId) {
    return (
      <div className="p-6 text-center">
        <MessageSquare className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No Project Selected</h3>
        <p className="text-gray-500">Please select a project to start chatting with your data.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full max-h-[800px]">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium text-gray-900">Natural Language to SQL</h3>
            <p className="text-sm text-gray-500">Ask questions about your data in plain English</p>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowSQL(!showSQL)}
              className="btn btn-secondary btn-sm"
            >
              {showSQL ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              {showSQL ? 'Hide' : 'Show'} SQL
            </button>
            <button
              onClick={resetSession}
              className="btn btn-secondary btn-sm"
            >
              <RefreshCw className="h-4 w-4 mr-1" />
              New Session
            </button>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <Brain className="mx-auto h-16 w-16 text-blue-500 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Start a conversation</h3>
            <p className="text-gray-500 mb-4">Try asking questions like:</p>
            <div className="space-y-2 text-sm text-gray-600 max-w-md mx-auto">
              <div className="bg-gray-50 p-2 rounded">"Show me all customers from New York"</div>
              <div className="bg-gray-50 p-2 rounded">"What are our top selling products this month?"</div>
              <div className="bg-gray-50 p-2 rounded">"Find orders with high values"</div>
              <div className="bg-gray-50 p-2 rounded">"List revenue by region"</div>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <MessageComponent
                key={message.id}
                message={message}
                onExecuteSQL={executeSQL}
                onFeedback={submitFeedback}
                onCopy={copyToClipboard}
                showSQL={showSQL}
                loading={loading}
              />
            ))}
            
            {loading && currentStep && (
              <div className="flex items-center space-x-2 text-gray-500">
                <InlineSpinner size="sm" color="gray" />
                <span className="text-sm">
                  {currentStep === 'entities' && 'Extracting entities...'}
                  {currentStep === 'mapping' && 'Mapping to schema...'}
                  {currentStep === 'sql' && 'Generating SQL...'}
                  {currentStep === 'execution' && 'Executing query...'}
                </span>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 border-t border-gray-200 p-4">
        <div className="flex space-x-2">
          <input
            type="text"
            value={currentQuery}
            onChange={(e) => setCurrentQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSubmitQuery()}
            placeholder="Ask a question about your data..."
            className="flex-1 form-input"
            disabled={loading}
          />
          <LoadingButton
            onClick={handleSubmitQuery}
            loading={loading}
            disabled={!currentQuery.trim()}
            className="btn-primary"
            loadingText=""
          >
            {loading ? <InlineSpinner size="sm" color="white" /> : <Send className="h-4 w-4" />}
          </LoadingButton>
        </div>
      </div>
    </div>
  );
};

// Message Component
const MessageComponent = ({ message, onExecuteSQL, onFeedback, onCopy, showSQL, loading }) => {
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackComment, setFeedbackComment] = useState('');

  const renderUserMessage = () => (
    <div className="flex justify-end">
      <div className="bg-blue-600 text-white rounded-lg px-4 py-2 max-w-lg">
        <p>{message.content}</p>
        <p className="text-xs text-blue-100 mt-1">
          {new Date(message.timestamp).toLocaleTimeString()}
        </p>
      </div>
    </div>
  );

  const renderSystemMessage = () => (
    <div className="flex items-center space-x-2 text-gray-500">
      <InlineSpinner size="sm" color="gray" />
      <span className="text-sm">{message.content}</span>
    </div>
  );

  const renderEntitiesMessage = () => (
    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
      <div className="flex items-center mb-2">
        <Zap className="h-5 w-5 text-green-600 mr-2" />
        <h4 className="font-medium text-green-900">Entities Extracted</h4>
        <span className="ml-auto text-xs text-green-600">
          {message.metadata.extractionTime}ms
        </span>
      </div>
      <div className="space-y-2">
        {message.metadata.entities.map((entity, index) => (
          <div key={index} className="flex items-center justify-between bg-white p-2 rounded border">
            <div>
              <span className="font-medium text-gray-900">{entity.entity}</span>
              <span className="ml-2 text-sm text-gray-500">({entity.type})</span>
            </div>
            <div className="flex items-center">
              <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                <div 
                  className="bg-green-600 h-2 rounded-full"
                  style={{ width: `${entity.confidence * 100}%` }}
                />
              </div>
              <span className="text-xs text-gray-600">{(entity.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderMappingsMessage = () => (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
      <div className="flex items-center mb-2">
        <Link className="h-5 w-5 text-blue-600 mr-2" />
        <h4 className="font-medium text-blue-900">Schema Mapping</h4>
        <span className="ml-auto text-xs text-blue-600">
          {message.metadata.mappingTime}ms
        </span>
      </div>
      <div className="space-y-3">
        {message.metadata.mappings.map((mapping, index) => (
          <div key={index} className="bg-white p-3 rounded border">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-gray-900">{mapping.entity}</span>
              <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                {mapping.entity_type}
              </span>
            </div>
            {mapping.best_match && (
              <div className="flex items-center text-sm text-gray-600">
                <span>→ </span>
                <span className="font-medium">{mapping.best_match.name}</span>
                {mapping.best_match.table && (
                  <span className="ml-1">({mapping.best_match.table})</span>
                )}
                <div className="ml-auto flex items-center">
                  <div className="w-12 bg-gray-200 rounded-full h-1 mr-2">
                    <div 
                      className="bg-blue-600 h-1 rounded-full"
                      style={{ width: `${mapping.best_match.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs">{(mapping.best_match.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  const renderSQLMessage = () => (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center">
          <Code className="h-5 w-5 text-gray-600 mr-2" />
          <h4 className="font-medium text-gray-900">Generated SQL</h4>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-500">
            Confidence: {(message.metadata.confidence * 100).toFixed(0)}%
          </span>
          <span className="text-xs text-gray-500">
            {message.metadata.generationTime}ms
          </span>
          <button
            onClick={() => onCopy(message.metadata.sql)}
            className="text-gray-400 hover:text-gray-600"
          >
            <Copy className="h-4 w-4" />
          </button>
        </div>
      </div>
      
      {showSQL && (
        <div className="bg-gray-900 text-gray-100 p-3 rounded font-mono text-sm overflow-x-auto mb-3">
          {message.metadata.sql}
        </div>
      )}
      
      {message.metadata.rationale && (
        <div className="text-sm text-gray-600 mb-3">
          <strong>Rationale:</strong> {message.metadata.rationale}
        </div>
      )}
      
      {message.metadata.tablesUsed && message.metadata.tablesUsed.length > 0 && (
        <div className="text-sm text-gray-600">
          <strong>Tables used:</strong> {message.metadata.tablesUsed.join(', ')}
        </div>
      )}
    </div>
  );

  const renderConfirmationMessage = () => (
    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
      <div className="flex items-center mb-3">
        <Play className="h-5 w-5 text-yellow-600 mr-2" />
        <h4 className="font-medium text-yellow-900">Ready to Execute</h4>
      </div>
      <p className="text-sm text-yellow-800 mb-4">{message.content}</p>
      <div className="flex space-x-2">
        <LoadingButton
          onClick={onExecuteSQL}
          loading={loading}
          className="btn-primary btn-sm"
          loadingText="Executing..."
        >
          <Play className="h-4 w-4 mr-1" />
          Execute SQL
        </LoadingButton>
        <button
          onClick={() => setShowFeedback(true)}
          className="btn btn-secondary btn-sm"
        >
          Modify Query
        </button>
      </div>
    </div>
  );

  const renderResultsMessage = () => (
    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center">
          <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
          <h4 className="font-medium text-green-900">Query Results</h4>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-green-600">
            {message.metadata.results.row_count} rows in {message.metadata.results.execution_time_ms}ms
          </span>
          <button
            onClick={() => onCopy(JSON.stringify(message.metadata.results.data, null, 2))}
            className="text-green-600 hover:text-green-800"
          >
            <Download className="h-4 w-4" />
          </button>
        </div>
      </div>
      
      {message.metadata.results.data && message.metadata.results.data.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-green-100">
                {message.metadata.results.columns.map((col, index) => (
                  <th key={index} className="px-3 py-2 text-left font-medium text-green-900">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white">
              {message.metadata.results.data.slice(0, 10).map((row, index) => (
                <tr key={index} className="border-t border-green-200">
                  {message.metadata.results.columns.map((col, colIndex) => (
                    <td key={colIndex} className="px-3 py-2 text-gray-900">
                      {row[col] !== null ? String(row[col]) : 'NULL'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {message.metadata.results.row_count > 10 && (
            <p className="text-xs text-green-600 mt-2">
              Showing first 10 of {message.metadata.results.row_count} rows
            </p>
          )}
        </div>
      ) : (
        <p className="text-sm text-green-700">No data returned</p>
      )}
      
      {/* Feedback Section */}
      <div className="mt-4 pt-3 border-t border-green-200">
        <p className="text-sm text-green-800 mb-2">Was this helpful?</p>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => onFeedback(5, 'Positive feedback')}
            className="flex items-center space-x-1 text-green-600 hover:text-green-800"
          >
            <ThumbsUp className="h-4 w-4" />
            <span className="text-sm">Yes</span>
          </button>
          <button
            onClick={() => setShowFeedback(true)}
            className="flex items-center space-x-1 text-red-600 hover:text-red-800"
          >
            <ThumbsDown className="h-4 w-4" />
            <span className="text-sm">No</span>
          </button>
        </div>
        
        {showFeedback && (
          <div className="mt-3 space-y-2">
            <textarea
              value={feedbackComment}
              onChange={(e) => setFeedbackComment(e.target.value)}
              placeholder="Tell us what went wrong or how we can improve..."
              className="w-full p-2 text-sm border border-gray-300 rounded"
              rows="2"
            />
            <div className="flex space-x-2">
              <button
                onClick={() => {
                  onFeedback(1, feedbackComment);
                  setShowFeedback(false);
                  setFeedbackComment('');
                }}
                className="btn btn-primary btn-sm"
              >
                Submit Feedback
              </button>
              <button
                onClick={() => setShowFeedback(false)}
                className="btn btn-secondary btn-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderErrorMessage = () => (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
      <div className="flex items-center">
        <XCircle className="h-5 w-5 text-red-600 mr-2" />
        <h4 className="font-medium text-red-900">Error</h4>
      </div>
      <p className="text-sm text-red-800 mt-1">{message.content}</p>
    </div>
  );

  const renderWarningMessage = () => (
    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
      <div className="flex items-center">
        <AlertTriangle className="h-5 w-5 text-yellow-600 mr-2" />
        <h4 className="font-medium text-yellow-900">Warning</h4>
      </div>
      <p className="text-sm text-yellow-800 mt-1">{message.content}</p>
    </div>
  );

  // Render based on message type
  switch (message.type) {
    case 'user':
      return renderUserMessage();
    case 'system':
      return renderSystemMessage();
    case 'entities':
      return renderEntitiesMessage();
    case 'mappings':
      return renderMappingsMessage();
    case 'sql':
      return renderSQLMessage();
    case 'confirmation':
      return renderConfirmationMessage();
    case 'results':
      return renderResultsMessage();
    case 'error':
      return renderErrorMessage();
    case 'warning':
      return renderWarningMessage();
    default:
      return (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <p className="text-sm text-gray-800">{message.content}</p>
        </div>
      );
  }
};

export default ChatTab;