// static/src/components/ChatTab.js
import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, 
  Send, 
  Bot, 
  User, 
  Database, 
  Search,
  Code,
  CheckCircle,
  AlertTriangle,
  Clock,
  Trash2,
  Download,
  Copy,
  Eye,
  EyeOff,
  Play,
  Pause,
  RotateCcw
} from 'lucide-react';
import LoadingSpinner, { LoadingButton } from './LoadingSpinner';

const ChatTab = ({ projectId, apiUrl, onNotification }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatState, setChatState] = useState('idle'); // idle, extracting, mapping, generating, executing
  const [currentContext, setCurrentContext] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (projectId) {
      // Load chat history for this project
      loadChatHistory();
    }
  }, [projectId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadChatHistory = async () => {
    try {
      const response = await fetch(`${apiUrl}/projects/${projectId}/chat/history`);
      if (response.ok) {
        const data = await response.json();
        setMessages(data.messages || []);
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || loading || !projectId) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);
    setChatState('extracting');

    try {
      // Step 1: Extract entities
      const entityResponse = await fetch(`${apiUrl}/chat/entities`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userMessage.content,
          project_id: projectId
        })
      });

      if (!entityResponse.ok) {
        throw new Error('Failed to extract entities');
      }

      const entityData = await entityResponse.json();
      
      // Add entity extraction message
      const entityMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        subtype: 'entities',
        content: 'I found these entities in your query:',
        entities: entityData.entities,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, entityMessage]);
      setChatState('mapping');

      // Step 2: Map entities to schema
      const mappingResponse = await fetch(`${apiUrl}/chat/mapping`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          entities: entityData.entities,
          project_id: projectId
        })
      });

      if (!mappingResponse.ok) {
        throw new Error('Failed to map entities');
      }

      const mappingData = await mappingResponse.json();
      
      // Add mapping message
      const mappingMessage = {
        id: Date.now() + 2,
        type: 'assistant',
        subtype: 'mapping',
        content: 'I mapped these entities to your database schema:',
        mappings: mappingData.mappings,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, mappingMessage]);
      setChatState('generating');

      // Step 3: Generate SQL
      const sqlResponse = await fetch(`${apiUrl}/chat/sql`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userMessage.content,
          entities: entityData.entities,
          mappings: mappingData.mappings,
          project_id: projectId
        })
      });

      if (!sqlResponse.ok) {
        throw new Error('Failed to generate SQL');
      }

      const sqlData = await sqlResponse.json();
      
      // Add SQL generation message
      const sqlMessage = {
        id: Date.now() + 3,
        type: 'assistant',
        subtype: 'sql',
        content: 'Here\'s the SQL query I generated:',
        sql: sqlData.sql_result.sql,
        explanation: sqlData.sql_result.explanation,
        confidence: sqlData.sql_result.confidence,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, sqlMessage]);
      
      // Store context for potential execution
      setCurrentContext({
        query: userMessage.content,
        entities: entityData.entities,
        mappings: mappingData.mappings,
        sql: sqlData.sql_result.sql
      });

    } catch (error) {
      onNotification('Error processing query: ' + error.message, 'error');
      const errorMessage = {
        id: Date.now() + 999,
        type: 'assistant',
        subtype: 'error',
        content: 'Sorry, I encountered an error processing your query. Please try again.',
        error: error.message,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
      setChatState('idle');
    }
  };

  const executeSQL = async (sql) => {
    if (!sql || !projectId) return;

    setLoading(true);
    setChatState('executing');

    try {
      const response = await fetch(`${apiUrl}/chat/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sql: sql,
          project_id: projectId
        })
      });

      if (!response.ok) {
        throw new Error('Failed to execute SQL');
      }

      const data = await response.json();
      
      // Add execution results message
      const resultMessage = {
        id: Date.now(),
        type: 'assistant',
        subtype: 'results',
        content: 'Query executed successfully:',
        results: data.results,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, resultMessage]);

    } catch (error) {
      onNotification('Error executing SQL: ' + error.message, 'error');
      const errorMessage = {
        id: Date.now(),
        type: 'assistant',
        subtype: 'error',
        content: 'Failed to execute the SQL query.',
        error: error.message,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
      setChatState('idle');
    }
  };

  const generateAnswer = async (sql, results) => {
    if (!sql || !results || !projectId) return;

    setLoading(true);

    try {
      const response = await fetch(`${apiUrl}/chat/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: currentContext?.query,
          sql: sql,
          results: results,
          project_id: projectId
        })
      });

      if (!response.ok) {
        throw new Error('Failed to generate answer');
      }

      const data = await response.json();
      
      // Add natural language answer
      const answerMessage = {
        id: Date.now(),
        type: 'assistant',
        subtype: 'answer',
        content: data.answer,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, answerMessage]);

    } catch (error) {
      onNotification('Error generating answer: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setCurrentContext(null);
    setChatState('idle');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!projectId) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <MessageSquare className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Project Selected</h3>
          <p className="text-gray-500">Please select a project to start chatting with your data.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-gray-200 bg-white">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium text-gray-900">Natural Language Query</h3>
            <p className="text-sm text-gray-500">Ask questions about your data in plain English</p>
          </div>
          <div className="flex items-center space-x-2">
            {chatState !== 'idle' && (
              <div className="flex items-center text-sm text-blue-600">
                <LoadingSpinner size="sm" className="mr-2" />
                <span className="capitalize">{chatState}...</span>
              </div>
            )}
            <button
              onClick={clearChat}
              className="btn btn-secondary btn-sm"
              disabled={loading}
            >
              <Trash2 className="h-4 w-4 mr-1" />
              Clear
            </button>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <Bot className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <h4 className="text-lg font-medium text-gray-900 mb-2">Start a Conversation</h4>
            <p className="text-gray-500 mb-6">
              Ask questions about your data. For example:
            </p>
            <div className="space-y-2 max-w-md mx-auto text-left">
              <div className="bg-gray-50 p-3 rounded-lg text-sm text-gray-700">
                "Show me total sales by region"
              </div>
              <div className="bg-gray-50 p-3 rounded-lg text-sm text-gray-700">
                "What are the top 10 customers by revenue?"
              </div>
              <div className="bg-gray-50 p-3 rounded-lg text-sm text-gray-700">
                "Find products with low inventory"
              </div>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              onExecuteSQL={executeSQL}
              onGenerateAnswer={generateAnswer}
              onNotification={onNotification}
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area - Fixed at bottom */}
      <div className="flex-shrink-0 border-t border-gray-200 bg-white p-4">
        <div className="flex items-end space-x-3">
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about your data..."
              className="form-textarea resize-none"
              rows="2"
              disabled={loading}
            />
          </div>
          <LoadingButton
            onClick={handleSendMessage}
            loading={loading}
            disabled={!inputValue.trim() || !projectId}
            className="btn-primary"
            loadingText=""
          >
            <Send className="h-4 w-4" />
          </LoadingButton>
        </div>
        
        {/* Status indicator */}
        <div className="mt-2 text-xs text-gray-500 text-center">
          {loading ? (
            <span className="flex items-center justify-center">
              <LoadingSpinner size="xs" className="mr-1" />
              Processing your query...
            </span>
          ) : (
            'Press Enter to send, Shift+Enter for new line'
          )}
        </div>
      </div>
    </div>
  );
};

// Message Bubble Component
const MessageBubble = ({ message, onExecuteSQL, onGenerateAnswer, onNotification }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      onNotification('Copied to clipboard', 'success');
    });
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  if (message.type === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-3xl">
          <div className="flex items-end space-x-2">
            <div className="bg-blue-600 text-white rounded-lg px-4 py-2 max-w-lg">
              <p className="text-sm">{message.content}</p>
            </div>
            <User className="h-6 w-6 text-gray-400 flex-shrink-0" />
          </div>
          <div className="text-xs text-gray-500 mt-1 text-right">
            {formatTimestamp(message.timestamp)}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-4xl w-full">
        <div className="flex items-start space-x-3">
          <Bot className="h-6 w-6 text-blue-500 flex-shrink-0 mt-1" />
          <div className="flex-1 bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-900 mb-3">{message.content}</p>

            {/* Entities Display */}
            {message.subtype === 'entities' && message.entities && (
              <div className="space-y-2">
                {message.entities.map((entity, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-white rounded border">
                    <div>
                      <span className="font-medium text-gray-900">{entity.text}</span>
                      <span className="ml-2 text-sm text-gray-500">({entity.type})</span>
                    </div>
                    <span className="text-xs text-green-600 font-medium">
                      {Math.round(entity.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Mappings Display */}
            {message.subtype === 'mapping' && message.mappings && (
              <div className="space-y-2">
                {message.mappings.map((mapping, index) => (
                  <div key={index} className="p-3 bg-white rounded border">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-gray-900">{mapping.entity}</span>
                      <span className="text-xs text-blue-600 font-medium">
                        {Math.round(mapping.confidence * 100)}% match
                      </span>
                    </div>
                    <div className="text-sm text-gray-600">
                      <Database className="h-4 w-4 inline mr-1" />
                      {mapping.table}.{mapping.column}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* SQL Display */}
            {message.subtype === 'sql' && message.sql && (
              <div className="space-y-3">
                <div className="bg-gray-900 text-gray-100 p-3 rounded font-mono text-sm overflow-x-auto">
                  <div className="flex justify-between items-start mb-2">
                    <Code className="h-4 w-4 text-gray-400" />
                    <button
                      onClick={() => copyToClipboard(message.sql)}
                      className="text-gray-400 hover:text-white"
                    >
                      {copied ? <CheckCircle className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </button>
                  </div>
                  <pre className="whitespace-pre-wrap">{message.sql}</pre>
                </div>
                
                {message.explanation && (
                  <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded">
                    <strong>Explanation:</strong> {message.explanation}
                  </div>
                )}
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-500">Confidence:</span>
                    <span className={`text-sm font-medium ${
                      message.confidence > 0.8 ? 'text-green-600' : 
                      message.confidence > 0.6 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {Math.round(message.confidence * 100)}%
                    </span>
                  </div>
                  <button
                    onClick={() => onExecuteSQL(message.sql)}
                    className="btn btn-primary btn-sm"
                  >
                    <Play className="h-4 w-4 mr-1" />
                    Execute Query
                  </button>
                </div>
              </div>
            )}

            {/* Results Display */}
            {message.subtype === 'results' && message.results && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-900">
                    {message.results.data.length} rows returned
                  </span>
                  <button
                    onClick={() => onGenerateAnswer(message.results.sql, message.results)}
                    className="btn btn-secondary btn-sm"
                  >
                    <MessageSquare className="h-4 w-4 mr-1" />
                    Generate Answer
                  </button>
                </div>
                
                {message.results.data.length > 0 && (
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead className="bg-gray-100">
                        <tr>
                          {message.results.columns.map((col, index) => (
                            <th key={index} className="px-3 py-2 text-left font-medium text-gray-900">
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {message.results.data.slice(0, 10).map((row, index) => (
                          <tr key={index}>
                            {message.results.columns.map((col, colIndex) => (
                              <td key={colIndex} className="px-3 py-2 text-gray-900">
                                {row[col] !== null ? String(row[col]) : 'NULL'}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {message.results.data.length > 10 && (
                      <div className="text-center py-2 text-sm text-gray-500">
                        ... and {message.results.data.length - 10} more rows
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Natural Language Answer */}
            {message.subtype === 'answer' && (
              <div className="bg-green-50 p-3 rounded border border-green-200">
                <div className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                  <div className="text-sm text-green-800">{message.content}</div>
                </div>
              </div>
            )}

            {/* Error Display */}
            {message.subtype === 'error' && (
              <div className="bg-red-50 p-3 rounded border border-red-200">
                <div className="flex items-start">
                  <AlertTriangle className="h-5 w-5 text-red-500 mr-2 flex-shrink-0 mt-0.5" />
                  <div>
                    <div className="text-sm text-red-800">{message.content}</div>
                    {message.error && (
                      <details className="mt-2">
                        <summary className="text-xs text-red-600 cursor-pointer">Error Details</summary>
                        <div className="text-xs text-red-600 mt-1 font-mono">{message.error}</div>
                      </details>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        <div className="text-xs text-gray-500 mt-1 ml-9">
          {formatTimestamp(message.timestamp)}
        </div>
      </div>
    </div>
  );
};

export default ChatTab;