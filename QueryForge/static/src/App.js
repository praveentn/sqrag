// static/src/App.js
import React, { useState, useEffect } from 'react';
import './App.css';

// Import components for each tab
import ProjectsTab from './components/ProjectsTab';
import DataSourcesTab from './components/DataSourcesTab';
import DictionaryTab from './components/DictionaryTab';
import EmbeddingsTab from './components/EmbeddingsTab';
import SearchTab from './components/SearchTab';
import ChatTab from './components/ChatTab';
import AdminTab from './components/AdminTab';
import LoadingSpinner from './components/LoadingSpinner';
import NotificationBanner from './components/NotificationBanner';

// Icons (using Lucide React icons)
import { 
  FolderOpen, 
  Database, 
  BookOpen, 
  Cpu, 
  Search, 
  MessageSquare, 
  Settings,
  Menu,
  X,
  ChevronDown
} from 'lucide-react';

// Environment configuration with fallbacks
const getApiBaseUrl = () => {
  // Try multiple sources for environment variables
  if (typeof process !== 'undefined' && process.env && process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }
  
  // Fallback to window environment variables (set by HTML template)
  if (typeof window !== 'undefined' && window.ENV && window.ENV.API_URL) {
    return window.ENV.API_URL;
  }
  
  // Final fallback based on current location
  if (typeof window !== 'undefined') {
    const { protocol, hostname, port } = window.location;
    const backendPort = '5000';
    return `${protocol}//${hostname}:${backendPort}/api`;
  }
  
  // Default fallback
  return 'http://localhost:5000/api';
};

const API_BASE_URL = getApiBaseUrl();

const App = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedProject, setSelectedProject] = useState(null);
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [notification, setNotification] = useState(null);

  // Tab configuration
  const tabs = [
    { 
      id: 0, 
      name: 'Projects', 
      icon: FolderOpen, 
      component: ProjectsTab,
      description: 'Manage and organize your data projects'
    },
    { 
      id: 1, 
      name: 'Data Sources', 
      icon: Database, 
      component: DataSourcesTab,
      description: 'Upload files and connect databases',
      requiresProject: true
    },
    { 
      id: 2, 
      name: 'Data Dictionary', 
      icon: BookOpen, 
      component: DictionaryTab,
      description: 'Manage table and column mappings',
      requiresProject: true
    },
    { 
      id: 3, 
      name: 'Embeddings', 
      icon: Cpu, 
      component: EmbeddingsTab,
      description: 'Generate and manage vector embeddings',
      requiresProject: true
    },
    { 
      id: 4, 
      name: 'Search', 
      icon: Search, 
      component: SearchTab,
      description: 'Perform intelligent data searches',
      requiresProject: true
    },
    { 
      id: 5, 
      name: 'Chat', 
      icon: MessageSquare, 
      component: ChatTab,
      description: 'Natural language chat interface',
      requiresProject: true
    },
    { 
      id: 6, 
      name: 'Admin', 
      icon: Settings, 
      component: AdminTab,
      description: 'System administration and monitoring'
    }
  ];

  // Load projects on component mount
  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/projects`);
      
      if (!response.ok) {
        throw new Error(`Failed to load projects: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setProjects(data.projects || []);
        
        // Auto-select first project if available
        if (data.projects && data.projects.length > 0 && !selectedProject) {
          setSelectedProject(data.projects[0]);
        }
      } else {
        throw new Error(data.error || 'Failed to load projects');
      }
    } catch (err) {
      console.error('Error loading projects:', err);
      setError(err.message);
      showNotification('Failed to load projects. Please check your connection.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const showNotification = (message, type = 'info', duration = 5000) => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), duration);
  };

  const handleTabChange = (tabId) => {
    const tab = tabs.find(t => t.id === tabId);
    
    // Check if tab requires a project
    if (tab && tab.requiresProject && !selectedProject) {
      showNotification('Please select a project first', 'warning');
      return;
    }
    
    setActiveTab(tabId);
  };

  const handleProjectSelect = (project) => {
    setSelectedProject(project);
    showNotification(`Switched to project: ${project.name}`, 'success');
  };

  const handleProjectCreate = (project) => {
    setProjects(prev => [...prev, project]);
    setSelectedProject(project);
    showNotification(`Created project: ${project.name}`, 'success');
  };

  const handleProjectUpdate = (updatedProject) => {
    setProjects(prev => prev.map(p => p.id === updatedProject.id ? updatedProject : p));
    if (selectedProject && selectedProject.id === updatedProject.id) {
      setSelectedProject(updatedProject);
    }
    showNotification(`Updated project: ${updatedProject.name}`, 'success');
  };

  const handleProjectDelete = (projectId) => {
    setProjects(prev => prev.filter(p => p.id !== projectId));
    if (selectedProject && selectedProject.id === projectId) {
      setSelectedProject(null);
      setActiveTab(0); // Switch back to Projects tab
    }
    showNotification('Project deleted successfully', 'success');
  };

  // In App.js, update getCurrentTabComponent to pass the correct props
  const getCurrentTabComponent = () => {
    const currentTab = tabs.find(tab => tab.id === activeTab);
    if (!currentTab) return null;

    const Component = currentTab.component;

    // Pass projectId and apiUrl, rename showNotification to onNotification
    const commonProps = {
      apiUrl: API_BASE_URL,
      onNotification: showNotification,
      projectId: selectedProject ? selectedProject.id : null,
      onProjectSelect: handleProjectSelect,
      onProjectCreate: handleProjectCreate,
      onProjectUpdate: handleProjectUpdate,
      onProjectDelete: handleProjectDelete
    };

    return <Component {...commonProps} />;
  };


  // // Get current tab component
  // const getCurrentTabComponent = () => {
  //   const currentTab = tabs.find(tab => tab.id === activeTab);
  //   if (!currentTab) return null;

  //   const Component = currentTab.component;
    
  //   // const commonProps = {
  //   //   selectedProject,
  //   //   projects,
  //   //   onProjectSelect: handleProjectSelect,
  //   //   onProjectCreate: handleProjectCreate,
  //   //   onProjectUpdate: handleProjectUpdate,
  //   //   onProjectDelete: handleProjectDelete,
  //   //   showNotification,
  //   //   apiBaseUrl: API_BASE_URL
  //   // };

  //   const commonProps = {
  //     apiUrl: API_BASE_URL,             // match ProjectsTab’s apiUrl prop
  //     onNotification: showNotification, // match ProjectsTab’s onNotification prop
  //     onProjectsChange: loadProjects,   // so parent refreshes its project list
  //     onProjectSelect: handleProjectSelect
  //   };

  //   return <Component {...commonProps} />;
  // };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="large" />
          <p className="mt-4 text-lg text-gray-600">Loading QueryForge Pro...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Notification Banner */}
      {notification && (
        <NotificationBanner
          message={notification.message}
          type={notification.type}
          onClose={() => setNotification(null)}
        />
      )}

      {/* Sidebar */}
      <div className={`bg-white shadow-lg transition-all duration-300 ${sidebarOpen ? 'w-64' : 'w-16'}`}>
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            {sidebarOpen && (
              <h1 className="text-xl font-bold text-gray-900">QueryForge Pro</h1>
            )}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-1 rounded-md hover:bg-gray-100 transition-colors"
            >
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>

        {/* Project Selector */}
        {sidebarOpen && (
          <div className="p-4 border-b border-gray-200">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Current Project
            </label>
            {projects.length > 0 ? (
              <div className="relative">
                <select
                  value={selectedProject?.id || ''}
                  onChange={(e) => {
                    const project = projects.find(p => p.id === parseInt(e.target.value));
                    if (project) handleProjectSelect(project);
                  }}
                  className="w-full p-2 border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select a project...</option>
                  {projects.map(project => (
                    <option key={project.id} value={project.id}>
                      {project.name}
                    </option>
                  ))}
                </select>
              </div>
            ) : (
              <p className="text-sm text-gray-500">No projects available</p>
            )}
          </div>
        )}

        {/* Navigation */}
        <nav className="p-4">
          <ul className="space-y-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              const isDisabled = tab.requiresProject && !selectedProject;
              
              return (
                <li key={tab.id}>
                  <button
                    onClick={() => handleTabChange(tab.id)}
                    disabled={isDisabled}
                    className={`w-full flex items-center px-3 py-2 rounded-md text-left transition-colors ${
                      isActive
                        ? 'bg-blue-100 text-blue-700 border border-blue-200'
                        : isDisabled
                        ? 'text-gray-400 cursor-not-allowed'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`}
                    title={sidebarOpen ? tab.description : tab.name}
                  >
                    <Icon size={20} className="flex-shrink-0" />
                    {sidebarOpen && (
                      <span className="ml-3 font-medium">{tab.name}</span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Footer */}
        {sidebarOpen && (
          <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200 bg-white">
            <div className="text-xs text-gray-500">
              <p>QueryForge Pro v1.0.0</p>
              <p>Enterprise RAG Platform</p>
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <div className="bg-white shadow-sm border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">
                {tabs.find(tab => tab.id === activeTab)?.name}
              </h2>
              <p className="text-sm text-gray-600 mt-1">
                {tabs.find(tab => tab.id === activeTab)?.description}
              </p>
            </div>
            
            {selectedProject && (
              <div className="text-right">
                <p className="text-sm text-gray-600">Active Project</p>
                <p className="font-medium text-gray-900">{selectedProject.name}</p>
              </div>
            )}
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 p-6 overflow-y-auto">
          {error ? (
            <div className="bg-red-50 border border-red-200 rounded-md p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <X className="h-5 w-5 text-red-400" />
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Error</h3>
                  <p className="mt-1 text-sm text-red-700">{error}</p>
                  <button
                    onClick={() => {
                      setError(null);
                      loadProjects();
                    }}
                    className="mt-2 text-sm text-red-600 hover:text-red-500 underline"
                  >
                    Try again
                  </button>
                </div>
              </div>
            </div>
          ) : (
            getCurrentTabComponent()
          )}
        </div>
      </div>
    </div>
  );
};

export default App;