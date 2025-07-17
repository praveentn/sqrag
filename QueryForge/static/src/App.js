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

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

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
      description: 'Define business terms and metadata',
      requiresProject: true
    },
    { 
      id: 3, 
      name: 'Embeddings & Indexing', 
      icon: Cpu, 
      component: EmbeddingsTab,
      description: 'Create embeddings and search indexes',
      requiresProject: true
    },
    { 
      id: 4, 
      name: 'Search Playground', 
      icon: Search, 
      component: SearchTab,
      description: 'Test different search methods',
      requiresProject: true
    },
    { 
      id: 5, 
      name: 'Chat (NL → SQL)', 
      icon: MessageSquare, 
      component: ChatTab,
      description: 'Natural language to SQL conversion',
      requiresProject: true
    },
    { 
      id: 6, 
      name: 'Admin Panel', 
      icon: Settings, 
      component: AdminTab,
      description: 'System administration and monitoring'
    }
  ];

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/projects`);
      if (!response.ok) throw new Error('Failed to fetch projects');
      
      const data = await response.json();
      setProjects(data.projects || []);
      
      // Auto-select first project if available
      if (data.projects && data.projects.length > 0 && !selectedProject) {
        setSelectedProject(data.projects[0]);
      }
    } catch (err) {
      setError(err.message);
      showNotification('Error fetching projects: ' + err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const showNotification = (message, type = 'info') => {
    setNotification({ message, type, id: Date.now() });
    setTimeout(() => setNotification(null), 5000);
  };

  const handleTabChange = (tabId) => {
    const tab = tabs.find(t => t.id === tabId);
    if (tab && tab.requiresProject && !selectedProject) {
      showNotification('Please select a project first', 'warning');
      return;
    }
    setActiveTab(tabId);
  };

  const handleProjectChange = (project) => {
    setSelectedProject(project);
    showNotification(`Switched to project: ${project.name}`, 'success');
  };

  const renderTabContent = () => {
    const currentTab = tabs.find(tab => tab.id === activeTab);
    if (!currentTab) return null;

    const Component = currentTab.component;
    return (
      <Component
        projectId={selectedProject?.id}
        project={selectedProject}
        apiUrl={API_BASE_URL}
        onNotification={showNotification}
        onProjectsChange={fetchProjects}
        onProjectSelect={handleProjectChange}
      />
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <LoadingSpinner size="large" message="Loading QueryForge Pro..." />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo and Title */}
            <div className="flex items-center">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 lg:hidden"
              >
                {sidebarOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </button>
              <div className="flex items-center ml-4">
                <div className="flex-shrink-0">
                  <div className="h-8 w-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-md flex items-center justify-center">
                    <Database className="h-5 w-5 text-white" />
                  </div>
                </div>
                <div className="ml-3">
                  <h1 className="text-xl font-semibold text-gray-900">QueryForge Pro</h1>
                  <p className="text-sm text-gray-500">Intelligent Data Querying Platform</p>
                </div>
              </div>
            </div>

            {/* Project Selector */}
            <div className="flex items-center space-x-4">
              <div className="relative">
                <select
                  value={selectedProject?.id || ''}
                  onChange={(e) => {
                    const project = projects.find(p => p.id === parseInt(e.target.value));
                    if (project) handleProjectChange(project);
                  }}
                  className="block w-64 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md bg-white"
                  disabled={projects.length === 0}
                >
                  <option value="">Select a project...</option>
                  {projects.map((project) => (
                    <option key={project.id} value={project.id}>
                      {project.name}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-3 h-4 w-4 text-gray-400 pointer-events-none" />
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <aside className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0 fixed lg:static inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-200 ease-in-out lg:transform-none`}>
          <div className="flex flex-col h-full">
            {/* Navigation */}
            <nav className="flex-1 pt-6 pb-4 overflow-y-auto">
              <div className="px-3">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  const isDisabled = tab.requiresProject && !selectedProject;
                  
                  return (
                    <button
                      key={tab.id}
                      onClick={() => handleTabChange(tab.id)}
                      disabled={isDisabled}
                      className={`
                        group flex items-center px-3 py-2 text-sm font-medium rounded-md w-full text-left mb-1
                        ${isActive 
                          ? 'bg-blue-50 text-blue-700 border-r-2 border-blue-700' 
                          : isDisabled
                            ? 'text-gray-400 cursor-not-allowed'
                            : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                        }
                      `}
                    >
                      <Icon className={`mr-3 h-5 w-5 ${isActive ? 'text-blue-500' : isDisabled ? 'text-gray-300' : 'text-gray-400'}`} />
                      <div className="flex-1">
                        <div className="text-sm font-medium">{tab.name}</div>
                        <div className={`text-xs ${isActive ? 'text-blue-600' : 'text-gray-500'}`}>
                          {tab.description}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </nav>

            {/* Footer */}
            <div className="flex-shrink-0 border-t border-gray-200 p-4">
              <div className="text-xs text-gray-500 text-center">
                <div>QueryForge Pro v1.0</div>
                <div>Enterprise RAG Platform</div>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 lg:ml-0">
          <div className="py-6">
            <div className="px-4 sm:px-6 lg:px-8">
              {/* Page Header */}
              <div className="mb-8">
                <div className="md:flex md:items-center md:justify-between">
                  <div className="flex-1 min-w-0">
                    <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
                      {tabs.find(tab => tab.id === activeTab)?.name}
                    </h2>
                    <div className="mt-1 flex flex-col sm:flex-row sm:flex-wrap sm:mt-0 sm:space-x-6">
                      <div className="mt-2 flex items-center text-sm text-gray-500">
                        {tabs.find(tab => tab.id === activeTab)?.description}
                      </div>
                      {selectedProject && (
                        <div className="mt-2 flex items-center text-sm text-gray-500">
                          <span className="font-medium">Project:</span>
                          <span className="ml-1 text-gray-900">{selectedProject.name}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Tab Content */}
              <div className="bg-white shadow rounded-lg min-h-[600px]">
                {renderTabContent()}
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* Notifications */}
      {notification && (
        <NotificationBanner
          message={notification.message}
          type={notification.type}
          onClose={() => setNotification(null)}
        />
      )}

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        >
          <div className="absolute inset-0 bg-gray-600 opacity-75"></div>
        </div>
      )}
    </div>
  );
};

export default App;