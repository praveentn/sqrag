// static/src/components/ProjectsTab.js
import React, { useState, useEffect } from 'react';
import { 
  Plus, 
  FolderOpen, 
  Edit3, 
  Trash2, 
  Copy, 
  Calendar,
  Database,
  FileText,
  Search,
  MoreVertical,
  Eye
} from 'lucide-react';
import LoadingSpinner, { LoadingButton } from './LoadingSpinner';

const ProjectsTab = ({ apiUrl, onNotification, onProjectsChange, onProjectSelect }) => {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [selectedProject, setSelectedProject] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('updated_at');
  const [sortOrder, setSortOrder] = useState('desc');

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiUrl}/projects`);
      if (!response.ok) throw new Error('Failed to fetch projects');
      
      const data = await response.json();
      setProjects(data.projects || []);
      
      if (onProjectsChange) {
        onProjectsChange();
      }
    } catch (error) {
      onNotification('Error fetching projects: ' + error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateProject = async (projectData) => {
    try {
      const response = await fetch(`${apiUrl}/projects`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(projectData),
      });

      if (!response.ok) throw new Error('Failed to create project');
      
      const data = await response.json();
      onNotification(`Project "${data.project.name}" created successfully`, 'success');
      setShowCreateModal(false);
      fetchProjects();
    } catch (error) {
      onNotification('Error creating project: ' + error.message, 'error');
    }
  };

  const handleUpdateProject = async (projectId, projectData) => {
    try {
      const response = await fetch(`${apiUrl}/projects/${projectId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(projectData),
      });

      if (!response.ok) throw new Error('Failed to update project');
      
      const data = await response.json();
      onNotification(`Project "${data.project.name}" updated successfully`, 'success');
      setShowEditModal(false);
      setSelectedProject(null);
      fetchProjects();
    } catch (error) {
      onNotification('Error updating project: ' + error.message, 'error');
    }
  };

  const handleDeleteProject = async (projectId, projectName) => {
    if (!window.confirm(`Are you sure you want to delete the project "${projectName}"? This action cannot be undone.`)) {
      return;
    }

    try {
      const response = await fetch(`${apiUrl}/projects/${projectId}`, {
        method: 'DELETE',
      });

      if (!response.ok) throw new Error('Failed to delete project');
      
      onNotification(`Project "${projectName}" deleted successfully`, 'success');
      fetchProjects();
    } catch (error) {
      onNotification('Error deleting project: ' + error.message, 'error');
    }
  };

  const handleCloneProject = async (projectId, projectName) => {
    const newName = prompt(`Enter name for cloned project:`, `${projectName} (Copy)`);
    if (!newName) return;

    try {
      const response = await fetch(`${apiUrl}/projects/${projectId}/clone`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: newName }),
      });

      if (!response.ok) throw new Error('Failed to clone project');
      
      const data = await response.json();
      onNotification(`Project cloned as "${data.project.name}"`, 'success');
      fetchProjects();
    } catch (error) {
      onNotification('Error cloning project: ' + error.message, 'error');
    }
  };

  const filteredAndSortedProjects = projects
    .filter(project => 
      project.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (project.description && project.description.toLowerCase().includes(searchTerm.toLowerCase()))
    )
    .sort((a, b) => {
      const aValue = a[sortBy];
      const bValue = b[sortBy];
      
      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <div className="p-6">
        <LoadingSpinner size="lg" message="Loading projects..." />
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
        <div>
          <h3 className="text-lg font-medium text-gray-900">Projects</h3>
          <p className="mt-1 text-sm text-gray-500">
            Manage your data projects and organize your work
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn btn-primary"
          >
            <Plus className="h-4 w-4 mr-2" />
            New Project
          </button>
        </div>
      </div>

      {/* Search and Sort */}
      <div className="flex flex-col sm:flex-row sm:items-center space-y-3 sm:space-y-0 sm:space-x-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search projects..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="form-input pl-10"
          />
        </div>
        <div className="flex items-center space-x-2">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="form-select"
          >
            <option value="name">Name</option>
            <option value="created_at">Created</option>
            <option value="updated_at">Updated</option>
            <option value="sources_count">Sources</option>
          </select>
          <button
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            className="btn btn-secondary text-xs"
          >
            {sortOrder === 'asc' ? '↑' : '↓'}
          </button>
        </div>
      </div>

      {/* Projects Grid */}
      {filteredAndSortedProjects.length === 0 ? (
        <div className="text-center py-12">
          <FolderOpen className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No projects</h3>
          <p className="mt-1 text-sm text-gray-500">
            {searchTerm ? 'No projects match your search.' : 'Get started by creating a new project.'}
          </p>
          {!searchTerm && (
            <div className="mt-6">
              <button
                onClick={() => setShowCreateModal(true)}
                className="btn btn-primary"
              >
                <Plus className="h-4 w-4 mr-2" />
                New Project
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {filteredAndSortedProjects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onEdit={() => {
                setSelectedProject(project);
                setShowEditModal(true);
              }}
              onDelete={() => handleDeleteProject(project.id, project.name)}
              onClone={() => handleCloneProject(project.id, project.name)}
              onSelect={() => onProjectSelect && onProjectSelect(project)}
              formatDate={formatDate}
            />
          ))}
        </div>
      )}

      {/* Create Project Modal */}
      {showCreateModal && (
        <ProjectModal
          title="Create New Project"
          onSubmit={handleCreateProject}
          onCancel={() => setShowCreateModal(false)}
        />
      )}

      {/* Edit Project Modal */}
      {showEditModal && selectedProject && (
        <ProjectModal
          title="Edit Project"
          project={selectedProject}
          onSubmit={(data) => handleUpdateProject(selectedProject.id, data)}
          onCancel={() => {
            setShowEditModal(false);
            setSelectedProject(null);
          }}
        />
      )}
    </div>
  );
};

// Project Card Component
const ProjectCard = ({ project, onEdit, onDelete, onClone, onSelect, formatDate }) => {
  const [showDropdown, setShowDropdown] = useState(false);

  return (
    <div className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow">
      <div className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="h-10 w-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <FolderOpen className="h-6 w-6 text-white" />
              </div>
            </div>
            <div className="ml-4 flex-1">
              <h3 className="text-lg font-medium text-gray-900 truncate">
                {project.name}
              </h3>
              <div className="flex items-center mt-1">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  project.status === 'active' 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-gray-100 text-gray-800'
                }`}>
                  {project.status}
                </span>
              </div>
            </div>
          </div>
          
          <div className="relative">
            <button
              onClick={() => setShowDropdown(!showDropdown)}
              className="p-2 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100"
            >
              <MoreVertical className="h-5 w-5" />
            </button>
            
            {showDropdown && (
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg z-10 border border-gray-200">
                <div className="py-1">
                  <button
                    onClick={() => {
                      onSelect();
                      setShowDropdown(false);
                    }}
                    className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full text-left"
                  >
                    <Eye className="h-4 w-4 mr-2" />
                    Select Project
                  </button>
                  <button
                    onClick={() => {
                      onEdit();
                      setShowDropdown(false);
                    }}
                    className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full text-left"
                  >
                    <Edit3 className="h-4 w-4 mr-2" />
                    Edit
                  </button>
                  <button
                    onClick={() => {
                      onClone();
                      setShowDropdown(false);
                    }}
                    className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full text-left"
                  >
                    <Copy className="h-4 w-4 mr-2" />
                    Clone
                  </button>
                  <button
                    onClick={() => {
                      onDelete();
                      setShowDropdown(false);
                    }}
                    className="flex items-center px-4 py-2 text-sm text-red-600 hover:bg-red-50 w-full text-left"
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {project.description && (
          <p className="mt-2 text-sm text-gray-600 line-clamp-2">
            {project.description}
          </p>
        )}

        <div className="mt-4">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <div className="flex items-center">
              <Database className="h-4 w-4 mr-1" />
              <span>{project.sources_count || 0} sources</span>
            </div>
            <div className="flex items-center">
              <FileText className="h-4 w-4 mr-1" />
              <span>{project.dictionary_entries_count || 0} terms</span>
            </div>
          </div>
          
          <div className="mt-2 flex items-center text-xs text-gray-500">
            <Calendar className="h-3 w-3 mr-1" />
            <span>Updated {formatDate(project.updated_at)}</span>
          </div>
        </div>

        <div className="mt-4">
          <button
            onClick={onSelect}
            className="w-full btn btn-primary btn-sm"
          >
            Select Project
          </button>
        </div>
      </div>
    </div>
  );
};

// Project Modal Component
const ProjectModal = ({ title, project, onSubmit, onCancel }) => {
  const [formData, setFormData] = useState({
    name: project?.name || '',
    description: project?.description || '',
    owner: project?.owner || 'current_user'
  });
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validation
    const newErrors = {};
    if (!formData.name.trim()) {
      newErrors.name = 'Project name is required';
    }
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setLoading(true);
    try {
      await onSubmit(formData);
    } catch (error) {
      // Error handled by parent component
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
              <div className="mb-4">
                <h3 className="text-lg leading-6 font-medium text-gray-900">
                  {title}
                </h3>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="form-label">
                    Project Name *
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className={`form-input ${errors.name ? 'border-red-300' : ''}`}
                    placeholder="Enter project name"
                  />
                  {errors.name && (
                    <p className="form-error">{errors.name}</p>
                  )}
                </div>

                <div>
                  <label className="form-label">
                    Description
                  </label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    className="form-textarea"
                    rows="3"
                    placeholder="Enter project description (optional)"
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
                {project ? 'Update' : 'Create'} Project
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

export default ProjectsTab;