// frontend/src/components/features/projects/ProjectsPage.tsx
import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Plus, 
  Search, 
  Filter, 
  FolderOpen, 
  Calendar,
  Users,
  Activity,
  MoreHorizontal,
  Database,
  Brain,
  MessageSquare
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import LoadingSpinner from '@/components/ui/LoadingSpinner'

export default function ProjectsPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [filterStatus, setFilterStatus] = useState('all')

  // Mock projects data
  const projects = [
    {
      id: 1,
      name: 'Customer Analytics Platform',
      description: 'Advanced analytics for customer behavior patterns and insights',
      status: 'active',
      owner: 'Alice Johnson',
      created_at: '2024-01-15',
      updated_at: '2024-01-20',
      sources_count: 5,
      queries_count: 128,
      team_size: 4,
      tags: ['analytics', 'customer', 'ml']
    },
    {
      id: 2,
      name: 'Financial Data Warehouse',
      description: 'Centralized financial reporting and compliance system',
      status: 'active',
      owner: 'Bob Smith',
      created_at: '2024-01-10',
      updated_at: '2024-01-19',
      sources_count: 12,
      queries_count: 256,
      team_size: 8,
      tags: ['finance', 'compliance', 'reporting']
    },
    {
      id: 3,
      name: 'HR Analytics Dashboard',
      description: 'Employee performance and workforce optimization insights',
      status: 'development',
      owner: 'Carol Davis',
      created_at: '2024-01-08',
      updated_at: '2024-01-18',
      sources_count: 3,
      queries_count: 45,
      team_size: 2,
      tags: ['hr', 'performance', 'dashboard']
    },
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success'
      case 'development': return 'warning'
      case 'archived': return 'secondary'
      default: return 'secondary'
    }
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  }

  return (
    <div className="flex-1 space-y-6 p-6">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
      >
        <div>
          <h1 className="text-3xl font-bold text-gradient-brand">Projects</h1>
          <p className="text-muted-foreground mt-1">
            Manage your AI-powered data projects
          </p>
        </div>
        
        <Button className="bg-gradient-brand hover:opacity-90">
          <Plus className="w-4 h-4 mr-2" />
          New Project
        </Button>
      </motion.div>

      {/* Stats Cards */}
      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-4 gap-4"
      >
        {[
          { label: 'Total Projects', value: '12', icon: FolderOpen, color: 'blue' },
          { label: 'Active Projects', value: '8', icon: Activity, color: 'green' },
          { label: 'Data Sources', value: '45', icon: Database, color: 'purple' },
          { label: 'Total Queries', value: '1.2K', icon: MessageSquare, color: 'orange' },
        ].map((stat, index) => (
          <motion.div
            key={stat.label}
            variants={itemVariants}
            className="glass-card p-6 rounded-xl"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">{stat.label}</p>
                <p className="text-2xl font-bold mt-1">{stat.value}</p>
              </div>
              <div className={`w-12 h-12 rounded-lg bg-${stat.color}-500/10 flex items-center justify-center`}>
                <stat.icon className={`w-6 h-6 text-${stat.color}-500`} />
              </div>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* Search and Filters */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="flex flex-col sm:flex-row gap-4"
      >
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search projects..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-background border border-input rounded-lg focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
        
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Filter className="w-4 h-4 mr-2" />
            Filter
          </Button>
        </div>
      </motion.div>

      {/* Projects Grid */}
      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6"
      >
        {projects.map((project) => (
          <motion.div
            key={project.id}
            variants={itemVariants}
            whileHover={{ y: -4, transition: { duration: 0.2 } }}
            className="group"
          >
            <Link to={`/projects/${project.id}`}>
              <div className="glass-card p-6 rounded-xl border border-border/40 hover:border-primary/50 transition-all duration-300 h-full">
                {/* Project Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold group-hover:text-primary transition-colors">
                      {project.name}
                    </h3>
                    <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                      {project.description}
                    </p>
                  </div>
                  
                  <Button variant="ghost" size="sm" className="opacity-0 group-hover:opacity-100 transition-opacity">
                    <MoreHorizontal className="w-4 h-4" />
                  </Button>
                </div>

                {/* Status and Tags */}
                <div className="flex items-center gap-2 mb-4">
                  <Badge variant={getStatusColor(project.status) as any}>
                    {project.status}
                  </Badge>
                  {project.tags.slice(0, 2).map((tag) => (
                    <Badge key={tag} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                  {project.tags.length > 2 && (
                    <Badge variant="outline" className="text-xs">
                      +{project.tags.length - 2}
                    </Badge>
                  )}
                </div>

                {/* Project Stats */}
                <div className="grid grid-cols-3 gap-4 mb-4 text-center">
                  <div>
                    <p className="text-sm font-medium">{project.sources_count}</p>
                    <p className="text-xs text-muted-foreground">Sources</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">{project.queries_count}</p>
                    <p className="text-xs text-muted-foreground">Queries</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">{project.team_size}</p>
                    <p className="text-xs text-muted-foreground">Team</p>
                  </div>
                </div>

                {/* Project Footer */}
                <div className="flex items-center justify-between text-xs text-muted-foreground pt-4 border-t border-border/40">
                  <div className="flex items-center gap-1">
                    <Users className="w-3 h-3" />
                    <span>{project.owner}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    <span>{new Date(project.updated_at).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
            </Link>
          </motion.div>
        ))}
      </motion.div>

      {/* Empty State */}
      {projects.length === 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex flex-col items-center justify-center py-16"
        >
          <div className="w-20 h-20 bg-muted rounded-full flex items-center justify-center mb-4">
            <FolderOpen className="w-10 h-10 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-semibold mb-2">No projects yet</h3>
          <p className="text-muted-foreground text-center max-w-md mb-6">
            Get started by creating your first project. Connect your data sources and start asking questions.
          </p>
          <Button className="bg-gradient-brand">
            <Plus className="w-4 h-4 mr-2" />
            Create Your First Project
          </Button>
        </motion.div>
      )}
    </div>
  )
}