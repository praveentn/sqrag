// frontend/src/components/layout/Sidebar.tsx
import React from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  FolderOpen,
  Database,
  BookOpen,
  Layers,
  Search,
  MessageSquare,
  Settings,
  BarChart3,
  Users,
  Shield,
  Home,
  ChevronRight,
  Plus,
  Activity
} from 'lucide-react'

import { cn } from '@/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

interface SidebarProps {
  isOpen: boolean
  isMobile: boolean
  onClose: () => void
}

export default function Sidebar({ isOpen, isMobile, onClose }: SidebarProps) {
  const location = useLocation()
  const { projectId } = useParams()

  const mainNavItems = [
    {
      title: 'Projects',
      href: '/projects',
      icon: FolderOpen,
      description: 'Manage your projects',
    },
  ]

  const projectNavItems = projectId ? [
    {
      title: 'Data Sources',
      href: `/projects/${projectId}/sources`,
      icon: Database,
      description: 'Upload & connect data',
      badge: '2 active',
    },
    {
      title: 'Dictionary',
      href: `/projects/${projectId}/dictionary`,
      icon: BookOpen,
      description: 'Business terms & definitions',
      badge: '45 terms',
    },
    {
      title: 'Embeddings',
      href: `/projects/${projectId}/embeddings`,
      icon: Layers,
      description: 'Vector indexing & search',
      badge: 'Building',
      badgeVariant: 'secondary' as const,
    },
    {
      title: 'Search Playground',
      href: `/projects/${projectId}/search`,
      icon: Search,
      description: 'Test queries & explore',
    },
    {
      title: 'Natural Language',
      href: `/projects/${projectId}/chat`,
      icon: MessageSquare,
      description: 'Ask questions naturally',
      isNew: true,
    },
  ] : []

  const adminNavItems = [
    {
      title: 'Admin Panel',
      href: '/admin',
      icon: Settings,
      description: 'System administration',
    },
  ]

  const isActive = (href: string) => {
    if (href === '/projects') {
      return location.pathname === '/projects' || 
             (location.pathname.startsWith('/projects/') && !location.pathname.includes('/', 10))
    }
    return location.pathname.startsWith(href)
  }

  const sidebarVariants = {
    open: {
      width: 256,
      transition: { duration: 0.3, ease: "easeInOut" }
    },
    closed: {
      width: 0,
      transition: { duration: 0.3, ease: "easeInOut" }
    }
  }

  const contentVariants = {
    open: {
      opacity: 1,
      x: 0,
      transition: { duration: 0.3, delay: 0.1 }
    },
    closed: {
      opacity: 0,
      x: -20,
      transition: { duration: 0.2 }
    }
  }

  return (
    <>
      {/* Sidebar */}
      <motion.aside
        variants={sidebarVariants}
        animate={isOpen ? "open" : "closed"}
        className={cn(
          "glass-sidebar flex-shrink-0 border-r border-border/40 overflow-hidden",
          isMobile && "fixed left-0 top-0 z-50 h-full"
        )}
      >
        <motion.div
          variants={contentVariants}
          animate={isOpen ? "open" : "closed"}
          className="flex flex-col h-full w-64"
        >
          {/* Sidebar Header */}
          <div className="p-6 border-b border-border/40">
            <h2 className="text-lg font-semibold text-gradient-brand">
              Navigation
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              {projectId ? `Project ${projectId}` : 'Workspace'}
            </p>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4 space-y-8 overflow-y-auto scrollbar-thin">
            {/* Main Navigation */}
            <div>
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                Workspace
              </h3>
              <ul className="space-y-1">
                {mainNavItems.map((item) => (
                  <NavItem
                    key={item.href}
                    item={item}
                    isActive={isActive(item.href)}
                    onClose={onClose}
                  />
                ))}
              </ul>
            </div>

            {/* Project Navigation */}
            {projectNavItems.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    Project Tools
                  </h3>
                  <Badge variant="outline" className="text-xs">
                    {projectNavItems.length}
                  </Badge>
                </div>
                <ul className="space-y-1">
                  {projectNavItems.map((item) => (
                    <NavItem
                      key={item.href}
                      item={item}
                      isActive={isActive(item.href)}
                      onClose={onClose}
                    />
                  ))}
                </ul>
              </div>
            )}

            {/* Admin Navigation */}
            <div>
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                Administration
              </h3>
              <ul className="space-y-1">
                {adminNavItems.map((item) => (
                  <NavItem
                    key={item.href}
                    item={item}
                    isActive={isActive(item.href)}
                    onClose={onClose}
                  />
                ))}
              </ul>
            </div>
          </nav>

          {/* Sidebar Footer */}
          <div className="p-4 border-t border-border/40">
            <div className="glass-card p-4 rounded-lg">
              <div className="flex items-center space-x-3 mb-3">
                <div className="w-8 h-8 bg-gradient-brand rounded-lg flex items-center justify-center">
                  <Activity className="w-4 h-4 text-white" />
                </div>
                <div>
                  <p className="text-sm font-medium">System Status</p>
                  <p className="text-xs text-muted-foreground">All systems operational</p>
                </div>
              </div>
              
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">API Status</span>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full" />
                    <span className="text-green-600">Online</span>
                  </div>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Query Engine</span>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full" />
                    <span className="text-green-600">Ready</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.aside>
    </>
  )
}

// Navigation Item Component
function NavItem({ 
  item, 
  isActive, 
  onClose 
}: { 
  item: any
  isActive: boolean
  onClose: () => void
}) {
  const Icon = item.icon

  return (
    <li>
      <Link
        to={item.href}
        onClick={onClose}
        className={cn(
          "group flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200",
          "hover:bg-accent hover:text-accent-foreground",
          isActive && "bg-accent text-accent-foreground shadow-sm"
        )}
      >
        <Icon className={cn(
          "w-4 h-4 mr-3 flex-shrink-0 transition-colors",
          isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
        )} />
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <span className="truncate">{item.title}</span>
            {item.isNew && (
              <Badge variant="destructive" className="text-xs ml-2">
                New
              </Badge>
            )}
            {item.badge && !item.isNew && (
              <Badge 
                variant={item.badgeVariant || "secondary"} 
                className="text-xs ml-2"
              >
                {item.badge}
              </Badge>
            )}
          </div>
          {item.description && (
            <p className="text-xs text-muted-foreground mt-0.5 truncate">
              {item.description}
            </p>
          )}
        </div>

        {isActive && (
          <motion.div
            layoutId="activeTab"
            className="w-1 h-6 bg-primary rounded-full ml-2"
            initial={false}
            transition={{ duration: 0.2 }}
          />
        )}
      </Link>
    </li>
  )
}