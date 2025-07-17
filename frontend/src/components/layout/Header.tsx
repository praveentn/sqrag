// frontend/src/components/layout/Header.tsx
import React from 'react'
import { useParams, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Menu, 
  Bell, 
  Search, 
  User, 
  Settings,
  ChevronDown,
  Database,
  Sparkles
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Badge } from '@/components/ui/badge'

interface HeaderProps {
  onMenuClick: () => void
  onSidebarToggle: () => void
  sidebarOpen: boolean
}

export default function Header({ onMenuClick, onSidebarToggle, sidebarOpen }: HeaderProps) {
  const { projectId } = useParams()
  const location = useLocation()

  // Get current page info
  const getPageInfo = () => {
    const path = location.pathname
    if (path.includes('/projects') && !projectId) {
      return { title: 'Projects', subtitle: 'Manage your AI projects' }
    }
    if (path.includes('/sources')) {
      return { title: 'Data Sources', subtitle: 'Connect and manage data' }
    }
    if (path.includes('/dictionary')) {
      return { title: 'Data Dictionary', subtitle: 'Define business terms' }
    }
    if (path.includes('/embeddings')) {
      return { title: 'Embeddings & Indexing', subtitle: 'Vector search setup' }
    }
    if (path.includes('/search')) {
      return { title: 'Search Playground', subtitle: 'Test your queries' }
    }
    if (path.includes('/chat')) {
      return { title: 'Natural Language Query', subtitle: 'Ask questions naturally' }
    }
    if (path.includes('/admin')) {
      return { title: 'Admin Panel', subtitle: 'System administration' }
    }
    return { title: 'StructuraAI', subtitle: 'Enterprise RAG Platform' }
  }

  const { title, subtitle } = getPageInfo()

  return (
    <header className="glass-nav sticky top-0 z-30 h-16 border-b border-border/40">
      <div className="flex items-center justify-between h-full px-4 lg:px-6">
        {/* Left Side */}
        <div className="flex items-center space-x-4">
          {/* Mobile Menu Button */}
          <Button
            variant="ghost"
            size="sm"
            onClick={onMenuClick}
            className="lg:hidden"
          >
            <Menu className="w-5 h-5" />
          </Button>

          {/* Logo & Brand */}
          <div className="flex items-center space-x-3">
            <motion.div
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.5 }}
              className="flex items-center justify-center w-8 h-8 bg-gradient-brand rounded-lg"
            >
              <Sparkles className="w-4 h-4 text-white" />
            </motion.div>
            
            <div className="hidden sm:block">
              <h1 className="text-xl font-bold text-gradient-brand">
                StructuraAI
              </h1>
            </div>
          </div>

          {/* Page Info */}
          <div className="hidden md:block border-l border-border/40 pl-4">
            <div className="flex items-center space-x-2">
              <h2 className="text-lg font-semibold">{title}</h2>
              {projectId && (
                <Badge variant="outline" className="text-xs">
                  Project {projectId}
                </Badge>
              )}
            </div>
            <p className="text-sm text-muted-foreground">{subtitle}</p>
          </div>
        </div>

        {/* Right Side */}
        <div className="flex items-center space-x-3">
          {/* Search */}
          <Button variant="ghost" size="sm" className="hidden sm:flex">
            <Search className="w-4 h-4 mr-2" />
            Search
            <kbd className="ml-2 px-1.5 py-0.5 text-xs bg-muted rounded">
              ⌘K
            </kbd>
          </Button>

          {/* Notifications */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="relative">
                <Bell className="w-4 h-4" />
                <Badge 
                  variant="destructive" 
                  className="absolute -top-1 -right-1 w-2 h-2 p-0 text-[10px]"
                >
                  3
                </Badge>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-80">
              <DropdownMenuLabel>Notifications</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <div className="space-y-2 p-2">
                <NotificationItem
                  title="Index Build Complete"
                  description="FAISS index for Project Alpha is ready"
                  time="2 min ago"
                  type="success"
                />
                <NotificationItem
                  title="New Data Source Added"
                  description="CSV file uploaded to Project Beta"
                  time="1 hour ago"
                  type="info"
                />
                <NotificationItem
                  title="Query Failed"
                  description="Invalid SQL generated for complex query"
                  time="3 hours ago"
                  type="error"
                />
              </div>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* User Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="flex items-center space-x-2">
                <div className="w-6 h-6 bg-gradient-brand rounded-full flex items-center justify-center">
                  <User className="w-3 h-3 text-white" />
                </div>
                <span className="hidden sm:inline text-sm">Demo User</span>
                <ChevronDown className="w-3 h-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>My Account</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <User className="w-4 h-4 mr-2" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Database className="w-4 h-4 mr-2" />
                API Keys
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="text-red-600">
                Sign Out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  )
}

// Notification Item Component
function NotificationItem({ 
  title, 
  description, 
  time, 
  type 
}: { 
  title: string
  description: string
  time: string
  type: 'success' | 'info' | 'error' | 'warning'
}) {
  const getIndicatorColor = () => {
    switch (type) {
      case 'success': return 'bg-green-500'
      case 'info': return 'bg-blue-500'
      case 'error': return 'bg-red-500'
      case 'warning': return 'bg-yellow-500'
      default: return 'bg-gray-500'
    }
  }

  return (
    <div className="flex items-start space-x-3 p-2 rounded-lg hover:bg-muted/50 cursor-pointer">
      <div className={`w-2 h-2 rounded-full mt-2 ${getIndicatorColor()}`} />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{title}</p>
        <p className="text-xs text-muted-foreground truncate">{description}</p>
        <p className="text-xs text-muted-foreground mt-1">{time}</p>
      </div>
    </div>
  )
}