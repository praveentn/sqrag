// frontend/src/components/features/placeholder-pages.tsx
import React from 'react'
import { useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Construction, Sparkles } from 'lucide-react'

// Reusable placeholder component
function PlaceholderPage({ 
  title, 
  description, 
  icon: Icon = Construction 
}: { 
  title: string
  description: string
  icon?: React.ComponentType<any>
}) {
  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="text-center space-y-6 max-w-md"
      >
        <motion.div
          animate={{ 
            rotate: [0, 10, -10, 0],
            scale: [1, 1.1, 1]
          }}
          transition={{ 
            duration: 2,
            repeat: Infinity,
            repeatType: "reverse"
          }}
          className="flex justify-center"
        >
          <div className="w-20 h-20 bg-gradient-brand rounded-2xl flex items-center justify-center shadow-lg">
            <Icon className="w-10 h-10 text-white" />
          </div>
        </motion.div>
        
        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-gradient-brand">{title}</h1>
          <p className="text-muted-foreground">{description}</p>
        </div>
        
        <div className="glass-card p-4 rounded-lg">
          <p className="text-sm text-muted-foreground">
            🚧 This feature is under development and will be available soon!
          </p>
        </div>

        {/* Floating animation elements */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          {[...Array(5)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-primary/30 rounded-full"
              style={{
                left: `${20 + i * 15}%`,
                top: `${30 + i * 10}%`,
              }}
              animate={{
                y: [0, -20, 0],
                opacity: [0.3, 0.8, 0.3],
              }}
              transition={{
                duration: 2 + i * 0.5,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
          ))}
        </div>
      </motion.div>
    </div>
  )
}

// Project Details Page
export function ProjectDetailsPage() {
  const { projectId } = useParams()
  
  return (
    <PlaceholderPage
      title={`Project ${projectId} Details`}
      description="Comprehensive project overview with analytics and insights"
      icon={Sparkles}
    />
  )
}

// Sources Page
export function SourcesPage() {
  const { projectId } = useParams()
  
  return (
    <PlaceholderPage
      title="Data Sources"
      description="Upload files, connect databases, and manage your data connections"
    />
  )
}

// Dictionary Page  
export function DictionaryPage() {
  const { projectId } = useParams()
  
  return (
    <PlaceholderPage
      title="Data Dictionary"
      description="Define business terms, manage glossaries, and maintain data definitions"
    />
  )
}

// Embeddings Page
export function EmbeddingsPage() {
  const { projectId } = useParams()
  
  return (
    <PlaceholderPage
      title="Embeddings & Indexing"
      description="Create vector embeddings and build search indexes for your data"
    />
  )
}

// Search Page
export function SearchPage() {
  const { projectId } = useParams()
  
  return (
    <PlaceholderPage
      title="Search Playground"
      description="Test semantic search, keyword matching, and explore your data"
    />
  )
}

// Chat Page
export function ChatPage() {
  const { projectId } = useParams()
  
  return (
    <PlaceholderPage
      title="Natural Language Query"
      description="Ask questions in plain English and get intelligent answers from your data"
    />
  )
}

// Admin Page
export function AdminPage() {
  return (
    <PlaceholderPage
      title="Admin Panel"
      description="System administration, user management, and platform configuration"
    />
  )
}