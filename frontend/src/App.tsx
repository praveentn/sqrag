// frontend/src/App.tsx
import React, { Suspense } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'

import Layout from '@/components/layout/Layout'
import LoadingSpinner from '@/components/ui/LoadingSpinner'
import ErrorBoundary from '@/components/ui/ErrorBoundary'

// Lazy load pages for better performance
const Projects = React.lazy(() => import('@/components/features/projects/ProjectsPage'))
const ProjectDetails = React.lazy(() => import('@/components/features/projects/ProjectDetailsPage'))
const Sources = React.lazy(() => import('@/components/features/sources/SourcesPage'))
const Dictionary = React.lazy(() => import('@/components/features/dictionary/DictionaryPage'))
const Embeddings = React.lazy(() => import('@/components/features/embeddings/EmbeddingsPage'))
const Search = React.lazy(() => import('@/components/features/search/SearchPage'))
const Chat = React.lazy(() => import('@/components/features/chat/ChatPage'))
const Admin = React.lazy(() => import('@/components/features/admin/AdminPage'))

function App() {
  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-background">
        {/* Background Pattern */}
        <div className="fixed inset-0 opacity-5 pointer-events-none">
          <div className="absolute inset-0 bg-grid-pattern bg-grid" />
          <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-transparent to-secondary/20" />
        </div>
        
        {/* Main Content */}
        <div className="relative z-10">
          <Routes>
            <Route path="/" element={<Layout />}>
              {/* Redirect root to projects */}
              <Route index element={<Navigate to="/projects" replace />} />
              
              {/* Projects Routes */}
              <Route
                path="projects"
                element={
                  <SuspenseWrapper>
                    <Projects />
                  </SuspenseWrapper>
                }
              />
              <Route
                path="projects/:projectId"
                element={
                  <SuspenseWrapper>
                    <ProjectDetails />
                  </SuspenseWrapper>
                }
              />
              
              {/* Data Sources */}
              <Route
                path="projects/:projectId/sources"
                element={
                  <SuspenseWrapper>
                    <Sources />
                  </SuspenseWrapper>
                }
              />
              
              {/* Dictionary */}
              <Route
                path="projects/:projectId/dictionary"
                element={
                  <SuspenseWrapper>
                    <Dictionary />
                  </SuspenseWrapper>
                }
              />
              
              {/* Embeddings & Indexing */}
              <Route
                path="projects/:projectId/embeddings"
                element={
                  <SuspenseWrapper>
                    <Embeddings />
                  </SuspenseWrapper>
                }
              />
              
              {/* Search */}
              <Route
                path="projects/:projectId/search"
                element={
                  <SuspenseWrapper>
                    <Search />
                  </SuspenseWrapper>
                }
              />
              
              {/* Chat */}
              <Route
                path="projects/:projectId/chat"
                element={
                  <SuspenseWrapper>
                    <Chat />
                  </SuspenseWrapper>
                }
              />
              
              {/* Admin */}
              <Route
                path="admin"
                element={
                  <SuspenseWrapper>
                    <Admin />
                  </SuspenseWrapper>
                }
              />
              
              {/* Catch all route */}
              <Route path="*" element={<NotFound />} />
            </Route>
          </Routes>
        </div>
      </div>
    </ErrorBoundary>
  )
}

// Suspense wrapper with beautiful loading animation
function SuspenseWrapper({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center min-h-[60vh]">
          <LoadingSpinner size="lg" />
        </div>
      }
    >
      <AnimatePresence mode="wait">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{
            duration: 0.3,
            ease: "easeInOut"
          }}
        >
          {children}
        </motion.div>
      </AnimatePresence>
    </Suspense>
  )
}

// 404 Not Found Component
function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="space-y-6"
      >
        {/* 404 Animation */}
        <div className="relative">
          <h1 className="text-9xl font-bold text-gradient-brand opacity-20">
            404
          </h1>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="glass-card p-8 rounded-2xl">
              <h2 className="text-2xl font-semibold mb-2">Page Not Found</h2>
              <p className="text-muted-foreground mb-6">
                The page you're looking for doesn't exist or has been moved.
              </p>
              <motion.button
                onClick={() => window.history.back()}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors"
              >
                Go Back
              </motion.button>
            </div>
          </div>
        </div>
        
        {/* Floating Elements */}
        <div className="absolute inset-0 pointer-events-none">
          {[...Array(5)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 bg-primary/30 rounded-full"
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

export default App