// frontend/src/components/features/sources/SourcesPage.tsx
import React from 'react'
import { useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Database } from 'lucide-react'

export default function SourcesPage() {
  const { projectId } = useParams()
  
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
            <Database className="w-10 h-10 text-white" />
          </div>
        </motion.div>
        
        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-gradient-brand">Data Sources</h1>
          <p className="text-muted-foreground">Upload files, connect databases, and manage your data connections</p>
        </div>
        
        <div className="glass-card p-4 rounded-lg">
          <p className="text-sm text-muted-foreground">
            🚧 This feature is under development and will be available soon!
          </p>
        </div>
      </motion.div>
    </div>
  )
}


