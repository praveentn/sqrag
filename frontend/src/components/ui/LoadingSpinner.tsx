// frontend/src/components/ui/LoadingSpinner.tsx
import React from 'react'
import { motion } from 'framer-motion'
import { Loader2, Database, Sparkles } from 'lucide-react'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/utils'

const spinnerVariants = cva(
  "flex items-center justify-center",
  {
    variants: {
      size: {
        sm: "w-4 h-4",
        md: "w-6 h-6", 
        lg: "w-8 h-8",
        xl: "w-12 h-12",
      },
      variant: {
        default: "text-primary",
        muted: "text-muted-foreground",
        white: "text-white",
        gradient: "text-transparent",
      }
    },
    defaultVariants: {
      size: "md",
      variant: "default"
    }
  }
)

interface LoadingSpinnerProps extends VariantProps<typeof spinnerVariants> {
  className?: string
  message?: string
  showMessage?: boolean
  type?: 'default' | 'dots' | 'pulse' | 'brand'
}

export default function LoadingSpinner({ 
  size, 
  variant, 
  className, 
  message = "Loading...",
  showMessage = false,
  type = 'default'
}: LoadingSpinnerProps) {
  
  if (type === 'dots') {
    return (
      <div className={cn("flex flex-col items-center space-y-3", className)}>
        <div className="flex space-x-1">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className={cn(
                "rounded-full bg-primary",
                size === 'sm' && "w-1 h-1",
                size === 'md' && "w-2 h-2",
                size === 'lg' && "w-3 h-3",
                size === 'xl' && "w-4 h-4"
              )}
              animate={{
                y: [0, -10, 0],
                opacity: [0.7, 1, 0.7],
              }}
              transition={{
                duration: 0.8,
                repeat: Infinity,
                delay: i * 0.2,
                ease: "easeInOut"
              }}
            />
          ))}
        </div>
        {showMessage && (
          <p className="text-sm text-muted-foreground">{message}</p>
        )}
      </div>
    )
  }

  if (type === 'pulse') {
    return (
      <div className={cn("flex flex-col items-center space-y-3", className)}>
        <motion.div
          className={cn(
            "rounded-full bg-primary/20 border-2 border-primary/30",
            size === 'sm' && "w-8 h-8",
            size === 'md' && "w-12 h-12",
            size === 'lg' && "w-16 h-16",
            size === 'xl' && "w-24 h-24"
          )}
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.5, 0.8, 0.5],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        >
          <div className={cn(
            "w-full h-full rounded-full bg-primary/40",
            "flex items-center justify-center"
          )}>
            <Database className={cn(
              "text-primary",
              size === 'sm' && "w-3 h-3",
              size === 'md' && "w-4 h-4",
              size === 'lg' && "w-6 h-6",
              size === 'xl' && "w-8 h-8"
            )} />
          </div>
        </motion.div>
        {showMessage && (
          <p className="text-sm text-muted-foreground">{message}</p>
        )}
      </div>
    )
  }

  if (type === 'brand') {
    return (
      <div className={cn("flex flex-col items-center space-y-4", className)}>
        <motion.div
          className="relative"
          animate={{ rotate: 360 }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "linear"
          }}
        >
          <div className={cn(
            "relative rounded-full border-2 border-transparent bg-gradient-brand",
            size === 'sm' && "w-8 h-8",
            size === 'md' && "w-12 h-12",
            size === 'lg' && "w-16 h-16",
            size === 'xl' && "w-24 h-24"
          )}>
            <div className="absolute inset-1 rounded-full bg-background flex items-center justify-center">
              <motion.div
                animate={{ rotate: -360 }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: "linear"
                }}
              >
                <Sparkles className={cn(
                  "text-primary",
                  size === 'sm' && "w-3 h-3",
                  size === 'md' && "w-4 h-4",
                  size === 'lg' && "w-6 h-6",
                  size === 'xl' && "w-8 h-8"
                )} />
              </motion.div>
            </div>
          </div>
          
          {/* Orbiting dots */}
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-primary rounded-full"
              style={{
                top: '50%',
                left: '50%',
                transformOrigin: `${size === 'xl' ? '48px' : size === 'lg' ? '32px' : size === 'md' ? '24px' : '16px'} 0`,
              }}
              animate={{ rotate: 360 }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                delay: i * 0.3,
                ease: "linear"
              }}
            />
          ))}
        </motion.div>
        
        {showMessage && (
          <motion.p 
            className="text-sm text-muted-foreground text-center"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            {message}
          </motion.p>
        )}
      </div>
    )
  }

  // Default spinner
  return (
    <div className={cn("flex flex-col items-center space-y-3", className)}>
      <Loader2 className={cn(
        spinnerVariants({ size, variant }),
        "animate-spin"
      )} />
      {showMessage && (
        <p className="text-sm text-muted-foreground">{message}</p>
      )}
    </div>
  )
}