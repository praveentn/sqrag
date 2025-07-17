// frontend/src/hooks/useTheme.ts
import { useState, useEffect, useCallback } from 'react'

type Theme = 'light' | 'dark' | 'system'

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    // Get initial theme from localStorage or default to system
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('theme') as Theme
      return stored || 'system'
    }
    return 'system'
  })

  // Get the actual theme (resolve 'system' to 'light' or 'dark')
  const getResolvedTheme = useCallback((): 'light' | 'dark' => {
    if (theme === 'system') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    }
    return theme
  }, [theme])

  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>(() => {
    if (typeof window !== 'undefined') {
      return getResolvedTheme()
    }
    return 'light'
  })

  // Apply theme to document
  const applyTheme = useCallback((newTheme: 'light' | 'dark') => {
    const root = document.documentElement
    root.classList.remove('light', 'dark')
    root.classList.add(newTheme)
    root.setAttribute('data-theme', newTheme)
    setResolvedTheme(newTheme)
  }, [])

  // Handle system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    
    const handleChange = () => {
      if (theme === 'system') {
        const newTheme = getResolvedTheme()
        applyTheme(newTheme)
      }
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [theme, getResolvedTheme, applyTheme])

  // Apply theme whenever it changes
  useEffect(() => {
    const newResolvedTheme = getResolvedTheme()
    applyTheme(newResolvedTheme)
    
    // Store in localStorage
    localStorage.setItem('theme', theme)
  }, [theme, getResolvedTheme, applyTheme])

  // Toggle between light and dark (skip system)
  const toggleTheme = useCallback(() => {
    setTheme(current => {
      if (current === 'light') return 'dark'
      if (current === 'dark') return 'light'
      // If system, toggle to opposite of current resolved theme
      return resolvedTheme === 'dark' ? 'light' : 'dark'
    })
  }, [resolvedTheme])

  // Set specific theme
  const setThemeMode = useCallback((newTheme: Theme) => {
    setTheme(newTheme)
  }, [])

  return {
    theme,
    resolvedTheme,
    setTheme: setThemeMode,
    toggleTheme,
    isDark: resolvedTheme === 'dark',
    isLight: resolvedTheme === 'light',
    isSystem: theme === 'system'
  }
}