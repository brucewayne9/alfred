import { useEffect } from 'react'
import { useAuthStore } from './stores/authStore'
import { LoginOverlay } from './components/auth/LoginOverlay'
import { AppLayout } from './components/layout/AppLayout'

export default function App() {
  const { isAuthenticated, isLoading, checkAuth } = useAuthStore()

  useEffect(() => {
    checkAuth()
  }, [checkAuth])

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center bg-alfred-bg">
        <div className="thinking-morph w-10 h-10 bg-alfred-accent" />
      </div>
    )
  }

  if (!isAuthenticated) {
    return <LoginOverlay />
  }

  return <AppLayout />
}
