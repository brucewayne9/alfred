import { useEffect } from 'react'
import { useAuthStore } from '../stores/authStore'

export function useAuth() {
  const store = useAuthStore()

  useEffect(() => {
    store.checkAuth()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return store
}
