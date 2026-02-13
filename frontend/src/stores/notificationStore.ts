import { create } from 'zustand'

export interface Toast {
  id: string
  type: 'info' | 'success' | 'error' | 'warning'
  title: string
  message: string
  duration?: number
}

interface NotificationState {
  toasts: Toast[]
  wsConnected: boolean

  addToast: (toast: Omit<Toast, 'id'>) => void
  removeToast: (id: string) => void
  setWsConnected: (v: boolean) => void
}

let toastId = 0

export const useNotificationStore = create<NotificationState>((set) => ({
  toasts: [],
  wsConnected: false,

  addToast: (toast) => {
    const id = `toast-${++toastId}`
    set(s => ({ toasts: [...s.toasts, { ...toast, id }] }))
    // Auto-dismiss
    const duration = toast.duration ?? 5000
    if (duration > 0) {
      setTimeout(() => {
        set(s => ({ toasts: s.toasts.filter(t => t.id !== id) }))
      }, duration)
    }
  },

  removeToast: (id) => set(s => ({ toasts: s.toasts.filter(t => t.id !== id) })),
  setWsConnected: (v) => set({ wsConnected: v }),
}))
