import { useEffect, useRef, useCallback } from 'react'
import { useNotificationStore } from '../stores/notificationStore'

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout>>()
  const addToast = useNotificationStore(s => s.addToast)
  const setWsConnected = useNotificationStore(s => s.setWsConnected)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    // Pass token as query param — cookies may not be sent for WebSocket in all browsers
    const token = document.cookie.split('; ').find(c => c.startsWith('alfred_token='))?.split('=')[1] || ''
    const url = token
      ? `${protocol}//${window.location.host}/ws/notifications?token=${token}`
      : `${protocol}//${window.location.host}/ws/notifications`
    const ws = new WebSocket(url)

    ws.onopen = () => {
      setWsConnected(true)
      // Start ping interval
      const ping = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping')
      }, 30000)
      ws.addEventListener('close', () => clearInterval(ping))
    }

    ws.onmessage = (event) => {
      if (event.data === 'pong') return
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'connected') return

        // Map notification types to toasts
        const typeMap: Record<string, 'success' | 'error' | 'info'> = {
          agent_completed: 'success',
          agent_failed: 'error',
          agent_started: 'info',
          system_alert: 'warning' as 'info',
          long_processing: 'info',
        }
        addToast({
          type: typeMap[data.type] || 'info',
          title: data.type.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase()),
          message: data.data?.result_preview || data.data?.message || data.data?.error || '',
        })
      } catch { /* ignore */ }
    }

    ws.onclose = () => {
      setWsConnected(false)
      // Reconnect with backoff
      reconnectTimeout.current = setTimeout(connect, 5000)
    }

    ws.onerror = () => ws.close()

    wsRef.current = ws
  }, [addToast, setWsConnected])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimeout.current)
      wsRef.current?.close()
    }
  }, [connect])
}
