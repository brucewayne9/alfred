import { useCallback } from 'react'
import { notificationsApi } from '../api/notifications'

export function usePushNotifications() {
  const subscribe = useCallback(async () => {
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) return false

    try {
      const { publicKey } = await notificationsApi.vapidKey()
      if (!publicKey) return false

      const registration = await navigator.serviceWorker.ready
      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(publicKey),
      })

      const json = subscription.toJSON()
      await notificationsApi.subscribe(json.endpoint!, json.keys as { p256dh: string; auth: string })
      return true
    } catch {
      return false
    }
  }, [])

  return { subscribe }
}

function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4)
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/')
  const rawData = atob(base64)
  return Uint8Array.from(rawData, (char) => char.charCodeAt(0))
}
