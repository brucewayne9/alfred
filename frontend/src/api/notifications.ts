import { apiFetch } from './client'

export const notificationsApi = {
  status: () =>
    apiFetch<{ connected_clients: number; push_subscriptions: number }>('/notifications/status'),

  vapidKey: () => apiFetch<{ publicKey: string }>('/push/vapid-key'),

  subscribe: (endpoint: string, keys: { p256dh: string; auth: string }) =>
    apiFetch('/push/subscribe', {
      method: 'POST',
      body: JSON.stringify({ endpoint, keys }),
    }),

  unsubscribe: (endpoint: string) =>
    apiFetch('/push/unsubscribe', {
      method: 'POST',
      body: JSON.stringify({ endpoint }),
    }),
}
