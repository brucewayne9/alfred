import { apiFetch } from './client'

export interface AuthUser {
  authenticated: boolean
  username?: string
  role?: string
}

export interface LoginResponse {
  token?: string
  username?: string
  role?: string
  requires_2fa?: boolean
}

export const authApi = {
  me: () => apiFetch<AuthUser>('/auth/me'),

  autoLogin: () => apiFetch<{ auto_login: boolean; username?: string }>('/auth/auto'),

  login: (username: string, password: string) =>
    apiFetch<LoginResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),

  login2fa: (username: string, code: string) =>
    apiFetch<LoginResponse>('/auth/2fa/login', {
      method: 'POST',
      body: JSON.stringify({ username, code }),
    }),

  logout: () => apiFetch('/auth/logout', { method: 'POST' }),

  passkeyLoginBegin: () =>
    apiFetch<Record<string, unknown>>('/auth/passkey/login/begin', { method: 'POST' }),

  passkeyLoginComplete: (credential: Record<string, unknown>) =>
    apiFetch<LoginResponse>('/auth/passkey/login/complete', {
      method: 'POST',
      body: JSON.stringify({ credential }),
    }),

  methods: (username: string) =>
    apiFetch<{ exists: boolean; totp_enabled: boolean; has_passkeys: boolean }>(
      `/auth/methods?username=${encodeURIComponent(username)}`,
    ),
}
