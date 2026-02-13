import { create } from 'zustand'
import { authApi, type AuthUser } from '../api/auth'

interface AuthState {
  user: AuthUser | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null

  checkAuth: () => Promise<void>
  login: (username: string, password: string) => Promise<{ requires_2fa?: boolean }>
  login2fa: (username: string, code: string) => Promise<void>
  loginWithPasskey: () => Promise<void>
  logout: () => Promise<void>
  clearError: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isAuthenticated: false,
  isLoading: true,
  error: null,

  checkAuth: async () => {
    set({ isLoading: true, error: null })
    try {
      // 1. Check existing session cookie
      const me = await authApi.me()
      if (me.authenticated) {
        set({ user: me, isAuthenticated: true, isLoading: false })
        return
      }
      // 2. Try auto-login (local network)
      try {
        const auto = await authApi.autoLogin()
        if (auto.auto_login) {
          const me2 = await authApi.me()
          if (me2.authenticated) {
            set({ user: me2, isAuthenticated: true, isLoading: false })
            return
          }
        }
      } catch {
        // auto-login not available
      }
      set({ isAuthenticated: false, isLoading: false })
    } catch {
      set({ isAuthenticated: false, isLoading: false })
    }
  },

  login: async (username, password) => {
    set({ error: null })
    try {
      const res = await authApi.login(username, password)
      if (res.requires_2fa) {
        return { requires_2fa: true }
      }
      set({ user: { authenticated: true, username: res.username, role: res.role }, isAuthenticated: true })
      return {}
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Login failed'
      set({ error: msg })
      throw e
    }
  },

  login2fa: async (username, code) => {
    set({ error: null })
    try {
      const res = await authApi.login2fa(username, code)
      set({ user: { authenticated: true, username: res.username, role: res.role }, isAuthenticated: true })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '2FA failed'
      set({ error: msg })
      throw e
    }
  },

  loginWithPasskey: async () => {
    set({ error: null })
    try {
      const options = await authApi.passkeyLoginBegin()
      // WebAuthn ceremony
      const publicKey = {
        challenge: base64urlToBuffer(options.challenge as string),
        rpId: options.rpId as string,
        timeout: options.timeout as number,
        userVerification: options.userVerification as UserVerificationRequirement,
        allowCredentials: (options.allowCredentials as Array<{ id: string; type: string; transports?: string[] }>)?.map(c => ({
          id: base64urlToBuffer(c.id),
          type: c.type as PublicKeyCredentialType,
          transports: c.transports as AuthenticatorTransport[],
        })) || [],
      }
      const credential = await navigator.credentials.get({ publicKey }) as PublicKeyCredential
      if (!credential) throw new Error('No credential returned')

      const response = credential.response as AuthenticatorAssertionResponse
      const credentialData = {
        id: credential.id,
        rawId: bufferToBase64url(credential.rawId),
        response: {
          clientDataJSON: bufferToBase64url(response.clientDataJSON),
          authenticatorData: bufferToBase64url(response.authenticatorData),
          signature: bufferToBase64url(response.signature),
          userHandle: response.userHandle ? bufferToBase64url(response.userHandle) : null,
        },
        type: credential.type,
      }

      const res = await authApi.passkeyLoginComplete(credentialData)
      set({ user: { authenticated: true, username: res.username, role: res.role }, isAuthenticated: true })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Passkey login failed'
      set({ error: msg })
      throw e
    }
  },

  logout: async () => {
    await authApi.logout().catch(() => {})
    set({ user: null, isAuthenticated: false })
  },

  clearError: () => set({ error: null }),
}))

// Base64url helpers for WebAuthn
function base64urlToBuffer(base64url: string): ArrayBuffer {
  const base64 = base64url.replace(/-/g, '+').replace(/_/g, '/')
  const pad = base64.length % 4 === 0 ? '' : '='.repeat(4 - (base64.length % 4))
  const binary = atob(base64 + pad)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes.buffer
}

function bufferToBase64url(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}
