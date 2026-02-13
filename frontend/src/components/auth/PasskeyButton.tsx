import { useState } from 'react'
import { useAuthStore } from '../../stores/authStore'
import { KeyRound } from 'lucide-react'

export function PasskeyButton() {
  const { loginWithPasskey } = useAuthStore()
  const [loading, setLoading] = useState(false)

  const handleClick = async () => {
    setLoading(true)
    try {
      await loginWithPasskey()
    } catch { /* error shown in store */ }
    setLoading(false)
  }

  // Only show if WebAuthn is available
  if (!window.PublicKeyCredential) return null

  return (
    <button
      onClick={handleClick}
      disabled={loading}
      className="w-full mt-4 py-3 bg-alfred-surface border border-alfred-border text-white rounded-xl font-medium flex items-center justify-center gap-2 hover:bg-alfred-hover disabled:opacity-50 transition-colors"
    >
      <KeyRound size={18} />
      {loading ? 'Authenticating...' : 'Sign in with Passkey'}
    </button>
  )
}
