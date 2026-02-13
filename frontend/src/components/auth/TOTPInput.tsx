import { useState, useCallback } from 'react'
import { useAuthStore } from '../../stores/authStore'
import { ShieldCheck, ArrowLeft } from 'lucide-react'

interface TOTPInputProps {
  username: string
  onBack: () => void
}

export function TOTPInput({ username, onBack }: TOTPInputProps) {
  const { login2fa, error, clearError } = useAuthStore()
  const [code, setCode] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    if (code.length !== 6) return
    setLoading(true)
    clearError()
    try {
      await login2fa(username, code)
    } catch { /* error shown via store */ }
    setLoading(false)
  }, [code, username, login2fa, clearError])

  return (
    <div className="relative z-10 w-full max-w-sm px-6">
      <div className="text-center mb-8">
        <ShieldCheck size={48} className="mx-auto mb-4 text-alfred-accent" />
        <h2 className="text-xl font-bold text-white">Two-Factor Authentication</h2>
        <p className="text-alfred-muted text-sm mt-1">Enter the code from your authenticator app</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="text"
          inputMode="numeric"
          pattern="[0-9]*"
          maxLength={6}
          placeholder="000000"
          value={code}
          onChange={e => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
          autoFocus
          className="w-full px-4 py-3 bg-alfred-surface border border-alfred-border rounded-xl text-white text-center text-2xl tracking-[0.3em] font-mono placeholder-alfred-muted focus:outline-none focus:border-alfred-accent transition-colors"
        />

        {error && <p className="text-red-400 text-sm text-center">{error}</p>}

        <button
          type="submit"
          disabled={loading || code.length !== 6}
          className="w-full py-3 bg-alfred-accent text-white rounded-xl font-medium hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Verifying...' : 'Verify'}
        </button>

        <button
          type="button"
          onClick={onBack}
          className="w-full py-2 text-alfred-muted text-sm flex items-center justify-center gap-1 hover:text-white transition-colors"
        >
          <ArrowLeft size={14} /> Back to login
        </button>
      </form>
    </div>
  )
}
