import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuthStore } from '../../stores/authStore'
import { KeyRound, LogIn } from 'lucide-react'
import { TOTPInput } from './TOTPInput'
import { PasskeyButton } from './PasskeyButton'

export function LoginOverlay() {
  const { login, error, clearError } = useAuthStore()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [needs2fa, setNeeds2fa] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Matrix rain effect
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#$%^&*()アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン'
    const fontSize = 16
    const columns = Math.floor(canvas.width / fontSize)
    const drops = new Array(columns).fill(0).map(() => Math.random() * -50)

    const draw = () => {
      // Slower fade = longer trails
      ctx.fillStyle = 'rgba(10, 10, 10, 0.03)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.font = `${fontSize}px monospace`

      for (let i = 0; i < drops.length; i++) {
        const text = chars[Math.floor(Math.random() * chars.length)]
        const y = drops[i] * fontSize

        // Bright head character (white-orange glow)
        ctx.shadowBlur = 8
        ctx.shadowColor = '#f97316'
        ctx.fillStyle = '#ffffff'
        ctx.fillText(text, i * fontSize, y)

        // Trail character just above (bright orange)
        ctx.shadowBlur = 0
        ctx.fillStyle = 'rgba(249, 115, 22, 0.8)'
        const trailChar = chars[Math.floor(Math.random() * chars.length)]
        ctx.fillText(trailChar, i * fontSize, y - fontSize)

        // Mid-trail (dimmer orange)
        ctx.fillStyle = 'rgba(249, 115, 22, 0.4)'
        const midChar = chars[Math.floor(Math.random() * chars.length)]
        ctx.fillText(midChar, i * fontSize, y - fontSize * 2)

        if (y > canvas.height && Math.random() > 0.95) {
          drops[i] = 0
        }
        drops[i]++
      }
      ctx.shadowBlur = 0
    }

    const interval = setInterval(draw, 45)
    return () => {
      clearInterval(interval)
      window.removeEventListener('resize', resize)
    }
  }, [])

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username || !password) return
    setLoading(true)
    clearError()
    try {
      const res = await login(username, password)
      if (res.requires_2fa) setNeeds2fa(true)
    } catch { /* error shown via store */ }
    setLoading(false)
  }, [username, password, login, clearError])

  if (needs2fa) {
    return (
      <div className="h-full flex items-center justify-center bg-alfred-bg relative">
        <canvas ref={canvasRef} className="absolute inset-0" />
        <TOTPInput username={username} onBack={() => setNeeds2fa(false)} />
      </div>
    )
  }

  return (
    <div className="h-full flex items-center justify-center bg-alfred-bg relative">
      <canvas ref={canvasRef} className="absolute inset-0" />

      <div className="relative z-10 w-full max-w-sm px-6 py-8 mx-4 bg-alfred-bg/80 backdrop-blur-sm rounded-2xl border border-alfred-border/50">
        <div className="text-center mb-8">
          <img src="/alfred-icon.jpg" alt="Alfred" className="w-20 h-20 rounded-full mx-auto mb-4 border-2 border-alfred-accent" />
          <h1 className="text-2xl font-bold text-white">Alfred</h1>
          <p className="text-alfred-muted text-sm mt-1">Personal AI Assistant</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={e => setUsername(e.target.value)}
            autoComplete="username"
            className="w-full px-4 py-3 bg-alfred-surface border border-alfred-border rounded-xl text-white placeholder-alfred-muted focus:outline-none focus:border-alfred-accent transition-colors"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            autoComplete="current-password"
            className="w-full px-4 py-3 bg-alfred-surface border border-alfred-border rounded-xl text-white placeholder-alfred-muted focus:outline-none focus:border-alfred-accent transition-colors"
          />

          {error && (
            <p className="text-red-400 text-sm text-center">{error}</p>
          )}

          <button
            type="submit"
            disabled={loading || !username || !password}
            className="w-full py-3 bg-alfred-accent text-white rounded-xl font-medium flex items-center justify-center gap-2 hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <LogIn size={18} />
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <div className="mt-4 flex items-center gap-3">
          <div className="flex-1 h-px bg-alfred-border" />
          <span className="text-alfred-muted text-xs">or</span>
          <div className="flex-1 h-px bg-alfred-border" />
        </div>

        <PasskeyButton />
      </div>
    </div>
  )
}
