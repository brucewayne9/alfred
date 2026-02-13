import { useState, useEffect, useRef } from 'react'
import { Menu, LogOut, Volume2, VolumeX, Headphones, AudioWaveform, Settings2 } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { useSidebarStore } from '../../stores/sidebarStore'
import { useVoiceStore } from '../../stores/voiceStore'

function TtsPopover({ onClose }: { onClose: () => void }) {
  const {
    ttsBackend, ttsVoice, voices, ttsLoading,
    loadTtsSettings, setTtsBackend, setTtsVoice,
  } = useVoiceStore()
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    loadTtsSettings()
  }, [loadTtsSettings])

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose()
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [onClose])

  return (
    <div
      ref={ref}
      className="absolute right-0 top-full mt-2 w-64 bg-alfred-surface border border-alfred-border rounded-xl shadow-2xl z-50 p-3 space-y-3"
    >
      <h4 className="text-xs font-medium text-alfred-muted uppercase tracking-wider">Voice Engine</h4>

      {/* Backend toggle */}
      <div className="flex gap-1.5">
        {['kokoro', 'qwen3'].map(b => (
          <button
            key={b}
            onClick={() => setTtsBackend(b)}
            className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-medium transition-colors ${
              ttsBackend === b
                ? 'bg-alfred-accent text-white'
                : 'bg-alfred-hover text-alfred-muted hover:text-white'
            }`}
          >
            {b === 'kokoro' ? 'Kokoro' : 'Qwen3'}
          </button>
        ))}
      </div>

      {/* Voice selector */}
      <div>
        <label className="text-xs text-alfred-muted block mb-1">Voice</label>
        {ttsLoading ? (
          <div className="text-xs text-alfred-muted py-2">Loading voices...</div>
        ) : (
          <select
            value={ttsVoice}
            onChange={e => setTtsVoice(e.target.value)}
            className="w-full px-2.5 py-1.5 bg-alfred-bg border border-alfred-border rounded-lg text-sm text-white outline-none focus:border-alfred-accent/50 transition-colors"
          >
            {voices.map(v => (
              <option key={v.id} value={v.id}>
                {v.name} — {v.desc}
              </option>
            ))}
          </select>
        )}
      </div>

      <p className="text-[10px] text-alfred-muted/60">
        {ttsBackend === 'kokoro' ? 'Fast, local TTS' : 'High quality, voice cloning'}
      </p>
    </div>
  )
}

export function Header() {
  const { logout, user } = useAuthStore()
  const toggleSidebar = useSidebarStore(s => s.toggle)
  const { autoSpeak, setAutoSpeak, handsFreeActive, setHandsFree, wakeWordActive, setWakeWord } = useVoiceStore()
  const [showTts, setShowTts] = useState(false)

  return (
    <header className="flex items-center justify-between px-4 py-2 border-b border-alfred-border bg-alfred-bg z-20 shrink-0">
      <div className="flex items-center gap-3">
        <button
          onClick={toggleSidebar}
          className="p-2 rounded-lg hover:bg-alfred-hover text-alfred-muted hover:text-white transition-colors"
          aria-label="Toggle sidebar"
        >
          <Menu size={20} />
        </button>
        <div className="flex items-center gap-2">
          <img src="/alfred-icon.jpg" alt="Alfred" className="w-7 h-7 rounded-full" />
          <span className="font-semibold text-white text-sm hidden sm:inline">Alfred</span>
        </div>
      </div>

      <div className="flex items-center gap-1.5 relative">
        <button
          onClick={() => setAutoSpeak(!autoSpeak)}
          className={`p-2 rounded-lg transition-colors ${autoSpeak ? 'text-alfred-accent bg-alfred-accent/10' : 'text-alfred-muted hover:text-white hover:bg-alfred-hover'}`}
          title={autoSpeak ? 'Auto-speak on' : 'Auto-speak off'}
        >
          {autoSpeak ? <Volume2 size={18} /> : <VolumeX size={18} />}
        </button>

        <button
          onClick={() => setHandsFree(!handsFreeActive)}
          className={`p-2 rounded-lg transition-colors ${handsFreeActive ? 'text-alfred-accent bg-alfred-accent/10' : 'text-alfred-muted hover:text-white hover:bg-alfred-hover'}`}
          title={handsFreeActive ? 'Hands-free active' : 'Hands-free off'}
        >
          <Headphones size={18} />
        </button>

        <button
          onClick={() => setWakeWord(!wakeWordActive)}
          className={`p-2 rounded-lg transition-colors ${wakeWordActive ? 'text-alfred-accent bg-alfred-accent/10' : 'text-alfred-muted hover:text-white hover:bg-alfred-hover'}`}
          title={wakeWordActive ? '"Hey Alfred" listening' : '"Hey Alfred" off'}
        >
          <AudioWaveform size={18} />
        </button>

        <button
          onClick={() => setShowTts(!showTts)}
          className={`p-2 rounded-lg transition-colors ${showTts ? 'text-alfred-accent bg-alfred-accent/10' : 'text-alfred-muted hover:text-white hover:bg-alfred-hover'}`}
          title="Voice settings"
        >
          <Settings2 size={18} />
        </button>

        {showTts && <TtsPopover onClose={() => setShowTts(false)} />}

        {user?.username && (
          <span className="text-alfred-muted text-xs hidden sm:inline ml-1">{user.username}</span>
        )}

        <button
          onClick={logout}
          className="p-2 rounded-lg text-alfred-muted hover:text-red-400 hover:bg-alfred-hover transition-colors"
          title="Sign out"
        >
          <LogOut size={18} />
        </button>
      </div>
    </header>
  )
}
