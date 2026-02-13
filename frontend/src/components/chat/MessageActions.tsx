import { useState, useCallback } from 'react'
import { Copy, Check, Volume2 } from 'lucide-react'
import { voiceApi } from '../../api/voice'
import { playAudioBuffer } from '../../lib/audio'

interface MessageActionsProps {
  content: string
}

export function MessageActions({ content }: MessageActionsProps) {
  const [copied, setCopied] = useState(false)
  const [speaking, setSpeaking] = useState(false)

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [content])

  const handleSpeak = useCallback(async () => {
    if (speaking) return
    setSpeaking(true)
    try {
      const audio = await voiceApi.speak(content)
      await playAudioBuffer(audio)
    } catch { /* ignore */ }
    setSpeaking(false)
  }, [content, speaking])

  return (
    <div className="flex items-center gap-1 mt-1">
      <button
        onClick={handleCopy}
        className="p-1 rounded text-alfred-muted hover:text-white hover:bg-alfred-hover transition-colors"
        title="Copy"
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
      <button
        onClick={handleSpeak}
        className={`p-1 rounded transition-colors ${speaking ? 'text-alfred-accent' : 'text-alfred-muted hover:text-white hover:bg-alfred-hover'}`}
        title="Speak"
      >
        <Volume2 size={14} />
      </button>
    </div>
  )
}
