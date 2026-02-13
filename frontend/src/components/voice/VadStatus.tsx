import { useVoiceStore } from '../../stores/voiceStore'
import { Radio } from 'lucide-react'

export function VadStatus() {
  const handsFreeActive = useVoiceStore(s => s.handsFreeActive)

  if (!handsFreeActive) return null

  return (
    <div className="fixed top-14 right-4 z-50 flex items-center gap-2 px-3 py-1.5 bg-alfred-accent/20 border border-alfred-accent/30 rounded-full">
      <Radio size={14} className="text-alfred-accent animate-pulse" />
      <span className="text-xs text-alfred-accent font-medium">Hands-free</span>
    </div>
  )
}
