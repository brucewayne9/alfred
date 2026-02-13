import { useVoice } from '../../hooks/useVoice'
import { Mic } from 'lucide-react'

export function MicButton() {
  const { isRecording, toggleRecording } = useVoice()

  return (
    <button
      onClick={toggleRecording}
      className={`p-1.5 rounded-lg transition-colors shrink-0 mb-0.5 ${
        isRecording
          ? 'text-red-400 bg-red-400/10 mic-active'
          : 'text-alfred-muted hover:text-white hover:bg-alfred-hover'
      }`}
      title={isRecording ? 'Stop recording' : 'Voice input'}
    >
      <Mic size={18} />
    </button>
  )
}
