import { apiUpload, apiFetch } from './client'

export const voiceApi = {
  transcribe: async (audioBlob: Blob): Promise<string> => {
    const formData = new FormData()
    formData.append('audio', audioBlob, 'recording.webm')
    const res = await apiUpload<{ text: string }>('/voice/transcribe', formData)
    return res.text
  },

  speak: async (text: string): Promise<ArrayBuffer> => {
    const res = await fetch('/voice/speak', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    })
    if (!res.ok) throw new Error('TTS failed')
    return res.arrayBuffer()
  },

  voices: () => apiFetch<{
    kokoro: Array<{ id: string; name: string; desc: string }>
    qwen3: Array<{ id: string; name: string; desc: string }>
    current_backend: string
    current_voice: string
  }>('/voice/voices'),
}
