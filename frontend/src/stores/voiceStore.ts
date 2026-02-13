import { create } from 'zustand'
import { apiFetch } from '../api/client'

interface Voice {
  id: string
  name: string
  desc: string
  type?: string
}

interface VoiceState {
  isRecording: boolean
  handsFreeActive: boolean
  wakeWordActive: boolean
  autoSpeak: boolean
  transcribedText: string | null

  ttsBackend: string
  ttsVoice: string
  voices: Voice[]
  ttsLoading: boolean

  setRecording: (v: boolean) => void
  setHandsFree: (v: boolean) => void
  setWakeWord: (v: boolean) => void
  setAutoSpeak: (v: boolean) => void
  setTranscribedText: (v: string | null) => void
  loadTtsSettings: () => Promise<void>
  setTtsBackend: (backend: string) => Promise<void>
  setTtsVoice: (voice: string) => Promise<void>
}

export const useVoiceStore = create<VoiceState>((set) => ({
  isRecording: false,
  handsFreeActive: false,
  wakeWordActive: false,
  autoSpeak: false,
  transcribedText: null,

  ttsBackend: 'kokoro',
  ttsVoice: '',
  voices: [],
  ttsLoading: false,

  setRecording: (v) => set({ isRecording: v }),
  setHandsFree: (v) => set({ handsFreeActive: v }),
  setWakeWord: (v) => set({ wakeWordActive: v }),
  setAutoSpeak: (v) => set({ autoSpeak: v }),
  setTranscribedText: (v) => set({ transcribedText: v }),

  loadTtsSettings: async () => {
    set({ ttsLoading: true })
    try {
      const data = await apiFetch<{ backend: string; voices: Voice[]; current: string }>('/settings/voices')
      set({
        ttsBackend: data.backend,
        ttsVoice: data.current,
        voices: data.voices,
        ttsLoading: false,
      })
    } catch {
      set({ ttsLoading: false })
    }
  },

  setTtsBackend: async (backend) => {
    try {
      await apiFetch(`/settings/tts?backend=${backend}`, { method: 'PUT' })
      // Reload voices for the new backend
      const data = await apiFetch<{ backend: string; voices: Voice[]; current: string }>('/settings/voices')
      set({
        ttsBackend: data.backend,
        ttsVoice: data.current,
        voices: data.voices,
      })
    } catch { /* ignore */ }
  },

  setTtsVoice: async (voice) => {
    try {
      await apiFetch(`/settings/voice?voice=${encodeURIComponent(voice)}`, { method: 'PUT' })
      set({ ttsVoice: voice })
    } catch { /* ignore */ }
  },
}))
