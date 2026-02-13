import { useCallback } from 'react'
import { useChatStore } from '../stores/chatStore'
import { useVoiceStore } from '../stores/voiceStore'
import { voiceApi } from '../api/voice'
import { playAudioBuffer, stopAudio } from '../lib/audio'

export function useChat() {
  const chat = useChatStore()
  const autoSpeak = useVoiceStore(s => s.autoSpeak)

  const send = useCallback(async (text: string, imageBase64?: string, imageMediaType?: string) => {
    // Stop any currently playing TTS before sending new message
    stopAudio()

    await chat.sendMessage(text, imageBase64, imageMediaType)

    // Auto-speak response if enabled
    if (autoSpeak) {
      const messages = useChatStore.getState().messages
      const lastMsg = messages[messages.length - 1]
      if (lastMsg?.role === 'assistant' && lastMsg.content) {
        try {
          const audio = await voiceApi.speak(lastMsg.content)
          await playAudioBuffer(audio)
        } catch { /* ignore TTS errors */ }
      }
    }
  }, [chat, autoSpeak])

  return { ...chat, send }
}
