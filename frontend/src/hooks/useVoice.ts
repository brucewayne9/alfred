import { useRef, useCallback } from 'react'
import { useVoiceStore } from '../stores/voiceStore'
import { voiceApi } from '../api/voice'

export function useVoice() {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const store = useVoiceStore()

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm',
      })
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(chunksRef.current, { type: mediaRecorder.mimeType })
        store.setRecording(false)

        try {
          const text = await voiceApi.transcribe(blob)
          if (text.trim()) store.setTranscribedText(text.trim())
        } catch {
          // ignore transcription errors
        }
      }

      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start()
      store.setRecording(true)

      // Request wake lock to keep screen on
      try {
        if ('wakeLock' in navigator) {
          await (navigator as unknown as { wakeLock: { request: (type: string) => Promise<unknown> } }).wakeLock.request('screen')
        }
      } catch { /* ignore */ }
    } catch {
      store.setRecording(false)
    }
  }, [store])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
  }, [])

  const toggleRecording = useCallback(() => {
    if (store.isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }, [store.isRecording, startRecording, stopRecording])

  return {
    isRecording: store.isRecording,
    transcribedText: store.transcribedText,
    clearTranscribedText: () => store.setTranscribedText(null),
    toggleRecording,
    startRecording,
    stopRecording,
  }
}
