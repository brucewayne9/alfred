import { useRef, useEffect, useCallback } from 'react'
import { useVoiceStore } from '../stores/voiceStore'

/**
 * "Hey Alfred" wake word detection.
 * Connects to /ws/wakeword, streams 16kHz PCM audio,
 * and triggers hands-free mode when wake word is detected.
 */
export function useWakeWord() {
  const { wakeWordActive, setWakeWord, setHandsFree } = useVoiceStore()
  const wsRef = useRef<WebSocket | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const activeRef = useRef(false)

  const startWakeWord = useCallback(async () => {
    try {
      // Get mic stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
      })
      streamRef.current = stream

      // Connect WebSocket with auth token
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const token = document.cookie.split('; ').find(c => c.startsWith('alfred_token='))?.split('=')[1] || ''
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/wakeword?token=${token}`)
      wsRef.current = ws

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'detected') {
            // Wake word detected — activate hands-free
            setHandsFree(true)
          }
        } catch { /* ignore */ }
      }

      ws.onclose = () => {
        // Reconnect if still active
        if (activeRef.current) {
          setTimeout(() => {
            if (activeRef.current) startWakeWord()
          }, 3000)
        }
      }

      ws.onerror = () => ws.close()

      ws.onopen = () => {
        // Set up audio processing to stream PCM to WebSocket
        const audioCtx = new AudioContext({ sampleRate: 16000 })
        audioCtxRef.current = audioCtx
        const source = audioCtx.createMediaStreamSource(stream)

        // ScriptProcessor for raw PCM access (deprecated but widely supported)
        const processor = audioCtx.createScriptProcessor(4096, 1, 1)
        processorRef.current = processor

        processor.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return
          const float32 = e.inputBuffer.getChannelData(0)
          // Convert float32 to int16 PCM
          const int16 = new Int16Array(float32.length)
          for (let i = 0; i < float32.length; i++) {
            const s = Math.max(-1, Math.min(1, float32[i]))
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
          }
          ws.send(int16.buffer)
        }

        source.connect(processor)
        processor.connect(audioCtx.destination)
      }
    } catch {
      setWakeWord(false)
    }
  }, [setHandsFree, setWakeWord])

  const stopWakeWord = useCallback(() => {
    processorRef.current?.disconnect()
    processorRef.current = null
    audioCtxRef.current?.close()
    audioCtxRef.current = null
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  useEffect(() => {
    activeRef.current = wakeWordActive
    if (wakeWordActive) {
      startWakeWord()
    } else {
      stopWakeWord()
    }
    return () => {
      if (activeRef.current) stopWakeWord()
    }
  }, [wakeWordActive]) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    wakeWordActive,
    toggle: () => setWakeWord(!wakeWordActive),
  }
}
