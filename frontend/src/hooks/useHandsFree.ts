import { useRef, useCallback, useEffect } from 'react'
import { useVoiceStore } from '../stores/voiceStore'
import { useChatStore } from '../stores/chatStore'
import { voiceApi } from '../api/voice'
import { playAudioBuffer } from '../lib/audio'

/**
 * Hands-free voice mode.
 * When active: continuously listens via MediaRecorder,
 * uses silence detection to detect speech end,
 * transcribes, sends to chat, and speaks the response.
 */
export function useHandsFree() {
  const { handsFreeActive, setHandsFree, setRecording } = useVoiceStore()
  const sendMessage = useChatStore(s => s.sendMessage)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const rafRef = useRef<number>(0)
  const isProcessingRef = useRef(false)
  const activeRef = useRef(false)
  const wakeLockRef = useRef<WakeLockSentinel | null>(null)

  const SILENCE_THRESHOLD = 15 // RMS below this = silence
  const SILENCE_DURATION = 1500 // ms of silence before processing

  const processRecording = useCallback(async (blob: Blob) => {
    if (blob.size < 1000) return // too small, skip
    isProcessingRef.current = true
    try {
      const text = await voiceApi.transcribe(blob)
      if (!text.trim()) return

      // Send to chat
      await sendMessage(text.trim())

      // Speak the response
      const messages = useChatStore.getState().messages
      const lastMsg = messages[messages.length - 1]
      if (lastMsg?.role === 'assistant' && lastMsg.content) {
        const audio = await voiceApi.speak(lastMsg.content)
        await playAudioBuffer(audio)
      }
    } catch {
      // ignore errors, keep listening
    } finally {
      isProcessingRef.current = false
      // Restart listening if still active
      if (activeRef.current) {
        startListening()
      }
    }
  }, [sendMessage])

  const checkSilence = useCallback(() => {
    if (!analyserRef.current || !activeRef.current) return

    const data = new Uint8Array(analyserRef.current.fftSize)
    analyserRef.current.getByteTimeDomainData(data)

    // Calculate RMS
    let sum = 0
    for (let i = 0; i < data.length; i++) {
      const val = (data[i] - 128) / 128
      sum += val * val
    }
    const rms = Math.sqrt(sum / data.length) * 100

    if (rms > SILENCE_THRESHOLD) {
      // Sound detected - reset silence timer
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current)
        silenceTimerRef.current = null
      }
      setRecording(true)
    } else if (!silenceTimerRef.current && mediaRecorderRef.current?.state === 'recording') {
      // Silence started - set timer to stop
      silenceTimerRef.current = setTimeout(() => {
        if (mediaRecorderRef.current?.state === 'recording' && !isProcessingRef.current) {
          mediaRecorderRef.current.stop()
          setRecording(false)
        }
      }, SILENCE_DURATION)
    }

    if (activeRef.current) {
      rafRef.current = requestAnimationFrame(checkSilence)
    }
  }, [setRecording])

  const startListening = useCallback(async () => {
    if (isProcessingRef.current) return

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Set up analyser for silence detection
      const audioCtx = new AudioContext()
      audioCtxRef.current = audioCtx
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 2048
      source.connect(analyser)
      analyserRef.current = analyser

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm',
      })
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mediaRecorder.mimeType })
        chunksRef.current = []
        processRecording(blob)
      }

      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start()

      // Start silence detection loop
      rafRef.current = requestAnimationFrame(checkSilence)
    } catch {
      // mic access denied
      setHandsFree(false)
    }
  }, [processRecording, checkSilence, setHandsFree])

  const stopListening = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current)
      silenceTimerRef.current = null
    }
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    audioCtxRef.current?.close()
    audioCtxRef.current = null
    analyserRef.current = null
    setRecording(false)
  }, [setRecording])

  // React to handsFreeActive toggle
  useEffect(() => {
    activeRef.current = handsFreeActive
    if (handsFreeActive) {
      startListening()
      // Request wake lock
      if ('wakeLock' in navigator) {
        (navigator as unknown as { wakeLock: { request: (type: string) => Promise<WakeLockSentinel> } })
          .wakeLock.request('screen')
          .then(lock => { wakeLockRef.current = lock })
          .catch(() => {})
      }
    } else {
      stopListening()
      wakeLockRef.current?.release()
      wakeLockRef.current = null
    }
    return () => {
      if (activeRef.current) stopListening()
    }
  }, [handsFreeActive]) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    handsFreeActive,
    toggle: () => setHandsFree(!handsFreeActive),
  }
}
