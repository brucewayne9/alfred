import { useState, useRef, useCallback, useEffect } from 'react'
import { Send, Paperclip, Loader2 } from 'lucide-react'
import { useChatStore } from '../../stores/chatStore'
import { useChat } from '../../hooks/useChat'
import { useVoice } from '../../hooks/useVoice'
import { useVoiceStore } from '../../stores/voiceStore'
import { MicButton } from '../voice/MicButton'
import { FilePreview } from './FilePreview'
import { apiUpload } from '../../api/client'

interface UploadedFile {
  name: string
  type: 'image' | 'document'
  // For images
  base64?: string
  mediaType?: string
  // For documents
  documentPath?: string
  textPreview?: string
}

const IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
const DOC_EXTENSIONS = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.csv', '.txt', '.md', '.json']

export function ChatInput() {
  const [text, setText] = useState('')
  const [file, setFile] = useState<UploadedFile | null>(null)
  const [uploading, setUploading] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const isThinking = useChatStore(s => s.isThinking)
  const { send } = useChat()
  const voice = useVoice()
  const transcribedText = useVoiceStore(s => s.transcribedText)

  // Insert transcribed text
  useEffect(() => {
    if (transcribedText) {
      setText(prev => prev ? `${prev} ${transcribedText}` : transcribedText)
      voice.clearTranscribedText()
      textareaRef.current?.focus()
    }
  }, [transcribedText, voice])

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }, [text])

  const handleSend = useCallback(async () => {
    const msg = text.trim()
    if (!msg && !file) return
    setText('')
    const currentFile = file
    setFile(null)
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    if (currentFile?.type === 'image') {
      await send(msg || 'Describe this image.', currentFile.base64, currentFile.mediaType)
    } else if (currentFile?.type === 'document') {
      // Prepend document context to message so LLM can read it
      const docMsg = `[Uploaded document: ${currentFile.name}]\n${currentFile.textPreview}\n\n${msg || 'Please analyze this document.'}`
      await send(docMsg)
    } else {
      await send(msg)
    }
  }, [text, file, send])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }, [handleSend])

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    e.target.value = '' // Reset input

    const ext = '.' + f.name.split('.').pop()?.toLowerCase()
    const isImage = IMAGE_TYPES.includes(f.type)
    const isDoc = DOC_EXTENSIONS.includes(ext)

    if (isImage) {
      // Read as base64 for inline vision
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        const base64 = result.split(',')[1]
        setFile({ name: f.name, type: 'image', base64, mediaType: f.type })
      }
      reader.readAsDataURL(f)
    } else if (isDoc) {
      // Upload to server, get parsed text back
      setUploading(true)
      try {
        const formData = new FormData()
        formData.append('file', f)
        const res = await apiUpload<{
          path: string
          filename: string
          text_preview: string
          error: string | null
        }>('/upload/document', formData)

        if (res.error) {
          alert(`Error reading file: ${res.error}`)
        } else {
          setFile({
            name: f.name,
            type: 'document',
            documentPath: res.path,
            textPreview: res.text_preview,
          })
        }
      } catch {
        alert('Failed to upload document')
      } finally {
        setUploading(false)
      }
    } else {
      alert('Unsupported file type. Supported: images, PDF, DOCX, XLSX, CSV, TXT, MD, JSON')
    }
  }, [])

  return (
    <div className="shrink-0 px-4 pb-4 safe-bottom">
      <div className="max-w-3xl mx-auto">
        {file && (
          <FilePreview name={file.name} onRemove={() => setFile(null)} />
        )}
        <div className="flex items-end gap-2 bg-alfred-input border border-alfred-border rounded-2xl px-3 py-2 focus-within:border-alfred-accent/50 transition-colors">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="p-1.5 rounded-lg text-alfred-muted hover:text-white hover:bg-alfred-hover transition-colors shrink-0 mb-0.5 disabled:opacity-50"
            title="Attach file"
          >
            {uploading ? <Loader2 size={18} className="animate-spin" /> : <Paperclip size={18} />}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.pdf,.doc,.docx,.xls,.xlsx,.csv,.txt,.md,.json"
            onChange={handleFileSelect}
            className="hidden"
          />

          <textarea
            ref={textareaRef}
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message Alfred..."
            rows={1}
            className="flex-1 bg-transparent text-white text-sm resize-none outline-none placeholder-alfred-muted py-1.5 max-h-[200px]"
          />

          <MicButton />

          <button
            onClick={handleSend}
            disabled={isThinking || (!text.trim() && !file)}
            className="p-1.5 rounded-lg bg-alfred-accent text-white hover:bg-orange-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0 mb-0.5"
            title="Send"
          >
            <Send size={18} />
          </button>
        </div>
        <p className="text-center text-[10px] text-alfred-muted/50 mt-2">
          Alfred can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  )
}
