import { create } from 'zustand'
import { chatApi, type ChatResponse } from '../api/chat'
import { conversationsApi, type Conversation, type Message } from '../api/conversations'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  tier?: string
  timestamp: string
  images?: ChatResponse['images']
  isStreaming?: boolean
}

interface ChatState {
  messages: ChatMessage[]
  currentConversationId: string | null
  isThinking: boolean
  error: string | null

  sendMessage: (text: string, imageBase64?: string, imageMediaType?: string, documentPath?: string) => Promise<void>
  loadConversation: (id: string) => Promise<void>
  newChat: () => void
  clearError: () => void
}

let messageCounter = 0

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  currentConversationId: null,
  isThinking: false,
  error: null,

  sendMessage: async (text, imageBase64, imageMediaType, documentPath) => {
    const state = get()
    let convId = state.currentConversationId

    // Create conversation if needed
    if (!convId) {
      try {
        const conv = await conversationsApi.create()
        convId = conv.id
        set({ currentConversationId: convId })
      } catch (e) {
        set({ error: 'Failed to create conversation' })
        return
      }
    }

    // Add user message to UI
    const userMsg: ChatMessage = {
      id: `msg-${++messageCounter}`,
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    }
    set(s => ({ messages: [...s.messages, userMsg], isThinking: true, error: null }))

    // Add placeholder for streaming
    const assistantId = `msg-${++messageCounter}`
    set(s => ({
      messages: [...s.messages, {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        isStreaming: true,
      }],
    }))

    try {
      // Try streaming first
      let fullText = ''
      await chatApi.stream(
        {
          message: text,
          session_id: convId!,
          image_base64: imageBase64,
          image_media_type: imageMediaType,
          document_path: documentPath,
        },
        (chunk) => {
          fullText += chunk
          set(s => ({
            messages: s.messages.map(m =>
              m.id === assistantId ? { ...m, content: fullText } : m,
            ),
          }))
        },
      )

      // Finalize
      set(s => ({
        messages: s.messages.map(m =>
          m.id === assistantId ? { ...m, isStreaming: false } : m,
        ),
        isThinking: false,
      }))
    } catch {
      // Fallback to non-streaming
      try {
        const res = await chatApi.send({
          message: text,
          session_id: convId!,
          image_base64: imageBase64,
          image_media_type: imageMediaType,
          document_path: documentPath,
        })
        set(s => ({
          messages: s.messages.map(m =>
            m.id === assistantId
              ? { ...m, content: res.response, tier: res.tier, images: res.images, isStreaming: false }
              : m,
          ),
          isThinking: false,
        }))
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Failed to send message'
        set(s => ({
          messages: s.messages.filter(m => m.id !== assistantId),
          isThinking: false,
          error: msg,
        }))
      }
    }
  },

  loadConversation: async (id) => {
    try {
      const conv = await conversationsApi.get(id)
      const msgs: ChatMessage[] = (conv.messages || []).map((m: Message, i: number) => ({
        id: `loaded-${i}`,
        role: m.role,
        content: m.content,
        tier: m.tier,
        timestamp: m.timestamp,
      }))
      set({ currentConversationId: id, messages: msgs, error: null })
    } catch {
      set({ error: 'Failed to load conversation' })
    }
  },

  newChat: () => set({ currentConversationId: null, messages: [], error: null }),

  clearError: () => set({ error: null }),
}))
