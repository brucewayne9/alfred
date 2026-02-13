import { apiFetch, apiStream } from './client'

export interface ChatResponse {
  response: string
  tier: string
  timestamp: string
  images?: Array<{ base64: string; filename: string; download_url?: string }>
  ui_action?: { action: string; value: unknown }
}

export interface ChatRequest {
  message: string
  session_id: string
  tier?: string
  image_base64?: string
  image_media_type?: string
  image_path?: string
  document_path?: string
}

export const chatApi = {
  send: (req: ChatRequest) =>
    apiFetch<ChatResponse>('/chat', { method: 'POST', body: JSON.stringify(req) }),

  stream: (req: ChatRequest, onChunk: (text: string) => void) =>
    apiStream('/chat/stream', req, onChunk),
}
