import { apiFetch, apiUpload } from './client'

export interface KnowledgeStatus {
  connected: boolean
  error?: string
  document_counts?: Record<string, number>
  details?: Record<string, unknown>
}

export interface KnowledgeDocument {
  id: string
  file_path?: string | null
  content_summary?: string
  content_length?: number
  status?: string
  created_at?: string
  updated_at?: string
  chunks_count?: number | null
  error_msg?: string | null
  metadata?: Record<string, unknown> | null
  [key: string]: unknown
}

export interface KnowledgeDocuments {
  documents: KnowledgeDocument[]
  pagination?: { total: number; page: number; page_size: number; total_pages: number }
  status_counts?: Record<string, number>
}

export interface KnowledgeEntity {
  label: string
  type?: string
  description?: string
  count?: number
  [key: string]: unknown
}

export interface QueryResult {
  response?: string
  sources?: unknown[]
  [key: string]: unknown
}

export const knowledgeApi = {
  status: () => apiFetch<KnowledgeStatus>('/knowledge/status'),

  listDocuments: (limit = 20, offset = 0) =>
    apiFetch<KnowledgeDocuments>(`/knowledge/documents?limit=${limit}&offset=${offset}`),

  uploadText: (text: string, description = '') =>
    apiFetch<{ message: string; result: unknown }>('/knowledge/upload/text', {
      method: 'POST',
      body: JSON.stringify({ text, description }),
    }),

  uploadFile: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return apiUpload<{ message: string; result: unknown }>('/knowledge/upload/file', fd)
  },

  query: (query: string, mode = 'hybrid', top_k = 10) =>
    apiFetch<QueryResult>('/knowledge/query', {
      method: 'POST',
      body: JSON.stringify({ query, mode, top_k }),
    }),

  entities: (limit = 20) =>
    apiFetch<KnowledgeEntity[]>(`/knowledge/entities?limit=${limit}`),

  searchEntities: (label: string) =>
    apiFetch<KnowledgeEntity[]>(`/knowledge/search?label=${encodeURIComponent(label)}`),

  deleteDocument: (docId: string) =>
    apiFetch<{ message: string }>(`/knowledge/document/${encodeURIComponent(docId)}`, {
      method: 'DELETE',
    }),
}
