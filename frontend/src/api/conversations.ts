import { apiFetch } from './client'

export interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  tier: string
  timestamp: string
}

export interface Conversation {
  id: string
  title: string
  created_at: string
  updated_at: string
  archived: boolean
  project_id?: string | null
  messages?: Message[]
}

export const conversationsApi = {
  list: (limit = 50, offset = 0, project_id?: string) => {
    let url = `/conversations?limit=${limit}&offset=${offset}`
    if (project_id) url += `&project_id=${project_id}`
    return apiFetch<Conversation[]>(url)
  },

  get: (id: string) => apiFetch<Conversation>(`/conversations/${id}`),

  create: () => apiFetch<Conversation>('/conversations', { method: 'POST' }),

  archive: (id: string) => apiFetch(`/conversations/${id}`, { method: 'DELETE' }),

  rename: (id: string, title: string) =>
    apiFetch(`/conversations/${id}/title`, {
      method: 'PUT',
      body: JSON.stringify({ title }),
    }),

  moveToProject: (id: string, project_id: string | null) =>
    apiFetch(`/conversations/${id}/project`, {
      method: 'PUT',
      body: JSON.stringify({ project_id }),
    }),

  restore: (id: string) =>
    apiFetch(`/conversations/${id}/restore`, { method: 'POST' }),

  deletePermanently: (id: string) =>
    apiFetch(`/conversations/${id}/permanent`, { method: 'DELETE' }),

  search: (q: string, limit = 20) =>
    apiFetch<Conversation[]>(`/conversations/search?q=${encodeURIComponent(q)}&limit=${limit}`),

  archived: (limit = 50, offset = 0) =>
    apiFetch<Conversation[]>(`/conversations/archived?limit=${limit}&offset=${offset}`),
}
