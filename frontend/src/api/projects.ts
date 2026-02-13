import { apiFetch, apiUpload } from './client'

export interface Project {
  id: string
  name: string
  description: string
  color: string
  created_at: string
}

export interface Reference {
  id: number
  project_id: string
  type: 'note' | 'file'
  title: string
  content?: string
  file_path?: string
  mime_type?: string
  file_size?: number
  created_at: string
}

export const projectsApi = {
  list: () => apiFetch<Project[]>('/projects'),

  get: (id: string) => apiFetch<Project>(`/projects/${id}`),

  create: (name: string, description = '', color = '#3b82f6') =>
    apiFetch<Project>('/projects', {
      method: 'POST',
      body: JSON.stringify({ name, description, color }),
    }),

  update: (id: string, data: { name?: string; description?: string; color?: string }) =>
    apiFetch(`/projects/${id}`, { method: 'PUT', body: JSON.stringify(data) }),

  delete: (id: string) => apiFetch(`/projects/${id}`, { method: 'DELETE' }),

  listReferences: (projectId: string) =>
    apiFetch<Reference[]>(`/projects/${projectId}/references`),

  addNote: (projectId: string, title: string, content: string) =>
    apiFetch<Reference>(`/projects/${projectId}/references`, {
      method: 'POST',
      body: JSON.stringify({ title, content }),
    }),

  uploadFile: (projectId: string, file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return apiUpload<Reference>(`/projects/${projectId}/references/upload`, formData)
  },

  searchReferences: (projectId: string, q: string) =>
    apiFetch<Reference[]>(`/projects/${projectId}/references/search?q=${encodeURIComponent(q)}`),

  listConversations: (projectId: string) =>
    apiFetch(`/projects/${projectId}/conversations`),
}
