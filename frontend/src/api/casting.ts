// frontend/src/api/casting.ts
import { apiFetch, apiUpload } from './client'

export interface DJ {
  id: number; name: string; role: 'host' | 'guest'; status: 'draft' | 'ready' | 'live'
  persona_prompt: string; archetype_tags: string[]; expertise: string
  voice_source: string; moods_present: string[]; avatar?: string | null
}
export interface MoodRead { mood: string; label: string; direction: string; script: string }
export interface Archetype { id: string; name: string; summary: string }
export interface Assignment { id: number; dj_id: number; dj_name: string; station_id: number; slot: string; effective_at: string; applied: boolean }
export interface PersonaDraft { persona_prompt: string; archetype_tags: string[] }

export const castingApi = {
  moodpack: () => apiFetch<{ moods: MoodRead[] }>('/api/casting/moodpack').then(d => d.moods),
  archetypes: () => apiFetch<Archetype[]>('/api/casting/archetypes'),
  listDJs: () => apiFetch<DJ[]>('/api/casting/djs'),
  createDJ: (body: Partial<DJ>) => apiFetch<DJ>('/api/casting/djs', { method: 'POST', body: JSON.stringify(body) }),
  draftPersona: (name: string, brief: string, archetype_id?: string) =>
    apiFetch<PersonaDraft>('/api/casting/persona/draft', { method: 'POST', body: JSON.stringify({ name, brief, archetype_id }) }),
  uploadMood: (djId: number, mood: string, file: File) => {
    const fd = new FormData(); fd.append('file', file)
    return apiUpload<DJ>(`/api/casting/djs/${djId}/voice/${mood}`, fd)
  },
  previewUrl: (djId: number) => `/api/casting/djs/${djId}/preview`,
  listAssignments: (stationId?: number) =>
    apiFetch<Assignment[]>(`/api/casting/assignments${stationId ? `?station_id=${stationId}` : ''}`),
  createAssignment: (dj_id: number, station_id: number, slot: string, effective_at: string) =>
    apiFetch<{ id: number }>('/api/casting/assignments', { method: 'POST', body: JSON.stringify({ dj_id, station_id, slot, effective_at }) }),
}
