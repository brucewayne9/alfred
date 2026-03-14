import { create } from 'zustand'
import {
  knowledgeApi,
  type KnowledgeStatus,
  type KnowledgeDocument,
  type KnowledgeEntity,
  type QueryResult,
} from '../api/knowledge'

interface KnowledgeState {
  status: KnowledgeStatus | null
  documents: KnowledgeDocument[]
  totalDocs: number
  entities: KnowledgeEntity[]
  queryResult: QueryResult | null
  isLoading: boolean
  isUploading: boolean
  isQuerying: boolean
  error: string | null

  loadStatus: () => Promise<void>
  loadDocuments: () => Promise<void>
  loadEntities: () => Promise<void>
  uploadText: (text: string, description: string) => Promise<void>
  uploadFile: (file: File) => Promise<void>
  deleteDocument: (docId: string) => Promise<void>
  query: (q: string, mode: string) => Promise<void>
  searchEntities: (label: string) => Promise<void>
  clearError: () => void
}

export const useKnowledgeStore = create<KnowledgeState>((set, get) => ({
  status: null,
  documents: [],
  totalDocs: 0,
  entities: [],
  queryResult: null,
  isLoading: false,
  isUploading: false,
  isQuerying: false,
  error: null,

  loadStatus: async () => {
    try {
      const status = await knowledgeApi.status()
      set({ status })
    } catch (e: any) {
      set({ error: e.message })
    }
  },

  loadDocuments: async () => {
    set({ isLoading: true })
    try {
      const result = await knowledgeApi.listDocuments(100)
      set({ documents: result.documents || [], totalDocs: result.pagination?.total || result.documents?.length || 0, isLoading: false })
    } catch (e: any) {
      set({ isLoading: false, error: e.message })
    }
  },

  loadEntities: async () => {
    try {
      const entities = await knowledgeApi.entities(50)
      set({ entities: Array.isArray(entities) ? entities : [] })
    } catch { /* ignore */ }
  },

  uploadText: async (text, description) => {
    set({ isUploading: true, error: null })
    try {
      await knowledgeApi.uploadText(text, description)
      set({ isUploading: false })
      get().loadDocuments()
      get().loadStatus()
    } catch (e: any) {
      set({ isUploading: false, error: e.message })
    }
  },

  uploadFile: async (file) => {
    set({ isUploading: true, error: null })
    try {
      await knowledgeApi.uploadFile(file)
      set({ isUploading: false })
      get().loadDocuments()
      get().loadStatus()
    } catch (e: any) {
      set({ isUploading: false, error: e.message })
    }
  },

  deleteDocument: async (docId) => {
    try {
      await knowledgeApi.deleteDocument(docId)
      set(s => ({
        documents: s.documents.filter(d => d.id !== docId),
        totalDocs: s.totalDocs - 1,
      }))
      get().loadStatus()
    } catch (e: any) {
      set({ error: e.message })
    }
  },

  query: async (q, mode) => {
    set({ isQuerying: true, error: null })
    try {
      const result = await knowledgeApi.query(q, mode)
      set({ queryResult: result, isQuerying: false })
    } catch (e: any) {
      set({ isQuerying: false, error: e.message })
    }
  },

  searchEntities: async (label) => {
    try {
      const entities = await knowledgeApi.searchEntities(label)
      set({ entities: Array.isArray(entities) ? entities : [] })
    } catch (e: any) {
      set({ error: e.message })
    }
  },

  clearError: () => set({ error: null }),
}))
