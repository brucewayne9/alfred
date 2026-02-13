import { create } from 'zustand'
import { conversationsApi, type Conversation } from '../api/conversations'
import { projectsApi, type Project } from '../api/projects'

interface SidebarState {
  isOpen: boolean
  conversations: Conversation[]
  projects: Project[]
  searchQuery: string
  searchResults: Conversation[]
  isSearching: boolean
  isLoading: boolean

  toggle: () => void
  open: () => void
  close: () => void
  loadConversations: () => Promise<void>
  loadProjects: () => Promise<void>
  search: (query: string) => Promise<void>
  clearSearch: () => void
  archiveConversation: (id: string) => Promise<void>
  deleteConversation: (id: string) => Promise<void>
  renameConversation: (id: string, title: string) => Promise<void>
  createProject: (name: string, color?: string) => Promise<void>
  deleteProject: (id: string) => Promise<void>
}

export const useSidebarStore = create<SidebarState>((set, get) => ({
  isOpen: false,
  conversations: [],
  projects: [],
  searchQuery: '',
  searchResults: [],
  isSearching: false,
  isLoading: false,

  toggle: () => set(s => ({ isOpen: !s.isOpen })),
  open: () => set({ isOpen: true }),
  close: () => set({ isOpen: false }),

  loadConversations: async () => {
    set({ isLoading: true })
    try {
      const convs = await conversationsApi.list(50)
      set({ conversations: convs, isLoading: false })
    } catch {
      set({ isLoading: false })
    }
  },

  loadProjects: async () => {
    try {
      const projects = await projectsApi.list()
      set({ projects })
    } catch { /* ignore */ }
  },

  search: async (query) => {
    set({ searchQuery: query })
    if (query.length < 2) {
      set({ searchResults: [], isSearching: false })
      return
    }
    set({ isSearching: true })
    try {
      const results = await conversationsApi.search(query)
      set({ searchResults: results, isSearching: false })
    } catch {
      set({ isSearching: false })
    }
  },

  clearSearch: () => set({ searchQuery: '', searchResults: [], isSearching: false }),

  archiveConversation: async (id) => {
    await conversationsApi.archive(id)
    set(s => ({ conversations: s.conversations.filter(c => c.id !== id) }))
  },

  deleteConversation: async (id) => {
    await conversationsApi.deletePermanently(id)
    set(s => ({ conversations: s.conversations.filter(c => c.id !== id) }))
  },

  renameConversation: async (id, title) => {
    await conversationsApi.rename(id, title)
    set(s => ({
      conversations: s.conversations.map(c => c.id === id ? { ...c, title } : c),
    }))
  },

  createProject: async (name, color = '#3b82f6') => {
    try {
      const project = await projectsApi.create(name, '', color)
      set(s => ({ projects: [...s.projects, project] }))
    } catch { /* ignore */ }
  },

  deleteProject: async (id) => {
    await projectsApi.delete(id)
    set(s => ({ projects: s.projects.filter(p => p.id !== id) }))
  },
}))
