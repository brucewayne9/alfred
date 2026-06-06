import { create } from 'zustand'
import { castingApi, type DJ, type Archetype, type MoodRead } from '../api/casting'

interface CastingState {
  djs: DJ[]; archetypes: Archetype[]; moodPack: MoodRead[]; loading: boolean
  refresh: () => Promise<void>
  loadStatic: () => Promise<void>
}

export const useCastingStore = create<CastingState>((set) => ({
  djs: [], archetypes: [], moodPack: [], loading: false,
  refresh: async () => {
    set({ loading: true })
    try { set({ djs: await castingApi.listDJs() }) } finally { set({ loading: false }) }
  },
  loadStatic: async () => {
    const [archetypes, moodPack] = await Promise.all([castingApi.archetypes(), castingApi.moodpack()])
    set({ archetypes, moodPack })
  },
}))
