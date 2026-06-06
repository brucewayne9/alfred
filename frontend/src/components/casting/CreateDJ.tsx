import { useState } from 'react'
import { useCastingStore } from '../../stores/castingStore'
import { castingApi } from '../../api/casting'
import type { DJ } from '../../api/casting'

export function CreateDJ({ onDone }: { onDone: () => void }) {
  const { archetypes, moodPack, refresh } = useCastingStore()
  const [name, setName] = useState('')
  const [brief, setBrief] = useState('')
  const [archetypeId, setArchetypeId] = useState('')
  const [persona, setPersona] = useState('')
  const [dj, setDj] = useState<DJ | null>(null)
  const [busy, setBusy] = useState(false)

  const draft = async () => {
    setBusy(true)
    try {
      const d = await castingApi.draftPersona(name, brief, archetypeId || undefined)
      setPersona(d.persona_prompt)
    } finally { setBusy(false) }
  }
  const save = async () => {
    const created = await castingApi.createDJ({
      name, role: 'host', persona_prompt: persona,
      archetype_tags: archetypeId ? [archetypeId] : [], voice_source: 'recorded',
    })
    setDj(created); await refresh()
  }
  const upload = async (mood: string, file: File) => {
    if (!dj) return
    const updated = await castingApi.uploadMood(dj.id, mood, file)
    setDj(updated); await refresh()
  }

  return (
    <div className="create-dj">
      <div className="create-dj-form">
        <label>Name<input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Sloan Rivera" /></label>
        <label>Archetype
          <select value={archetypeId} onChange={e => setArchetypeId(e.target.value)}>
            <option value="">(none)</option>
            {archetypes.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
          </select>
        </label>
        <label>Brief
          <textarea value={brief} onChange={e => setBrief(e.target.value)} rows={3}
            placeholder="warm Latina, mid-30s, sales-driven, sports-obsessed, big-sister energy" />
        </label>
        <button className="btn-primary" disabled={busy || !name || !brief} onClick={draft}>
          {busy ? 'Drafting…' : 'Draft persona'}
        </button>
        <textarea className="persona-out" value={persona} onChange={e => setPersona(e.target.value)} rows={12}
          placeholder="The drafted persona brief appears here. Edit freely before saving." />
        <button className="btn-primary" disabled={!name || !persona} onClick={save}>{dj ? 'Saved ✓' : 'Save DJ'}</button>
      </div>

      {dj && (
        <section className="mood-capture">
          <h3>Record the Mood Pack — same mic, same distance, one session</h3>
          {moodPack.map(m => {
            const captured = dj.moods_present.includes(m.mood)
            return (
              <div className={`mood-row${captured ? ' captured' : ''}`} key={m.mood}>
                <div className="mood-meta">
                  <strong>{m.label}</strong>
                  <em>{m.direction}</em>
                  <p>{m.script}</p>
                </div>
                <div className="mood-upload">
                  {captured ? <span className="mood-done">✓ captured</span> : null}
                  <input type="file" accept="audio/*"
                    onChange={e => e.target.files?.[0] && upload(m.mood, e.target.files[0])} />
                </div>
              </div>
            )
          })}
          <button className="btn-primary" onClick={onDone}>Done</button>
        </section>
      )}
    </div>
  )
}
